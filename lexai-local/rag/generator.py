"""LLM-2 legal reasoning generator for LexAI."""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    Llama = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_LLM2_MODEL = str(MODELS_DIR / "mistral-7b-instruct.Q4_K_M.gguf")
LLM2_MODEL_PATH = os.environ.get("LEXAI_LLM2_MODEL", DEFAULT_LLM2_MODEL).strip()

USE_LLM2 = os.environ.get("LEXAI_USE_LLM", "1").strip().lower() not in {"0", "false", "no"}
LLM2_POLISH_ENABLED = os.environ.get("LEXAI_LLM2_POLISH_ENABLED", "1").strip().lower() in {"1", "true", "yes"}
DEFAULT_PROFILE = os.environ.get("LEXAI_PROFILE", os.environ.get("LEXAI_MODEL_PROFILE", "balanced")).strip().lower()

N_CTX = int(os.environ.get("LEXAI_LLM2_N_CTX", "3072"))
N_THREADS = int(os.environ.get("LEXAI_LLM2_N_THREADS", str(max(4, min(12, (os.cpu_count() or 8) - 1)))))
N_BATCH = int(os.environ.get("LEXAI_LLM2_N_BATCH", "256"))

PROFILE_CFG = {
    "balanced": {
        "max_tokens": int(os.environ.get("LEXAI_LLM2_MAX_TOKENS_BALANCED", os.environ.get("LEXAI_LLM2_MAX_TOKENS", "180"))),
        "temperature": float(os.environ.get("LEXAI_LLM2_TEMP_BALANCED", "0.12")),
        "top_p": float(os.environ.get("LEXAI_LLM2_TOP_P_BALANCED", "0.9")),
        "repeat_penalty": float(os.environ.get("LEXAI_LLM2_REPEAT_PENALTY_BALANCED", "1.12")),
        "timeout_s": float(os.environ.get("LEXAI_LLM2_TIMEOUT_BALANCED", "12")),
    },
    "high_accuracy": {
        "max_tokens": int(os.environ.get("LEXAI_LLM2_MAX_TOKENS_HIGH", os.environ.get("LEXAI_LLM2_MAX_TOKENS", "180"))),
        "temperature": float(os.environ.get("LEXAI_LLM2_TEMP_HIGH", "0.08")),
        "top_p": float(os.environ.get("LEXAI_LLM2_TOP_P_HIGH", "0.9")),
        "repeat_penalty": float(os.environ.get("LEXAI_LLM2_REPEAT_PENALTY_HIGH", "1.10")),
        "timeout_s": float(os.environ.get("LEXAI_LLM2_TIMEOUT_HIGH", "14")),
    },
}
MAX_CONTEXT_FOR_LLM2 = int(os.environ.get("LEXAI_MAX_CONTEXT_FOR_LLM2", "4500"))

_ACTIVE_LLM: Optional[Any] = None
_ACTIVE_MODEL_PATH: Optional[Path] = None
_INIT_ERROR = "" if Llama is not None else "llama_cpp is not installed. Install llama-cpp-python."


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_profile(profile: str | None) -> str:
    p = (profile or DEFAULT_PROFILE or "balanced").strip().lower()
    return p if p in PROFILE_CFG else "balanced"


def _extract_refs_from_context(context: str) -> List[str]:
    refs: List[str] = []
    seen = set()
    for line in (context or "").splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^\[\d+\]\s*(.+?)\s*\|\s*(.+)$", s)
        if m:
            ref = f"{m.group(1).strip()} - {m.group(2).strip()}"
            k = ref.lower()
            if k not in seen:
                seen.add(k)
                refs.append(ref)
    return refs[:10]


def _dedupe_refs(refs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for ref in refs:
        r = _clean_text(ref)
        if not r:
            continue
        k = r.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _fallback(reason: str, refs: List[str]) -> str:
    refs_block = "\n".join([f"- {r}" for r in refs]) if refs else "- Not available"
    return (
        "Answer:\n"
        "The answer is not found in the provided legal context.\n\n"
        "Relevant Sections:\n"
        f"{refs_block}\n\n"
        "Legal Interpretation:\n"
        f"{reason}\n\n"
        "Conclusion:\n"
        "Use the cited sections and verify the primary legal text."
    )


def _extractive_draft(question: str, context: str, refs: List[str], analysis: Dict | None = None) -> str:
    """Produce a conservative answer directly from ranked evidence when no local LLM is available."""
    q = _clean_text(question).lower()
    raw_lines = [x.strip() for x in (context or "").splitlines() if x.strip()]
    evidence: List[str] = []
    for line in raw_lines:
        if not line.startswith("- "):
            continue
        item = _clean_text(line[2:])
        if len(item) < 35:
            continue
        low = item.lower()
        if any(noise in low for noise in ["signature and seal", "ditto", "on or about the day of"]):
            continue
        evidence.append(item)

    # Preserve ranking while removing duplicate/near-duplicate evidence.
    unique: List[str] = []
    seen = set()
    for item in evidence:
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    evidence = unique[:5]

    act = _clean_text(str((analysis or {}).get("possible_act", "") or ""))
    domain = _clean_text(str((analysis or {}).get("legal_domain", "") or ""))
    intent = _clean_text(str((analysis or {}).get("intent", "") or ""))
    sections = [str(x).upper() for x in ((analysis or {}).get("possible_sections") or [])]

    if not evidence:
        summary = "The retrieved context does not contain a usable evidence passage for this question."
    else:
        # Prefer passages that answer the user's actual intent.
        intent_terms = []
        if intent in {"penalty", "environmental_offence"}:
            intent_terms = ["penalty", "fine", "punishable", "offence", "offense", "imprisonment", "compensation"]
        elif intent == "compliance_remedy":
            intent_terms = ["permission", "clearance", "prohibited", "shall", "required", "offence", "penalty", "restoration", "compensation"]
        elif "what" in q or "how" in q:
            intent_terms = ["shall", "must", "required", "permission", "procedure", "penalty", "offence"]
        ranked = sorted(
            evidence,
            key=lambda s: (
                sum(1 for term in intent_terms if term in s.lower()),
                sum(1 for term in ["section", "act", "shall", "liable", "punishable"] if term in s.lower()),
            ),
            reverse=True,
        )
        selected = ranked[:3]
        summary = " ".join(selected)
        if act:
            summary = f"Based on the retrieved {act} material, {summary}"
        elif domain:
            summary = f"Based on the retrieved {domain.replace('_', ' ')} material, {summary}"

    refs_block = "\n".join([f"- {r}" for r in refs]) if refs else "- Not available"
    section_hint = ", ".join(sections[:4]) if sections else "Not explicitly identified"
    if intent == "compliance_remedy":
        conclusion = (
            "The retrieved text does not establish a personalized next-step procedure. "
            "It should not be used to infer a permit, settlement, or penalty that is not stated in the cited evidence."
        )
    else:
        conclusion = "The answer above is limited to the retrieved evidence; verify the complete primary provision before relying on it."

    return (
        "Answer:\n"
        f"{summary}\n\n"
        "Relevant Sections:\n"
        f"{refs_block}\n\n"
        "Legal Interpretation:\n"
        f"Detected domain: {domain or 'general'}; intent: {intent or 'general'}; section hints: {section_hint}. "
        "Only retrieved evidence is used; no unsupported statutory provision is added.\n\n"
        "Conclusion:\n"
        f"{conclusion}"
    )

def _build_prompt(question: str, context: str, refs: List[str], target_words: int, confidence_bucket: str) -> str:
    refs_block = "\n".join([f"- {r}" for r in refs]) if refs else "- Not available"
    caution = "Use cautious wording and start Answer with 'As per retrieved context,'." if confidence_bucket == "medium" else ""
    return f"""[INST]
You are a legal assistant specializing in Indian law.
Use ONLY the legal context provided.
Do not invent legal provisions.
If the answer is not clearly present in context, say: "The answer is not found in the provided legal context."
{caution}

Question:
{question}

Legal Context:
{context}

Citations:
{refs_block}

Instructions:
1. Identify relevant act and section.
2. Explain the law clearly.
3. State penalty/provision only if present in context.

Output format (strict):
Answer:
...

Relevant Sections:
- Act Name - Section Number

Legal Interpretation:
...

Conclusion:
...
[/INST]"""


def _normalize_output(text: str, refs: List[str]) -> str:
    cleaned = _clean_text(text)
    required = ["Answer:", "Relevant Sections:", "Legal Interpretation:", "Conclusion:"]
    if all(x in cleaned for x in required):
        return cleaned
    refs_block = "\n".join([f"- {r}" for r in refs]) if refs else "- Not available"
    core = cleaned or "The answer is not found in the provided legal context."
    return (
        "Answer:\n"
        f"{core}\n\n"
        "Relevant Sections:\n"
        f"{refs_block}\n\n"
        "Legal Interpretation:\n"
        "This response is constrained to retrieved legal context.\n\n"
        "Conclusion:\n"
        "Verify exact statutory language in cited provisions."
    )


def _looks_bad_output(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 60:
        return True
    low = t.lower()
    if low.count("section") > 18 and len(low) > 1400:
        return True
    if "legal context start" in low or "next section" in low:
        return True
    return False


def _load_llm2() -> Optional[Any]:
    global _ACTIVE_LLM, _ACTIVE_MODEL_PATH, _INIT_ERROR
    if Llama is None:
        _INIT_ERROR = "llama_cpp is not installed. Install llama-cpp-python."
        return None
    model_path = Path(LLM2_MODEL_PATH)
    if not model_path.exists():
        _INIT_ERROR = f"LLM2 model missing: {model_path}"
        return None
    if _ACTIVE_LLM is not None and _ACTIVE_MODEL_PATH == model_path:
        return _ACTIVE_LLM

    _ACTIVE_LLM = None
    _ACTIVE_MODEL_PATH = None
    gc.collect()
    try:
        _ACTIVE_LLM = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            n_gpu_layers=0,
            verbose=False,
        )
        _ACTIVE_MODEL_PATH = model_path
        _INIT_ERROR = ""
        return _ACTIVE_LLM
    except Exception as exc:
        _INIT_ERROR = str(exc)
        _ACTIVE_LLM = None
        return None


def _stream_infer(
    llm: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    timeout_s: float,
) -> Tuple[str, bool]:
    import time

    parts: List[str] = []
    timed_out = False
    t0 = time.perf_counter()
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=["</s>", "[INST]"],
        stream=True,
    )
    for item in stream:
        txt = str(item.get("choices", [{}])[0].get("text", ""))
        if txt:
            parts.append(txt)
        if (time.perf_counter() - t0) > timeout_s:
            timed_out = True
            break
    return "".join(parts).strip(), timed_out


def generate_answer(
    context: str,
    question: str,
    source_refs: List[str] | None = None,
    style: str = "normal",
    target_words: int = 140,
    confidence_reason: str = "",
    confidence_bucket: str = "high",
    profile: str | None = None,
    analysis: Dict | None = None,
) -> str:
    """Generate answer from deterministic draft + optional LLM2 polish."""
    q = _clean_text(question)
    if not q:
        raise ValueError("question cannot be empty")
    ctx = _clean_text(context)
    refs = _dedupe_refs(list(source_refs or _extract_refs_from_context(context)))
    if len(ctx) < 120:
        return _fallback("Retrieved context is too short for reliable legal interpretation.", refs)

    draft = _extractive_draft(q, context, refs, analysis=analysis)
    p = _safe_profile(profile)
    cfg = PROFILE_CFG[p]

    if not USE_LLM2:
        return draft
    if not LLM2_POLISH_ENABLED:
        return draft
    if confidence_bucket == "low":
        return draft

    llm = _load_llm2()
    if llm is None:
        return draft

    llm_context = context if len(context) <= MAX_CONTEXT_FOR_LLM2 else context[:MAX_CONTEXT_FOR_LLM2]

    prompt = _build_prompt(
        question=q,
        context=llm_context,
        refs=refs,
        target_words=max(40, min(320, int(target_words or 140))),
        confidence_bucket=confidence_bucket,
    )

    raw, timed_out = _stream_infer(
        llm=llm,
        prompt=prompt,
        max_tokens=int(cfg["max_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        repeat_penalty=float(cfg["repeat_penalty"]),
        timeout_s=float(cfg["timeout_s"]),
    )
    if raw.startswith(prompt):
        raw = raw[len(prompt) :].strip()
    if timed_out:
        return draft
    if re.search(r"the answer is not found in the provided legal context", raw, flags=re.IGNORECASE):
        return draft
    if _looks_bad_output(raw):
        return draft

    polished = _normalize_output(raw, refs)
    if len(polished) < len(draft) * 0.7:
        return draft
    return polished


def generator_status() -> Dict:
    model_path = Path(LLM2_MODEL_PATH)
    ready = bool(model_path.exists() or not USE_LLM2)
    return {
        "ready": ready,
        "error": _INIT_ERROR if not ready else "",
        "use_llm2": USE_LLM2,
        "llm2_polish_enabled": LLM2_POLISH_ENABLED,
        "default_profile": _safe_profile(DEFAULT_PROFILE),
        "llm2_model": str(model_path),
        "active_model_path": str(_ACTIVE_MODEL_PATH) if _ACTIVE_MODEL_PATH else "",
        "max_context_for_llm2": MAX_CONTEXT_FOR_LLM2,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_batch": N_BATCH,
    }
