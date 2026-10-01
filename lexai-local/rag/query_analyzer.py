"""LLM-1 query analyzer for LexAI (deterministic JSON hints)."""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    Llama = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_LLM1_MODEL = str(MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf")
LLM1_MODEL_PATH = os.environ.get("LEXAI_LLM1_MODEL", DEFAULT_LLM1_MODEL).strip()
LLM1_N_CTX = int(os.environ.get("LEXAI_LLM1_N_CTX", "2048"))
LLM1_N_THREADS = int(os.environ.get("LEXAI_LLM1_N_THREADS", str(max(2, min(8, (os.cpu_count() or 8) - 1)))))
LLM1_N_BATCH = int(os.environ.get("LEXAI_LLM1_N_BATCH", "128"))
LLM1_MAX_TOKENS = int(os.environ.get("LEXAI_LLM1_MAX_TOKENS", "72"))
LLM1_TIMEOUT_S = float(os.environ.get("LEXAI_LLM1_TIMEOUT_S", "4"))
LLM1_USE = os.environ.get("LEXAI_USE_LLM1", "1").strip().lower() not in {"0", "false", "no"}
LLM1_ONLY_AMBIGUOUS = os.environ.get("LEXAI_LLM1_ONLY_AMBIGUOUS", "1").strip().lower() not in {"0", "false", "no"}
ALLOW_LLM1_IN_FAST = os.environ.get("LEXAI_ALLOW_LLM1_IN_FAST", "1").strip().lower() in {"1", "true", "yes"}
FAST_MODE = os.environ.get("LEXAI_FAST_MODE", "1").strip().lower() not in {"0", "false", "no"}

ACT_ALIASES = {
    "Indian Penal Code": ["ipc", "indian penal code", "penal code", "bns", "bharatiya nyaya sanhita"],
    "Code of Criminal Procedure": ["crpc", "code of criminal procedure", "bnss"],
    "Code of Civil Procedure": ["cpc", "code of civil procedure"],
    "Motor Vehicles Act": ["motor vehicles act", "mv act", "mva", "traffic act"],
    "Indian Contract Act": ["contract act", "indian contract act"],
    "Companies Act": ["companies act", "company law", "corporate law"],
    "Constitution of India": ["constitution", "constitution of india", "article"],
    "Muslim Personal Law (Shariat) Application Act": ["muslim personal law", "shariat"],
    "Environment (Protection) Act": ["environment protection act", "environment protection", "environmental protection", "environment department"],
    "Forest (Conservation) Act": ["forest conservation act", "forest conservation", "forest clearance"],
    "Indian Forest Act": ["indian forest act", "forest act", "forest offence", "reserved forest", "protected forest"],
    "Wild Life (Protection) Act": ["wildlife protection act", "wild life protection", "wildlife offence", "protected species"],
    "Biological Diversity Act": ["biological diversity act", "biodiversity act"],
}

QUERY_REWRITE_RULES = [
    (r"\bhalf\s*murder\b", "attempt to murder ipc section 307 punishment"),
    (r"\bhelmet\s*fine\b", "motor vehicles act helmet penalty section 129 section 177 section 194d"),
    (r"\bnot\s+wearing\s+(a\s+)?helmet\b", "motor vehicles act helmet penalty section 129 section 177"),
    (r"\bno\s*helmet\b", "motor vehicles act helmet penalty section 129"),
    (
        r"\b(kiss(?:ed|ing)?|physical contact)\b.*\b(without|no)\b.*\b(consent|permission)\b",
        "indian penal code section 354A punishment without consent",
    ),
    (
        r"\b(run\s*over|ran\s*over|hit\s*and\s*run|road\s*accident|rash\s*driving|negligent\s*driving)\b",
        "indian penal code section 279 section 304A punishment for rash and negligent driving causing death",
    ),
]

TYPO_MAP = {
    "acident": "accident",
    "accidant": "accident",
    "punisment": "punishment",
    "punishmnt": "punishment",
    "helmate": "helmet",
    "helment": "helmet",
    "permision": "permission",
    "conscent": "consent",
    "agrement": "agreement",
    "contrct": "contract",
    "muder": "murder",
    "ipc302": "ipc 302",
}

_LLM1: Optional[Any] = None
_INIT_ERROR = "" if Llama is not None else "llama_cpp is not installed. Analyzer will use fallback parser."


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_intent(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["kiss", "consent", "sexual harassment", "molestation", "outraging modesty"]):
        return "sexual_offence"
    if any(k in q for k in ["helmet", "headgear", "two-wheeler", "motorcycle"]):
        return "penalty"
    if any(k in q for k in ["tree", "trees", "cut tree", "cutting trees", "felling", "forest", "environment", "pollution", "wildlife"]):
        if any(k in q for k in ["what should i do", "what can i do", "now what", "how do i", "what to do", "permission", "clearance"]):
            return "compliance_remedy"
        return "environmental_offence"
    if any(k in q for k in ["penalty", "penalties", "fine", "fines", "punishment", "punishments", "liable", "liability", "imprisonment"]):
        return "penalty"
    if any(k in q for k in ["case law", "judgment", "precedent", "citation"]):
        return "case_law"
    if any(k in q for k in ["what is", "define", "meaning"]):
        return "definition"
    return "general"


def _safe_domain(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["helmet", "traffic", "motor vehicle", "driving", "challan"]):
        return "traffic_rules"
    if any(k in q for k in ["ipc", "bns", "murder", "assault", "stab", "stabbing", "stabbed", "knife", "weapon", "injury", "hurt", "consent", "punishment", "punishments", "penalty", "penalties", "crime", "offence", "offense"]):
        return "criminal_law"
    if any(k in q for k in ["marriage", "divorce", "maintenance", "succession", "shariat", "family"]):
        return "family_law"
    if any(k in q for k in ["contract", "company", "agreement", "damages", "civil"]):
        return "civil_law"
    if any(k in q for k in ["constitution", "article", "fundamental right", "writ"]):
        return "constitutional_law"
    if any(k in q for k in ["tree", "trees", "cutting trees", "felling", "forest", "environment", "pollution", "wildlife", "biodiversity"]):
        return "environmental_law"
    return "general"


def _detect_act(query: str) -> str:
    q = (query or "").lower()
    for act, aliases in ACT_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return act
    if _safe_intent(q) == "sexual_offence":
        return "Indian Penal Code"
    if _safe_domain(q) == "traffic_rules":
        return "Motor Vehicles Act"
    if _safe_domain(q) == "environmental_law":
        for act in ["Environment (Protection) Act", "Forest (Conservation) Act", "Indian Forest Act", "Wild Life (Protection) Act", "Biological Diversity Act"]:
            if any(alias in q for alias in {
                "Environment (Protection) Act": ["environment protection", "environmental protection", "environment department"],
                "Forest (Conservation) Act": ["forest conservation", "forest clearance"],
                "Indian Forest Act": ["indian forest act", "forest act"],
                "Wild Life (Protection) Act": ["wildlife protection", "wild life protection"],
                "Biological Diversity Act": ["biological diversity", "biodiversity act"],
            }[act]):
                return act
    return ""


def _extract_sections(query: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in re.finditer(r"\b(?:section|sec\.?|s\.)\s*([0-9]{1,4}[A-Za-z]?)\b", query or "", flags=re.IGNORECASE):
        s = m.group(1).upper()
        if s not in seen:
            seen.add(s)
            out.append(s)
    if not out and re.search(r"\b(ipc|crpc|cpc|act|code|penalty|punishment)\b", query or "", flags=re.IGNORECASE):
        for m in re.finditer(r"\b([0-9]{2,4}[A-Za-z]?)\b", query or ""):
            s = m.group(1).upper()
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out[:5]


def _normalize_query(query: str) -> str:
    out = _clean_text(query)
    qlow = out.lower()
    if any(k in qlow for k in ["cut many trees", "cut trees", "cut a tree", "cutting trees", "felled trees", "felling trees"]):
        out = f"{out} tree felling forest clearance environmental offence permission"
    elif any(k in qlow for k in ["environment department", "environmental department"]):
        out = f"{out} environmental law forest tree regulation"
    if out:
        tokens = out.split()
        fixed = [TYPO_MAP.get(t.lower(), t) for t in tokens]
        out = " ".join(fixed)
    for patt, repl in QUERY_REWRITE_RULES:
        if re.search(patt, out, flags=re.IGNORECASE):
            out = re.sub(patt, repl, out, flags=re.IGNORECASE)
            break
    return _clean_text(out)


def _fallback_analysis(query: str) -> Dict:
    raw = _clean_text(query)
    norm = _normalize_query(query)
    rewrite_applied = raw.lower() != norm.lower()
    conf = 0.58
    if rewrite_applied and re.search(r"\bsection\s*\d", norm, flags=re.IGNORECASE):
        conf = 0.78
    return {
        "normalized_query": norm,
        "possible_sections": _extract_sections(norm),
        "possible_act": _detect_act(norm),
        "legal_domain": _safe_domain(norm),
        "intent": _safe_intent(norm),
        "confidence": conf,
    }


def fallback_analyze_query(query: str) -> Dict:
    """Deterministic parser path used when LLM-1 is disabled/slow."""
    return _fallback_analysis(query)


def _detect_query_tone(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["urgent", "immediately", "now", "help", "emergency", "arrest"]):
        return "urgent"
    if any(k in q for k in ["confused", "dont know", "not sure", "what should i do"]):
        return "confused"
    if any(k in q for k in ["please", "kindly", "can you"]):
        return "neutral_polite"
    return "neutral"


def _needs_llm_rewrite(raw_query: str, fallback: Dict) -> bool:
    q = _clean_text(raw_query)
    toks = re.findall(r"[a-z0-9]+", q.lower())
    if len(toks) <= 5:
        return True
    if any(t in TYPO_MAP for t in toks):
        return True
    if not str(fallback.get("possible_act", "")).strip() and not list(fallback.get("possible_sections") or []):
        if str(fallback.get("legal_domain", "general")).lower() == "general":
            return True
    if re.search(r"\b(i|me|my)\b", q.lower()) and not re.search(r"\b(section|ipc|act|law|penalty|punishment|offence|offense)\b", q.lower()):
        return True
    return False


def _build_rewrite_prompt(query: str) -> str:
    return (
        "You are an Indian legal query rewriter.\n"
        "Rewrite the user query into a clear legal research query using correct grammar and legal terms.\n"
        "Do not answer the question. Output only one rewritten query line.\n"
        "Prefer adding likely legal terms only when strongly implied by user intent.\n"
        f"User query: {query}\n"
        "Rewritten query:"
    )


def _clean_rewrite_output(text: str, original: str) -> str:
    out = _clean_text(text)
    out = re.sub(r"^(rewritten query|query|answer)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    out = out.strip("`\"' ")
    if not out or len(out) < 8:
        return _normalize_query(original)
    return out


def _should_use_llm1(raw_query: str, fallback: Dict) -> bool:
    if not LLM1_USE:
        return False
    if FAST_MODE and not ALLOW_LLM1_IN_FAST:
        return False
    if not LLM1_ONLY_AMBIGUOUS:
        return True
    q = _clean_text(raw_query)
    if len(q) < 14:
        return False
    if fallback.get("possible_sections"):
        return False
    if str(fallback.get("possible_act", "")).strip():
        return False
    if str(fallback.get("legal_domain", "general")).strip().lower() != "general":
        return False
    if str(fallback.get("intent", "general")).strip().lower() in {"sexual_offence", "penalty"}:
        return False
    return True


def _safe_json_extract(text: str) -> Dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _normalize_output(obj: Dict, query: str) -> Dict:
    fallback = _fallback_analysis(query)
    norm = _clean_text(str(obj.get("normalized_query", "") or fallback["normalized_query"]))
    secs = obj.get("possible_sections", fallback["possible_sections"])
    if not isinstance(secs, list):
        secs = fallback["possible_sections"]
    secs_out: List[str] = []
    for s in secs:
        val = _clean_text(str(s)).upper()
        if re.match(r"^[0-9]{1,4}[A-Z]?$", val) and val not in secs_out:
            secs_out.append(val)
    act = _clean_text(str(obj.get("possible_act", "") or fallback["possible_act"]))
    domain = _clean_text(str(obj.get("legal_domain", "") or fallback["legal_domain"])).lower()
    intent = _clean_text(str(obj.get("intent", "") or fallback["intent"])).lower()
    try:
        conf = float(obj.get("confidence", fallback["confidence"]))
    except Exception:
        conf = float(fallback["confidence"])
    conf = max(0.0, min(1.0, conf))
    return {
        "normalized_query": norm or fallback["normalized_query"],
        "possible_sections": secs_out[:5],
        "possible_act": act,
        "legal_domain": domain or fallback["legal_domain"],
        "intent": intent or fallback["intent"],
        "confidence": conf,
    }


def _build_prompt(query: str) -> str:
    return (
        "You are a legal query analyzer.\n"
        "Rewrite the user's question into a precise legal research query.\n"
        "Identify section numbers, act names, legal domain, and query intent.\n"
        "Return ONLY valid JSON with keys:\n"
        "normalized_query, possible_sections, possible_act, legal_domain, intent, confidence\n"
        "User Query:\n"
        f"{query}\n"
    )


def _load_llm1() -> Optional[Any]:
    global _LLM1, _INIT_ERROR
    if _LLM1 is not None:
        return _LLM1
    if not LLM1_USE:
        _INIT_ERROR = "LLM1 disabled by env"
        return None
    if Llama is None:
        _INIT_ERROR = "llama_cpp is not installed. Analyzer will use fallback parser."
        return None
    model_path = Path(LLM1_MODEL_PATH)
    if not model_path.exists():
        _INIT_ERROR = f"LLM1 model missing: {model_path}"
        return None
    try:
        _LLM1 = Llama(
            model_path=str(model_path),
            n_ctx=LLM1_N_CTX,
            n_threads=LLM1_N_THREADS,
            n_batch=LLM1_N_BATCH,
            n_gpu_layers=0,
            verbose=False,
        )
        _INIT_ERROR = ""
        return _LLM1
    except Exception as exc:
        _INIT_ERROR = str(exc)
        _LLM1 = None
        return None


@lru_cache(maxsize=512)
def analyze_query(query: str) -> Dict:
    """Analyze query using LLM-1 with deterministic JSON output and robust fallback."""
    q = _clean_text(query)
    if not q:
        out = _fallback_analysis("")
        out["query_tone"] = "neutral"
        out["rewritten_by_llm1"] = False
        return out

    fallback = _fallback_analysis(q)
    fallback["query_tone"] = _detect_query_tone(q)
    fallback["rewritten_by_llm1"] = False

    if _needs_llm_rewrite(q, fallback) and _should_use_llm1(q, fallback):
        llm = _load_llm1()
        if llm is not None:
            try:
                t0 = time.perf_counter()
                out = llm(
                    _build_rewrite_prompt(q),
                    max_tokens=max(24, min(96, LLM1_MAX_TOKENS)),
                    temperature=0.0,
                    top_p=0.95,
                    repeat_penalty=1.05,
                    stop=["\n\n", "</s>"],
                    stream=False,
                )
                if (time.perf_counter() - t0) <= LLM1_TIMEOUT_S:
                    rewritten = _clean_rewrite_output(str(out.get("choices", [{}])[0].get("text", "")), q)
                    improved = _fallback_analysis(rewritten)
                    improved["query_tone"] = fallback["query_tone"]
                    improved["rewritten_by_llm1"] = rewritten.lower() != q.lower()
                    improved["original_query"] = q
                    improved["normalized_query"] = rewritten
                    improved["confidence"] = max(float(improved.get("confidence", 0.0)), 0.70)
                    return improved
            except Exception:
                pass

    if not _should_use_llm1(q, fallback):
        return fallback

    llm = _load_llm1()
    if llm is None:
        return fallback

    prompt = _build_prompt(q)
    t0 = time.perf_counter()
    try:
        out = llm(
            prompt,
            max_tokens=LLM1_MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=["</s>", "\n\nUser Query:"],
            stream=False,
        )
        elapsed = time.perf_counter() - t0
        if elapsed > LLM1_TIMEOUT_S:
            return fallback
        txt = str(out.get("choices", [{}])[0].get("text", "")).strip()
        parsed = _safe_json_extract(txt)
        if not parsed:
            return fallback
        out = _normalize_output(parsed, q)
        out["query_tone"] = fallback["query_tone"]
        out["rewritten_by_llm1"] = out.get("normalized_query", "").lower() != q.lower()
        out["original_query"] = q
        return out
    except Exception:
        return fallback


def query_analyzer_status() -> Dict:
    model_path = Path(LLM1_MODEL_PATH)
    ready = bool(model_path.exists() and LLM1_USE)
    return {
        "ready": ready,
        "error": _INIT_ERROR,
        "use_llm1": LLM1_USE,
        "llm1_model": str(model_path),
        "llm1_max_tokens": LLM1_MAX_TOKENS,
        "llm1_timeout_s": LLM1_TIMEOUT_S,
        "llm1_n_ctx": LLM1_N_CTX,
        "llm1_only_ambiguous": LLM1_ONLY_AMBIGUOUS,
        "allow_llm1_in_fast": ALLOW_LLM1_IN_FAST,
        "fast_mode": FAST_MODE,
        "llm1_rewrite_enabled": True,
    }
