"""LexAI dual-LLM orchestration pipeline."""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List

from rag.context_builder import build_context, build_evidence_context, extract_citations
from rag.generator import generate_answer, generator_status
from rag.query_analyzer import analyze_query, fallback_analyze_query, query_analyzer_status
from rag.reranker import rerank_candidates, reranker_status
from rag.retriever import expand_section_matches, retrieve_chunks, retriever_status


DEFAULT_TOP_K = int(os.environ.get("LEXAI_FINAL_K", "5"))
DEFAULT_INITIAL_K = int(os.environ.get("LEXAI_INITIAL_K", "15"))
DEFAULT_PROFILE = os.environ.get("LEXAI_PROFILE", os.environ.get("LEXAI_MODEL_PROFILE", "balanced")).strip().lower()
DEFAULT_RETRIEVAL_MODE = os.environ.get("LEXAI_RETRIEVAL_MODE", "hybrid").strip().lower() or "hybrid"
MAX_CONTEXT_CHARS = int(os.environ.get("LEXAI_MAX_CONTEXT_CHARS", "1500"))
EVIDENCE_CONTEXT_CHARS = int(os.environ.get("LEXAI_EVIDENCE_CONTEXT_CHARS", "1200"))
FAST_MODE = os.environ.get("LEXAI_FAST_MODE", "1").strip().lower() not in {"0", "false", "no"}
FAST_SKIP_DENSE = os.environ.get("LEXAI_FAST_SKIP_DENSE", "1").strip().lower() not in {"0", "false", "no"}
FAST_DISABLE_POLISH = os.environ.get("LEXAI_FAST_DISABLE_POLISH", "1").strip().lower() not in {"0", "false", "no"}
MAX_TOTAL_MS = float(os.environ.get("LEXAI_MAX_TOTAL_MS", "30000"))
MAX_STAGE_LLM1_MS = float(os.environ.get("LEXAI_MAX_STAGE_LLM1_MS", "2500"))
MAX_STAGE_RETRIEVE_MS = float(os.environ.get("LEXAI_MAX_STAGE_RETRIEVE_MS", "9000"))
MAX_STAGE_RERANK_MS = float(os.environ.get("LEXAI_MAX_STAGE_RERANK_MS", "2500"))

CONF_THRESH = {
    "balanced": {
        "high": float(os.environ.get("LEXAI_CONF_BALANCED_HIGH", "0.66")),
        "medium": float(os.environ.get("LEXAI_CONF_BALANCED_MED", "0.45")),
    },
    "high_accuracy": {
        "high": float(os.environ.get("LEXAI_CONF_HIGH_HIGH", "0.62")),
        "medium": float(os.environ.get("LEXAI_CONF_HIGH_MED", "0.40")),
    },
}

_LAT_HIST: List[Dict[str, float]] = []


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _safe_profile(profile: str | None) -> str:
    p = (profile or DEFAULT_PROFILE or "balanced").strip().lower()
    return p if p in {"balanced", "high_accuracy"} else "balanced"


def _confidence_bucket(confidence: float, profile: str) -> str:
    cfg = CONF_THRESH[_safe_profile(profile)]
    if confidence >= cfg["high"]:
        return "high"
    if confidence >= cfg["medium"]:
        return "medium"
    return "low"


def _dedupe_rows(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for row in rows:
        key = str(row.get("chunk_id", "")).strip()
        if not key:
            key = f"{row.get('act_name','')}|{row.get('section_number','')}|{row.get('chunk_text','')[:80]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _record_latency(meta: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    _LAT_HIST.append(dict(meta))
    if len(_LAT_HIST) > 120:
        del _LAT_HIST[:-120]
    out = {}
    for key in ["llm1_ms", "retrieve_ms", "rerank_ms", "llm2_ms", "total_ms"]:
        vals = sorted(float(x.get(key, 0.0)) for x in _LAT_HIST if key in x)
        if not vals:
            out[key] = {"p50": 0.0, "p95": 0.0, "n": 0}
            continue
        p50_idx = int(0.5 * (len(vals) - 1))
        p95_idx = int(0.95 * (len(vals) - 1))
        out[key] = {"p50": round(vals[p50_idx], 2), "p95": round(vals[p95_idx], 2), "n": len(vals)}
    return out


def ask_lexai(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    initial_k: int = DEFAULT_INITIAL_K,
    style: str | None = None,
    target_words: int | None = None,
    profile: str = DEFAULT_PROFILE,
    retrieval_mode: str | None = None,
) -> Dict:
    """Run dual-LLM legal QA pipeline."""
    total_t0 = time.perf_counter()
    raw_query = (question or "").strip()
    if not raw_query:
        return {
            "answer": "Query cannot be empty.",
            "sources": [],
            "meta": {"error": "empty_query", "total_ms": 0.0},
        }

    resolved_profile = _safe_profile(profile)
    resolved_mode = (retrieval_mode or DEFAULT_RETRIEVAL_MODE).strip().lower()
    resolved_top_k = max(1, int(top_k))
    shortlist_n = max(5, resolved_top_k)
    resolved_initial_k = max(15, int(initial_k))
    resolved_target_words = max(40, min(320, int(target_words if target_words is not None else 140)))
    resolved_style = (style or "normal").strip().lower()

    # Step 1: LLM-1 analysis
    t_llm1 = time.perf_counter()
    analysis = analyze_query(raw_query)
    llm1_ms = (time.perf_counter() - t_llm1) * 1000.0
    if llm1_ms > MAX_STAGE_LLM1_MS and FAST_MODE:
        analysis = fallback_analyze_query(raw_query)

    # Step 2: retrieval top-15 candidates
    t_ret = time.perf_counter()
    retrieved, ret_meta = retrieve_chunks(
        question=analysis.get("normalized_query", raw_query),
        top_k=resolved_initial_k,
        initial_k=resolved_initial_k,
        profile=resolved_profile,
        retrieval_mode=resolved_mode,
        analysis=analysis,
        skip_dense=(FAST_MODE and FAST_SKIP_DENSE),
        return_meta=True,
    )
    if (
        FAST_MODE
        and float(ret_meta.get("confidence", 0.0) or 0.0) < 0.40
        and _elapsed_ms(total_t0) < (MAX_TOTAL_MS - 7000)
    ):
        # Escalate once to dense retrieval when fast lexical path is weak.
        retrieved2, ret_meta2 = retrieve_chunks(
            question=analysis.get("normalized_query", raw_query),
            top_k=resolved_initial_k,
            initial_k=min(resolved_initial_k, 18),
            profile=resolved_profile,
            retrieval_mode=resolved_mode,
            analysis=analysis,
            skip_dense=False,
            return_meta=True,
        )
        if float(ret_meta2.get("confidence", 0.0) or 0.0) > float(ret_meta.get("confidence", 0.0) or 0.0):
            retrieved, ret_meta = retrieved2, ret_meta2
    retrieve_ms = (time.perf_counter() - t_ret) * 1000.0
    if retrieve_ms > MAX_STAGE_RETRIEVE_MS and FAST_MODE:
        resolved_top_k = max(1, min(resolved_top_k, 2))
        shortlist_n = max(5, resolved_top_k)
        retrieved = retrieved[:shortlist_n]

    # Step 3: rerank to top_n shortlist
    t_rerank = time.perf_counter()
    if FAST_MODE:
        reranked = retrieved[:shortlist_n]
        rr_meta = {"rerank_used": False, "rerank_ms": 0.0, "model_name": "", "error": "fast_mode_skip"}
    else:
        reranked, rr_meta = rerank_candidates(
            query=analysis.get("normalized_query", raw_query),
            rows=retrieved,
            top_n=shortlist_n,
            profile=resolved_profile,
        )
    rerank_ms = (time.perf_counter() - t_rerank) * 1000.0
    if rerank_ms > MAX_STAGE_RERANK_MS and FAST_MODE:
        reranked = retrieved[:shortlist_n]
        rr_meta = {"rerank_used": False, "rerank_ms": round(rerank_ms, 2), "model_name": "", "error": "stage_budget_exceeded"}

    # Step 4: section expansion
    explicit_section_in_query = bool(
        analysis.get("possible_sections")
        and re.search(r"\b(?:section|sec\.?|s\.)\s*[0-9]{1,4}[A-Za-z]?\b", raw_query, flags=re.IGNORECASE)
    )
    strong_inferred_section = bool(
        analysis.get("possible_sections")
        and analysis.get("possible_act")
        and float(analysis.get("confidence", 0.0) or 0.0) >= 0.75
    )
    if explicit_section_in_query or strong_inferred_section:
        expanded = expand_section_matches(
            possible_sections=list(analysis.get("possible_sections") or []),
            possible_act=str(analysis.get("possible_act", "") or ""),
            limit=30 if FAST_MODE else 60,
        )
    else:
        expanded = []
    merged_sources = _dedupe_rows(reranked + expanded)
    final_sources = merged_sources[:shortlist_n]

    # Step 5: context build
    context = build_context(final_sources, max_sections=shortlist_n, max_chars=MAX_CONTEXT_CHARS)
    evidence_context = build_evidence_context(
        query=analysis.get("normalized_query", raw_query),
        rows=final_sources,
        max_blocks=max(5, shortlist_n),
        max_chars=EVIDENCE_CONTEXT_CHARS,
    )
    gen_context = evidence_context or context
    citations = extract_citations(final_sources, analysis=analysis)

    # Step 6: generation
    confidence = float(ret_meta.get("confidence", 0.0))
    conf_bucket = _confidence_bucket(confidence, resolved_profile)
    if (
        conf_bucket == "low"
        and analysis.get("possible_act")
        and analysis.get("possible_sections")
        and float(analysis.get("confidence", 0.0) or 0.0) >= 0.75
    ):
        conf_bucket = "medium"

    if not gen_context:
        answer = (
            "Answer:\n"
            "The answer is not found in the provided legal context.\n\n"
            "Relevant Sections:\n"
            + ("\n".join([f"- {c}" for c in citations]) if citations else "- Not available")
            + "\n\nLegal Interpretation:\n"
            + str(ret_meta.get("confidence_reason", "Low confidence retrieval."))
            + "\n\nConclusion:\nUse the cited sections and ask a narrower legal query."
        )
        llm2_ms = 0.0
        quality_mode = "draft_only"
    else:
        t_llm2 = time.perf_counter()
        if FAST_MODE and _elapsed_ms(total_t0) > (MAX_TOTAL_MS - 3000):
            conf_bucket = "medium"
        if FAST_MODE and FAST_DISABLE_POLISH and conf_bucket == "high":
            conf_bucket = "medium"
        answer = generate_answer(
            context=gen_context,
            question=raw_query,
            source_refs=citations,
            style=resolved_style,
            target_words=resolved_target_words,
            confidence_reason=str(ret_meta.get("confidence_reason", "")),
            confidence_bucket=conf_bucket,
            profile=resolved_profile,
            analysis=analysis,
        )
        llm2_ms = (time.perf_counter() - t_llm2) * 1000.0
        quality_mode = "draft_plus_polish" if conf_bucket in {"high", "medium"} else "draft_only"

    total_ms = (time.perf_counter() - total_t0) * 1000.0
    roll = _record_latency(
        {
            "llm1_ms": round(llm1_ms, 2),
            "retrieve_ms": round(retrieve_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "llm2_ms": round(llm2_ms, 2),
            "total_ms": round(total_ms, 2),
        }
    )

    return {
        "answer": answer,
        "sources": final_sources,
        "meta": {
            "analysis": analysis,
            "confidence": round(confidence, 4),
            "confidence_bucket": conf_bucket,
            "quality_mode": quality_mode,
            "profile": resolved_profile,
            "retrieval_mode": resolved_mode,
            "llm1_ms": round(llm1_ms, 2),
            "retrieve_ms": round(retrieve_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "context_chars": len(gen_context),
            "llm2_ms": round(llm2_ms, 2),
            "generate_ms": round(llm2_ms, 2),
            "total_ms": round(total_ms, 2),
            "top_k": resolved_top_k,
            "initial_k": resolved_initial_k,
            "shortlist_n": shortlist_n,
            "llm1_model": query_analyzer_status().get("llm1_model", ""),
            "llm2_model": generator_status().get("llm2_model", ""),
            "reranker_model": rr_meta.get("model_name", ""),
            "fast_mode": FAST_MODE,
            "fast_skip_dense": FAST_SKIP_DENSE,
            "fast_disable_polish": FAST_DISABLE_POLISH,
            "retrieval": ret_meta,
            "reranker": rr_meta,
            "status": {
                "query_analyzer": query_analyzer_status(),
                "retriever": retriever_status(),
                "reranker": reranker_status(),
                "generator": generator_status(),
            },
            "latency_profile": roll,
        },
    }
