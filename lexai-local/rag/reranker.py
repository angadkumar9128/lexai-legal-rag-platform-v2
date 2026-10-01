"""Reranker module for LexAI local pipeline."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None  # type: ignore


DEFAULT_MODEL = os.environ.get("LEXAI_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
HIGH_MODEL = os.environ.get("LEXAI_RERANK_MODEL_HIGH", "BAAI/bge-reranker-base").strip()
USE_RERANKER = os.environ.get("LEXAI_USE_RERANKER", "0").strip().lower() not in {"0", "false", "no"}
RERANK_BALANCED = os.environ.get("LEXAI_RERANK_BALANCED", "0").strip().lower() in {"1", "true", "yes"}
RERANK_MAX_POOL = int(os.environ.get("LEXAI_RERANK_MAX_POOL", "12"))

_MODELS = {}
_INIT_ERROR = "" if CrossEncoder is not None else "sentence-transformers is not installed. Reranker disabled."


def _model_name_for_profile(profile: str) -> str:
    p = (profile or "balanced").strip().lower()
    if p == "high_accuracy" and os.environ.get("LEXAI_USE_BGE_RERANKER_HIGH", "0").strip().lower() in {"1", "true", "yes"}:
        return HIGH_MODEL
    return DEFAULT_MODEL


def _load_model(model_name: str) -> CrossEncoder | None:
    global _INIT_ERROR
    if CrossEncoder is None:
        _INIT_ERROR = "sentence-transformers is not installed. Reranker disabled."
        return None
    if model_name in _MODELS:
        return _MODELS[model_name]
    try:
        _MODELS[model_name] = CrossEncoder(model_name)
        _INIT_ERROR = ""
        return _MODELS[model_name]
    except Exception as exc:
        _INIT_ERROR = str(exc)
        return None


def rerank_candidates(query: str, rows: List[Dict], top_n: int, profile: str) -> Tuple[List[Dict], Dict]:
    """Rerank rows with cross-encoder. Falls back gracefully when unavailable."""
    t0 = time.perf_counter()
    if not rows:
        return [], {"rerank_used": False, "rerank_ms": 0.0, "model_name": "", "error": ""}
    if not USE_RERANKER:
        return rows[: max(1, int(top_n))], {"rerank_used": False, "rerank_ms": 0.0, "model_name": "", "error": "disabled"}
    p = (profile or "balanced").strip().lower()
    if p == "balanced" and not RERANK_BALANCED:
        return rows[: max(1, int(top_n))], {"rerank_used": False, "rerank_ms": 0.0, "model_name": "", "error": "balanced_disabled"}

    model_name = _model_name_for_profile(profile)
    model = _load_model(model_name)
    if model is None:
        return rows[: max(1, int(top_n))], {
            "rerank_used": False,
            "rerank_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "model_name": model_name,
            "error": _INIT_ERROR,
        }

    pool = rows[: max(1, min(len(rows), min(RERANK_MAX_POOL, max(6, int(top_n) * 3))))]
    pairs = [(query, str(r.get("chunk_text", ""))[:700]) for r in pool]
    try:
        ce_scores = model.predict(pairs)
        for i, s in enumerate(ce_scores):
            ce = float(s)
            pool[i]["rerank_score"] = ce
            pool[i]["score"] = (0.72 * float(pool[i].get("score", 0.0))) + (0.28 * ce)
        pool.sort(key=lambda x: x.get("score", -1e9), reverse=True)
        out = pool[: max(1, int(top_n))]
        return out, {
            "rerank_used": True,
            "rerank_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "model_name": model_name,
            "error": "",
        }
    except Exception as exc:
        return rows[: max(1, int(top_n))], {
            "rerank_used": False,
            "rerank_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "model_name": model_name,
            "error": str(exc),
        }


def reranker_status() -> Dict:
    return {
        "use_reranker": USE_RERANKER,
        "default_model": DEFAULT_MODEL,
        "high_model": HIGH_MODEL,
        "error": _INIT_ERROR,
        "loaded_models": list(_MODELS.keys()),
        "rerank_balanced": RERANK_BALANCED,
        "rerank_max_pool": RERANK_MAX_POOL,
    }
