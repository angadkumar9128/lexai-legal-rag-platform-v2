"""Central configuration for the production-oriented local LexAI stack."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VECTOR_DIR = ROOT / "vector_store"
MODELS_DIR = ROOT / "models"

EMBED_MODEL = os.getenv("LEXAI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
LLAMA_BASE_URL = os.getenv("LEXAI_LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLAMA_MODEL = os.getenv("LEXAI_LLM_MODEL", "qwen2.5-3b-instruct")
LLAMA_TIMEOUT = float(os.getenv("LEXAI_LLM_TIMEOUT", "60"))
LLAMA_CONTEXT = int(os.getenv("LEXAI_LLM_CONTEXT", "8192"))
LLAMA_MAX_TOKENS = int(os.getenv("LEXAI_LLM_MAX_TOKENS", "500"))
RETRIEVAL_TOP_K = int(os.getenv("LEXAI_TOP_K", "8"))
RERANK_TOP_N = int(os.getenv("LEXAI_RERANK_TOP_N", "20"))
USE_RERANKER = os.getenv("LEXAI_USE_RERANKER", "0").lower() not in {"0","false","no"}
RERANK_MODEL = os.getenv("LEXAI_RERANK_MODEL", "BAAI/bge-reranker-base")
USE_QWEN_PLANNER = os.getenv("LEXAI_USE_QWEN_PLANNER", "0").lower() not in {"0","false","no"}
MIN_EVIDENCE_SCORE = float(os.getenv("LEXAI_MIN_EVIDENCE_SCORE", "0.34"))
MAX_HISTORY_TURNS = int(os.getenv("LEXAI_MAX_HISTORY_TURNS", "8"))

def env_summary() -> dict:
    return {
        "llm_url": LLAMA_BASE_URL,
        "llm_model": LLAMA_MODEL,
        "llm_context": LLAMA_CONTEXT,
        "embed_model": EMBED_MODEL,
        "top_k": RETRIEVAL_TOP_K,
        "reranker": RERANK_MODEL if USE_RERANKER else "disabled",
        "qwen_planner": USE_QWEN_PLANNER,
    }
