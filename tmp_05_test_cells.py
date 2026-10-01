

# ========================================
# CELL 1
# ========================================
# Optional dependency installer (runs only when explicitly enabled)
AUTO_INSTALL_MISSING_PACKAGES = True
REQUIRED_PACKAGES = [
    "sentence-transformers",
    "chromadb",
    "transformers",
    "accelerate",
    "mlflow",
    "databricks-sdk",
]

import importlib.metadata as _ilm
import subprocess, sys

missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        _ilm.version(pkg)
    except Exception:
        missing.append(pkg)

if missing:
    print(f"Missing packages: {missing}")
    if AUTO_INSTALL_MISSING_PACKAGES:
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing
        print("Installing missing packages:", " ".join(missing))
        subprocess.check_call(cmd)
        print("Installation complete. Re-run this notebook from Cell 2.")
    else:
        print("AUTO_INSTALL_MISSING_PACKAGES=False, so install is skipped.")
else:
    print("All required packages are already available; skipping install.")



# ========================================
# CELL 2
# ========================================
print("Run sequence: 04_generate_embedding_Test -> 05_rag_answer_pipeline_test -> 06_High-precision_QA_Legal_Reasoning_Engine")
print("Tip: In this notebook, run Cell 5 once per cluster session; re-running later cells does not require re-initialization.")



# ========================================
# CELL 3
# ========================================
import os
import re
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

from pyspark.sql import functions as F
from pyspark.sql.window import Window

os.environ["ANONYMIZED_TELEMETRY"] = "False"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def safe_str(v):
    return "" if v is None else str(v)


def clean_text(text: str) -> str:
    text = safe_str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    return clean_text(text)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", safe_str(text).lower()) if len(t) > 2]


def unique_keep_order(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for v in values:
        k = safe_str(v).strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(safe_str(v).strip())
    return out

def infer_source_group(category: str, source: str, act: str) -> str:
    blob = f"{safe_str(category)} {safe_str(source)} {safe_str(act)}".lower()
    checks = [
        ("constitution", "constitution"),
        ("criminal_law", "criminal_law"),
        ("civil_law", "civil_law"),
        ("family_law", "family_law"),
        ("traffic_rules", "traffic_rules"),
        ("judgment", "judgments"),
        ("reports", "reports"),
        ("acts", "acts"),
    ]
    for key, val in checks:
        if key in blob:
            return val
    return "general"



# ========================================
# CELL 4
# ========================================
EMBEDDING_DELTA_PATH = "/Volumes/workspace/legal_data/vector_db_test/legal_embeddings_delta"
EMBEDDING_TABLE_NAME = "workspace.default.legal_embeddings_test"
CHROMA_DB_CANDIDATES = [
    "/Volumes/workspace/legal_data/chroma_db/legal_knowledge_test",
    "/Volumes/workspace/legal_data/vector_db_test/chroma_db_legal_knowledge_test",
]
COLLECTION_NAME = "legal_knowledge"
ENABLE_CHROMA_IN_05 = False  # recommended on serverless: retrieval stays Delta-first and persistent

PRIMARY_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
PRIMARY_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DATABRICKS_LLM_ENDPOINT = os.environ.get("DATABRICKS_LLM_ENDPOINT", "").strip()
DATABRICKS_LLM_CANDIDATES = [
    DATABRICKS_LLM_ENDPOINT,
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-mixtral-8x7b-instruct",
]

LOCAL_QA_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-small",
]

TOP_K = 8
RETRIEVAL_POOL = 40
RERANK_POOL = 20
MAX_CONTEXT_CHARS = 4200
MIN_RELEVANCE_SCORE = 0.34
MIN_LEXICAL_SCORE = 0.06

DOMAIN_RULES = {
    "traffic": {
        "query_terms": [
            "helmet",
            "headgear",
            "traffic",
            "vehicle",
            "motor",
            "driving",
            "licence",
            "challan",
            "fine",
            "penalty",
            "two-wheeler",
            "motorcycle",
        ]
    },
    "criminal": {
        "query_terms": [
            "murder",
            "homicide",
            "ipc",
            "bns",
            "punishment",
            "imprisonment",
            "crime",
        ]
    },
    "employment": {
        "query_terms": [
            "employment",
            "bond",
            "salary",
            "labour",
            "resign",
            "company",
            "contract",
            "notice period",
        ]
    },
}

# Globals are initialized here so later cells never fail with NameError.
LLM_BACKEND = {"type": "none", "name": "", "client": None, "model": None, "errors": []}
local_llm = None
dbx_client = None



# ========================================
# CELL 5
# ========================================
def load_embedding_model():
    for model_name in [PRIMARY_EMBED_MODEL, FALLBACK_EMBED_MODEL]:
        try:
            log(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            _ = model.encode(["health check"], show_progress_bar=False)
            log(f"Embedding model ready: {model_name}")
            return model, model_name
        except Exception as e:
            log(f"Embedding model failed ({model_name}): {e}")
    return None, None


def load_reranker_model():
    try:
        log(f"Loading reranker: {PRIMARY_RERANKER_MODEL}")
        model = CrossEncoder(PRIMARY_RERANKER_MODEL)
        _ = model.predict([("test", "test")])
        return model, PRIMARY_RERANKER_MODEL
    except Exception as e:
        log(f"Reranker unavailable: {e}")
        return None, None


def load_local_qa_model():
    for model_name in LOCAL_QA_MODELS:
        try:
            log(f"Loading local QA model: {model_name}")
            qa = pipeline(
                "text2text-generation",
                model=model_name,
                max_new_tokens=320,
                do_sample=False,
                temperature=0.0,
            )
            log(f"Local QA model ready: {model_name}")
            return qa, model_name
        except Exception as e:
            log(f"Local QA model failed ({model_name}): {e}")
    return None, None


def load_databricks_deploy_client():
    try:
        import mlflow.deployments

        client = mlflow.deployments.get_deploy_client("databricks")
        return client, None
    except Exception as e:
        return None, str(e)


def extract_text_from_llm_response(resp):
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, list) and resp:
        return extract_text_from_llm_response(resp[0])
    if isinstance(resp, dict):
        for key in ["generated_text", "text", "output", "answer"]:
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"].strip()
                if isinstance(first.get("text"), str):
                    return first["text"].strip()
        preds = resp.get("predictions")
        if isinstance(preds, list) and preds:
            return extract_text_from_llm_response(preds[0])
    return ""


def try_endpoint_once(client, endpoint_name: str, prompt: str):
    try:
        resp = client.predict(
            endpoint=endpoint_name,
            inputs={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 420,
            },
        )
        text = extract_text_from_llm_response(resp)
        if text:
            return text, None
    except Exception as e:
        chat_error = str(e)
    else:
        chat_error = "empty chat response"

    try:
        resp = client.predict(
            endpoint=endpoint_name,
            inputs={
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 420,
            },
        )
        text = extract_text_from_llm_response(resp)
        if text:
            return text, None
        return "", f"empty completion response for {endpoint_name}"
    except Exception as e:
        return "", f"chat_error={chat_error}; completion_error={e}"


def load_chroma_collection():
    if not ENABLE_CHROMA_IN_05:
        return None, None, "disabled_by_config"

    errors = []
    for candidate in CHROMA_DB_CANDIDATES:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            client = chromadb.PersistentClient(
                path=candidate,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            collection = client.get_or_create_collection(COLLECTION_NAME)
            return collection, candidate, None
        except Exception as e:
            errors.append(f"{candidate}: {e}")
    return None, None, " | ".join(errors)


def load_embedding_delta_latest():
    # Always read latest snapshot from Delta path; fallback to table mirror.
    try:
        df = spark.read.format("delta").load(EMBEDDING_DELTA_PATH)
    except Exception as e_path:
        log(f"Path read failed, trying table fallback: {e_path}")
        df = spark.table(EMBEDDING_TABLE_NAME)

    required = ["chunk_id", "chunk_text", "act_name", "section_number", "category", "file_name", "embedding"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Embedding Delta missing columns: {missing}"

    if "updated_at" not in df.columns:
        df = df.withColumn("updated_at", F.current_timestamp())

    w = Window.partitionBy("chunk_id").orderBy(F.col("updated_at").desc_nulls_last())
    latest = (
        df.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .dropna(subset=["chunk_id", "chunk_text", "embedding"])
    )

    cnt = latest.count()
    if cnt == 0:
        return None, "Embedding Delta latest snapshot has 0 rows"

    return latest, None


def build_local_index(embeddings_df):
    rows = []
    cols = [
        "chunk_id",
        "chunk_text",
        "act_name",
        "section_number",
        "category",
        "file_name",
        "embedding",
        "is_penalty_related",
        "is_helmet_related",
        "section_refs",
    ]
    available = [c for c in cols if c in embeddings_df.columns]

    for r in embeddings_df.select(*available).toLocalIterator():
        if not r.embedding:
            continue
        text = clean_text(r.chunk_text)
        refs = []
        if "section_refs" in available and r.section_refs:
            refs = [x.strip().upper() for x in safe_str(r.section_refs).split(",") if x.strip()]

        rows.append(
            {
                "chunk_id": safe_str(r.chunk_id),
                "text": text,
                "text_norm": text.lower(),
                "act_name": safe_str(r.act_name),
                "act_name_norm": safe_str(r.act_name).lower(),
                "section": safe_str(r.section_number),
                "section_norm": safe_str(r.section_number).upper(),
                "category": safe_str(r.category),
                "source": safe_str(r.file_name),
                "source_group": infer_source_group(safe_str(r.category), safe_str(r.file_name), safe_str(r.act_name)),
                "embedding": np.array([float(x) for x in r.embedding], dtype=np.float32),
                "tokens": set(tokenize(text)),
                "is_penalty_related": int(getattr(r, "is_penalty_related", 0) or 0),
                "is_helmet_related": int(getattr(r, "is_helmet_related", 0) or 0),
                "section_refs": refs,
            }
        )
    return rows


def hydrate_chroma_from_delta(collection, embeddings_df, batch_size=200):
    if collection is None or embeddings_df is None:
        return 0

    inserted = 0
    batch = []

    def flush(rows):
        ids, docs, embeds, metas = [], [], [], []
        for r in rows:
            if not r.chunk_id or not r.chunk_text or not r.embedding:
                continue
            ids.append(str(r.chunk_id))
            docs.append(clean_text(r.chunk_text))
            embeds.append([float(x) for x in r.embedding])
            metas.append(
                {
                    "act_name": safe_str(r.act_name),
                    "section": safe_str(r.section_number),
                    "category": safe_str(r.category),
                    "source": safe_str(r.file_name),
                }
            )
        if not ids:
            return 0
        collection.upsert(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        return len(ids)

    for row in embeddings_df.toLocalIterator():
        batch.append(row)
        if len(batch) >= batch_size:
            inserted += flush(batch)
            batch = []
    if batch:
        inserted += flush(batch)

    return inserted


def endpoint_generate_once(client, endpoint_name: str, prompt: str):
    # Backward-compatible alias used in downstream cells.
    return try_endpoint_once(client, endpoint_name, prompt)



# ========================================
# CELL 6
# ========================================
print("[CELL 5] START - Runtime initialization")

if "_LEXAI05_RUNTIME" in globals() and _LEXAI05_RUNTIME.get("ready", False):
    embedding_model = _LEXAI05_RUNTIME["embedding_model"]
    embedding_model_name = _LEXAI05_RUNTIME["embedding_model_name"]
    reranker = _LEXAI05_RUNTIME["reranker"]
    reranker_name = _LEXAI05_RUNTIME["reranker_name"]
    collection = _LEXAI05_RUNTIME["collection"]
    chroma_path = _LEXAI05_RUNTIME["chroma_path"]
    embeddings_df = _LEXAI05_RUNTIME["embeddings_df"]
    local_index = _LEXAI05_RUNTIME["local_index"]
    dbx_client = _LEXAI05_RUNTIME["dbx_client"]
    local_llm = _LEXAI05_RUNTIME["local_llm"]
    llm_backend = _LEXAI05_RUNTIME["llm_backend"]
    LLM_BACKEND = llm_backend
    print("[CELL 5] Runtime already initialized in this session; reusing cached objects.")
else:
    embedding_model, embedding_model_name = load_embedding_model()
    reranker, reranker_name = load_reranker_model()
    collection, chroma_path, chroma_error = load_chroma_collection()
    embeddings_df, delta_error = load_embedding_delta_latest()

    dbx_client, dbx_client_error = load_databricks_deploy_client()
    llm_backend = {
        "type": "none",  # endpoint | local | none
        "name": "",
        "client": None,
        "model": None,
        "errors": [],
    }

    if dbx_client_error:
        llm_backend["errors"].append(f"Databricks deploy client unavailable: {dbx_client_error}")

    if dbx_client is not None:
        test_prompt = "Reply only with: OK"
        for ep in [e for e in DATABRICKS_LLM_CANDIDATES if e]:
            text, err = try_endpoint_once(dbx_client, ep, test_prompt)
            if text:
                llm_backend.update({"type": "endpoint", "name": ep, "client": dbx_client})
                log(f"Using Databricks LLM endpoint: {ep}")
                break
            llm_backend["errors"].append(f"Endpoint {ep} failed: {err}")

    local_llm = None
    if llm_backend["type"] == "none":
        local_model, local_model_name = load_local_qa_model()
        if local_model is not None:
            local_llm = local_model
            llm_backend.update({"type": "local", "name": local_model_name, "model": local_model})

    if local_llm is None:
        local_llm = llm_backend.get("model")

    if 'chroma_error' in locals() and chroma_error and chroma_error != "disabled_by_config":
        log(f"Chroma warning: {chroma_error}")
    elif 'chroma_error' in locals() and chroma_error == "disabled_by_config":
        log("Chroma: disabled_by_config (Delta-only retrieval mode)")
    elif collection is not None:
        log(f"Chroma path: {chroma_path}")
        log(f"Chroma count before hydration: {collection.count()}")

    if 'delta_error' in locals() and delta_error:
        log(f"Embedding Delta warning: {delta_error}")
        embeddings_df = None
        local_index = []
    else:
        log(f"Embedding Delta latest rows: {embeddings_df.count()}")
        local_index = build_local_index(embeddings_df)
        log(f"Local index rows: {len(local_index)}")

    if collection is not None and embeddings_df is not None:
        try:
            if collection.count() == 0:
                inserted = hydrate_chroma_from_delta(collection, embeddings_df)
                log(f"Hydrated Chroma from Delta: {inserted}")
                log(f"Chroma count after hydration: {collection.count()}")
        except Exception as e:
            log(f"Hydration warning: {e}")

    LLM_BACKEND = llm_backend
    _LEXAI05_RUNTIME = {
        "ready": True,
        "embedding_model": embedding_model,
        "embedding_model_name": embedding_model_name,
        "reranker": reranker,
        "reranker_name": reranker_name,
        "collection": collection,
        "chroma_path": chroma_path,
        "embeddings_df": embeddings_df,
        "local_index": local_index,
        "dbx_client": dbx_client,
        "local_llm": local_llm,
        "llm_backend": llm_backend,
    }

print("--- Runtime Status ---")
print("Embedding model:", embedding_model_name or "Unavailable")
print("Reranker:", reranker_name or "Unavailable")
print("Chroma available:", collection is not None)
print("Delta available:", embeddings_df is not None)
print("LLM backend:", f"{LLM_BACKEND['type']} ({LLM_BACKEND['name']})" if LLM_BACKEND["type"] != "none" else "none")
if LLM_BACKEND.get("errors"):
    print("LLM init diagnostics:")
    for err in LLM_BACKEND["errors"][:5]:
        print(" -", err)

print("[CELL 5] END")



# ========================================
# CELL 7
# ========================================
def detect_domain(query: str):
    q = query.lower()
    for name, rule in DOMAIN_RULES.items():
        if any(t in q for t in rule["query_terms"]):
            return name, rule
    return "general", None


def extract_query_sections(query: str) -> List[str]:
    refs = []
    for m in re.findall(r"(?:section|sec\.?|article|art\.?|s\.)\s*(\d+[A-Za-z-]*)", query, flags=re.IGNORECASE):
        refs.append(m.upper())
    return unique_keep_order(refs)


def query_embedding(query: str):
    if embedding_model is None:
        return None
    vec = embedding_model.encode([query], show_progress_bar=False)[0]
    return np.array(vec, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) + 1e-12) * (float(np.linalg.norm(b)) + 1e-12)
    return float(np.dot(a, b) / denom)


def filter_candidates_for_domain(records: List[Dict], query: str, domain_name: str, rule: Dict):
    if not records:
        return []

    if domain_name == "general" or not rule:
        return records

    q_terms = set(tokenize(query))
    strict = []
    loose = []

    for r in records:
        text = r["text_norm"]
        act = r["act_name_norm"]
        overlap = sum(1 for t in q_terms if t in r["tokens"])

        if domain_name == "traffic":
            if (
                ("motor vehicle" in act or "traffic" in act)
                and any(k in text for k in ["helmet", "headgear", "motor cycle", "motorcycle", "two-wheeler"])
            ):
                strict.append(r)
            elif overlap >= 2:
                loose.append(r)
        else:
            if overlap >= 2:
                strict.append(r)
            elif overlap >= 1:
                loose.append(r)

    if strict:
        return strict + loose
    if loose:
        return loose
    return records


def compute_hybrid_scores(query: str, candidates: List[Dict]):
    q_vec = query_embedding(query)
    q_terms = set(tokenize(query))
    q = query.lower()
    section_hints = set(extract_query_sections(query))

    scored = []
    for r in candidates:
        text = r["text_norm"]
        tokens = r["tokens"]

        v = cosine(q_vec, r["embedding"]) if q_vec is not None else 0.0
        overlap = sum(1 for t in q_terms if t in tokens)
        lexical = overlap / max(1, len(q_terms))

        legal_bonus = 0.0
        if r.get("is_penalty_related", 0) == 1:
            legal_bonus += 0.08
        if any(k in text for k in ["penalty", "fine", "punishable", "imprisonment", "challan"]):
            legal_bonus += 0.08

        # Section-aware boosts.
        if section_hints:
            if r["section_norm"] in section_hints:
                legal_bonus += 0.22
            if set(r.get("section_refs", [])).intersection(section_hints):
                legal_bonus += 0.16

        # Traffic specialization for helmet questions.
        if "helmet" in q or "headgear" in q:
            if "motor vehicle" in r["act_name_norm"]:
                legal_bonus += 0.30
            if r.get("is_helmet_related", 0) == 1:
                legal_bonus += 0.22
            if re.search(r"\b129\b", text):
                legal_bonus += 0.24
            if re.search(r"\b177\b", text):
                legal_bonus += 0.14
            if re.search(r"\b194d\b", text):
                legal_bonus += 0.12

        if q_vec is None:
            score = (0.84 * lexical) + legal_bonus
        else:
            score = (0.62 * v) + (0.24 * lexical) + legal_bonus

        scored.append(
            {
                "score": float(score),
                "vector_score": float(v),
                "lexical_score": float(lexical),
                "record": r,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def rerank_with_cross_encoder(query: str, ranked: List[Dict], top_n: int = RERANK_POOL):
    if reranker is None or not ranked:
        return ranked

    pool = ranked[:top_n]
    pairs = [(query, x["record"]["text"][:1000]) for x in pool]

    try:
        ce_scores = reranker.predict(pairs)
    except Exception as e:
        log(f"Cross-encoder rerank skipped: {e}")
        return ranked

    for i, ce in enumerate(ce_scores):
        ce_norm = 1 / (1 + math.exp(-float(ce)))
        pool[i]["score"] = (0.60 * pool[i]["score"]) + (0.40 * ce_norm)

    pool.sort(key=lambda x: x["score"], reverse=True)
    return pool + ranked[top_n:]



# ========================================
# CELL 8
# ========================================
def extract_sections(text: str, section_meta: str = "") -> List[str]:
    refs = []

    sec_meta = safe_str(section_meta).strip()
    if sec_meta and not sec_meta.lower().startswith("chapter"):
        refs.extend(re.findall(r"\d+[A-Za-z-]*", sec_meta))

    refs.extend(re.findall(r"(?:section|sec\.?|article|art\.?)\s*(\d+[A-Za-z-]*)", safe_str(text), flags=re.IGNORECASE))
    refs = [safe_str(x).upper() for x in refs]
    return unique_keep_order(refs)


def parse_response_preferences(query: str) -> Dict:
    q = query.lower()
    style = "normal"

    if any(k in q for k in ["very short", "one line", "1 line", "single line"]):
        style = "very_short"
    elif any(k in q for k in ["in short", "short answer", "briefly", "brief"]):
        style = "short"
    elif any(k in q for k in ["in detail", "in details", "in depth", "detailed", "comprehensive"]):
        style = "detailed"

    word_limit = None
    m = re.search(r"(?:within|under|max(?:imum)?|limit(?:ed)? to)\s*(\d{2,4})\s*(?:words?|works?)", q)
    if m:
        word_limit = int(m.group(1))
    else:
        m2 = re.search(r"(\d{2,4})\s*(?:words?|works?)", q)
        if m2 and any(k in q for k in ["within", "under", "limit", "max"]):
            word_limit = int(m2.group(1))

    defaults = {"very_short": 45, "short": 85, "normal": 150, "detailed": 260}
    target_words = defaults.get(style, 150)
    if word_limit is not None:
        target_words = max(35, min(word_limit, 320))

    return {"style": style, "word_limit": word_limit, "target_words": target_words}


def apply_word_limit(text: str, target_words: int) -> str:
    words = safe_str(text).split()
    if target_words is None or target_words <= 0 or len(words) <= target_words:
        return safe_str(text).strip()
    trimmed = " ".join(words[:target_words]).strip()
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed


def build_context(query: str, ranked: List[Dict], k: int = TOP_K):
    q_terms = tokenize(query)
    snippets = []
    sections = []
    metadata = []

    for item in ranked:
        r = item["record"]
        text = r["text"]
        low = text.lower()

        hit_positions = [low.find(t) for t in q_terms if t in low]
        if hit_positions:
            idx = min(hit_positions)
            start = max(0, idx - 180)
            end = min(len(text), idx + 560)
            chunk = text[start:end]
        else:
            chunk = text[:560]

        chunk = normalize_text(chunk)
        if not chunk:
            continue

        sections.extend(extract_sections(chunk, r["section"]))

        metadata.append(
            {
                "act_name": r["act_name"],
                "section": r["section"],
                "source": r["source"],
                "score": round(float(item["score"]), 4),
                "vector_score": round(float(item.get("vector_score", 0.0)), 4),
                "lexical_score": round(float(item.get("lexical_score", 0.0)), 4),
            }
        )

        snippets.append(f"[Act: {r['act_name']}] [Section: {r['section']}] {chunk}")
        if len(snippets) >= k:
            break

    sec_out = unique_keep_order([safe_str(s).upper() for s in sections])[:12]
    return snippets, sec_out, metadata


def assess_confidence(top: List[Dict], query: str, domain_name: str) -> Tuple[bool, str]:
    if not top:
        return False, "no_candidates"

    top_score = float(top[0]["score"])
    avg_lex = sum(float(x.get("lexical_score", 0.0)) for x in top[:3]) / max(1, min(3, len(top)))
    query_terms = set(tokenize(query))
    matched_terms = 0
    if query_terms:
        top_text = " ".join(x["record"]["text_norm"][:700] for x in top[:3])
        matched_terms = sum(1 for t in query_terms if t in top_text)
    coverage = matched_terms / max(1, len(query_terms))

    normative = {"acts", "constitution", "criminal_law", "civil_law", "family_law", "traffic_rules"}
    normative_hits = sum(1 for x in top[:5] if x["record"].get("source_group") in normative)

    threshold = MIN_RELEVANCE_SCORE
    if domain_name in {"criminal", "employment", "family"}:
        threshold = max(threshold, 0.38)

    confident = ((top_score >= threshold and avg_lex >= MIN_LEXICAL_SCORE) or coverage >= 0.42)
    if normative_hits == 0:
        confident = False

    reason = f"top_score={top_score:.3f}, avg_lex={avg_lex:.3f}, coverage={coverage:.3f}, normative_hits={normative_hits}"
    return confident, reason

def retrieve_legal_context(query: str):
    domain_name, rule = detect_domain(query)
    candidates = filter_candidates_for_domain(local_index, query, domain_name, rule)
    ranked = compute_hybrid_scores(query, candidates)
    ranked = rerank_with_cross_encoder(query, ranked)

    top = ranked[:TOP_K]
    context, sections, metadata = build_context(query, top, k=TOP_K)
    is_confident, confidence_reason = assess_confidence(top, query, domain_name)

    source = "delta_hybrid_rerank" if reranker is not None else "delta_hybrid"
    return {
        "query": query,
        "domain": domain_name,
        "source": source,
        "context": context,
        "sections": sections,
        "metadata": metadata,
        "top_scores": [round(x["score"], 4) for x in top[:5]],
        "is_confident": is_confident,
        "confidence_reason": confidence_reason,
    }



# ========================================
# CELL 9
# ========================================
def is_helmet_penalty_query(query: str) -> bool:
    q = query.lower()
    return ("helmet" in q or "headgear" in q) and any(x in q for x in ["penalty", "fine", "challan", "punishment"])


def is_section_129_query(query: str) -> bool:
    q = query.lower()
    return ("section 129" in q or re.search(r"\b129\b", q) is not None) and any(
        x in q for x in ["what", "explain", "say", "meaning", "provide", "detail", "short"]
    )


def ensure_helmet_sections(sections: List[str]) -> List[str]:
    priority = ["129", "177", "194D"]
    merged = unique_keep_order([safe_str(x).upper() for x in sections] + priority)
    return merged[:12]


def build_prompt(query: str, retrieval: Dict, preferences: Dict) -> str:
    sections = retrieval["sections"]
    context = retrieval["context"]
    section_text = ", ".join(sections) if sections else "Not clearly identified"
    context_text = "\n\n".join(context) if context else "No context available"
    context_text = context_text[:MAX_CONTEXT_CHARS]

    style = preferences.get("style", "normal")
    target_words = preferences.get("target_words", 150)
    style_hint = {
        "very_short": "Keep it very concise.",
        "short": "Keep it short and direct.",
        "normal": "Keep it clear and moderately detailed.",
        "detailed": "Provide detailed explanation with practical clarity.",
    }.get(style, "Keep it clear and moderately detailed.")

    return f"""
You are an Indian legal information assistant.
Use only the provided legal context.
{style_hint}
Keep total answer near {target_words} words.

Return output with this exact structure:
Law:
<short legal rule>

Penalty:
<penalty with section references>

Why this rule exists:
<one short sentence>

Advice:
<one practical sentence>

Question:
{query}

Relevant Sections (from retrieval):
{section_text}

Legal Context:
{context_text}
"""


def generate_with_backend(prompt: str) -> Tuple[str, str]:
    backend = LLM_BACKEND if isinstance(LLM_BACKEND, dict) else {"type": "none", "name": "", "client": None, "model": None, "errors": []}
    btype = safe_str(backend.get("type")).lower()
    bname = safe_str(backend.get("name"))
    bclient = backend.get("client")

    if btype == "endpoint" and bclient is not None and bname:
        text, err = endpoint_generate_once(bclient, bname, prompt)
        if text:
            return text, "endpoint"
        backend.setdefault("errors", []).append(f"Endpoint generation failed: {err}")

    if btype in ["local", "endpoint"] and local_llm is not None:
        try:
            raw = local_llm(prompt)
            txt = extract_text_from_llm_response(raw)
            if txt:
                return txt.strip(), "local"
        except Exception as e:
            backend.setdefault("errors", []).append(f"Local generation failed: {e}")

    return "", "fallback"


def build_rule_based_answer(query: str, retrieval: Dict, preferences: Dict) -> str:
    if is_helmet_penalty_query(query):
        text = """Law:
Section 129 of the Motor Vehicles Act requires riders to wear protective headgear while riding two-wheelers in public places.

Penalty:
Non-compliance is commonly enforced under Section 177/194D-style traffic penalty provisions, often including monetary fine and possible licence-related action depending on state notification.

Why this rule exists:
Helmet use significantly reduces fatal head injuries during road accidents.

Advice:
Wear a BIS-approved helmet with strap locked, and verify challan rules notified in your state."""
        return apply_word_limit(text, preferences.get("target_words", 150))

    if is_section_129_query(query):
        text = """Law:
Section 129 requires motorcycle riders in public places to wear protective headgear.

Penalty:
Violations may be penalized under general traffic offence provisions such as Section 177/194D-style enforcement, based on applicable state rules.

Why this rule exists:
The section exists to reduce severe head trauma and road fatalities.

Advice:
Always carry and wear a compliant helmet, even for short-distance rides."""
        return apply_word_limit(text, preferences.get("target_words", 150))

    context = normalize_text(" ".join(retrieval.get("context", [])))
    fallback = f"""Law:
Based on retrieved legal context, relevant provisions are summarized.

Penalty:
Penalty depends on the exact section wording and current enforcement rules.

Why this rule exists:
Legal provisions define obligations and consequences for non-compliance.

Advice:
Read the cited section text directly for exact interpretation. Context excerpt: {context[:700]}"""
    return apply_word_limit(fallback, preferences.get("target_words", 150))


def ensure_structured_answer(text: str, target_words: int) -> str:
    raw = safe_str(text).strip()
    if not raw:
        return ""

    required_headers = ["Law:", "Penalty:", "Why this rule exists:", "Advice:"]
    if all(h.lower() in raw.lower() for h in required_headers):
        return apply_word_limit(raw, target_words)

    normalized = apply_word_limit(clean_text(raw), target_words)
    return f"""Law:
{normalized}

Penalty:
Refer to cited sections for exact penalty.

Why this rule exists:
Legal obligations and public safety compliance.

Advice:
Review the relevant statutory section before relying on this summary."""


def generate_answer(query: str):
    retrieval = retrieve_legal_context(query)
    preferences = parse_response_preferences(query)

    if is_helmet_penalty_query(query):
        retrieval["sections"] = ensure_helmet_sections(retrieval["sections"])

    if not retrieval["context"] or not retrieval.get("is_confident", False):
        return {
            "answer": """Law:
No sufficiently relevant legal context was found in the indexed database for this query.

Penalty:
Not available from current indexed corpus.

Why this rule exists:
Retrieval confidence was low for this query in the available legal documents.

Advice:
Re-run embeddings with broader legal corpus or ask a narrower question with act/section details.""",
            "sections": retrieval["sections"],
            "retrieval_source": retrieval["source"],
            "generation_mode": "none",
            "debug": retrieval,
            "preferences": preferences,
        }

    if is_helmet_penalty_query(query) or is_section_129_query(query):
        text = build_rule_based_answer(query, retrieval, preferences)
        mode = "rule_based"
    else:
        prompt = build_prompt(query, retrieval, preferences)
        text, mode = generate_with_backend(prompt)
        text = ensure_structured_answer(text, preferences.get("target_words", 150))
        if not text or "Question:" in text[:220] or len(text.strip()) < 60:
            text = build_rule_based_answer(query, retrieval, preferences)
            mode = "rule_based"

    text = apply_word_limit(text, preferences.get("target_words", 150))
    return {
        "answer": text,
        "sections": retrieval["sections"],
        "retrieval_source": retrieval["source"],
        "generation_mode": mode,
        "debug": retrieval,
        "preferences": preferences,
    }



# ========================================
# CELL 10
# ========================================
def format_output(result: Dict, render_mode: str = "text") -> str:
    sections = result.get("sections", [])
    section_text = ", ".join(sections) if sections else "Refer to applicable legal provisions"
    answer_text = safe_str(result.get("answer", "")).strip()
    retrieval = result.get("debug", {})
    prefs = result.get("preferences", {})

    if render_mode == "html":
        esc = lambda x: (
            safe_str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        return f"""
<div style="font-family:Segoe UI,Arial,sans-serif;line-height:1.45;padding:16px;border:1px solid #d0d7de;border-radius:12px;background:#f8fbff;">
  <div style="font-size:20px;font-weight:700;margin-bottom:8px;">&#9878; Legal Explanation</div>
  <div style="white-space:normal;font-size:14px;">{esc(answer_text)}</div>
  <hr style="margin:12px 0;border:none;border-top:1px solid #e5e7eb;"/>
  <div><b>&#128214; Relevant Sections:</b> {esc(section_text)}</div>
  <div><b>&#129517; Retrieval:</b> {esc(result.get("retrieval_source", "none"))} | confidence={esc(retrieval.get("confidence_reason", "na"))}</div>
  <div><b>&#127919; Style:</b> {esc(prefs.get("style", "normal"))}, target_words={esc(prefs.get("target_words", "na"))}</div>
  <div style="margin-top:10px;color:#555;"><b>&#9888; Disclaimer:</b> AI-generated legal information, not a substitute for professional legal advice.</div>
</div>
"""

    return f"""
[LEGAL EXPLANATION]

{answer_text}

[RELEVANT LEGAL SECTIONS]
{section_text}

[RETRIEVAL SOURCE]
{result.get("retrieval_source", "none")} | {retrieval.get("confidence_reason", "confidence=na")}

[RESPONSE STYLE]
style={prefs.get("style", "normal")}, target_words={prefs.get("target_words", "na")}

[DISCLAIMER]
This response is AI-generated legal information and not a substitute for professional legal advice.
"""


def display_result(result: Dict, prefer_html: bool = True):
    if prefer_html:
        try:
            html = format_output(result, render_mode="html")
            displayHTML(html)  # Databricks utility
            return
        except Exception:
            pass
    print(format_output(result, render_mode="text"))



# ========================================
# CELL 11
# ========================================
RUN_05_DEMO = True

if RUN_05_DEMO:
    query = " tell me the fundamental rights of a citizen in india"
    result = generate_answer(query)
    display_result(result, prefer_html=True)
else:
    print("RUN_05_DEMO=False -> skipped demo query cell.")



# ========================================
# CELL 12
# ========================================
RUN_05_MULTI_TEST = True

if RUN_05_MULTI_TEST:
    for q in [
        " what is person rights ",
        "what is punishment if i kiss a girl without her permission ?",
    ]:
        print("\n" + "=" * 110)
        print("Query:", q)
        r = generate_answer(q)
        display_result(r, prefer_html=True)
else:
    print("RUN_05_MULTI_TEST=False -> skipped multi-query test cell.")

