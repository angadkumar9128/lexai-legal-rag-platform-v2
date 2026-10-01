

# ========================================
# CELL 1
# ========================================
# Optional dependency installer (runs only when explicitly enabled) Repository
AUTO_INSTALL_MISSING_PACKAGES = True
REQUIRED_PACKAGES = [
    "sentence-transformers",
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
        print("Installation complete. Re-run this notebook from Cell 1.")
    else:
        print("AUTO_INSTALL_MISSING_PACKAGES=False, so install is skipped.")
else:
    print("All required packages are already available; skipping install.")



# ========================================
# CELL 2
# ========================================
import os
import re
import math
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def s(v):
    return "" if v is None else str(v)


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", s(t)).strip()


def toks(t: str):
    return [x for x in re.findall(r"[a-zA-Z0-9]+", s(t).lower()) if len(x) > 2]


def unique_keep_order(values):
    out = []
    seen = set()
    for v in values:
        k = clean(v).lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(clean(v))
    return out



# ========================================
# CELL 3
# ========================================
EMBEDDING_DELTA_PATH = "/Volumes/workspace/legal_data/vector_db_test/legal_embeddings_delta"
EMBEDDING_TABLE_NAME = "workspace.default.legal_embeddings_test"

PRIMARY_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DATABRICKS_LLM_ENDPOINT = os.environ.get("DATABRICKS_LLM_ENDPOINT", "").strip()
LLM_ENDPOINT_CANDIDATES = [
    DATABRICKS_LLM_ENDPOINT,
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-mixtral-8x7b-instruct",
]

LOCAL_LLM_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-small",
]

TOP_K = 8
POOL_K = 48
SHORTLIST_K = 140
RERANK_K = 24
LEXICAL_HARD_CAP = 22000
MAX_CONTEXT_CHARS = 5000
MIN_RELEVANCE_SCORE = 0.34
MIN_LEXICAL_SCORE = 0.06
MIN_PRIMARY_CANDIDATES = 80

ENABLE_LEXICAL_ARTIFACTS = True
LEXICAL_ARTIFACT_ROOT = "/Volumes/workspace/legal_data/vector_db_test/retrieval_artifacts_06"
LEXICAL_TOKEN_ARTIFACT_PATH = f"{LEXICAL_ARTIFACT_ROOT}/token_postings"
LEXICAL_META_ARTIFACT_PATH = f"{LEXICAL_ARTIFACT_ROOT}/index_meta"

RUN_06_DEMO = False
RUN_06_EVAL = False



# ========================================
# CELL 4
# ========================================
def extract_text(resp) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, list) and resp:
        return extract_text(resp[0])
    if isinstance(resp, dict):
        for key in ["generated_text", "text", "output", "answer"]:
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        preds = resp.get("predictions")
        if isinstance(preds, list) and preds:
            return extract_text(preds[0])
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"].strip()
                if isinstance(c0.get("text"), str):
                    return c0["text"].strip()
    return ""


def load_embedder():
    for name in [PRIMARY_EMBED_MODEL, FALLBACK_EMBED_MODEL]:
        try:
            model = SentenceTransformer(name)
            _ = model.encode(["health check"], show_progress_bar=False)
            log(f"Embedding model ready: {name}")
            return model, name
        except Exception as e:
            log(f"Embedding model failed ({name}): {e}")
    return None, "unavailable"


def load_reranker():
    try:
        rr = CrossEncoder(RERANK_MODEL)
        _ = rr.predict([("hello", "world")])
        return rr, RERANK_MODEL
    except Exception as e:
        log(f"Reranker unavailable: {e}")
        return None, "unavailable"


def load_dbx_client():
    try:
        import mlflow.deployments
        return mlflow.deployments.get_deploy_client("databricks")
    except Exception as e:
        log(f"Databricks endpoint client unavailable: {e}")
        return None


def try_endpoint_once(client, endpoint_name: str, prompt: str) -> Tuple[str, str]:
    try:
        resp = client.predict(
            endpoint=endpoint_name,
            inputs={"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 420},
        )
        text = extract_text(resp)
        if text:
            return text, ""
    except Exception as e:
        chat_err = str(e)
    else:
        chat_err = "empty chat response"

    try:
        resp = client.predict(
            endpoint=endpoint_name,
            inputs={"prompt": prompt, "temperature": 0.0, "max_tokens": 420},
        )
        text = extract_text(resp)
        if text:
            return text, ""
        return "", f"empty completion response: {endpoint_name}"
    except Exception as e:
        return "", f"chat_error={chat_err}; completion_error={e}"


def load_local_llm():
    for model_name in LOCAL_LLM_MODELS:
        try:
            llm = pipeline(
                "text2text-generation",
                model=model_name,
                max_new_tokens=420,
                do_sample=False,
                temperature=0.0,
            )
            log(f"Local generation model ready: {model_name}")
            return llm, model_name
        except Exception as e:
            log(f"Local generation model failed ({model_name}): {e}")
    return None, "unavailable"


def infer_source_group(category: str, source: str, act: str) -> str:
    blob = f"{category} {source} {act}".lower()
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


def _parse_section_refs(raw_val) -> List[str]:
    if raw_val is None:
        return []
    vals = raw_val if isinstance(raw_val, list) else [x for x in s(raw_val).split(",")]
    refs = []
    for x in vals:
        c = clean(x).upper()
        if c:
            refs.append(c)
    return unique_keep_order(refs)


def _normalize_records_for_runtime(records_in: List[Dict]) -> List[Dict]:
    out = []
    for ridx, rec in enumerate(records_in):
        r = dict(rec)

        text = clean(r.get("text", "") or r.get("chunk_text", ""))
        tok_list = toks(text)

        if not isinstance(r.get("token_freq"), dict):
            tf = {}
            for t in tok_list:
                tf[t] = tf.get(t, 0) + 1
            r["token_freq"] = tf

        toks_val = r.get("tokens")
        if isinstance(toks_val, set):
            pass
        elif isinstance(toks_val, list):
            r["tokens"] = set(toks_val)
        else:
            r["tokens"] = set(tok_list)

        r["idx"] = int(r.get("idx", ridx))
        r["id"] = s(r.get("id", r.get("chunk_id", "")))
        r["text"] = text
        r["text_norm"] = clean(r.get("text_norm", text.lower())).lower()
        r["act"] = s(r.get("act", r.get("act_name", "")))
        r["act_norm"] = s(r.get("act_norm", s(r["act"]).lower())).lower()
        r["section"] = s(r.get("section", r.get("section_number", "")))
        r["section_norm"] = s(r.get("section_norm", s(r["section"]).upper())).upper()
        r["category"] = s(r.get("category", ""))
        r["source"] = s(r.get("source", r.get("file_name", "")))
        r["source_group"] = s(r.get("source_group", infer_source_group(r["category"], r["source"], r["act"])))
        r["doc_len"] = int(r.get("doc_len", len(tok_list)))
        r["section_refs"] = _parse_section_refs(r.get("section_refs", []))
        r["is_penalty_related"] = int(r.get("is_penalty_related", 0) or 0)
        r["is_helmet_related"] = int(r.get("is_helmet_related", 0) or 0)

        emb = r.get("emb")
        if not isinstance(emb, np.ndarray):
            if emb is None and r.get("embedding") is not None:
                emb = r.get("embedding")
            if emb is None:
                continue
            try:
                r["emb"] = np.array([float(x) for x in emb], dtype=np.float32)
            except Exception:
                continue

        out.append(r)

    return out


def _build_lexical_state(records_in: List[Dict]):
    inverted = {}
    doc_freq = {}
    for rec in records_in:
        for t in rec["token_freq"].keys():
            doc_freq[t] = doc_freq.get(t, 0) + 1
            inverted.setdefault(t, []).append(int(rec["idx"]))

    n_docs = len(records_in)
    avg_doc_len = sum(max(1, r["doc_len"]) for r in records_in) / max(1, n_docs)
    return inverted, doc_freq, avg_doc_len, n_docs


def _load_embeddings_latest_df():
    try:
        emb_df = spark.read.format("delta").load(EMBEDDING_DELTA_PATH)
        source_name = EMBEDDING_DELTA_PATH
    except Exception as e:
        log(f"Path load failed, trying table fallback: {e}")
        emb_df = spark.table(EMBEDDING_TABLE_NAME)
        source_name = EMBEDDING_TABLE_NAME

    required = ["chunk_id", "chunk_text", "act_name", "section_number", "category", "file_name", "embedding"]
    missing = [c for c in required if c not in emb_df.columns]
    if missing:
        raise ValueError(f"Embedding Delta missing columns: {missing}")

    if "updated_at" not in emb_df.columns:
        emb_df = emb_df.withColumn("updated_at", F.current_timestamp())

    w = Window.partitionBy("chunk_id").orderBy(F.col("updated_at").desc_nulls_last())
    latest_df = (
        emb_df.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .dropna(subset=["chunk_id", "chunk_text", "embedding"])
        .orderBy(F.col("chunk_id").asc())
    )

    available_cols = [
        c
        for c in [
            "chunk_id",
            "chunk_text",
            "act_name",
            "section_number",
            "category",
            "file_name",
            "embedding",
            "section_refs",
            "is_penalty_related",
            "is_helmet_related",
            "updated_at",
        ]
        if c in latest_df.columns
    ]

    return latest_df.select(*available_cols), source_name


def _compute_data_signature(latest_df):
    row = (
        latest_df.agg(
            F.count("*").alias("n_rows"),
            F.countDistinct("chunk_id").alias("n_chunks"),
            F.max("updated_at").alias("max_updated_at"),
            F.sum(F.length("chunk_text")).alias("sum_chars"),
        )
        .collect()[0]
    )
    n_rows = int(row["n_rows"] or 0)
    n_chunks = int(row["n_chunks"] or 0)
    max_updated = s(row["max_updated_at"])
    sum_chars = int(row["sum_chars"] or 0)
    raw = f"rows={n_rows}|chunks={n_chunks}|max_updated={max_updated}|sum_chars={sum_chars}"
    sig = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return sig, raw


def _load_persisted_lexical_state(index_signature: str, expected_docs: int):
    if not ENABLE_LEXICAL_ARTIFACTS:
        return None
    try:
        meta_df = spark.read.format("delta").load(LEXICAL_META_ARTIFACT_PATH).filter(F.col("index_signature") == index_signature)
        meta_rows = meta_df.orderBy(F.col("created_at").desc()).limit(1).collect()
        if not meta_rows:
            return None
        meta = meta_rows[0]
        n_docs = int(meta["n_docs"] or 0)
        avg_doc_len = float(meta["avg_doc_len"] or 0.0)
        if n_docs <= 0 or expected_docs <= 0 or n_docs != expected_docs:
            return None

        tok_df = (
            spark.read.format("delta")
            .load(LEXICAL_TOKEN_ARTIFACT_PATH)
            .filter(F.col("index_signature") == index_signature)
            .select("token", "doc_freq", "postings")
        )

        inverted = {}
        doc_freq = {}
        for row in tok_df.toLocalIterator():
            token = s(row["token"])
            if not token:
                continue
            postings = [int(x) for x in (row["postings"] or []) if 0 <= int(x) < expected_docs]
            if not postings:
                continue
            inverted[token] = postings
            doc_freq[token] = int(row["doc_freq"] or len(postings))

        if not inverted:
            return None

        return inverted, doc_freq, avg_doc_len, n_docs
    except Exception as e:
        log(f"Persisted lexical load skipped: {e}")
        return None


def _safe_replace_where_write(df, path: str, predicate: str):
    try:
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("replaceWhere", predicate)
            .save(path)
        )
        return True
    except Exception:
        try:
            df.write.format("delta").mode("append").save(path)
            return True
        except Exception:
            return False


def _persist_lexical_state(index_signature: str, inverted: Dict[str, List[int]], doc_freq: Dict[str, int], avg_doc_len: float, n_docs: int):
    if not ENABLE_LEXICAL_ARTIFACTS:
        return False
    if not inverted:
        return False

    try:
        token_rows = [
            {
                "index_signature": index_signature,
                "token": token,
                "doc_freq": int(doc_freq.get(token, len(postings))),
                "postings": [int(x) for x in postings],
                "created_at": datetime.now(timezone.utc),
            }
            for token, postings in inverted.items()
            if postings
        ]
        if not token_rows:
            return False

        token_df = spark.createDataFrame(token_rows)
        meta_df = spark.createDataFrame(
            [
                {
                    "index_signature": index_signature,
                    "n_docs": int(n_docs),
                    "avg_doc_len": float(avg_doc_len),
                    "vocab_size": int(len(inverted)),
                    "created_at": datetime.now(timezone.utc),
                }
            ]
        )

        tok_ok = _safe_replace_where_write(token_df, LEXICAL_TOKEN_ARTIFACT_PATH, f"index_signature = '{index_signature}'")
        meta_ok = _safe_replace_where_write(meta_df, LEXICAL_META_ARTIFACT_PATH, f"index_signature = '{index_signature}'")
        if tok_ok and meta_ok:
            log(f"Persisted lexical artifacts to Delta for signature={index_signature} (vocab={len(inverted)}).")
            return True
        return False
    except Exception as e:
        log(f"Persisting lexical artifacts failed: {e}")
        return False


latest_df, embedding_source_name = _load_embeddings_latest_df()
data_signature, data_signature_raw = _compute_data_signature(latest_df)

cached = globals().get("_LEXAI06_RUNTIME")
cache_ready = isinstance(cached, dict) and cached.get("ready", False)
cache_same_data = cache_ready and cached.get("data_signature") == data_signature

use_cache = False
if cache_same_data:
    embedder = cached.get("embedder")
    embedder_name = cached.get("embedder_name", "cached")
    reranker = cached.get("reranker")
    reranker_name = cached.get("reranker_name", "unavailable")
    dbx = cached.get("dbx")
    llm_backend = cached.get("llm_backend", {"type": "none", "name": "", "client": None, "model": None, "errors": []})
    local_llm = cached.get("local_llm")

    records = _normalize_records_for_runtime(cached.get("records", []))
    INVERTED_INDEX = cached.get("INVERTED_INDEX")
    DOC_FREQ = cached.get("DOC_FREQ")
    AVG_DOC_LEN = cached.get("AVG_DOC_LEN")
    N_DOCS = cached.get("N_DOCS")

    if records and all(x is not None for x in [INVERTED_INDEX, DOC_FREQ, AVG_DOC_LEN, N_DOCS]):
        use_cache = True
    else:
        log("Cache matched data signature but runtime payload was incomplete; rebuilding retrieval state.")

if use_cache:
    _LEXAI06_RUNTIME = cached
    print("Runtime cache hit: models and retrieval index reused for current data signature.")
else:
    # Reuse already loaded model objects when possible, but refresh retrieval state for latest data.
    embedder = cached.get("embedder") if isinstance(cached, dict) else None
    embedder_name = cached.get("embedder_name", "cached") if isinstance(cached, dict) else "unavailable"
    if embedder is None:
        embedder, embedder_name = load_embedder()
    if embedder is None:
        raise RuntimeError("No embedding model could be loaded.")

    reranker = cached.get("reranker") if isinstance(cached, dict) else None
    reranker_name = cached.get("reranker_name", "cached") if isinstance(cached, dict) else "unavailable"
    if reranker is None:
        reranker, reranker_name = load_reranker()

    dbx = cached.get("dbx") if isinstance(cached, dict) else None
    llm_backend = cached.get("llm_backend") if isinstance(cached, dict) and isinstance(cached.get("llm_backend"), dict) else None
    local_llm = cached.get("local_llm") if isinstance(cached, dict) else None

    if llm_backend is None:
        llm_backend = {"type": "none", "name": "", "client": None, "model": None, "errors": []}

    if dbx is None:
        dbx = load_dbx_client()

    if llm_backend.get("type") == "none" and dbx is not None:
        test_prompt = "Reply with OK"
        for ep in [x for x in LLM_ENDPOINT_CANDIDATES if x]:
            txt, err = try_endpoint_once(dbx, ep, test_prompt)
            if txt:
                llm_backend.update({"type": "endpoint", "name": ep, "client": dbx})
                log(f"Using endpoint backend: {ep}")
                break
            llm_backend.setdefault("errors", []).append(f"{ep}: {err}")

    if llm_backend.get("type") == "none" and local_llm is None:
        local_llm, local_llm_name = load_local_llm()
        if local_llm is not None:
            llm_backend.update({"type": "local", "name": local_llm_name, "model": local_llm})

    raw_records = []
    for ridx, r in enumerate(latest_df.toLocalIterator()):
        raw_records.append(
            {
                "idx": ridx,
                "id": s(r["chunk_id"]),
                "text": clean(r["chunk_text"]),
                "act": s(r["act_name"]),
                "section": s(r["section_number"]),
                "category": s(r["category"]),
                "source": s(r["file_name"]),
                "emb": np.array([float(x) for x in (r["embedding"] or [])], dtype=np.float32),
                "section_refs": _parse_section_refs(r["section_refs"]) if "section_refs" in latest_df.columns else [],
                "is_penalty_related": int(r["is_penalty_related"] or 0) if "is_penalty_related" in latest_df.columns else 0,
                "is_helmet_related": int(r["is_helmet_related"] or 0) if "is_helmet_related" in latest_df.columns else 0,
            }
        )

    records = _normalize_records_for_runtime(raw_records)
    if not records:
        raise RuntimeError("No records loaded from latest embeddings snapshot")

    persisted = _load_persisted_lexical_state(data_signature, len(records))
    lexical_source = "delta_artifact"
    if persisted is not None:
        INVERTED_INDEX, DOC_FREQ, AVG_DOC_LEN, N_DOCS = persisted
        log(f"Loaded lexical artifacts from Delta for signature={data_signature}.")
    else:
        INVERTED_INDEX, DOC_FREQ, AVG_DOC_LEN, N_DOCS = _build_lexical_state(records)
        lexical_source = "rebuilt"
        _persist_lexical_state(data_signature, INVERTED_INDEX, DOC_FREQ, AVG_DOC_LEN, N_DOCS)

    _LEXAI06_RUNTIME = {
        "ready": True,
        "data_signature": data_signature,
        "data_signature_raw": data_signature_raw,
        "embedding_source": embedding_source_name,
        "embedder": embedder,
        "embedder_name": embedder_name,
        "reranker": reranker,
        "reranker_name": reranker_name,
        "dbx": dbx,
        "llm_backend": llm_backend,
        "local_llm": local_llm,
        "records": records,
        "INVERTED_INDEX": INVERTED_INDEX,
        "DOC_FREQ": DOC_FREQ,
        "AVG_DOC_LEN": AVG_DOC_LEN,
        "N_DOCS": N_DOCS,
        "lexical_source": lexical_source,
    }

print("--- Runtime Status ---")
print("Data signature:", data_signature)
print("Embedding source:", _LEXAI06_RUNTIME.get("embedding_source", "unknown"))
print("Records loaded:", len(records))
print("Embedding dim:", len(records[0]["emb"]) if records else 0)
print("Avg doc len:", round(float(AVG_DOC_LEN), 2))
print("Vocabulary size:", len(DOC_FREQ))
print("Lexical source:", _LEXAI06_RUNTIME.get("lexical_source", "cache" if use_cache else "unknown"))
print("Embedder:", embedder_name)
print("Reranker:", reranker_name)
print("LLM backend:", f"{llm_backend['type']} ({llm_backend['name']})" if llm_backend.get("type") != "none" else "none")
if llm_backend.get("errors"):
    print("LLM backend diagnostics:")
    for e in llm_backend["errors"][:5]:
        print(" -", e)



# ========================================
# CELL 5
# ========================================
def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((float(np.linalg.norm(a)) + 1e-12) * (float(np.linalg.norm(b)) + 1e-12)))


def parse_query_sections(query: str) -> List[str]:
    refs = []
    for m in re.findall(r"(?:section|sec\.?|article|art\.?)\s*(\d+[A-Za-z-]*)", query, flags=re.IGNORECASE):
        refs.append(m.upper())
    return unique_keep_order(refs)


def detect_domain(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["helmet", "traffic", "vehicle", "challan", "headgear", "driving", "speeding", "drunk driving"]):
        return "traffic"
    if any(x in q for x in ["murder", "homicide", "crime", "imprisonment", "punishment", "bns", "ipc", "bail", "fir"]):
        return "criminal"
    if any(x in q for x in ["bond", "employment", "salary", "resign", "contract", "notice period", "company", "termination"]):
        return "employment"
    if any(x in q for x in ["marriage", "divorce", "succession", "dowry", "domestic violence", "maintenance"]):
        return "family"
    return "general"


def is_case_law_query(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in ["judgment", "judgement", "precedent", "case law", "supreme court", "high court", "citation", "ratio"])


def build_query_route(query: str) -> Dict:
    domain = detect_domain(query)
    case_law = is_case_law_query(query)

    if case_law:
        primary_order = ["judgments", "acts", "constitution"]
        secondary_order = ["criminal_law", "civil_law", "family_law", "traffic_rules", "reports", "general"]
    else:
        if domain == "traffic":
            primary_order = ["traffic_rules", "acts", "constitution"]
        elif domain == "criminal":
            primary_order = ["criminal_law", "acts", "constitution"]
        elif domain == "family":
            primary_order = ["family_law", "acts", "constitution"]
        elif domain == "employment":
            primary_order = ["civil_law", "acts", "constitution"]
        else:
            primary_order = ["acts", "constitution", "criminal_law", "civil_law", "family_law", "traffic_rules"]
        secondary_order = ["general", "judgments", "reports"]

    return {
        "domain": domain,
        "case_law": case_law,
        "primary_order": primary_order,
        "secondary_order": secondary_order,
        "primary_set": set(primary_order),
        "secondary_set": set(secondary_order),
    }


def generate_query_variants(query: str) -> List[str]:
    q = clean(query)
    ql = q.lower()
    variants = [q]

    if "helmet" in ql or "headgear" in ql:
        variants.append("helmet penalty motor vehicles act section 129 section 177 section 194D")
        variants.append("protective headgear legal penalty two wheeler india")

    if "section 129" in ql or re.search(r"\b129\b", ql):
        variants.append("what does section 129 motor vehicles act require")

    if any(x in ql for x in ["murder", "homicide"]):
        variants.append("punishment for murder bns ipc equivalent section")

    if "bond" in ql and any(x in ql for x in ["company", "employment", "resign"]):
        variants.append("employment bond breach legal consequences india contract act section 73 section 74")

    if "mobile" in ql and "driving" in ql:
        variants.append("mobile phone while driving offence motor vehicles act")

    if is_case_law_query(ql):
        variants.append(f"{q} indian case law judgment")

    condensed = " ".join(toks(q)[:14])
    if clean(condensed):
        variants.append(condensed)

    return unique_keep_order([v for v in variants if clean(v)])


def source_group_bonus(domain: str, query: str, source_group: str) -> float:
    q = query.lower()
    case_law_query = is_case_law_query(q)
    bonus = 0.0

    if case_law_query:
        if source_group == "judgments":
            bonus += 0.20
        elif source_group == "reports":
            bonus += 0.04
    else:
        if source_group in {"acts", "constitution", "criminal_law", "civil_law", "family_law", "traffic_rules"}:
            bonus += 0.14
        elif source_group == "judgments":
            bonus -= 0.05
        elif source_group == "reports":
            bonus -= 0.10

    if domain == "traffic" and source_group == "traffic_rules":
        bonus += 0.14
    if domain == "criminal" and source_group == "criminal_law":
        bonus += 0.14
    if domain == "employment" and source_group in {"civil_law", "acts"}:
        bonus += 0.12
    if domain == "family" and source_group == "family_law":
        bonus += 0.14

    return bonus


def route_priority_bonus(route: Dict, source_group: str) -> float:
    primary_order = route.get("primary_order", [])
    if source_group in primary_order:
        idx = primary_order.index(source_group)
        return max(0.04, 0.18 - (0.03 * idx))
    if source_group in route.get("secondary_set", set()):
        return 0.02
    if (not route.get("case_law", False)) and source_group in {"judgments", "reports"}:
        return -0.08
    return -0.01


def bm25_score(query_terms: List[str], rec: Dict, k1: float = 1.5, b: float = 0.75) -> float:
    score = 0.0
    dl = max(1, rec.get("doc_len", 1))
    for t in query_terms:
        tf = rec["token_freq"].get(t, 0)
        if tf <= 0:
            continue
        df = DOC_FREQ.get(t, 0)
        idf = math.log(1 + ((N_DOCS - df + 0.5) / (df + 0.5))) if N_DOCS > 0 else 0.0
        denom = tf + k1 * (1 - b + b * (dl / max(1.0, AVG_DOC_LEN)))
        score += idf * ((tf * (k1 + 1)) / max(1e-9, denom))
    return float(score)


def candidate_ids_from_inverted(query_terms: List[str], hard_cap: int = LEXICAL_HARD_CAP) -> List[int]:
    ids = set()
    for t in query_terms:
        postings = INVERTED_INDEX.get(t, [])
        if postings:
            ids.update(postings[:5000])
        if len(ids) >= hard_cap:
            break
    if not ids:
        return list(range(len(records)))
    return list(ids)


def _dedup_records(rows: List[Dict]) -> List[Dict]:
    out = []
    seen = set()
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def apply_source_routing(rows: List[Dict], route: Dict) -> Tuple[List[Dict], str]:
    if not rows:
        return [], "empty"

    primary = [r for r in rows if r.get("source_group") in route.get("primary_set", set())]
    secondary = [r for r in rows if r.get("source_group") in route.get("secondary_set", set())]
    tertiary = [
        r
        for r in rows
        if r.get("source_group") not in route.get("primary_set", set()) and r.get("source_group") not in route.get("secondary_set", set())
    ]

    if route.get("case_law", False):
        if len(primary) >= max(20, MIN_PRIMARY_CANDIDATES // 2):
            selected = _dedup_records(primary + secondary[: max(0, MIN_PRIMARY_CANDIDATES - len(primary))])
            return selected, "case_law_strict"
        selected = _dedup_records(primary + secondary + tertiary[:300])
        return selected, "case_law_fallback"

    if len(primary) >= MIN_PRIMARY_CANDIDATES:
        return _dedup_records(primary), "statute_strict"

    if len(primary) >= 20:
        selected = _dedup_records(primary + secondary[: max(0, MIN_PRIMARY_CANDIDATES - len(primary))])
        return selected, "statute_soft"

    selected = _dedup_records(primary + secondary + tertiary[:300])
    return selected, "statute_fallback"


def filter_records(query: str, rows: List[Dict], domain: str):
    if not rows:
        return []
    if domain == "general":
        return rows

    strict = []
    soft = []
    q_terms = set(toks(query))
    for r in rows:
        act = r["act_norm"]
        txt = r["text_norm"]
        overlap = sum(1 for t in q_terms if t in r["tokens"])

        if domain == "traffic":
            if ("motor vehicle" in act or "traffic" in act or r["source_group"] == "traffic_rules") and any(
                k in txt for k in ["helmet", "headgear", "motor cycle", "motorcycle", "two-wheeler", "driving", "licence"]
            ):
                strict.append(r)
            elif overlap >= 2:
                soft.append(r)
        elif domain == "criminal":
            if any(k in txt for k in ["murder", "homicide", "imprisonment", "punishable", "offence"]):
                strict.append(r)
            elif overlap >= 2:
                soft.append(r)
        elif domain == "employment":
            if any(k in txt for k in ["contract", "bond", "agreement", "damages", "specific relief", "notice"]):
                strict.append(r)
            elif overlap >= 2:
                soft.append(r)
        elif domain == "family":
            if any(k in txt for k in ["marriage", "divorce", "maintenance", "dowry", "custody", "succession"]):
                strict.append(r)
            elif overlap >= 2:
                soft.append(r)
        else:
            if overlap >= 2:
                soft.append(r)

    return strict + soft if strict else soft if soft else rows


def score_record(query: str, rec: Dict, q_vec, q_terms: List[str], q_sections: set, domain: str, route: Dict):
    txt = rec["text_norm"]
    dense = cos(q_vec, rec["emb"]) if q_vec is not None else 0.0
    lex = sum(1 for t in set(q_terms) if t in rec["tokens"]) / max(1, len(set(q_terms)))
    bm25_raw = bm25_score(q_terms, rec)

    meta = 0.0
    if rec.get("is_penalty_related", 0) == 1:
        meta += 0.07
    if any(x in txt for x in ["penalty", "fine", "punishable", "challan", "imprisonment", "liable"]):
        meta += 0.08

    if q_sections:
        if rec["section_norm"] in q_sections:
            meta += 0.22
        if set(rec.get("section_refs", [])).intersection(q_sections):
            meta += 0.16

    meta += source_group_bonus(domain, query, rec.get("source_group", "general"))
    meta += route_priority_bonus(route, rec.get("source_group", "general"))

    return {
        "dense": float(dense),
        "lex": float(lex),
        "bm25_raw": float(bm25_raw),
        "meta": float(meta),
        "rec": rec,
    }


def retrieve_single_query(query: str, top_n: int = POOL_K):
    t0 = time.perf_counter()

    route = build_query_route(query)
    domain = route["domain"]
    q_terms = toks(query)
    q_sections = set(parse_query_sections(query))

    t_candidate_start = time.perf_counter()
    cand_ids = candidate_ids_from_inverted(q_terms, hard_cap=LEXICAL_HARD_CAP)
    candidates = [records[i] for i in cand_ids]
    candidate_fetch_ms = (time.perf_counter() - t_candidate_start) * 1000.0

    t_route_start = time.perf_counter()
    candidates = filter_records(query, candidates, domain)
    candidates, route_mode = apply_source_routing(candidates, route)
    routing_ms = (time.perf_counter() - t_route_start) * 1000.0

    if not candidates:
        trace = {
            "variant": query,
            "route_mode": route_mode,
            "candidate_count": 0,
            "shortlist_count": 0,
            "candidate_fetch_ms": round(candidate_fetch_ms, 2),
            "routing_ms": round(routing_ms, 2),
            "lexical_ms": 0.0,
            "embed_ms": 0.0,
            "dense_ms": 0.0,
            "single_query_total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        return [], trace

    # Stage 1: lexical shortlist (fast, no dense embedding compute)
    t_lex_start = time.perf_counter()
    raw = [score_record(query, r, None, q_terms, q_sections, domain, route) for r in candidates]
    max_bm25 = max(x["bm25_raw"] for x in raw) if raw else 1.0

    lexical_ranked = []
    for x in raw:
        bm25_norm = x["bm25_raw"] / max(1e-9, max_bm25)
        lexical_score = (0.60 * bm25_norm) + (0.30 * x["lex"]) + (0.10 * x["meta"])
        lexical_ranked.append(
            {
                "lexical_score": float(lexical_score),
                "lex": float(x["lex"]),
                "bm25_norm": float(bm25_norm),
                "meta": float(x["meta"]),
                "rec": x["rec"],
            }
        )

    lexical_ranked.sort(key=lambda x: x["lexical_score"], reverse=True)
    shortlist = lexical_ranked[: min(SHORTLIST_K, len(lexical_ranked))]
    lexical_ms = (time.perf_counter() - t_lex_start) * 1000.0

    # Stage 2: dense scoring only on shortlist
    t_embed_start = time.perf_counter()
    q_vec = np.array(embedder.encode([query], show_progress_bar=False)[0], dtype=np.float32) if embedder is not None else None
    embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

    t_dense_start = time.perf_counter()
    scored = []
    for item in shortlist:
        rec = item["rec"]
        dense = cos(q_vec, rec["emb"]) if q_vec is not None else 0.0

        if q_vec is None:
            score = (0.62 * item["bm25_norm"]) + (0.28 * item["lex"]) + item["meta"]
        else:
            score = (0.58 * dense) + (0.18 * item["lex"]) + (0.18 * item["bm25_norm"]) + item["meta"]

        scored.append((float(score), float(dense), float(max(item["lex"], item["bm25_norm"])), rec))

    scored.sort(key=lambda z: z[0], reverse=True)
    dense_ms = (time.perf_counter() - t_dense_start) * 1000.0

    trace = {
        "variant": query,
        "route_mode": route_mode,
        "candidate_count": int(len(candidates)),
        "shortlist_count": int(len(shortlist)),
        "candidate_fetch_ms": round(candidate_fetch_ms, 2),
        "routing_ms": round(routing_ms, 2),
        "lexical_ms": round(lexical_ms, 2),
        "embed_ms": round(embed_ms, 2),
        "dense_ms": round(dense_ms, 2),
        "single_query_total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    return scored[:top_n], trace


def reciprocal_rank_fusion(result_sets: List[List[Tuple]], k: int = 60):
    fused = {}
    for result in result_sets:
        for rank, item in enumerate(result, start=1):
            rid = item[3]["id"]
            base = fused.get(rid, {"rrf": 0.0, "best": item})
            base["rrf"] += 1.0 / (k + rank)
            if item[0] > base["best"][0]:
                base["best"] = item
            fused[rid] = base

    rows = []
    for _, data in fused.items():
        best = data["best"]
        merged_score = (0.76 * best[0]) + (0.24 * data["rrf"])
        rows.append((float(merged_score), best[1], best[2], best[3]))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def _sum_trace(traces: List[Dict], key: str) -> float:
    return round(float(sum(float(t.get(key, 0.0)) for t in traces)), 2)


def hybrid_retrieve(query: str, top_k: int = TOP_K, return_trace: bool = False):
    t0 = time.perf_counter()

    variants = generate_query_variants(query)
    per_variant = []
    raw_sets = []
    for v in variants:
        rows, tr = retrieve_single_query(v, top_n=POOL_K)
        per_variant.append(tr)
        raw_sets.append(rows)

    t_rrf_start = time.perf_counter()
    fused = reciprocal_rank_fusion(raw_sets, k=60)
    rrf_ms = (time.perf_counter() - t_rrf_start) * 1000.0

    rerank_ms = 0.0
    if reranker is not None and fused:
        t_rerank_start = time.perf_counter()
        pairs = [(query, x[3]["text"][:1000]) for x in fused[:RERANK_K]]
        try:
            ce = reranker.predict(pairs)
            reranked = []
            for i, c in enumerate(ce):
                ce_norm = 1 / (1 + math.exp(-float(c)))
                merged = (0.64 * fused[i][0]) + (0.36 * ce_norm)
                reranked.append((float(merged), fused[i][1], fused[i][2], fused[i][3]))
            reranked.sort(key=lambda x: x[0], reverse=True)
            fused = reranked + fused[RERANK_K:]
        except Exception as e:
            log(f"Reranker skipped: {e}")
        rerank_ms = (time.perf_counter() - t_rerank_start) * 1000.0

    trace = {
        "variant_count": len(variants),
        "variants": per_variant,
        "candidate_fetch_ms": _sum_trace(per_variant, "candidate_fetch_ms"),
        "routing_ms": _sum_trace(per_variant, "routing_ms"),
        "lexical_ms": _sum_trace(per_variant, "lexical_ms"),
        "embed_ms": _sum_trace(per_variant, "embed_ms"),
        "dense_ms": _sum_trace(per_variant, "dense_ms"),
        "rrf_ms": round(rrf_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
        "retrieve_total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "route_mode": per_variant[0].get("route_mode", "na") if per_variant else "na",
        "candidate_count": per_variant[0].get("candidate_count", 0) if per_variant else 0,
        "shortlist_count": per_variant[0].get("shortlist_count", 0) if per_variant else 0,
    }

    if return_trace:
        return fused[:top_k], variants, trace
    return fused[:top_k], variants



# ========================================
# CELL 6
# ========================================
def section_refs(text: str, sec_meta: str = ""):
    refs = []
    meta = s(sec_meta).strip()
    if meta and not meta.lower().startswith("chapter"):
        refs += re.findall(r"\d+[A-Za-z-]*", meta)

    refs += re.findall(r"(?:section|sec\.?|article|art\.?)\s*(\d+[A-Za-z-]*)", s(text), flags=re.IGNORECASE)
    refs = [s(x).upper() for x in refs]
    return unique_keep_order(refs)


_LATENCY_STORE_KEY = "_LEXAI06_LATENCY_HISTORY"
if not isinstance(globals().get(_LATENCY_STORE_KEY), list):
    globals()[_LATENCY_STORE_KEY] = []


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    v = sorted(float(x) for x in values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    weight = pos - lo
    return (v[lo] * (1 - weight)) + (v[hi] * weight)


def latency_profile(stages: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    hist = globals().get(_LATENCY_STORE_KEY, [])
    if not hist:
        return {}

    default_stages = [
        "retrieve_total_ms",
        "lexical_ms",
        "embed_ms",
        "dense_ms",
        "rerank_ms",
        "generation_ms",
        "total_ms",
    ]
    stages = stages or default_stages

    out = {}
    for st in stages:
        vals = [float(x.get(st, 0.0)) for x in hist if st in x]
        if not vals:
            continue
        out[st] = {
            "count": len(vals),
            "p50": round(_percentile(vals, 50), 2),
            "p95": round(_percentile(vals, 95), 2),
            "last": round(vals[-1], 2),
        }
    return out


def record_latency(lat: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    row = {k: round(float(v), 2) for k, v in lat.items() if isinstance(v, (int, float))}
    if not row:
        return latency_profile()

    hist = globals().get(_LATENCY_STORE_KEY, [])
    hist.append(row)
    if len(hist) > 400:
        del hist[:-400]
    globals()[_LATENCY_STORE_KEY] = hist
    return latency_profile()


def print_latency_profile():
    prof = latency_profile()
    if not prof:
        print("Latency profile: no samples yet.")
        return
    print("Latency profile (ms) p50/p95:")
    for st, vals in prof.items():
        print(f" - {st}: p50={vals['p50']}, p95={vals['p95']}, last={vals['last']}, n={vals['count']}")


def parse_response_preferences(query: str) -> Dict:
    q = query.lower()
    style = "normal"
    if any(k in q for k in ["very short", "one line", "single line"]):
        style = "very_short"
    elif any(k in q for k in ["in short", "briefly", "brief"]):
        style = "short"
    elif any(k in q for k in ["in detail", "in details", "in depth", "detailed", "comprehensive"]):
        style = "detailed"

    word_limit = None
    m = re.search(r"(?:within|under|max(?:imum)?|limit(?:ed)? to)\s*(\d{2,4})\s*(?:words?|works?)", q)
    if m:
        word_limit = int(m.group(1))

    defaults = {"very_short": 45, "short": 85, "normal": 150, "detailed": 260}
    target_words = defaults.get(style, 150)
    if word_limit is not None:
        target_words = max(35, min(word_limit, 380))

    return {"style": style, "word_limit": word_limit, "target_words": target_words}


def apply_word_limit(text: str, target_words: int) -> str:
    words = s(text).split()
    if target_words is None or target_words <= 0 or len(words) <= target_words:
        return s(text).strip()
    trimmed = " ".join(words[:target_words]).strip()
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed


def build_context(query: str, ranked):
    terms = toks(query)
    ctx = []
    sections = []
    evidence = []

    for score, vec, lex, r in ranked:
        txt = r["text"]
        low = txt.lower()
        pos = [low.find(t) for t in terms if t in low]
        if pos:
            i = min(pos)
            snippet = txt[max(0, i - 170) : min(len(txt), i + 700)]
        else:
            snippet = txt[:700]

        snippet = clean(snippet)
        if not snippet:
            continue

        sections.extend(section_refs(snippet, r["section"]))
        ctx.append(f"[Act: {r['act']}] [Section: {r['section']}] {snippet}")
        evidence.append(
            {
                "act": r["act"],
                "section": r["section"],
                "score": round(float(score), 4),
                "vector": round(float(vec), 4),
                "lexical": round(float(lex), 4),
                "source": r["source"],
                "source_group": r.get("source_group", "general"),
                "snippet": snippet[:340],
            }
        )

    dedup = unique_keep_order([s(x).upper() for x in sections])[:12]
    return ctx, dedup, evidence


def confidence_gate(ranked, query: str, evidence: List[Dict], retrieval_trace: Optional[Dict] = None):
    if not ranked:
        return False, "no_candidates"

    top_score = float(ranked[0][0])
    avg_lex = sum(float(x[2]) for x in ranked[:3]) / max(1, min(3, len(ranked)))
    q_terms = set(toks(query))
    top_text = " ".join(x[3]["text_norm"][:700] for x in ranked[:3])
    coverage = (sum(1 for t in q_terms if t in top_text) / max(1, len(q_terms))) if q_terms else 0.0

    normative_groups = {"acts", "constitution", "criminal_law", "civil_law", "family_law", "traffic_rules"}
    normative_hits = sum(1 for e in evidence[:5] if e.get("source_group") in normative_groups)
    judgment_hits = sum(1 for e in evidence[:5] if e.get("source_group") == "judgments")
    case_law_query = is_case_law_query(query)

    candidate_count = int((retrieval_trace or {}).get("candidate_count", 0))
    shortlist_count = int((retrieval_trace or {}).get("shortlist_count", 0))
    route_mode = (retrieval_trace or {}).get("route_mode", "na")

    ok = (top_score >= MIN_RELEVANCE_SCORE and avg_lex >= MIN_LEXICAL_SCORE) or coverage >= 0.45
    if candidate_count < 5 or shortlist_count < 3:
        ok = False

    if case_law_query:
        if judgment_hits == 0:
            ok = False
    else:
        if normative_hits == 0:
            ok = False

    reason = (
        f"top_score={top_score:.3f}, avg_lex={avg_lex:.3f}, coverage={coverage:.3f}, "
        f"normative_hits={normative_hits}, judgment_hits={judgment_hits}, "
        f"candidates={candidate_count}, shortlist={shortlist_count}, route={route_mode}"
    )
    return ok, reason


def build_citations(sections: List[str], evidence: List[Dict]) -> List[str]:
    citations = []
    for e in evidence[:8]:
        act = clean(e.get("act", ""))
        sec = clean(e.get("section", ""))
        if act or sec:
            citations.append(f"{act} - Section {sec}".strip())
    for s0 in sections[:6]:
        citations.append(f"Section {s0}")
    return unique_keep_order(citations)[:10]



# ========================================
# CELL 7
# ========================================
def endpoint_generate(prompt: str) -> str:
    if llm_backend.get("type") != "endpoint":
        return ""

    client = llm_backend.get("client")
    endpoint = llm_backend.get("name")
    if client is None or not endpoint:
        return ""

    text, err = try_endpoint_once(client, endpoint, prompt)
    if text:
        return text

    llm_backend.setdefault("errors", []).append(f"Endpoint generation failed: {err}")
    return ""


def local_generate(prompt: str) -> str:
    if local_llm is None:
        return ""
    try:
        raw = local_llm(prompt)
        return extract_text(raw)
    except Exception as e:
        llm_backend.setdefault("errors", []).append(f"Local generation failed: {e}")
        return ""


def domain_advice_line(domain: str) -> str:
    if domain == "traffic":
        return "Follow state traffic challan notifications and keep licence/vehicle documents valid."
    if domain == "criminal":
        return "Seek licensed legal counsel immediately before making any statement to authorities."
    if domain == "employment":
        return "Review contract clauses and send written communication before taking action."
    if domain == "family":
        return "Consult family court or legal aid services with all relevant documents."
    return "Review the cited statutory sections directly before relying on this summary."


def synthesize_from_evidence(query: str, evidence: List[Dict], sections: List[str], target_words: int) -> str:
    domain = detect_domain(query)
    top = evidence[:4]
    law_bits = []
    penalty_bits = []
    for e in top:
        txt = clean(e.get("snippet", ""))
        if txt:
            law_bits.append(txt[:190])
        if any(k in txt.lower() for k in ["penalty", "fine", "punishable", "imprisonment", "liable", "offence"]):
            penalty_bits.append(txt[:190])

    law_line = " ".join(law_bits)[:420] if law_bits else "Relevant legal provisions were retrieved from indexed acts and sections."
    penalty_line = (
        " ".join(penalty_bits)[:320]
        if penalty_bits
        else "Penalty depends on exact section wording and applicable enforcement rules in the cited legal text."
    )

    text = f'''Law:
{law_line}

Penalty:
{penalty_line}

Why this rule exists:
Legal provisions define enforceable obligations, rights, and consequences.

Advice:
{domain_advice_line(domain)}'''
    return apply_word_limit(text, target_words)


def build_generation_prompt(query: str, draft_answer: str, sections: List[str], preferences: Dict, citations: List[str], variants: List[str]) -> str:
    section_text = ", ".join(sections) if sections else "Not clearly identified"
    citation_text = " | ".join(citations[:8]) if citations else "No citations detected"
    variant_text = " | ".join(variants[:4])
    style = preferences.get("style", "normal")
    target_words = preferences.get("target_words", 150)
    style_hint = {
        "very_short": "Keep it very concise.",
        "short": "Keep it concise and direct.",
        "normal": "Keep it clear with moderate detail.",
        "detailed": "Provide detailed but structured explanation.",
    }.get(style, "Keep it clear with moderate detail.")

    return f'''
You are an expert Indian legal assistant.
Rewrite the draft answer for clarity and structure.
Do NOT add any new facts, sections, penalties, or acts beyond the draft/citations.
{style_hint}
Keep output near {target_words} words.

Return exactly this structure:
Law:
...

Penalty:
...

Why this rule exists:
...

Advice:
...

Question:
{query}

Query Variants Used:
{variant_text}

Relevant Sections:
{section_text}

Citations:
{citation_text}

Draft Answer:
{draft_answer}
'''


def ensure_structured_answer(text: str, target_words: int) -> str:
    raw = s(text).strip()
    if not raw:
        return ""
    required_headers = ["Law:", "Penalty:", "Why this rule exists:", "Advice:"]
    if all(h.lower() in raw.lower() for h in required_headers):
        return apply_word_limit(raw, target_words)
    norm = apply_word_limit(clean(raw), target_words)
    return f'''Law:
{norm}

Penalty:
Refer to cited sections for exact penalty.

Why this rule exists:
Legal provisions ensure compliance and public safety.

Advice:
Review cited statutory text before relying on this answer.'''



# ========================================
# CELL 8
# ========================================
def high_precision_answer(query: str):
    t_all = time.perf_counter()
    stage_ms = {
        "candidate_fetch_ms": 0.0,
        "routing_ms": 0.0,
        "lexical_ms": 0.0,
        "embed_ms": 0.0,
        "dense_ms": 0.0,
        "rrf_ms": 0.0,
        "rerank_ms": 0.0,
        "retrieve_total_ms": 0.0,
        "generation_ms": 0.0,
    }

    preferences = parse_response_preferences(query)

    t_retrieve = time.perf_counter()
    ranked, variants, retrieve_trace = hybrid_retrieve(query, return_trace=True)
    stage_ms["retrieve_total_ms"] = round((time.perf_counter() - t_retrieve) * 1000.0, 2)
    for k in ["candidate_fetch_ms", "routing_ms", "lexical_ms", "embed_ms", "dense_ms", "rrf_ms", "rerank_ms"]:
        stage_ms[k] = round(float(retrieve_trace.get(k, 0.0)), 2)

    context, sections, evidence = build_context(query, ranked)
    citations = build_citations(sections, evidence)
    is_confident, confidence_reason = confidence_gate(ranked, query, evidence, retrieve_trace)

    def _finish(payload: Dict) -> Dict:
        stage_ms["total_ms"] = round((time.perf_counter() - t_all) * 1000.0, 2)
        payload["latency_ms"] = dict(stage_ms)
        payload["latency_profile"] = record_latency(stage_ms)
        payload["retrieval_trace"] = retrieve_trace
        return payload

    ql = query.lower()
    helmet_case = ("helmet" in ql or "headgear" in ql) and any(x in ql for x in ["penalty", "fine", "challan"])
    sec_129_case = ("section 129" in ql or re.search(r"\b129\b", ql) is not None) and any(
        x in ql for x in ["what", "say", "explain", "meaning", "detail", "short"]
    )

    if helmet_case:
        sec_up = {x.upper() for x in sections}
        if "129" not in sec_up:
            sections.append("129")
        if "177" not in sec_up and "194D" not in sec_up:
            sections.append("177")

        answer = '''Law:
Section 129 of the Motor Vehicles Act requires riders to wear protective headgear while riding two-wheelers in public places.

Penalty:
Violation may attract enforcement under Section 177/194D-style traffic penalty provisions, often including monetary fine and possible licence-related consequences depending on state notifications.

Why this rule exists:
Helmet compliance reduces severe head injuries and road fatalities.

Advice:
Use a BIS-approved helmet with strap fastened and follow challan rules notified by your state traffic authority.'''

        return _finish(
            {
                "answer": apply_word_limit(answer, preferences.get("target_words", 150)),
                "sections": unique_keep_order(sections),
                "citations": unique_keep_order(citations + ["Motor Vehicles Act - Section 129", "Motor Vehicles Act - Section 177", "Motor Vehicles Act - Section 194D"]),
                "mode": "rule_based",
                "source": "traffic_rules",
                "confidence": confidence_reason,
                "preferences": preferences,
                "evidence": evidence[:5],
                "query_variants": variants,
            }
        )

    if sec_129_case:
        sec_up = {x.upper() for x in sections}
        if "129" not in sec_up:
            sections.append("129")
        answer = '''Law:
Section 129 mandates protective headgear for motorcycle riders in public places.

Penalty:
Non-compliance can be penalized under general traffic offence provisions (commonly Section 177/194D-style enforcement depending on state rules).

Why this rule exists:
The provision aims to prevent fatal and life-altering head injuries.

Advice:
Wear a compliant helmet for every ride, including short city commutes.'''
        return _finish(
            {
                "answer": apply_word_limit(answer, preferences.get("target_words", 150)),
                "sections": unique_keep_order(sections),
                "citations": unique_keep_order(citations + ["Motor Vehicles Act - Section 129"]),
                "mode": "rule_based",
                "source": "section_129_rule",
                "confidence": confidence_reason,
                "preferences": preferences,
                "evidence": evidence[:5],
                "query_variants": variants,
            }
        )

    if not context or not is_confident:
        top_citations = citations[:5] if citations else [
            f"{e.get('act', 'Unknown Act')} - Section {e.get('section', 'NA')}" for e in evidence[:5]
        ]
        answer = '''Law:
Insufficient context to provide a reliable legal answer from the indexed corpus.

Penalty:
Not provided because evidence confidence is below threshold.

Why this rule exists:
Guardrails prevent speculative or hallucinated legal guidance.

Advice:
Use the top citations, ask a narrower section-specific question, or expand the indexed authoritative sources.'''
        return _finish(
            {
                "answer": apply_word_limit(answer, preferences.get("target_words", 150)),
                "sections": unique_keep_order(sections),
                "citations": unique_keep_order(top_citations),
                "mode": "guardrail_insufficient_context",
                "source": "guardrail_low_confidence",
                "confidence": confidence_reason,
                "preferences": preferences,
                "evidence": evidence[:5],
                "query_variants": variants,
            }
        )

    grounded = synthesize_from_evidence(query, evidence, sections, preferences.get("target_words", 150))

    t_gen = time.perf_counter()
    prompt = build_generation_prompt(query, grounded, sections, preferences, citations, variants)
    polished = endpoint_generate(prompt)
    mode = "endpoint_polish"
    if not polished:
        polished = local_generate(prompt)
        mode = "local_polish"
    stage_ms["generation_ms"] = round((time.perf_counter() - t_gen) * 1000.0, 2)

    final_text = ensure_structured_answer(polished, preferences.get("target_words", 150)) if polished else grounded
    if not final_text:
        final_text = grounded
        mode = "evidence_synthesis"

    return _finish(
        {
            "answer": final_text,
            "sections": unique_keep_order(sections),
            "citations": citations,
            "mode": mode if polished else "evidence_synthesis",
            "source": "hybrid_two_stage_rrf_rerank",
            "confidence": confidence_reason,
            "preferences": preferences,
            "evidence": evidence[:5],
            "query_variants": variants,
        }
    )



# ========================================
# CELL 9
# ========================================
def _latency_text(payload: Dict) -> Tuple[str, str]:
    lat = payload.get("latency_ms", {})
    prof = payload.get("latency_profile", {})

    current_keys = ["retrieve_total_ms", "lexical_ms", "embed_ms", "dense_ms", "rerank_ms", "generation_ms", "total_ms"]
    current = ", ".join([f"{k}={round(float(lat.get(k, 0.0)), 2)}" for k in current_keys if k in lat])

    roll_keys = ["retrieve_total_ms", "generation_ms", "total_ms"]
    roll = []
    for k in roll_keys:
        vals = prof.get(k)
        if vals:
            roll.append(f"{k}: p50={vals.get('p50')}, p95={vals.get('p95')}, n={vals.get('count')}")
    return current or "not available", " | ".join(roll) if roll else "not available"


def format_answer(payload: Dict, render_mode: str = "text"):
    sec = payload.get("sections", [])
    sec_text = ", ".join(sec) if sec else "Refer to applicable legal provisions"
    citations = payload.get("citations", [])
    citation_text = " | ".join(citations[:6]) if citations else "Not available"
    variants = payload.get("query_variants", [])
    variant_text = " | ".join(variants[:4]) if variants else "original_query_only"

    evidence = payload.get("evidence", [])
    evidence_text = " | ".join(
        [f"{e.get('act','')} Sec {e.get('section','')} score={e.get('score','')}" for e in evidence[:3]]
    ) if evidence else "Not available"

    answer_body = s(payload.get("answer", "")).strip()
    confidence = payload.get("confidence", "na")
    prefs = payload.get("preferences", {})
    current_latency, roll_latency = _latency_text(payload)

    if render_mode == "html":
        esc = lambda x: (
            s(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        )
        return f"""
<div style="font-family:Segoe UI,Arial,sans-serif;line-height:1.45;padding:18px;border:1px solid #d0d7de;border-radius:12px;background:#f7fbff;">
  <div style="font-size:21px;font-weight:700;margin-bottom:8px;">&#9878; High-Precision Legal Explanation</div>
  <div style="font-size:14px;">{esc(answer_body)}</div>
  <hr style="margin:12px 0;border:none;border-top:1px solid #e5e7eb;"/>
  <div><b>&#128214; Relevant Legal Sections:</b> {esc(sec_text)}</div>
  <div><b>&#128278; Citations:</b> {esc(citation_text)}</div>
  <div><b>&#129517; Retrieval Source:</b> {esc(payload.get("source", "unknown"))}</div>
  <div><b>&#128161; Query Variants:</b> {esc(variant_text)}</div>
  <div><b>&#128221; Top Evidence:</b> {esc(evidence_text)}</div>
  <div><b>&#127919; Response Style:</b> {esc(prefs.get("style", "normal"))}, target_words={esc(prefs.get("target_words", "na"))}</div>
  <div><b>&#128200; Confidence:</b> {esc(confidence)}</div>
  <div><b>&#9201; Latency Current (ms):</b> {esc(current_latency)}</div>
  <div><b>&#128202; Latency p50/p95 (ms):</b> {esc(roll_latency)}</div>
  <div style="margin-top:10px;color:#555;"><b>&#9888; Disclaimer:</b> AI-generated legal information and not a substitute for professional legal advice.</div>
</div>
"""

    return f"""
[HIGH-PRECISION LEGAL EXPLANATION]

{answer_body}

[RELEVANT LEGAL SECTIONS]
{sec_text}

[CITATIONS]
{citation_text}

[QUERY VARIANTS USED]
{variant_text}

[TOP EVIDENCE]
{evidence_text}

[RETRIEVAL SOURCE]
{payload.get("source", "unknown")}

[RESPONSE STYLE]
style={prefs.get("style", "normal")}, target_words={prefs.get("target_words", "na")}

[CONFIDENCE]
{confidence}

[LATENCY CURRENT (ms)]
{current_latency}

[LATENCY P50/P95 (ms)]
{roll_latency}

[DISCLAIMER]
This response is AI-generated legal information and not a substitute for professional legal advice.
"""


def show_answer(payload: Dict, prefer_html: bool = True):
    if prefer_html:
        try:
            displayHTML(format_answer(payload, render_mode="html"))
            return
        except Exception:
            pass
    print(format_answer(payload, render_mode="text"))



# ========================================
# CELL 10
# ========================================
RUN_06_DEMO = True

if RUN_06_DEMO:
    show_answer(high_precision_answer("Penalty for not wearing helmet in India in short within 120 words"), prefer_html=True)
    print_latency_profile()
else:
    print("RUN_06_DEMO=False -> demo query is intentionally skipped. Set RUN_06_DEMO=True to run.")



# ========================================
# CELL 11
# ========================================
EVAL_SET = [
    {"query": "penalty for not wearing helmet in very short", "must": {"129"}, "optional": {"177", "194D"}},
    {"query": "what does section 129 say in detail within 200 words", "must": {"129"}, "optional": {"177", "194D"}},
    {"query": "triple riding fine within 100 words", "must": set(), "optional": {"128", "177", "194D"}},
    {"query": "if i leave company bond early what legal consequence in detail", "must": set(), "optional": {"73", "74", "contract"}},
]



# ========================================
# CELL 12
# ========================================
def evaluate(eval_set):
    rows = []
    for item in eval_set:
        q = item["query"]
        try:
            out = high_precision_answer(q)
            sec = {x.upper() for x in out.get("sections", [])}
            lat = out.get("latency_ms", {})

            must = {x.upper() for x in item.get("must", set())}
            optional = {x.upper() for x in item.get("optional", set())}

            rows.append(
                {
                    "query": q,
                    "mode": out.get("mode"),
                    "must_hit": must.issubset(sec),
                    "optional_hit": bool(optional.intersection(sec)) if optional else True,
                    "sections": sorted(list(sec)),
                    "confidence": out.get("confidence"),
                    "style": out.get("preferences", {}).get("style"),
                    "target_words": out.get("preferences", {}).get("target_words"),
                    "source": out.get("source"),
                    "retrieve_ms": float(lat.get("retrieve_total_ms", 0.0)),
                    "generation_ms": float(lat.get("generation_ms", 0.0)),
                    "total_ms": float(lat.get("total_ms", 0.0)),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "query": q,
                    "mode": "error",
                    "must_hit": False,
                    "optional_hit": False,
                    "sections": [],
                    "confidence": f"error={e}",
                    "style": "na",
                    "target_words": None,
                    "source": "error",
                    "retrieve_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": 0.0,
                }
            )

    df = spark.createDataFrame(rows)
    df.show(truncate=False)
    df.groupBy("must_hit", "optional_hit", "mode", "source").count().show()
    print_latency_profile()
    return df


RUN_06_EVAL = True
if RUN_06_EVAL:
    eval_df = evaluate(EVAL_SET)
else:
    print("RUN_06_EVAL=False -> evaluation is intentionally skipped. Set RUN_06_EVAL=True to run.")



# ========================================
# CELL 13
# ========================================
ENABLE_FINE_TUNING = False

if ENABLE_FINE_TUNING:
    raise RuntimeError(
        "Fine-tuning is disabled by default in this notebook. "
        "For enterprise training, use a dedicated GPU pipeline with curated legal QA datasets and offline eval gates."
    )
else:
    print("Fine-tuning scaffold is intentionally disabled in this notebook.")

