#!/usr/bin/env python
"""Build a local FAISS + lexical artifact store for LexAI from gold_chunks.parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "legal_embeddings_delta.parquet"
VECTOR_DIR = ROOT_DIR / "vector_store"
INDEX_PATH = VECTOR_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_DIR / "metadata.pkl"
LEXICAL_PATH = VECTOR_DIR / "lexical_artifacts.pkl"
MANIFEST_PATH = VECTOR_DIR / "build_manifest.json"

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_TYPE = "IndexFlatIP"
METRIC_NAME = "cosine_via_inner_product"
NORMALIZED = True
LEXICAL_VERSION = 1
MANIFEST_VERSION = 4
REQUIRED_COLUMNS = ["chunk_id", "act_name", "section_number", "chunk_text"]
REQUIRED_PRECOMPUTED_COLUMNS = ["chunk_id", "chunk_text", "embedding"]
TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_gold_chunks(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    if "char_count" not in out.columns:
        out["char_count"] = pd.NA

    out["chunk_id"] = out["chunk_id"].astype(str).str.strip()
    out["act_name"] = out["act_name"].fillna("").astype(str).str.strip()
    out["section_number"] = out["section_number"].fillna("").astype(str).str.strip()
    out["chunk_text"] = out["chunk_text"].fillna("").astype(str).str.strip()
    out["char_count"] = pd.to_numeric(out["char_count"], errors="coerce")

    out = out[(out["chunk_id"] != "") & (out["chunk_text"] != "")]
    if out.empty:
        raise ValueError("No usable rows remain after dropping null/empty chunk_text or chunk_id.")

    out = out.sort_values(["chunk_id", "act_name", "section_number"], kind="mergesort")
    out = out.drop_duplicates(subset=["chunk_id"], keep="first")
    out = out.reset_index(drop=True)
    return out


def _clean_precomputed_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_PRECOMPUTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for precomputed mode: {missing}")

    out = df.copy()
    for col, default_val in [
        ("act_name", ""),
        ("section_number", ""),
        ("category", ""),
        ("file_name", ""),
        ("section_refs", ""),
        ("is_penalty_related", 0),
        ("is_helmet_related", 0),
    ]:
        if col not in out.columns:
            out[col] = default_val
    if "char_count" not in out.columns:
        out["char_count"] = pd.NA

    out["chunk_id"] = out["chunk_id"].astype(str).str.strip()
    out["chunk_text"] = out["chunk_text"].fillna("").astype(str).str.strip()
    out["act_name"] = out["act_name"].fillna("").astype(str).str.strip()
    out["section_number"] = out["section_number"].fillna("").astype(str).str.strip()
    out["category"] = out["category"].fillna("").astype(str).str.strip()
    out["file_name"] = out["file_name"].fillna("").astype(str).str.strip()
    out["section_refs"] = out["section_refs"].fillna("").astype(str)
    out["is_penalty_related"] = pd.to_numeric(out["is_penalty_related"], errors="coerce").fillna(0).astype(int)
    out["is_helmet_related"] = pd.to_numeric(out["is_helmet_related"], errors="coerce").fillna(0).astype(int)
    out["char_count"] = pd.to_numeric(out["char_count"], errors="coerce")

    out = out[(out["chunk_id"] != "") & (out["chunk_text"] != "")]
    if out.empty:
        raise ValueError("No usable rows remain after dropping null/empty chunk_text or chunk_id.")

    if "updated_at" in out.columns:
        out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
        out = out.sort_values(["chunk_id", "updated_at"], ascending=[True, False], kind="mergesort")
    else:
        out = out.sort_values(["chunk_id"], kind="mergesort")
    out = out.drop_duplicates(subset=["chunk_id"], keep="first").reset_index(drop=True)
    return out


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _infer_source_group(act_name: str) -> str:
    a = (act_name or "").lower()
    if any(k in a for k in ["constitution", "article "]):
        return "constitution"
    if any(k in a for k in ["motor vehicle", "traffic", "road", "driving licence", "driving license"]):
        return "traffic_rules"
    if any(k in a for k in ["penal code", "nyaya sanhita", "criminal", "evidence act", "crpc", "bnss"]):
        return "criminal_law"
    if any(k in a for k in ["marriage", "divorce", "succession", "family", "domestic violence"]):
        return "family_law"
    if any(k in a for k in ["contract", "civil", "property", "partnership", "arbitration"]):
        return "civil_law"
    if any(k in a for k in ["judgment", "vs.", " v. "]):
        return "judgments"
    if any(k in a for k in ["report", "commission"]):
        return "reports"
    return "acts"


def _extract_section_tokens(section_number: str, chunk_text: str) -> List[str]:
    toks: Set[str] = set()
    sec = section_number or ""
    txt = chunk_text or ""

    for m in re.findall(r"\b([0-9]{1,4}[A-Za-z]?)\b", sec):
        toks.add(m.upper())

    for m in re.finditer(r"\b(?:section|sec\.?|s\.)\s*([0-9]{1,4}[A-Za-z]?)\b", txt, flags=re.IGNORECASE):
        toks.add(m.group(1).upper())

    for m in re.finditer(r"(?m)^\s*([0-9]{1,4}[A-Za-z]?)\s*[\.\-)]", txt[:1400]):
        toks.add(m.group(1).upper())

    return sorted(toks)


def _metadata_records(df: pd.DataFrame) -> List[Dict]:
    recs: List[Dict] = []
    for _, row in df.iterrows():
        cc = row.get("char_count")
        act_name = str(row["act_name"])
        section_number = str(row["section_number"])
        chunk_text = str(row["chunk_text"])

        source_group = _infer_source_group(act_name)
        section_tokens = _extract_section_tokens(section_number, chunk_text)
        text_lower = chunk_text.lower()
        is_helmet_related = int(
            any(k in text_lower for k in ["helmet", "headgear", "motor cycle", "motorcycle", "two-wheeler"])
            and any(k in text_lower for k in ["penalty", "fine", "section 129", "section 177", "194d"])
        )
        if "is_helmet_related" in df.columns:
            is_helmet_related = int(getattr(row, "is_helmet_related", is_helmet_related) or is_helmet_related)
        is_penalty_related = int(getattr(row, "is_penalty_related", 0) or 0)
        section_refs_raw = str(getattr(row, "section_refs", "") or "")
        section_refs = [x.strip().upper() for x in re.split(r"[,\|;]", section_refs_raw) if x.strip()]
        lexical_text = f"{act_name} {section_number} {chunk_text}"
        lexical_tokens = _tokenize(lexical_text)

        recs.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "act_name": act_name,
                "section_number": section_number,
                "chunk_text": chunk_text,
                "char_count": int(cc) if pd.notna(cc) else None,
                "source_group": source_group,
                "section_tokens": section_tokens,
                "is_helmet_related": is_helmet_related,
                "is_penalty_related": is_penalty_related,
                "section_refs": section_refs,
                "category": str(getattr(row, "category", "") or ""),
                "file_name": str(getattr(row, "file_name", "") or ""),
                "lexical_tokens": lexical_tokens,
            }
        )
    return recs


def _coerce_precomputed_embeddings(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    vectors: List[np.ndarray] = []
    keep_idx: List[int] = []
    dim = 0
    for i, emb in enumerate(df["embedding"].tolist()):
        try:
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            if dim == 0:
                dim = int(arr.size)
            if int(arr.size) != dim:
                continue
            vectors.append(arr)
            keep_idx.append(i)
        except Exception:
            continue

    if not vectors:
        raise ValueError("Precomputed embedding column exists but no valid vectors were found.")

    cleaned_df = df.iloc[keep_idx].reset_index(drop=True)
    emb = np.vstack(vectors).astype(np.float32)
    return cleaned_df, emb


def _build_lexical_artifacts(records: List[Dict]) -> Dict:
    tokenized_metadata: List[List[str]] = []
    inverted_tf: Dict[str, Dict[int, int]] = defaultdict(dict)
    doc_len: List[int] = []

    for doc_id, rec in enumerate(records):
        tokens = list(rec.get("lexical_tokens") or [])
        tokenized_metadata.append(tokens)
        doc_len.append(len(tokens))
        tf = Counter(tokens)
        for token, count in tf.items():
            inverted_tf[token][doc_id] = int(count)

    inverted_index: Dict[str, List[tuple[int, int]]] = {}
    doc_freq: Dict[str, int] = {}
    for token, postings in inverted_tf.items():
        posting_list = sorted(postings.items(), key=lambda x: x[0])
        inverted_index[token] = posting_list
        doc_freq[token] = len(posting_list)

    avg_doc_len = float(sum(doc_len) / max(1, len(doc_len)))
    return {
        "version": LEXICAL_VERSION,
        "doc_count": len(records),
        "avg_doc_len": avg_doc_len,
        "doc_len": doc_len,
        "doc_freq": doc_freq,
        "inverted_index": inverted_index,
        "tokenized_metadata": tokenized_metadata,
        "vocab_size": len(inverted_index),
    }


def _manifest_matches(existing: Dict, source_sha: str, row_count: int, model_name: str) -> bool:
    files_ready = INDEX_PATH.exists() and METADATA_PATH.exists() and LEXICAL_PATH.exists()
    if not files_ready:
        return False
    return (
        int(existing.get("version", -1)) >= MANIFEST_VERSION
        and existing.get("source_sha256") == source_sha
        and int(existing.get("row_count", -1)) == int(row_count)
        and existing.get("embedding_model") == model_name
        and existing.get("index_type") == INDEX_TYPE
        and existing.get("metric") == METRIC_NAME
        and bool(existing.get("normalized_embeddings")) == NORMALIZED
        and int(existing.get("lexical_artifact_version", -1)) == LEXICAL_VERSION
    )


def _can_reuse_dense(existing: Dict, source_sha: str, row_count: int, model_name: str) -> bool:
    return (
        INDEX_PATH.exists()
        and METADATA_PATH.exists()
        and existing.get("source_sha256") == source_sha
        and int(existing.get("row_count", -1)) == int(row_count)
        and existing.get("embedding_model") == model_name
        and existing.get("index_type") == INDEX_TYPE
        and existing.get("metric") == METRIC_NAME
        and bool(existing.get("normalized_embeddings")) == NORMALIZED
    )


def _build(parquet_path: Path, model_name: str, batch_size: int, if_needed: bool) -> None:
    total_start = time.perf_counter()
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {parquet_path}\n"
            "Copy the repository dataset to lexai-local/data/legal_embeddings_delta.parquet first."
        )

    source_sha = _file_sha256(parquet_path)
    print(f"[BUILD] Source parquet: {parquet_path}")
    print(f"[BUILD] Source SHA-256: {source_sha}")

    t_load = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    using_precomputed = "embedding" in df.columns
    df = _clean_precomputed_embeddings(df) if using_precomputed else _clean_gold_chunks(df)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    row_count = len(df)
    print(f"[BUILD] Rows after cleaning/dedupe: {row_count}")

    existing = _load_manifest(MANIFEST_PATH)
    if if_needed and existing and _manifest_matches(existing, source_sha, row_count, model_name):
        print("[BUILD] No changes detected; skipping rebuild due to --if-needed.")
        return

    if if_needed and existing and _can_reuse_dense(existing, source_sha, row_count, model_name):
        print("[BUILD] Reusing existing dense artifacts; rebuilding lexical artifacts only.")
        with METADATA_PATH.open("rb") as f:
            records = pickle.load(f)
        t_lexical = time.perf_counter()
        lexical_artifacts = _build_lexical_artifacts(records)
        lexical_ms = (time.perf_counter() - t_lexical) * 1000.0
        with LEXICAL_PATH.open("wb") as f:
            pickle.dump(lexical_artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

        index_sha = _file_sha256(INDEX_PATH)
        metadata_sha = _file_sha256(METADATA_PATH)
        lexical_sha = _file_sha256(LEXICAL_PATH)
        manifest = {
            "version": MANIFEST_VERSION,
            "built_at": _now_utc(),
            "source_path": str(parquet_path),
            "source_sha256": source_sha,
            "row_count": int(row_count),
            "embedding_model": model_name,
            "embedding_dim": int(existing.get("embedding_dim", 768)),
            "batch_size": int(existing.get("batch_size", batch_size)),
            "index_type": INDEX_TYPE,
            "metric": METRIC_NAME,
            "normalized_embeddings": NORMALIZED,
            "index_path": str(INDEX_PATH),
            "metadata_path": str(METADATA_PATH),
            "lexical_path": str(LEXICAL_PATH),
            "lexical_artifact_version": LEXICAL_VERSION,
            "index_sha256": index_sha,
            "metadata_sha256": metadata_sha,
            "lexical_sha256": lexical_sha,
            "metadata_schema": {
                "chunk_id": "str",
                "act_name": "str",
                "section_number": "str",
                "chunk_text": "str",
                "char_count": "int|None",
                "source_group": "str",
                "section_tokens": "list[str]",
                "is_helmet_related": "int",
                "lexical_tokens": "list[str]",
            },
            "lexical_schema": {
                "inverted_index": "dict[token, list[(doc_id, tf)]]",
                "doc_freq": "dict[token, int]",
                "avg_doc_len": "float",
                "doc_len": "list[int]",
                "tokenized_metadata": "list[list[str]]",
            },
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        total_ms = (time.perf_counter() - total_start) * 1000.0
        print(
            json.dumps(
                {
                    "mode": "lexical_only_refresh",
                    "load_ms": round(load_ms, 2),
                    "lexical_ms": round(lexical_ms, 2),
                    "total_ms": round(total_ms, 2),
                },
                indent=2,
            )
        )
        return

    t_embed = time.perf_counter()
    if using_precomputed:
        df, emb = _coerce_precomputed_embeddings(df)
        faiss.normalize_L2(emb)
        dim = int(emb.shape[1])
        if model_name == DEFAULT_MODEL and dim == 384:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"[BUILD] Using precomputed embeddings from parquet (dim={dim}, rows={len(df)})")
    else:
        print(f"[BUILD] Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        dim = int(model.get_sentence_embedding_dimension())
        print(f"[BUILD] Embedding dimension: {dim}")

        texts = df["chunk_text"].tolist()
        emb = model.encode(
            texts,
            batch_size=int(batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        faiss.normalize_L2(emb)
    embed_ms = (time.perf_counter() - t_embed) * 1000.0

    t_index = time.perf_counter()
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    index_ms = (time.perf_counter() - t_index) * 1000.0
    print(f"[BUILD] FAISS index size: {index.ntotal}")

    records = _metadata_records(df)

    t_lexical = time.perf_counter()
    lexical_artifacts = _build_lexical_artifacts(records)
    lexical_ms = (time.perf_counter() - t_lexical) * 1000.0
    print(
        f"[BUILD] Lexical artifacts: vocab={lexical_artifacts['vocab_size']}, "
        f"avg_doc_len={lexical_artifacts['avg_doc_len']:.2f}"
    )

    t_save = time.perf_counter()
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    with LEXICAL_PATH.open("wb") as f:
        pickle.dump(lexical_artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    index_sha = _file_sha256(INDEX_PATH)
    metadata_sha = _file_sha256(METADATA_PATH)
    lexical_sha = _file_sha256(LEXICAL_PATH)

    manifest = {
        "version": MANIFEST_VERSION,
        "built_at": _now_utc(),
        "source_path": str(parquet_path),
        "source_sha256": source_sha,
        "row_count": int(row_count),
        "embedding_model": model_name,
        "embedding_dim": int(dim),
        "build_mode": "precomputed_embeddings" if using_precomputed else "generated_embeddings",
        "batch_size": int(batch_size),
        "index_type": INDEX_TYPE,
        "metric": METRIC_NAME,
        "normalized_embeddings": NORMALIZED,
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "lexical_path": str(LEXICAL_PATH),
        "lexical_artifact_version": LEXICAL_VERSION,
        "index_sha256": index_sha,
        "metadata_sha256": metadata_sha,
        "lexical_sha256": lexical_sha,
        "metadata_schema": {
            "chunk_id": "str",
            "act_name": "str",
            "section_number": "str",
            "chunk_text": "str",
            "char_count": "int|None",
                "source_group": "str",
                "section_tokens": "list[str]",
                "is_helmet_related": "int",
                "is_penalty_related": "int",
                "section_refs": "list[str]",
                "category": "str",
                "file_name": "str",
                "lexical_tokens": "list[str]",
            },
        "lexical_schema": {
            "inverted_index": "dict[token, list[(doc_id, tf)]]",
            "doc_freq": "dict[token, int]",
            "avg_doc_len": "float",
            "doc_len": "list[int]",
            "tokenized_metadata": "list[list[str]]",
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_ms = (time.perf_counter() - t_save) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    print("[BUILD] Completed successfully.")
    print(
        json.dumps(
            {
                "load_ms": round(load_ms, 2),
                "embed_ms": round(embed_ms, 2),
                "index_ms": round(index_ms, 2),
                "lexical_ms": round(lexical_ms, 2),
                "save_ms": round(save_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LexAI FAISS + lexical artifacts from parquet.")
    p.add_argument(
        "--parquet",
        type=str,
        default=str(DATA_PATH),
        help="Path to input parquet. Defaults to the repository Databricks embedding export; also supports gold-chunk parquet.",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    p.add_argument(
        "--if-needed",
        action="store_true",
        help="Skip rebuild if source hash and build config match manifest.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _build(
        parquet_path=Path(args.parquet),
        model_name=str(args.model).strip(),
        batch_size=int(args.batch_size),
        if_needed=bool(args.if_needed),
    )


if __name__ == "__main__":
    main()
