"""Retriever for LexAI using FAISS + metadata-aware hybrid scoring."""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency in lexical mode
    SentenceTransformer = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "vector_store" / "faiss_index.bin"
METADATA_PATH = ROOT_DIR / "vector_store" / "metadata.pkl"
MANIFEST_PATH = ROOT_DIR / "vector_store" / "build_manifest.json"

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_MODEL_NAME = os.environ.get("LEXAI_EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_FINAL_K = int(os.environ.get("LEXAI_FINAL_K", "5"))
DEFAULT_INITIAL_K = int(os.environ.get("LEXAI_INITIAL_K", "15"))
DEFAULT_RETRIEVAL_MODE = os.environ.get("LEXAI_RETRIEVAL_MODE", "hybrid").strip().lower() or "hybrid"
SKIP_DENSE_FAST_DEFAULT = os.environ.get("LEXAI_SKIP_DENSE_FAST", "1").strip().lower() not in {"0", "false", "no"}

SECTION_MATCH_BONUS = float(os.environ.get("LEXAI_SECTION_MATCH_BONUS", "0.5"))
ACT_MATCH_BONUS = float(os.environ.get("LEXAI_ACT_MATCH_BONUS", "0.3"))
KEYWORD_WEIGHT = float(os.environ.get("LEXAI_KEYWORD_WEIGHT", "0.2"))

ACT_ALIASES: Dict[str, List[str]] = {
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

PENAL_ACT_HINTS = ("penal code", "indian penal code", "bharatiya nyaya sanhita", "bns")
PROCEDURAL_ACT_HINTS = ("code of criminal procedure", "crpc", "bnss")
SEXUAL_SECTIONS = {"354", "354A", "354B", "354C", "354D", "509", "376", "376A", "376AB", "376B", "376C", "376D"}
TRAFFIC_SECTIONS = {"129", "177", "194D", "194"}

TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

_INDEX = None
_METADATA: List[Dict] | None = None
_EMBEDDER = None
_INIT_ERROR = "" if SentenceTransformer is not None else "sentence-transformers is not installed. Dense retrieval disabled."
_EMBED_MODEL_RESOLVED = EMBED_MODEL_NAME


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _split_sentences(text: str) -> List[str]:
    src = _clean_text(text)
    if not src:
        return []
    return [s.strip() for s in SENTENCE_SPLIT.split(src) if len(s.strip()) >= 35]


def _safe_section(section: str | None) -> str:
    return (section or "").strip().upper()


def _extract_sections(query: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for m in re.finditer(r"\b(?:section|sec\.?|s\.)\s*([0-9]{1,4}[A-Za-z]?)\b", query or "", flags=re.IGNORECASE):
        s = _safe_section(m.group(1))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if not out and re.search(r"\b(ipc|crpc|cpc|act|code|penalty|punishment)\b", query or "", flags=re.IGNORECASE):
        for m in re.finditer(r"\b([0-9]{2,4}[A-Za-z]?)\b", query or ""):
            s = _safe_section(m.group(1))
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out[:6]


def _detect_domain(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["helmet", "traffic", "motor vehicle", "driving", "challan"]):
        return "traffic_rules"
    if any(k in q for k in ["ipc", "bns", "murder", "assault", "consent", "punishment", "crime"]):
        return "criminal_law"
    if any(k in q for k in ["contract", "company", "agreement", "damages", "civil"]):
        return "civil_law"
    if any(k in q for k in ["marriage", "divorce", "maintenance", "succession", "shariat", "family"]):
        return "family_law"
    if any(k in q for k in ["constitution", "article", "fundamental right", "writ"]):
        return "constitutional_law"
    if any(k in q for k in ["tree", "trees", "cutting trees", "felling", "forest", "environment", "pollution", "wildlife", "biodiversity"]):
        return "environmental_law"
    return "general"


def _detect_intent(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["kiss", "consent", "sexual harassment", "outraging modesty", "molestation", "assault"]):
        return "sexual_offence"
    if any(k in q for k in ["tree", "trees", "cutting trees", "felling", "forest", "environment", "pollution", "wildlife"]):
        return "environmental_offence"
    if any(k in q for k in ["penalty", "fine", "punishment", "liable", "imprisonment"]):
        return "penalty"
    if any(k in q for k in ["case law", "judgment", "precedent", "citation"]):
        return "case_law"
    if any(k in q for k in ["what is", "define", "meaning"]):
        return "definition"
    return "general"


def _detect_act(query: str) -> str | None:
    q = (query or "").lower()
    for act, aliases in ACT_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return act
    if _detect_intent(q) == "sexual_offence":
        return "Indian Penal Code"
    if _detect_domain(q) == "traffic_rules":
        return "Motor Vehicles Act"
    if _detect_domain(q) == "environmental_law":
        for act, aliases in ACT_ALIASES.items():
            if any(alias in q for alias in aliases):
                return act
    return None


def query_normalizer(question: str) -> str:
    out = _clean_text(question)
    qlow = out.lower()
    if any(k in qlow for k in ["cut many trees", "cut trees", "cutting trees", "felled trees", "felling trees"]):
        out = f"{out} tree felling forest clearance environmental offence permission"
    elif "environment department" in qlow or "environmental department" in qlow:
        out = f"{out} environmental law forest tree regulation"
    for patt, repl in QUERY_REWRITE_RULES:
        if re.search(patt, out, flags=re.IGNORECASE):
            out = re.sub(patt, repl, out, flags=re.IGNORECASE)
            break
    return _clean_text(out)


def query_parser(question: str) -> Dict:
    norm = query_normalizer(question)
    sections = _extract_sections(norm)
    return {
        "normalized_query": norm,
        "possible_sections": sections,
        "possible_act": _detect_act(norm) or "",
        "legal_domain": _detect_domain(norm),
        "intent": _detect_intent(norm),
        "confidence": 0.55,
    }


def _resolve_embed_model_name() -> str:
    if EMBED_MODEL_NAME:
        return EMBED_MODEL_NAME
    try:
        if MANIFEST_PATH.exists():
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            model = str(payload.get("embedding_model", "")).strip()
            if model:
                return model
    except Exception:
        pass
    return DEFAULT_EMBED_MODEL


def _ensure_row_features(row: Dict) -> Dict:
    row["act_name"] = _clean_text(str(row.get("act_name", "")))
    row["section_number"] = _clean_text(str(row.get("section_number", "")))
    row["chunk_text"] = _clean_text(str(row.get("chunk_text", "")))
    if "_tokens" not in row:
        row["_tokens"] = set(_tokenize(" ".join([row["act_name"], row["section_number"], row["chunk_text"]])))
    if "_sections" not in row:
        secs = set()
        sec = _safe_section(row["section_number"])
        if sec:
            secs.add(sec)
        for s in row.get("section_tokens", []) or []:
            s_norm = _safe_section(str(s))
            if s_norm:
                secs.add(s_norm)
        for m in re.finditer(r"\b(?:section|sec\.?|s\.)\s*([0-9]{1,4}[A-Za-z]?)\b", row["chunk_text"], flags=re.IGNORECASE):
            secs.add(_safe_section(m.group(1)))
        # Many legal chunks are of the form "302. Punishment for murder." (without explicit "Section 302").
        for m in re.finditer(r"\b([0-9]{1,4}[A-Za-z]{0,2})\.\s*[A-Z][a-z]", row["chunk_text"]):
            secs.add(_safe_section(m.group(1)))
        row["_sections"] = secs
    if "_source_group" not in row:
        row["_source_group"] = _clean_text(str(row.get("source_group", "acts"))) or "acts"
    return row


def _load_resources(load_embedder: bool = True) -> None:
    global _INDEX, _METADATA, _EMBEDDER, _INIT_ERROR, _EMBED_MODEL_RESOLVED
    try:
        if _INDEX is None or _METADATA is None:
            if not INDEX_PATH.exists() or not METADATA_PATH.exists():
                raise FileNotFoundError("Missing vector artifacts. Run: python vector_store/build_vector_db.py --if-needed")
            _INDEX = faiss.read_index(str(INDEX_PATH))
            with METADATA_PATH.open("rb") as f:
                raw = pickle.load(f)
            _METADATA = [_ensure_row_features(dict(r)) for r in raw]
        if load_embedder and _EMBEDDER is None:
            _EMBED_MODEL_RESOLVED = _resolve_embed_model_name()
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed. Dense retrieval disabled.")
            _EMBEDDER = SentenceTransformer(_EMBED_MODEL_RESOLVED)
        _INIT_ERROR = ""
    except Exception as exc:
        _INIT_ERROR = str(exc)
        if _INDEX is None or _METADATA is None:
            _INDEX = None
            _METADATA = None
        if load_embedder:
            _EMBEDDER = None


def _ensure_ready(require_embedder: bool = True):
    if _INDEX is None or _METADATA is None or (require_embedder and _EMBEDDER is None):
        _load_resources(load_embedder=require_embedder)
    if _INDEX is None or _METADATA is None or (require_embedder and _EMBEDDER is None):
        raise RuntimeError(
            f"Retriever is not ready: {_INIT_ERROR}\nBuild vector store first with: python vector_store/build_vector_db.py --if-needed"
        )
    if require_embedder:
        return _INDEX, _METADATA, _EMBEDDER
    return _INDEX, _METADATA


@lru_cache(maxsize=1024)
def _encode_query_cached(query_text: str) -> tuple[float, ...]:
    _, _, embedder = _ensure_ready(require_embedder=True)
    if embedder is None:
        raise RuntimeError("Dense embedder is not available.")
    model_low = (_EMBED_MODEL_RESOLVED or EMBED_MODEL_NAME or "").lower()
    input_text = (QUERY_PREFIX + query_text) if "bge" in model_low else query_text
    vec = embedder.encode([input_text], convert_to_numpy=True, normalize_embeddings=False)
    arr = np.asarray(vec, dtype=np.float32)
    faiss.normalize_L2(arr)
    return tuple(float(x) for x in arr[0])


def _row_act_match(row: Dict, act_hint: str) -> bool:
    if not act_hint:
        return False
    act_low = str(row.get("act_name", "")).lower()
    hint_low = act_hint.lower()
    if hint_low in act_low:
        return True
    for canonical, aliases in ACT_ALIASES.items():
        if canonical.lower() == hint_low:
            return any(alias in act_low for alias in aliases) or hint_low in act_low
    return False


def _row_section_match(row: Dict, sections: List[str]) -> bool:
    if not sections:
        return False
    row_secs = set(row.get("_sections") or set())
    return bool(row_secs.intersection({s.upper() for s in sections}))


def _keyword_overlap(query_terms: List[str], row: Dict) -> float:
    if not query_terms:
        return 0.0
    row_terms = row.get("_tokens") or set()
    overlap = len(set(query_terms).intersection(row_terms))
    return float(overlap) / float(max(1, len(set(query_terms))))


def _extract_evidence_spans(query: str, row: Dict, max_spans: int = 3) -> List[str]:
    q_terms = set(_tokenize(query))
    scored = []
    for sent in _split_sentences(str(row.get("chunk_text", ""))):
        sl = sent.lower()
        overlap = len(q_terms.intersection(set(_tokenize(sl)))) / max(1, len(q_terms)) if q_terms else 0.0
        legal = 0.0
        if re.search(r"\b(?:section|sec\.?|s\.)\s*\d+[a-z]?\b", sl):
            legal += 0.35
        if any(k in sl for k in ["penalty", "fine", "punishable", "imprisonment", "liable"]):
            legal += 0.30
        if any(k in sl for k in ["shall", "must", "required", "prohibited", "offence", "offense"]):
            legal += 0.15
        score = overlap + legal
        if score > 0:
            scored.append((score, sent[:280].strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return [s[:220].strip() for s in _split_sentences(str(row.get("chunk_text", "")))[:2]]
    out = []
    seen = set()
    for _, s in scored:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max(1, int(max_spans)):
            break
    return out


def _intent_adjust(intent: str, row: Dict, text_low: str) -> float:
    delta = 0.0
    act_low = str(row.get("act_name", "")).lower()
    secs = {str(x).upper() for x in (row.get("_sections") or set())}
    if intent == "sexual_offence":
        if any(h in act_low for h in PENAL_ACT_HINTS):
            delta += 0.35
        if any(h in act_low for h in PROCEDURAL_ACT_HINTS):
            delta -= 0.40
        if secs.intersection(SEXUAL_SECTIONS):
            delta += 0.45
        if any(k in text_low for k in ["without consent", "consent", "sexual harassment", "outraging modesty", "woman"]):
            delta += 0.20
    elif intent == "penalty":
        if any(k in text_low for k in ["penalty", "fine", "punishable", "liable", "imprisonment"]):
            delta += 0.14
    elif intent in {"environmental_offence", "compliance_remedy"}:
        if any(k in text_low for k in [
            "tree", "forest", "environment", "felling", "cutting", "permission",
            "clearance", "offence", "penalty", "compensation", "restoration"
        ]):
            delta += 0.20
    elif intent == "general" and secs.intersection(TRAFFIC_SECTIONS):
        delta += 0.02
    return float(delta)


def _domain_adjust(domain: str, row: Dict) -> float:
    d = (domain or "").strip().lower()
    src = str(row.get("_source_group", "")).strip().lower()
    act_low = str(row.get("act_name", "")).lower()
    if d == "criminal_law":
        if src == "criminal_law":
            return 0.24
        if "penal code" in act_low or "nyaya sanhita" in act_low:
            return 0.28
        if "code of criminal procedure" in act_low:
            return -0.18
    if d == "traffic_rules":
        if src == "traffic_rules":
            return 0.26
        if "motor vehicles act" in act_low or "central motor vehicle rules" in act_low:
            return 0.22
    if d == "family_law" and src == "family_law":
        return 0.20
    if d == "civil_law" and src == "civil_law":
        return 0.20
    if d == "constitutional_law" and src == "constitution":
        return 0.24
    if d == "environmental_law":
        if src in {"environmental_law", "forest_law"}:
            return 0.32
        if any(k in act_low for k in ["environment", "forest", "wild life", "wildlife", "biological diversity", "biodiversity"]):
            return 0.28
        if src in {"traffic_rules", "criminal_law", "civil_law", "family_law", "constitutional_law"}:
            return -0.12
    return 0.0


def _route_filter(rows: List[Dict], analysis: Dict) -> List[Dict]:
    intent = str(analysis.get("intent", "")).lower()
    if intent != "sexual_offence":
        return rows
    focused = []
    for row in rows:
        act_low = str(row.get("act_name", "")).lower()
        secs = {str(x).upper() for x in (row.get("_sections") or set())}
        if any(h in act_low for h in PENAL_ACT_HINTS) or secs.intersection(SEXUAL_SECTIONS):
            focused.append(row)
    return focused or rows


def _score_rows(rows: List[Dict], sem_by_id: Dict[int, float], query_terms: List[str], analysis: Dict, retrieval_mode: str) -> List[Dict]:
    sections = list(analysis.get("possible_sections") or [])
    act_hint = str(analysis.get("possible_act", "") or "")
    intent = str(analysis.get("intent", "general") or "general")
    domain = str(analysis.get("legal_domain", "general") or "general")

    if retrieval_mode == "dbx_parity":
        sem_w, kw_w = 0.62, 0.18
    else:
        sem_w, kw_w = 0.68, 0.14

    out = []
    for row in rows:
        doc_id = int(row.get("_doc_id", -1))
        sem = float(sem_by_id.get(doc_id, 0.0))
        kw = _keyword_overlap(query_terms, row)
        sec_match = _row_section_match(row, sections)
        act_match = _row_act_match(row, act_hint)
        score = (sem_w * sem) + (kw_w * kw)
        if sec_match:
            score += SECTION_MATCH_BONUS
            if sections:
                score += 0.30
        if act_match:
            score += ACT_MATCH_BONUS
            if act_hint:
                score += 0.18
        score += _domain_adjust(domain, row)
        score += _intent_adjust(intent, row, str(row.get("chunk_text", "")).lower())
        act_low = str(row.get("act_name", "")).lower()
        if act_hint and ("indian penal code" in act_hint.lower() or "bharatiya nyaya sanhita" in act_hint.lower()):
            if any(h in act_low for h in PROCEDURAL_ACT_HINTS):
                score -= 0.35
        if sections and (not sec_match) and any(h in act_low for h in PROCEDURAL_ACT_HINTS):
            score -= 0.20
        out.append(
            {
                **row,
                "score": float(score),
                "semantic_similarity": float(sem),
                "keyword_overlap": float(kw),
                "section_match": bool(sec_match),
                "act_match": bool(act_match),
            }
        )
    out.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    return out


def expand_section_matches(possible_sections: List[str], possible_act: str = "", limit: int = 80) -> List[Dict]:
    """Retrieve additional rows for section expansion."""
    if not possible_sections:
        return []
    _, metadata = _ensure_ready(require_embedder=False)
    target_secs = {s.upper() for s in possible_sections if s}
    out: List[Dict] = []
    for i, row in enumerate(metadata):
        if len(out) >= max(1, int(limit)):
            break
        if not set(row.get("_sections") or set()).intersection(target_secs):
            continue
        if possible_act and not _row_act_match(row, possible_act):
            continue
        item = dict(row)
        item["_doc_id"] = i
        item["score"] = 0.0
        item["semantic_similarity"] = 0.0
        item["keyword_overlap"] = 0.0
        item["section_match"] = True
        item["act_match"] = _row_act_match(row, possible_act)
        item["evidence_spans"] = _extract_evidence_spans("", row, max_spans=3)
        out.append(item)
    return out


def _lexical_shortlist(metadata: List[Dict], query_terms: List[str], analysis: Dict, limit: int) -> Tuple[List[Dict], Dict[int, float]]:
    sections = list(analysis.get("possible_sections") or [])
    act_hint = str(analysis.get("possible_act", "") or "")
    intent = str(analysis.get("intent", "general") or "general")

    scored: List[Tuple[float, int, Dict]] = []
    for doc_id, row in enumerate(metadata):
        kw = _keyword_overlap(query_terms, row)
        sec = _row_section_match(row, sections)
        act = _row_act_match(row, act_hint)
        txt_low = str(row.get("chunk_text", "")).lower()
        score = (0.70 * kw)
        if sec:
            score += SECTION_MATCH_BONUS
        if act:
            score += ACT_MATCH_BONUS
        score += _intent_adjust(intent, row, txt_low)
        if score <= 0.0 and kw < 0.03:
            continue
        item = dict(row)
        item["_doc_id"] = doc_id
        scored.append((float(score), doc_id, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    trimmed = scored[: max(1, int(limit))]
    sem_by_id: Dict[int, float] = {doc_id: 0.0 for _, doc_id, _ in trimmed}
    rows = [r for _, _, r in trimmed]
    return rows, sem_by_id


def retrieve_chunks(
    question: str,
    top_k: int = DEFAULT_FINAL_K,
    initial_k: int = DEFAULT_INITIAL_K,
    profile: str = "balanced",
    retrieval_mode: str | None = None,
    analysis: Dict | None = None,
    skip_dense: bool | None = None,
    return_meta: bool = False,
) -> List[Dict] | tuple[List[Dict], Dict]:
    """Retrieve legal chunks with dense search + metadata-aware scoring."""
    t_total = time.perf_counter()
    query = _clean_text(question)
    if not query:
        raise ValueError("question cannot be empty")
    if int(top_k) <= 0:
        raise ValueError("top_k must be > 0")

    retrieval_mode = (retrieval_mode or DEFAULT_RETRIEVAL_MODE).strip().lower()
    analysis_used = dict(analysis or query_parser(query))
    normalized_query = _clean_text(str(analysis_used.get("normalized_query", "") or query_normalizer(query)))
    analysis_used["normalized_query"] = normalized_query

    index, metadata = _ensure_ready(require_embedder=False)
    corpus_size = len(metadata)
    if corpus_size == 0:
        empty_meta = {
            "confidence": 0.0,
            "confidence_reason": "empty_corpus",
            "analysis_used": analysis_used,
            "candidate_count": 0,
            "retrieval_mode": retrieval_mode,
            "latency_ms": {"retrieve_total_ms": 0.0},
        }
        return ([], empty_meta) if return_meta else []

    initial_k = min(max(int(initial_k), max(10, int(top_k) * 3)), corpus_size)
    query_terms = [t for t in _tokenize(normalized_query) if t]

    dense_ms = 0.0
    use_dense = not (SKIP_DENSE_FAST_DEFAULT if skip_dense is None else bool(skip_dense))
    candidates: List[Dict] = []
    sem_by_id: Dict[int, float] = {}
    dense_error = ""

    # Always combine lexical and dense candidates when dense retrieval is available.
    # A weak lexical score must never discard semantically relevant legal passages.
    if use_dense:
        t_dense = time.perf_counter()
        try:
            q_vec = np.asarray([_encode_query_cached(normalized_query)], dtype=np.float32)
            d_k = min(corpus_size, max(initial_k, 30))
            d_scores, d_ids = index.search(q_vec, d_k)
            for rank, doc_id in enumerate(d_ids[0].tolist(), start=1):
                if doc_id < 0 or doc_id >= corpus_size:
                    continue
                sem_by_id[int(doc_id)] = max(float(d_scores[0][rank - 1]), sem_by_id.get(int(doc_id), 0.0))
                row = dict(metadata[int(doc_id)])
                row["_doc_id"] = int(doc_id)
                candidates.append(row)
            # Add a lexical pool so exact legal terminology can rescue dense misses.
            lexical_rows, lexical_sem = _lexical_shortlist(
                metadata, query_terms, analysis_used, limit=min(corpus_size, max(initial_k * 4, 60))
            )
            for row in lexical_rows:
                doc_id = int(row.get("_doc_id", -1))
                if doc_id < 0:
                    continue
                sem_by_id.setdefault(doc_id, float(lexical_sem.get(doc_id, 0.0)))
                candidates.append(row)
        except Exception as exc:
            dense_error = str(exc)
            use_dense = False
        dense_ms = (time.perf_counter() - t_dense) * 1000.0

    if not use_dense:
        t_dense = time.perf_counter()
        shortlist_k = min(corpus_size, max(initial_k * 4, 60))
        candidates, sem_by_id = _lexical_shortlist(metadata, query_terms, analysis_used, limit=shortlist_k)
        dense_ms = (time.perf_counter() - t_dense) * 1000.0

    # Deduplicate the merged candidate pool before scoring.
    unique_candidates = {}
    for row in candidates:
        unique_candidates[int(row.get("_doc_id", -1))] = row
    candidates = [row for doc_id, row in unique_candidates.items() if doc_id >= 0]

    candidates = _route_filter(candidates, analysis_used)
    scored = _score_rows(candidates, sem_by_id, query_terms, analysis_used, retrieval_mode)

    top_n = min(len(scored), max(int(top_k), 3))
    selected = scored[:top_n]

    out: List[Dict] = []
    for rank, row in enumerate(selected, start=1):
        out.append(
            {
                "chunk_id": row.get("chunk_id"),
                "act_name": row.get("act_name"),
                "section_number": row.get("section_number"),
                "chunk_text": row.get("chunk_text"),
                "char_count": row.get("char_count"),
                "source_group": row.get("_source_group", row.get("source_group", "acts")),
                "score": float(row.get("score", 0.0)),
                "semantic_similarity": float(row.get("semantic_similarity", 0.0)),
                "keyword_overlap": float(row.get("keyword_overlap", 0.0)),
                "section_match": bool(row.get("section_match", False)),
                "act_match": bool(row.get("act_match", False)),
                "rank": rank,
                "analysis_used": analysis_used,
                "score_breakdown": {
                    "semantic_similarity": float(row.get("semantic_similarity", 0.0)),
                    "keyword_overlap": float(row.get("keyword_overlap", 0.0)),
                    "section_bonus": SECTION_MATCH_BONUS if row.get("section_match", False) else 0.0,
                    "act_bonus": ACT_MATCH_BONUS if row.get("act_match", False) else 0.0,
                },
                "source_priority_reason": str(analysis_used.get("intent", "general")),
                "evidence_spans": _extract_evidence_spans(normalized_query, row, max_spans=3),
            }
        )

    if out:
        top = out[0]
        sem = float(top.get("semantic_similarity", 0.0))
        kw = float(top.get("keyword_overlap", 0.0))
        sec_hit = 1.0 if bool(top.get("section_match", False)) else 0.0
        act_hit = 1.0 if bool(top.get("act_match", False)) else 0.0
        domain_hit = 1.0 if _domain_adjust(str(analysis_used.get("legal_domain", "")), top) > 0 else 0.0
        confidence = (0.50 * max(0.0, sem)) + (0.25 * max(0.0, kw)) + (0.10 * sec_hit) + (0.10 * act_hit) + (0.05 * domain_hit)
    else:
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    conf_reason = (
        f"top_score={out[0]['score']:.3f}, top_sem={out[0]['semantic_similarity']:.3f}, "
        f"top_kw={out[0]['keyword_overlap']:.3f}, candidates={len(scored)}"
        if out
        else "no_candidates"
    )

    meta = {
        "analysis_used": analysis_used,
        "confidence": round(confidence, 4),
        "confidence_reason": conf_reason,
        "candidate_count": len(scored),
        "retrieval_mode": retrieval_mode,
        "latency_ms": {
            "dense_ms": round(dense_ms, 2),
            "retrieve_total_ms": round((time.perf_counter() - t_total) * 1000.0, 2),
        },
        "dense_used": bool(use_dense),
        "dense_error": dense_error,
        "profile": profile,
    }
    if return_meta:
        return out, meta
    return out


def retriever_status() -> Dict:
    # Retry lightweight load so status can recover after artifacts are built while app is running.
    if _INDEX is None or _METADATA is None:
        try:
            _load_resources(load_embedder=False)
        except Exception:
            pass

    base_ready = _INDEX is not None and _METADATA is not None
    dense_ready = _EMBEDDER is not None
    ready = bool(base_ready)
    return {
        "ready": bool(ready),
        "error": _INIT_ERROR,
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "corpus_size": len(_METADATA) if _METADATA is not None else 0,
        "embed_model": _EMBED_MODEL_RESOLVED,
        "default_top_k": DEFAULT_FINAL_K,
        "initial_k": DEFAULT_INITIAL_K,
        "default_retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "skip_dense_fast_default": SKIP_DENSE_FAST_DEFAULT,
        "dense_embedder_available": SentenceTransformer is not None,
        "dense_embedder_ready": dense_ready,
        "base_ready": base_ready,
    }


_load_resources(load_embedder=False)
