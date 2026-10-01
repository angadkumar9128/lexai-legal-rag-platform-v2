"""Context construction helpers for LexAI."""

from __future__ import annotations

import re
from typing import Dict, List


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[\.\!\?])\s+")
SECTION_RE_1 = re.compile(r"\b(?:section|sec\.?|s\.)\s*([0-9]{1,4}[A-Za-z]{0,2})\b", flags=re.IGNORECASE)
SECTION_RE_2 = re.compile(r"\b([0-9]{1,4}[A-Za-z]{0,2})\.\s*[A-Z]")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_sentences(text: str) -> List[str]:
    src = _clean_text(text)
    if not src:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(src) if len(s.strip()) >= 35]


def _extract_evidence_spans(query: str, row: Dict, max_spans: int = 2) -> List[str]:
    spans = list(row.get("evidence_spans") or [])
    if spans:
        return [str(s).strip() for s in spans if str(s).strip()][: max(1, int(max_spans))]
    q_terms = set(re.findall(r"[a-z0-9]{2,}", (query or "").lower()))
    scored = []
    for sent in _split_sentences(str(row.get("chunk_text", ""))):
        sl = sent.lower()
        overlap = len(q_terms.intersection(set(re.findall(r"[a-z0-9]{2,}", sl)))) / max(1, len(q_terms)) if q_terms else 0.0
        legal = 0.0
        if re.search(r"\b(?:section|sec\.?|s\.)\s*\d+[a-z]?\b", sl):
            legal += 0.35
        if any(k in sl for k in ["penalty", "fine", "punishable", "imprisonment", "liable"]):
            legal += 0.28
        if any(k in sl for k in ["shall", "must", "required", "prohibited", "offence", "offense"]):
            legal += 0.16
        score = overlap + legal
        if score > 0:
            scored.append((score, sent[:260]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[: max(1, int(max_spans))]]


def _extract_sections_from_text(text: str) -> List[str]:
    found: List[str] = []
    seen = set()
    for patt in (SECTION_RE_1, SECTION_RE_2):
        for m in patt.finditer(text or ""):
            s = str(m.group(1)).upper().strip()
            if s and s not in seen:
                seen.add(s)
                found.append(s)
    return found[:8]


def _pick_section_label(row: Dict, analysis: Dict | None = None) -> str:
    analysis_sections = [str(x).upper() for x in (analysis or {}).get("possible_sections", []) if str(x).strip()]
    row_section = _clean_text(str(row.get("section_number", "")))
    row_tokens = [str(x).upper() for x in (row.get("section_tokens") or row.get("_sections") or []) if str(x).strip()]
    text_sections = _extract_sections_from_text(str(row.get("chunk_text", "")))

    for s in analysis_sections:
        if s in row_tokens or s in text_sections:
            return f"Section {s}"

    for s in row_tokens:
        if re.match(r"^[0-9]{1,4}[A-Z]{0,2}$", s):
            return f"Section {s}"

    for s in text_sections:
        if re.match(r"^[0-9]{1,4}[A-Z]{0,2}$", s):
            return f"Section {s}"

    if row_section and not row_section.lower().startswith("chapter"):
        return row_section
    return row_section or "N/A"


def extract_citations(rows: List[Dict], analysis: Dict | None = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        act = _clean_text(str(row.get("act_name", ""))) or "Unknown Act"
        sec = _pick_section_label(row, analysis=analysis)
        ref = f"{act} - {sec}"
        key = ref.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out[:10]


def build_context(rows: List[Dict], max_sections: int = 3, max_chars: int = 1500) -> str:
    max_sections = max(1, int(max_sections))
    max_chars = max(400, int(max_chars))
    parts = ["--- LEGAL CONTEXT START ---\n\n"]
    used = len(parts[0])

    chosen = rows[:max_sections]
    for i, row in enumerate(chosen):
        act = _clean_text(str(row.get("act_name", ""))) or "Unknown Act"
        sec = _pick_section_label(row, analysis=None)
        txt = _clean_text(str(row.get("chunk_text", "")))
        if not txt:
            continue
        block = f"[{act} | {sec}]\n{txt}\n\n"
        if i < len(chosen) - 1:
            block += "--- NEXT SECTION ---\n\n"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip() + "\n"
        parts.append(block)
        used += len(block)
        if used >= max_chars:
            break
    out = "".join(parts).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def build_evidence_context(query: str, rows: List[Dict], max_blocks: int = 4, max_chars: int = 1200) -> str:
    max_blocks = max(1, int(max_blocks))
    max_chars = max(300, int(max_chars))
    parts = ["--- LEGAL CONTEXT START ---\n\n"]
    used = len(parts[0])
    blocks = 0

    for idx, row in enumerate(rows[:max_blocks], start=1):
        act = _clean_text(str(row.get("act_name", ""))) or "Unknown Act"
        sec = _pick_section_label(row, analysis=None)
        spans = _extract_evidence_spans(query, row, max_spans=2)
        if not spans:
            continue
        bullet = "".join([f"- {_clean_text(s)}\n" for s in spans])
        block = f"[{idx}] {act} | {sec}\n{bullet}\n"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip() + "\n"
        parts.append(block)
        used += len(block)
        blocks += 1
        if used >= max_chars:
            break

    if blocks == 0:
        return ""
    out = "".join(parts).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out
