"""Hybrid local retrieval + resilient web research."""
from __future__ import annotations
import re
import time
from rag.query_analyzer import analyze
from rag.retriever import retrieve
from rag.llm_service import answer
from rag.web_legal_search import search_legal_web
from config import WEB_MAX_RESULTS,RETRIEVAL_TOP_K

def _fallback_web_answer(question, web_rows):
    if not web_rows:
        return ("No web sources could be retrieved. Check the web-search diagnostics for "
                "network/proxy/DNS restrictions.")
    terms=[x for x in re.findall(r"[a-z0-9]{3,}",question.lower()) if x not in {"what","with","the","for","and","does","this","that"}]
    blocks=[]
    for i,r in enumerate(web_rows,1):
        text=str(r.get("content",""))
        sentences=re.split(r"(?<=[.!?])\s+",text)
        relevant=[s.strip() for s in sentences if any(t in s.lower() for t in terms)]
        excerpt=" ".join(relevant[:3]) or text[:700]
        blocks.append(f"**[WEB {i}] {r.get('source_name',r.get('domain'))}** — {r.get('title','Untitled')}\n"
                      f"{excerpt[:1200]}\nSource: {r.get('url')}")
    return ("### Web research result\n\n"
            "Qwen synthesis was unavailable, so LexAI is showing only relevance-checked web evidence rather than inventing a legal conclusion.\n\n"
            + "\n\n".join(blocks))

def ask_hybrid(question, history=None, top_k=RETRIEVAL_TOP_K, web_max_results=WEB_MAX_RESULTS):
    t0=time.perf_counter()
    q=(question or "").strip()
    plan=analyze(q,history or [])
    local_sources,ret=retrieve(str(plan.get("standalone_query") or q),plan,min(top_k,6))
    web_rows,web_meta=search_legal_web(str(plan.get("standalone_query") or q),max_results=web_max_results)

    web_parts=[]
    for i,r in enumerate(web_rows,1):
        web_parts.append(f"[WEB {i}] {r.get('source_name',r.get('domain'))} | {r.get('title')} | {r.get('url')}\n{str(r.get('content',''))[:5000]}")
    local_parts=[]
    for i,r in enumerate(local_sources,1):
        local_parts.append(f"[SOURCE {i}] {r.get('act_name')} | Section {r.get('section_number')}\n{str(r.get('chunk_text',''))[:1100]}")
    prompt=f"""Question: {q}
Research plan: {plan}

Local RAG evidence:
{chr(10).join(local_parts)}

Authoritative/preferred web evidence:
{chr(10).join(web_parts)}

Produce a structured Indian legal research response.
Use only supplied evidence. Prefer current government/court evidence when it conflicts with local corpus and explicitly identify conflicts.
Do not invent sections, penalties, dates, cases, authorities, or procedures.
Include direct answer, applicable supported law/sections, plain-language explanation, important exceptions/missing facts, practical next steps when supported, and [SOURCE n]/[WEB n] citations.
Include URLs for cited web sources."""

    text,llm=answer(
        [{"role":"system","content":"You are LexAI, a careful Indian legal research assistant. This is legal information, not a substitute for a qualified advocate."},
         {"role":"user","content":prompt}],
        max_tokens=450, timeout=35)
    fallback_used=False
    if not llm.get("ok") or not text:
        text=_fallback_web_answer(q,web_rows)
        fallback_used=True
    return {"answer":text,"sources":local_sources,"web_sources":web_rows,
            "meta":{"mode":"hybrid_web","plan":plan,"retrieval":ret,"web":web_meta,
                    "llm":llm,"fallback_used":fallback_used,
                    "total_ms":round((time.perf_counter()-t0)*1000,2)}}
