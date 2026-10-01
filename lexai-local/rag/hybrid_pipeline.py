"""Hybrid local retrieval + authoritative legal web research."""
from __future__ import annotations
import time
from rag.query_analyzer import analyze
from rag.retriever import retrieve
from rag.llm_service import answer
from rag.web_legal_search import search_legal_web
from config import WEB_MAX_RESULTS,RETRIEVAL_TOP_K

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

Authoritative web evidence:
{chr(10).join(web_parts)}

Produce a structured Indian legal research response.
Use only the supplied evidence. Prefer current authoritative web evidence if it conflicts with the local corpus, and explicitly identify the conflict.
Do not invent sections, penalties, dates, cases, authorities, or procedures.
Separate legal rules from general practical information.
Include:
1. Direct answer
2. Applicable Act/section, only when supported
3. Plain-language explanation
4. Important exceptions or missing facts
5. Practical next steps when supported
6. Citations using [SOURCE n] and [WEB n]
7. A short Sources section with the URLs for cited web sources
Do not treat local metadata as authoritative when it conflicts with the source text."""
    text,llm=answer([{"role":"system","content":"You are LexAI, a careful Indian legal research assistant. This is legal information, not a substitute for a qualified advocate."},{"role":"user","content":prompt}],max_tokens=500)
    if not llm.get("ok") or not text:
        text=("Qwen answer generation is unavailable. Authoritative web sources found:\n\n"+
              "\n".join(f"- {r.get('title')} — {r.get('url')}" for r in web_rows))
    return {"answer":text,"sources":local_sources,"web_sources":web_rows,
            "meta":{"mode":"hybrid_web","plan":plan,"retrieval":ret,"web":web_meta,
                    "llm":llm,"total_ms":round((time.perf_counter()-t0)*1000,2)}}
