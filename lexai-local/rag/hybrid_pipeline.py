"""Hybrid local RAG + authoritative legal web research."""
from __future__ import annotations
import time
from rag.rag_pipeline import ask_lexai
from rag.llm_service import answer
from rag.web_legal_search import search_legal_web
from config import WEB_MAX_RESULTS

def ask_hybrid(question, history=None, top_k=6, web_max_results=WEB_MAX_RESULTS):
    t0=time.perf_counter()
    local=ask_lexai(question, history=history, top_k=top_k)
    web_rows, web_meta=search_legal_web(question, max_results=web_max_results)
    local_answer=local.get("answer","")
    local_sources=local.get("sources",[])
    web_parts=[]
    for i,r in enumerate(web_rows,1):
        web_parts.append(f"[WEB {i}] {r.get('source_name',r.get('domain'))} | {r.get('title')} | {r.get('url')}\n{str(r.get('content',''))[:5000]}")
    local_parts=[]
    for i,r in enumerate(local_sources,1):
        local_parts.append(f"[SOURCE {i}] {r.get('act_name')} | Section {r.get('section_number')}\n{str(r.get('chunk_text',''))[:1100]}")
    prompt=f"""Question: {question}

Local RAG evidence:
{chr(10).join(local_parts)}

Authoritative web evidence:
{chr(10).join(web_parts)}

Local preliminary result:
{local_answer}

Produce a structured Indian legal research response.
Use only the supplied evidence. Prefer current authoritative web evidence if it conflicts with the local corpus, and explicitly identify the conflict.
Do not invent sections, penalties, dates, cases, authorities, or procedures.
Separate legal rules from general practical information.
Include:
1. Direct answer
2. Applicable law/sections, only when supported
3. Plain-language explanation
4. Important exceptions or missing facts
5. Practical next steps when supported
6. Sources with [SOURCE n] and [WEB n] citations
7. URL for every web source cited
Do not treat the local corpus as authoritative when its metadata conflicts with its text."""
    text,llm=answer([{"role":"system","content":"You are LexAI, a careful Indian legal research assistant. This is legal information, not a substitute for a qualified advocate."},{"role":"user","content":prompt}],max_tokens=500)
    if not llm.get("ok") or not text:
        text=("Qwen answer generation is unavailable. Web research found:\n\n"+
              "\n".join(f"- {r.get('title')} — {r.get('url')}" for r in web_rows))
    return {"answer":text,"sources":local_sources,"web_sources":web_rows,
            "meta":{"mode":"hybrid_web","web":web_meta,"local":local.get("meta",{}),
                    "llm":llm,"total_ms":round((time.perf_counter()-t0)*1000,2)}}
