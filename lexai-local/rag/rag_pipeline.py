"""End-to-end conversational legal QA engine."""
from __future__ import annotations
import time
from rag.query_analyzer import analyze
from rag.retriever import retrieve
from rag.llm_service import answer,health
from config import MAX_HISTORY_TURNS,RETRIEVAL_TOP_K,MIN_EVIDENCE_SCORE

def _evidence(rows):
 parts=[]
 for i,r in enumerate(rows,1):
  parts.append(f"[SOURCE {i}] {r.get('act_name','Unknown Act')} | Section {r.get('section_number','N/A')}\n{str(r.get('chunk_text',''))[:1800]}")
 return "\n\n".join(parts)

def _history_text(history):
 return "\n".join(f"{x['role']}: {x['content']}" for x in (history or [])[-MAX_HISTORY_TURNS:])

def ask_lexai(question,history=None,top_k=RETRIEVAL_TOP_K,**kwargs):
 q=(question or "").strip()
 if not q: return {"answer":"Please enter a legal question.","sources":[],"meta":{}}
 t0=time.perf_counter(); history=history or []
 plan=analyze(q,history)
 rows,ret=retrieve(str(plan.get("standalone_query") or q),plan,top_k)
 evidence=_evidence(rows); confidence=float(ret.get("confidence",0))
 if not rows or confidence < MIN_EVIDENCE_SCORE:
  return {"answer":"I could not find sufficiently relevant legal provisions in the local corpus for this question. I have not generated a legal conclusion from unrelated sections. Try adding the Act, section, offence, State, date, or facts involved.","sources":[],"meta":{"plan":plan,"retrieval":ret,"confidence":confidence,"mode":"insufficient_evidence","total_ms":round((time.perf_counter()-t0)*1000,2)}}
 system="""You are LexAI, a careful Indian legal research assistant.
Use ONLY the supplied retrieved legal sources for legal propositions.
Do not invent sections, penalties, dates, cases, authorities, or procedural steps.
If the sources are insufficient, say so explicitly.
Distinguish law stated in the source from practical guidance.
For current Indian law, pay attention to IPC/CrPC versus post-2024 BNS/BNSS; never silently treat them as identical.
Answer in the user's language when practical (English, Hindi, or Hinglish).
This is legal information, not a substitute for a qualified advocate."""
 user=f"""Question: {q}
Research plan: {plan}
Conversation:
{_history_text(history)}
Retrieved legal sources:
{evidence}
Write a useful day-to-day legal answer with:
- direct answer first
- applicable Act/section only when supported
- plain-language meaning
- practical next steps only when supported or clearly labeled general information
- important uncertainty or missing facts
- citations as [SOURCE n]
Do not cite a source that does not support the statement."""
 text,llm=answer([{"role":"system","content":system},{"role":"user","content":user}],max_tokens=900)
 if not llm.get("ok") or not text: text="The local Qwen service is unavailable. Retrieval found these potentially relevant provisions:\n\n"+evidence
 return {"answer":text,"sources":rows,"meta":{"plan":plan,"retrieval":ret,"confidence":confidence,"mode":"grounded_qwen" if llm.get("ok") else "retrieval_only","llm":llm,"total_ms":round((time.perf_counter()-t0)*1000,2)}}

def runtime_status():
 return {"llm":health()}
