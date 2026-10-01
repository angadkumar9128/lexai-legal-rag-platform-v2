"""Compatibility wrapper around the persistent Qwen generation service."""
from rag.llm_service import health
def generator_status():
 h=health()
 return {"ready":bool(h.get("ready")),"error":h.get("error",""),"mode":"llama.cpp persistent Qwen API"}
def generate_answer(context,question,**kwargs):
 from rag.llm_service import answer
 text,meta=answer([{"role":"system","content":"Answer only from the supplied legal context. Do not invent law."},{"role":"user","content":f"Question: {question}\nContext:\n{context}"}],max_tokens=kwargs.get("target_words",900))
 return text if meta.get("ok") else "Qwen service unavailable."
