"""LexAI local legal assistant UI."""
from __future__ import annotations
import streamlit as st
from rag.rag_pipeline import ask_lexai,runtime_status
from rag.retriever import retriever_status
from config import env_summary,RETRIEVAL_TOP_K

st.set_page_config(page_title="LexAI Legal Assistant",page_icon="⚖️",layout="wide")
st.title("⚖️ LexAI — Indian Legal Research Assistant")
st.caption("Local, evidence-grounded legal Q&A powered by Qwen2.5 + hybrid retrieval.")

if "messages" not in st.session_state: st.session_state.messages=[]
if "last_meta" not in st.session_state: st.session_state.last_meta={}

with st.sidebar:
 st.header("System")
 rs=retriever_status(); ls=runtime_status()["llm"]
 st.write("Retriever:", "🟢 Ready" if rs.get("ready") else "🔴 Not ready")
 st.write("Qwen:", "🟢 Connected" if ls.get("ready") else "🔴 Offline")
 st.write("Corpus:", rs.get("corpus_size",0))
 st.write("Embedding:", env_summary()["embed_model"])
 st.write("Reranker:", env_summary()["reranker"])
 top_k=st.slider("Sources",3,10,RETRIEVAL_TOP_K)
 st.divider()
 st.info("Legal information only. Verify important matters with primary law or a qualified advocate.")
 if st.button("Clear conversation"):
  st.session_state.messages=[]; st.rerun()

for m in st.session_state.messages:
 with st.chat_message(m["role"]): st.markdown(m["content"])

q=st.chat_input("Ask a legal question in English, Hindi, or Hinglish…")
if q:
 st.session_state.messages.append({"role":"user","content":q})
 with st.chat_message("user"): st.markdown(q)
 with st.chat_message("assistant"):
  with st.spinner("Researching the legal corpus…"):
   result=ask_lexai(q,st.session_state.messages[:-1],top_k=top_k)
  st.markdown(result["answer"])
  st.session_state.last_meta=result.get("meta",{})
  sources=result.get("sources",[])
  if sources:
   st.markdown("### Sources")
   for i,s in enumerate(sources,1):
    with st.expander(f"[SOURCE {i}] {s.get('act_name','Unknown Act')} — Section {s.get('section_number','N/A')}"):
     st.write(s.get("chunk_text",""))
     st.caption(f"dense={s.get('dense_score',0):.3f} | bm25={s.get('bm25_score',0):.3f} | score={s.get('score',0):.3f}")
st.divider()
meta=st.session_state.last_meta
if meta:
 with st.expander("Research diagnostics"):
  st.json(meta)
