"""Streamlit UI for LexAI local enterprise RAG."""

from __future__ import annotations

import streamlit as st

from rag.generator import generator_status
from rag.rag_pipeline import ask_lexai
from rag.query_analyzer import query_analyzer_status
from rag.reranker import reranker_status
from rag.retriever import retriever_status


st.set_page_config(page_title="LexAI Legal RAG - Local Edition", layout="wide")

st.title("LexAI Legal RAG - Local Edition")
st.caption("Evidence-grounded legal QA with hybrid retrieval, reranking, and local LLM profiles.")

ret_status = retriever_status()
gen_status = generator_status()
qa_status = query_analyzer_status()
rr_status = reranker_status()

with st.sidebar:
    st.subheader("Runtime Status")
    st.write(f"Retriever Ready: `{ret_status.get('ready')}`")
    st.write(f"Generator Ready: `{gen_status.get('ready')}`")
    st.write(f"Analyzer Ready: `{qa_status.get('ready')}`")
    st.write(f"Corpus Size: `{ret_status.get('corpus_size', 0)}`")
    st.write(f"Embedder: `{ret_status.get('embed_model', 'n/a')}`")
    st.write(f"Default Profile: `{gen_status.get('default_profile', 'balanced')}`")
    st.write(f"LLM-1 Model: `{qa_status.get('llm1_model', 'n/a')}`")
    st.write(f"LLM-2 Model: `{gen_status.get('llm2_model', 'n/a')}`")
    st.write(f"Reranker Model: `{rr_status.get('default_model', 'n/a')}`")
    st.write(f"Default Retrieval Mode: `{ret_status.get('default_retrieval_mode', 'hybrid')}`")

    profile = st.selectbox("Inference Profile", options=["balanced", "high_accuracy"], index=1)
    retrieval_mode = st.selectbox("Retrieval Mode", options=["hybrid", "dbx_parity"], index=0)
    top_k = st.slider("Top-K Retrieval", min_value=3, max_value=8, value=6)
    style = st.selectbox("Answer Style", options=["normal", "short", "very_short", "detailed"], index=0)
    target_words = st.slider("Target Words", min_value=40, max_value=320, value=180, step=10)

if not ret_status.get("ready"):
    st.error(
        "Retriever is not ready. Build artifacts first:\n\n"
        "`python vector_store/build_vector_db.py --if-needed`"
    )
    st.stop()

if not gen_status.get("ready"):
    st.warning(
        "LLM-2 model is not ready. LexAI will continue in deterministic draft mode.\n\n"
        f"Details: {gen_status.get('error', 'unknown error')}"
    )

question = st.text_area(
    "Ask a legal question",
    value="What is the penalty for not wearing a helmet?",
    height=120,
)

if st.button("Ask LexAI", type="primary"):
    with st.spinner("Retrieving legal evidence and generating response..."):
        try:
            result = ask_lexai(
                question=question,
                top_k=top_k,
                style=style,
                target_words=target_words,
                profile=profile,
                retrieval_mode=retrieval_mode,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Generated Answer")
    st.markdown(result["answer"])

    meta = result.get("meta", {})
    retrieval = meta.get("retrieval", {})
    rerank = meta.get("reranker", {})
    analysis = meta.get("analysis", {})
    conf = meta.get("confidence", retrieval.get("confidence", 0.0))
    bucket = meta.get("confidence_bucket", "low")
    mode = meta.get("quality_mode", "n/a")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LLM-1 (ms)", meta.get("llm1_ms", 0.0))
    c2.metric("Retrieve (ms)", meta.get("retrieve_ms", 0.0))
    c3.metric("Rerank (ms)", meta.get("rerank_ms", 0.0))
    c4.metric("LLM-2 (ms)", meta.get("llm2_ms", 0.0))
    c5.metric("Total (ms)", meta.get("total_ms", 0.0))
    st.metric("Confidence", conf)

    st.caption(
        " | ".join(
            [
                f"profile={meta.get('profile', profile)}",
                f"retrieval_mode={meta.get('retrieval_mode', retrieval_mode)}",
                f"bucket={bucket}",
                f"mode={mode}",
                f"top_k={meta.get('top_k', 0)}",
                f"context_chars={meta.get('context_chars', 0)}",
                f"rerank_used={retrieval.get('rerank_used', False)}",
            ]
        )
    )

    st.subheader("Query Analysis (LLM-1)")
    a1, a2 = st.columns(2)
    a1.write(f"Normalized Query: `{analysis.get('normalized_query', '')}`")
    a1.write(f"Possible Act: `{analysis.get('possible_act', '')}`")
    a1.write(f"Sections: `{analysis.get('possible_sections', [])}`")
    a2.write(f"Legal Domain: `{analysis.get('legal_domain', '')}`")
    a2.write(f"Intent: `{analysis.get('intent', '')}`")
    a2.write(f"Analyzer Confidence: `{analysis.get('confidence', 0.0)}`")

    profile_roll = meta.get("latency_profile", {})
    if profile_roll:
        st.subheader("Latency Profile (rolling)")
        p1, p2, p3 = st.columns(3)
        p1.write(
            f"Retrieve p50/p95: `{profile_roll.get('retrieve_ms', {}).get('p50', 0)}` / "
            f"`{profile_roll.get('retrieve_ms', {}).get('p95', 0)}`"
        )
        p2.write(
            f"Generate p50/p95: `{profile_roll.get('generate_ms', {}).get('p50', 0)}` / "
            f"`{profile_roll.get('generate_ms', {}).get('p95', 0)}`"
        )
        p3.write(
            f"Total p50/p95: `{profile_roll.get('total_ms', {}).get('p50', 0)}` / "
            f"`{profile_roll.get('total_ms', {}).get('p95', 0)}`"
        )

    st.subheader("Retrieval Diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.write(f"Candidates: `{retrieval.get('candidate_count', 0)}`")
    d2.write(f"Reason: `{retrieval.get('confidence_reason', '')}`")
    d3.write(f"Signals: `{retrieval.get('analysis_used', retrieval.get('signals', {}))}`")
    st.write(f"Reranker: `{rerank.get('model_name', '')}` | used=`{rerank.get('rerank_used', False)}` | error=`{rerank.get('error', '')}`")

    st.subheader("Retrieved Sources")
    sources = result.get("sources", [])
    if not sources:
        st.info("No relevant sources were retrieved.")
    else:
        for i, src in enumerate(sources, start=1):
            title = (
                f"{i}. {src.get('act_name', 'Unknown Act')} | "
                f"Section {src.get('section_number', 'N/A')} | chunk_id={src.get('chunk_id', 'N/A')}"
            )
            with st.expander(title):
                st.write(src.get("chunk_text", ""))
                st.caption(
                    " | ".join(
                        [
                            f"rank={src.get('rank')}",
                            f"score={src.get('score', 0):.4f}",
                            f"semantic={src.get('semantic_similarity', 0):.4f}",
                            f"keyword={src.get('keyword_overlap', 0):.4f}",
                            f"section_match={src.get('section_match', False)}",
                            f"act_match={src.get('act_match', False)}",
                            f"group={src.get('source_group', 'n/a')}",
                        ]
                    )
                )
