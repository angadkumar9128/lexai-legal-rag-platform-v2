# LexAI Local — Production-Oriented Legal RAG

LexAI is a local Indian-legal research assistant. The application is intentionally **evidence-first**: retrieval decides what legal material is available, and Qwen is allowed to explain only that material.

## Architecture

1. **Query planner** — deterministic routing + Qwen structured JSON.
2. **Hybrid retrieval** — BGE dense retrieval + BM25-style lexical retrieval + Reciprocal Rank Fusion.
3. **Legal routing** — act/domain/section/intent boosts; English + Hindi/Hinglish expansions.
4. **Reranking** — optional BAAI/bge-reranker-v2-m3 for multilingual relevance.
5. **Grounding gate** — weak retrieval produces an explicit insufficient-evidence response instead of invented law.
6. **Qwen generation** — one persistent llama.cpp server; the same Qwen2.5 model performs planning and answer generation.
7. **Conversation UI** — follow-up questions, source inspection and diagnostics.

## Why this replaces the old design

The old stack loaded separate local LLM bindings for analyzer/generator and mixed unrelated confidence signals. On Windows, llama-cpp-python could also be unavailable, silently forcing deterministic fallback. This version uses the official llama.cpp OpenAI-compatible server so model loading is persistent and independent of the Python environment.

Qwen2.5-3B-Instruct GGUF supports structured output/JSON and a 32K context in the official model card. llama.cpp supports JSON-schema constrained responses and OpenAI-compatible chat completions.

## Windows setup

Open PowerShell in `lexai-local`.

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Build the vector store:

    python vector_store/build_vector_db.py --if-needed

## Start Qwen

Install llama.cpp using the official Windows package:

    winget install llama.cpp

Then start the local server. CPU-safe example:

    llama-server -m "C:\path\to\qwen2.5-3b-instruct-q4_k_m.gguf" --host 127.0.0.1 --port 8080 -c 32768 -ngl 0

If you have a supported NVIDIA/AMD GPU build, use the appropriate GPU offload setting instead of `-ngl 0`. Do not guess the value; start with the default supported by your installed build and increase only when stable.

Verify:

    Invoke-RestMethod http://127.0.0.1:8080/health

Expected response is an HTTP success/healthy status.

In another PowerShell window:

    cd C:\Users\kumar\Downloads\lexai-legal-rag-platform\lexai-local
    .\.venv\Scripts\Activate.ps1
    streamlit run app.py

## Qwen settings

The app sends:

- 32K context window
- deterministic structured JSON for query planning
- low-temperature grounded answer generation
- top-p sampling
- repeat penalty
- thinking disabled for predictable latency
- JSON schema constraints for planner output

Qwen2.5-3B-Instruct-GGUF officially documents 32,768 context and up to 8,192 generated tokens; the Q4_K_M variant is the practical balanced local choice.

## Better quality option

If the computer has enough RAM/VRAM, Qwen2.5-7B-Instruct-Q4_K_M is the first upgrade to try. The official Qwen GGUF repository provides Q4_K_M and larger quantizations. Change only:

    LEXAI_LLM_MODEL=qwen2.5-7b-instruct

and start llama-server with the 7B GGUF file.

The application architecture does not change.

For English/Hindi/Hinglish retrieval, BAAI/bge-reranker-v2-m3 is a stronger multilingual reranking option than the previous MS MARCO reranker. It is optional and consumes additional RAM.

## Important: legal corpus quality

No RAG system can answer a law that is absent, outdated, incomplete or incorrectly chunked in its corpus. Before relying on LexAI, make sure the parquet corpus contains the statutes you expect, especially current BNS/BNSS provisions and relevant state laws.

For high-stakes matters, verify the cited provision against the current primary legal text.

## Troubleshooting

### Qwen is offline

Check:

    Invoke-RestMethod http://127.0.0.1:8080/health

If it fails, llama-server is not running or is listening on another port.

### Same answer for unrelated questions

This architecture does not reuse generated answers. Each turn gets a new query plan and retrieval pass. If results are still wrong, open Research diagnostics and inspect:

- expanded query
- detected domain/intent
- candidate count
- dense score
- BM25 score
- reranker result
- retrieved source text

### Rebuild retrieval after corpus changes

    python vector_store/build_vector_db.py --if-needed

If the source parquet changed but the manifest did not invalidate correctly, force a rebuild by deleting only:

    vector_store/faiss_index.bin
    vector_store/metadata.pkl
    vector_store/lexical_artifacts.pkl
    vector_store/build_manifest.json

then run the build command again.

## Daily-use workflow

Ask normal questions directly:

- What is the punishment for stabbing someone?
- What should I do if I cut trees without permission?
- What is Section 103 of BNS?
- Mere landlord ne deposit return nahi kiya, kya kar sakta hoon?
- Helmet nahi pehna to kitna fine hai?
- Is this a civil or criminal matter?

For follow-ups:

- What if the victim is seriously injured?
- What if it happened in Karnataka?
- Which section applies?
- What should I do next?

LexAI uses conversation history to turn follow-ups into standalone retrieval queries.
