# LexAI Local Setup

This is the repository's CPU-oriented local serving profile.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11
- 16 GB RAM recommended
- Disk for Python packages, embedding models and optional GGUF models

## Bootstrap data

The repository currently includes:

    data/legal_embeddings_delta.parquet

The local vector builder uses this file by default. Do not rename it to gold_chunks.parquet.

## Setup

PowerShell:

    cd lexai-local
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Build

    python vector_store/build_vector_db.py --if-needed

Generated Git-ignored files:

    vector_store/faiss_index.bin
    vector_store/metadata.pkl
    vector_store/lexical_artifacts.pkl
    vector_store/build_manifest.json

## Run

    streamlit run app.py

## Default local stack

- Embeddings: BAAI/bge-base-en-v1.5
- Retrieval: FAISS inner-product/cosine + lexical metadata scoring
- Reranker: enabled by default for higher retrieval precision
- Generation: optional local GGUF models; deterministic grounded fallback is used when llama-cpp-python is unavailable

Expected models:

    models/qwen2.5-3b-instruct-q4_k_m.gguf
    models/mistral-7b-instruct.Q4_K_M.gguf

Override model paths:

    $env:LEXAI_LLM1_MODEL="C:\path\to\qwen2.5-3b-instruct-q4_k_m.gguf"
    $env:LEXAI_LLM2_MODEL="C:\path\to\mistral-7b-instruct.Q4_K_M.gguf"

Disable local LLM generation if needed:

    $env:LEXAI_USE_LLM="0"
    $env:LEXAI_USE_LLM1="0"

The deterministic fallback path remains available.

## One-command startup after dependencies are installed

    python vector_store/build_vector_db.py --if-needed
    if ($LASTEXITCODE -eq 0) { streamlit run app.py }

## Troubleshooting

### Input parquet not found

Run from lexai-local or pass the repository file explicitly:

    python vector_store/build_vector_db.py --parquet .\data\legal_embeddings_delta.parquet --if-needed

### Retriever is not ready

Rebuild:

    python vector_store/build_vector_db.py --if-needed

### llama-cpp-python installation fails on Windows

The local LLM is optional:

    $env:LEXAI_USE_LLM="0"
    $env:LEXAI_USE_LLM1="0"

Then run the Streamlit app again.

### Slow responses

Keep fast mode enabled and use top_k around 3–5. Reranking and local generation are the expensive optional stages.
