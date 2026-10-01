# LexAI — Legal RAG & Semantic Intelligence Platform

LexAI has two execution profiles:

1. Databricks: PDF ingestion → Bronze → Silver → Gold → embeddings → high-precision legal QA.
2. Local: persisted legal embeddings → FAISS/lexical retrieval → optional reranking → optional local GGUF generation → Streamlit UI.

## Current layout

    apps/
      fastapi_app.py
      streamlit_app.py
      lexai06_notebook_adapter.py
      notebook_06_snapshot.*
      requirements.txt

    notebooks/
      01_bronze_ingestion.ipynb
      02_silver_processing.ipynb
      03_gold_chunking.ipynb
      04_generate_embeddings.ipynb
      05_rag_answer_pipeline.ipynb
      06_High-precision_QA_Legal_Reasoning_Engine.ipynb
      07_one_click_lexai_runner.ipynb
      08_export_gold_chunks_for_local.ipynb

    lexai-local/
      app.py
      data/legal_embeddings_delta.parquet
      models/
      rag/
      vector_store/
      requirements.txt

## 1. Run locally

The repository already contains the bootstrap embedding export:

    lexai-local/data/legal_embeddings_delta.parquet

PowerShell:

    cd lexai-local
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

    python vector_store/build_vector_db.py --if-needed
    streamlit run app.py

Open the Streamlit URL shown in the terminal, normally http://localhost:8501.

After dependencies are installed, the one-command startup is:

    cd lexai-local
    python vector_store/build_vector_db.py --if-needed
    if ($LASTEXITCODE -eq 0) { streamlit run app.py }

The build is idempotent. With --if-needed it reuses matching vector artifacts.

## 2. Local model files

GGUF files are intentionally not tracked.

Default paths:

    lexai-local/models/qwen2.5-3b-instruct-q4_k_m.gguf
    lexai-local/models/mistral-7b-instruct.Q4_K_M.gguf

Override them:

    $env:LEXAI_LLM1_MODEL="C:\path\to\qwen2.5-3b-instruct-q4_k_m.gguf"
    $env:LEXAI_LLM2_MODEL="C:\path\to\mistral-7b-instruct.Q4_K_M.gguf"

The UI can still start without GGUF files; deterministic/extractive fallbacks are used. To disable local LLM usage:

    $env:LEXAI_USE_LLM="0"
    $env:LEXAI_USE_LLM1="0"

## 3. Local architecture

    legal_embeddings_delta.parquet
              ↓
    vector_store/build_vector_db.py
              ↓
    FAISS + metadata + lexical artifacts
              ↓
    query analyzer
              ↓
    hybrid retrieval
              ↓
    optional cross-encoder reranking
              ↓
    optional local GGUF generation
              ↓
    Streamlit UI

Default local embedding model: BAAI/bge-base-en-v1.5.

The builder also supports parquet exports that already contain an embedding column.

## 4. Databricks pipeline

Run notebooks in this order:

    01_bronze_ingestion.ipynb
    02_silver_processing.ipynb
    03_gold_chunking.ipynb
    04_generate_embeddings.ipynb
    05_rag_answer_pipeline.ipynb
    06_High-precision_QA_Legal_Reasoning_Engine.ipynb

Notebook 07 is a convenience runner. Notebook 08 exports data for the local stack.

## 5. Databricks API + UI

The apps/ stack is not a pure-local replacement for notebook 06. The adapter executes selected notebook-06 cells and expects Spark/Databricks access.

Install:

    pip install -r apps/requirements.txt

Run the API from the repository root inside a Spark-enabled/Databricks runtime:

    uvicorn apps.fastapi_app:app --host 0.0.0.0 --port 8000

Health:

    curl http://127.0.0.1:8000/health

Run Streamlit:

    streamlit run apps/streamlit_app.py

Optional API target:

    export LEXAI_API_BASE_URL=http://127.0.0.1:8000

## Important notes

- Legal answers are retrieval-grounded assistance, not legal advice.
- Scanned PDFs still require OCR in the Databricks ingestion stage.
- Local vector artifacts are generated and ignored by Git.
- The tracked parquet file is the current local bootstrap dataset.
