# LexAI Local Setup (Enterprise CPU Mode)

This setup runs LexAI fully local on Windows CPU with:

- Embeddings: `BAAI/bge-base-en-v1.5`
- Retrieval: FAISS cosine + persisted lexical artifacts
- Optional reranker: cross-encoder profile-based
- Generation: `llama-cpp-python` with GGUF model profiles

## 1) System Requirements

- Windows 10/11
- Python 3.10 or 3.11
- RAM: 16 GB recommended
- Disk: 15+ GB free

## 2) Project Layout

```text
lexai-local/
  data/
    gold_chunks.parquet
  models/
    qwen2.5-3b-instruct-q4_k_m.gguf                 # balanced profile (recommended)
    mistral-7b-instruct.Q4_K_M.gguf                 # high_accuracy profile (optional)
  vector_store/
    build_vector_db.py
    faiss_index.bin                                 # generated
    metadata.pkl                                    # generated
    lexical_artifacts.pkl                           # generated
    build_manifest.json                             # generated
  rag/
    retriever.py
    generator.py
    rag_pipeline.py
  app.py
  requirements.txt
```

## 3) Create Environment

```powershell
cd lexai-local
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Build Artifacts

```powershell
python vector_store/build_vector_db.py --if-needed
```

Generated files:

- `vector_store/faiss_index.bin`
- `vector_store/metadata.pkl`
- `vector_store/lexical_artifacts.pkl`
- `vector_store/build_manifest.json`

## 5) Run App

```powershell
streamlit run app.py
```

## 6) One-Command Flow

```powershell
python vector_store/build_vector_db.py --if-needed; if ($LASTEXITCODE -eq 0) { streamlit run app.py }
```

## 7) Recommended Environment Variables

```powershell
$env:LEXAI_MODEL_PROFILE="balanced"
$env:LEXAI_USE_CROSS_ENCODER="1"
$env:LEXAI_USE_LLM="1"
$env:LEXAI_MAX_CONTEXT_CHARS="1500"
```

Optional model path overrides:

```powershell
$env:LEXAI_GGUF_BALANCED="C:\path\to\qwen2.5-3b-instruct-q4_k_m.gguf"
$env:LEXAI_GGUF_HIGH="C:\path\to\mistral-7b-instruct.Q4_K_M.gguf"
```

## 8) Performance Targets

- Retrieval p95: `< 500 ms`
- Generation p95: `< 45 s`
- Total p95: `< 60 s`

## 9) Troubleshooting

### Missing model file

Set `LEXAI_GGUF_BALANCED` (and optionally `LEXAI_GGUF_HIGH`) to exact GGUF path.

### Missing parquet

Place `gold_chunks.parquet` at `lexai-local/data/gold_chunks.parquet`.

### Build fails due to columns

Parquet must include:

- `chunk_id`
- `act_name`
- `section_number`
- `chunk_text`

### Slow responses

1. Use profile `balanced` in UI.
2. Keep `top_k` between 3 and 4.
3. Keep context cap at 1500 chars.
4. Keep reranker enabled only when needed.

