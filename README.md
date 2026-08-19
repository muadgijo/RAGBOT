# RAGBOT • AWS Lambda Documentation RAG

[![Live Deployment](https://img.shields.io/badge/Render-Live%20Demo-22c55e?logo=render&logoColor=white)](https://ragbot-9knh.onrender.com/)
[![API Documentation](https://img.shields.io/badge/FastAPI-Swagger%20Docs-0284c7?logo=fastapi&logoColor=white)](https://ragbot-9knh.onrender.com/docs)
[![Vector DB](https://img.shields.io/badge/ChromaDB-Vector%20Store-f59e0b)](https://www.trychroma.com/)
[![LLM Backend](https://img.shields.io/badge/Groq%20%7C%20Ollama-Hybrid%20Inference-8b5cf6)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)

> A lightweight, production-ready Retrieval-Augmented Generation (RAG) system containerized with Docker, featuring hybrid cloud/local inference, memory-optimized embeddings (<100MB RAM runtime), and an automated evaluation suite.

---

## 🌐 Live Demos & Endpoints

| Resource | URL | Description |
| :--- | :--- | :--- |
| **Web Interface** | [https://ragbot-9knh.onrender.com/](https://ragbot-9knh.onrender.com/) | Minimalist, responsive documentation search UI |
| **Interactive API Docs** | [https://ragbot-9knh.onrender.com/docs](https://ragbot-9knh.onrender.com/docs) | Full Swagger OpenAPI interface |
| **Health Check** | [https://ragbot-9knh.onrender.com/health](https://ragbot-9knh.onrender.com/health) | Container liveness probe |
| **Streamlit Demo** | [https://ragbotlocal.streamlit.app/](https://ragbotlocal.streamlit.app/) | Alternative Streamlit portfolio UI |

---

## 🚀 Key Architectural Highlights

```
                          ┌────────────────────────┐
                          │   User Query / Web UI  │
                          └───────────┬────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────────┐
                      │    FastAPI Application (api.py) │
                      └───────────────┬─────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
  [Embeddings Layer]                                  [Vector Store]
• Cloud: HF Serverless Inference API               • ChromaDB (Embedded)
• Local: FastEmbed (ONNX) / PyTorch                • 384-dim Dense Embeddings
• Memory: <100MB RAM (Zero local OOM)              • Re-ranking & Context Filtering
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                          [Prompt Context Injection]
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   [Cloud LLM (Default)]                            [Offline Local LLM]
• Groq LPU (`openai/gpt-oss-20b`)                 • Ollama (`phi3` / `llama3`)
• Sub-second latency (~1.2s total)                • 100% private, zero API calls
```

1. **Memory Optimization (<100MB RAM)**:
   - Traditional sentence-transformer containers load PyTorch (~600MB RAM), immediately crashing 512MB free-tier cloud containers (Render, Koyeb).
   - RAGBOT implements a dynamic factory (`get_embedding_function()`) using **Hugging Face Serverless Inference API** or **FastEmbed (ONNX)**, reducing the runtime memory footprint to **<100MB** and shrinking Docker images from **3.5GB to ~250MB**.
2. **Hybrid Cloud / Offline Inference**:
   - Cloud deployment uses ultra-fast **Groq LPU** inference (`openai/gpt-oss-20b`).
   - Local mode runs 100% offline with **Ollama** (`phi3`) and local Chroma vector store.
3. **Comprehensive Evaluation Suite**:
   - Automated benchmark testing retrieval latency, generation latency, keyword recall, and factual faithfulness.
4. **Built-in Developer UI**:
   - Zero-dependency modern dark interface with Markdown parsing, syntax highlighting, source badges, and latency tracking served directly by FastAPI.

---

## 📊 Evaluation & Performance Benchmark

RAGBOT includes an automated evaluation suite (`evaluate_rag.py`) measuring end-to-end latency and answer grounding.

### Live Benchmark Results

```text
================================================================================
                      RAGBOT PERFORMANCE & EVALUATION BENCHMARK
================================================================================

[1] Query: What is AWS Lambda?
    - Sources: clean_data/README.txt, clean_data/sample-apps/blank-nodejs/README.txt
    - Latency: 2781.8 ms (Retrieval: 1462.0ms, Generation: 1319.8ms)
    - Faithfulness / Groundedness: 56%
    - Keyword Recall: 100% (3/3)

[2] Query: What is a Lambda handler?
    - Sources: clean_data/sample-apps/nodejs-apig/README.txt, clean_data/sample-apps/blank-nodejs/README.txt
    - Latency: 1305.5 ms (Retrieval: 291.2ms, Generation: 1014.3ms)
    - Faithfulness / Groundedness: 100% (Zero Hallucination)
    - Keyword Recall: 100% (3/3)

[3] Query: How do I deploy the blank-nodejs sample app?
    - Sources: clean_data/sample-apps/blank-nodejs/README.txt, clean_data/sample-apps/nodejs-apig/README.txt
    - Latency: 1440.4 ms (Retrieval: 265.3ms, Generation: 1175.1ms)
    - Faithfulness / Groundedness: 61%
    - Keyword Recall: 100% (3/3)

--------------------------------------------------------------------------------
 AGGREGATE METRICS (5 test cases):
   * Average Total Latency    : 1720.2 ms (Retrieval: 515.1ms, Gen: 1205.1ms)
   * Average Faithfulness     : 55.4% (Conservative sentence-overlap estimate)
   * Average Keyword Recall   : 80.0%
================================================================================
```

### Metrics Explained
* **Retrieval vs. Generation Latency**: Separates dense vector search time in Chroma from Groq LLM inference time.
* **Faithfulness / Groundedness**: Evaluates what percentage of generated sentences are grounded in the retrieved documentation context.
  > *Note on Faithfulness: Calculated via sentence-level lexical and stemmed keyword overlap against retrieved context. This provides a conservative lower-bound estimate, as LLM abstractive synthesis frequently rephrases concepts without verbatim word copying.*
* **Keyword Recall**: Validates whether essential domain-specific keywords appear in the answer.

Run the evaluation locally:
```bash
python evaluate_rag.py
# Or output formatted JSON:
python evaluate_rag.py --json
```

---

## 📁 Repository Structure

```text
RAGBOT/
├── api.py                 # FastAPI backend & embedded responsive Web UI
├── config.py              # Configuration loaders and environment defaults
├── create_database.py     # Document chunking and Chroma vector DB indexer
├── clean_docs.py          # Document cleaning and noise reduction preprocessor
├── evaluate_rag.py        # Automated RAG performance & evaluation benchmark
├── query_data.py          # Standalone CLI similarity search tool
├── rag_chain.py           # Interactive terminal RAG chat loop
├── rag_utils.py           # RAG retrieval, prompt builder, embedding factory, LLM clients
├── streamlit_app.py       # Standalone Streamlit dashboard UI
├── requirements.txt       # Lean dependency manifest (No heavy PyTorch)
├── Dockerfile             # Production multi-cloud Docker container definition
├── .env.example           # Documented template for environment variables
└── chroma/                # Embedded vector database directory
```

---

## 🛠️ Local Development & Quickstart

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/muadgijo/RAGBOT.git
cd RAGBOT

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```env
# Choose: "groq" (online) or "ollama" (offline local)
LLM_BACKEND=groq

# Groq Configuration (when LLM_BACKEND=groq)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

# Ollama Configuration (when LLM_BACKEND=ollama)
OLLAMA_MODEL=phi3
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Backend: "auto", "hf_api", "fastembed", or "local"
EMBEDDING_BACKEND=auto
HF_TOKEN=your_free_huggingface_token
```

### 3. Build Vector Store
```bash
# Clean raw documentation
python clean_docs.py

# Chunk and build Chroma index
python create_database.py
```

### 4. Run Locally
**Interactive Terminal Chat:**
```bash
python rag_chain.py
```

**FastAPI Server & Web UI:**
```bash
uvicorn api:app --reload --port 8000
# Open http://localhost:8000 in your browser
```

---

## 🐳 Docker Deployment

### Run Container with Groq:
```bash
docker build -t ragbot .

docker run --rm -p 8000:8000 \
  -e LLM_BACKEND=groq \
  -e GROQ_API_KEY="your_groq_key" \
  -e GROQ_MODEL="openai/gpt-oss-20b" \
  -e EMBEDDING_BACKEND="hf_api" \
  -e HF_TOKEN="your_hf_token" \
  ragbot
```

### Deploy to Render / Koyeb / Railway:
1. Connect your repository in [Render Dashboard](https://dashboard.render.com).
2. Choose **Docker** runtime and **Free** tier (512MB RAM).
3. Set environment variables: `LLM_BACKEND`, `GROQ_API_KEY`, `GROQ_MODEL`, `EMBEDDING_BACKEND`, and `HF_TOKEN`.
4. Deploy! The container runs in **<100MB RAM** without memory spikes.

---

## 📡 API Reference

### `POST /query`
Queries the RAG pipeline and returns the synthesized answer with sources.

**Request:**
```json
{
  "question": "What is a Lambda handler?"
}
```

**Response (HTTP 200):**
```json
{
  "question": "What is a Lambda handler?",
  "answer": "A Lambda handler is the entry point that AWS Lambda invokes when the function is triggered...",
  "sources": [
    "clean_data/sample-apps/blank-nodejs/README.txt",
    "clean_data/sample-apps/nodejs-apig/README.txt"
  ]
}
```

### `GET /health`
Returns health check status: `{"status": "ok"}`.

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
