# Local RAG Project

https://ragbotlocal.streamlit.app/

This project is a beginner-friendly local RAG system built in Python. It uses:

- LangChain
- ChromaDB
- HuggingFace embeddings
- Ollama
- Phi-3 running locally

The goal is simple:

1. Clean the AWS Lambda documentation.
2. Build a vector database from the cleaned text.
3. Ask questions against the database.
4. Generate answers locally with Ollama.

## What the Project Does

The original AWS Lambda docs contain a lot of noise for RAG:

- code blocks
- shell commands
- JSON and config dumps
- images
- raw markdown links
- setup instructions

Those details are useful for humans, but they can hurt retrieval quality. This project cleans the docs first, then indexes the cleaned text.

## Main Files

### `clean_docs.py`

Reads all markdown files under `data/aws-lambda-developer-guide-main/` and writes cleaned `.txt` files into `clean_data/`.

What it removes or reduces:

- fenced code blocks
- inline code markers
- markdown images
- markdown links, while keeping readable link text
- shell commands
- JSON-like or config-like lines
- repeated blank lines
- version-heavy boilerplate

This step improves retrieval quality by reducing chunk noise.

### `create_database.py`

Reads the cleaned `.txt` files from `clean_data/`, splits them into chunks, and stores the embeddings in ChromaDB.

It also deletes the old `chroma/` folder before rebuilding the database so the index always matches the latest cleaned data.

### `rag_chain.py`

This is the main interactive RAG script.

It:

- loads embeddings
- loads the Chroma database
- loads the local Ollama Phi-3 model
- asks for a question in a loop
- retrieves the most relevant chunks
- prints the source files
- sends the retrieved context to the model
- prints the final answer

Important settings in this file:

- `temperature=0.1` for more factual answers
- `stop_tokens=["\nAsk:", "\nQuestion:"]` to reduce prompt echoing
- `num_gpu=0` to force CPU-only Ollama mode
- `num_predict=256` to limit answer length

### `query_data.py`

A smaller helper script that searches the Chroma database and prints the top matches.

This is useful when you want to inspect retrieval results without running the full generation loop.

### `data/`

Contains the raw AWS Lambda documentation source files.

### `clean_data/`

Contains the cleaned `.txt` files created by `clean_docs.py`.

### `chroma/`

Contains the local vector database built by `create_database.py`.

## Pipeline Overview

The flow is:

1. Read raw markdown documents.
2. Clean the documents.
3. Split cleaned text into chunks.
4. Convert chunks into embeddings.
5. Store embeddings in ChromaDB.
6. Ask a question.
7. Retrieve the most relevant chunks.
8. Inject retrieved context into the prompt.
9. Generate a local answer with Ollama and Phi-3.

## Setup

Activate the virtual environment first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .venv\Scripts\Activate.ps1
```

If you want CPU-only Ollama mode, set:

```powershell
$env:OLLAMA_NO_GPU="1"
```

## Typical Run Order

### 1. Clean the docs

```powershell
python clean_docs.py
```

### 2. Build the Chroma database

```powershell
python create_database.py
```

### 3. Ask questions with the RAG chain

```powershell
python rag_chain.py
```

### 4. Inspect retrieval matches directly

```powershell
python query_data.py "What is AWS Lambda?"
```

## Example Usage

When you run `rag_chain.py`, you can type questions like:

- What is AWS Lambda?
- Why do we use AWS Lambda?
- How does Lambda scale?

The script prints the source file names before the answer, which helps you see where the retrieved context came from.

## Notes

- The project is designed to stay local.
- The model runs through Ollama, not a cloud API.
- The embedding model is downloaded from HuggingFace the first time you run it.
- The README and scripts are intentionally simple so they are easy to learn from.

## Configuration Files

- `.env.example` is the tracked template for local settings.
- `.env` is the private local override file and is ignored by git.
- `config.py` loads `.env` and exposes the shared backend and model defaults.

To switch to Groq, set `LLM_BACKEND=groq` in `.env` and provide `GROQ_API_KEY`.

## FastAPI And Docker

The repo now includes a small FastAPI app in `api.py`.

What it does:

- loads embeddings and the Chroma database once at startup
- exposes `GET /health` for a quick check
- exposes `POST /query` for RAG answers over HTTP
- reuses the same retrieval and prompt logic as the terminal script

Run it locally:

```powershell
uvicorn api:app --reload
```

Open the Swagger UI at:

```text
http://127.0.0.1:8000/docs
```

Build the Docker image:

```powershell
docker build -t ragbot .
```

Run the container with Groq:

```powershell
docker run --rm -p 8000:8000 `
	-e LLM_BACKEND=groq `
	-e GROQ_API_KEY=YOUR_NEW_KEY `
	-e GROQ_MODEL=llama-3.1-8b-instant `
	ragbot
```

Run the container with host Ollama:

```powershell
docker run --rm -p 8000:8000 `
	-e LLM_BACKEND=ollama `
	-e OLLAMA_MODEL=phi3 `
	-e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
	ragbot
```

Test the API:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body '{"question":"What is a Lambda handler?"}'
```

If retrieval looks empty in Docker, make sure `chroma/` is not excluded from the build context and that the container can see the database files.

## Troubleshooting

### Ollama tries to use GPU

If you see a CUDA-related error, run Ollama in CPU-only mode:

```powershell
$env:OLLAMA_NO_GPU="1"
python rag_chain.py
```

### HuggingFace warning about unauthenticated requests

You may see a warning about the HuggingFace Hub. That usually means no `HF_TOKEN` is set in your environment. The project can still run, but authentication can reduce rate limits.

### Old results look noisy

If the answers seem stale or messy, rebuild everything in this order:

```powershell
python clean_docs.py
python create_database.py
python rag_chain.py
```

## Suggested Git Practice

Do not commit these generated or local files unless you specifically want them in the repo:

- `.venv/`
- `chroma/`
- `clean_data/`
- `.env`

The raw docs under `data/` are the source material, but you may choose to keep them out of the repo if that matches your workflow.

## Git Push Troubleshooting

`git bash` is fine to use. The shell was not the issue.

If you see this:

```text
fatal: Could not read from remote repository.
```

check your remote name first. A typo like `orgin` instead of `origin` causes that exact error.

Quick checks:

```bash
git remote -v
git branch --show-current
```

If local and remote both have new commits, use rebase before pushing:

```bash
git pull --rebase origin main
git push origin main
```

Why rebase: it replays your local commits on top of the latest remote commits so history stays linear and avoids unnecessary merge commits.

## What To Learn From This Project

If you are learning RAG, this repo shows a simple and practical pattern:

- clean the source text before embedding it
- keep the database rebuild step separate from the query step
- use retrieval context directly in the prompt
- keep generation settings conservative for factual answers
- print sources so answers are easier to trust

## Short Version

If you only remember three commands, use these:

```powershell
python clean_docs.py
python create_database.py
python rag_chain.py
```

