import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_chroma import Chroma
from pydantic import BaseModel

from config import get_llm_backend
from rag_utils import (
    build_prompt,
    get_embedding_function,
    get_llm_response,
    retrieve_context,
)

load_dotenv()

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embeddings...")
    resources["embeddings"] = get_embedding_function()
    print("Loading Chroma DB...")
    resources["db"] = Chroma(
        persist_directory="chroma",
        embedding_function=resources["embeddings"],
    )
    print("Ready.")
    yield
    resources.clear()


app = FastAPI(title="RAGBOT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGBOT • AWS Lambda Documentation</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-main: #0B0F17;
            --bg-card: #111827;
            --bg-subtle: #1F2937;
            --border-color: #374151;
            --border-hover: #4B5563;
            --text-primary: #F9FAFB;
            --text-secondary: #9CA3AF;
            --text-muted: #6B7280;
            --accent-blue: #38BDF8;
            --accent-blue-hover: #0EA5E9;
            --code-bg: #030712;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2.5rem 1rem;
            line-height: 1.5;
        }

        .layout {
            width: 100%;
            max-width: 820px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        /* Top Navigation Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--border-color);
        }

        .brand-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 34px;
            height: 34px;
            background: #1E293B;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-blue);
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }

        .brand-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .nav-links {
            display: flex;
            gap: 1rem;
        }

        .nav-link {
            color: var(--text-secondary);
            font-size: 0.82rem;
            text-decoration: none;
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }

        .nav-link:hover {
            color: var(--text-primary);
            background: var(--bg-subtle);
            border-color: var(--border-color);
        }

        /* Search Section */
        .search-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        .search-form {
            display: flex;
            gap: 0.75rem;
        }

        .input-wrapper {
            position: relative;
            flex: 1;
            display: flex;
            align-items: center;
        }

        .search-icon {
            position: absolute;
            left: 1rem;
            color: var(--text-muted);
            pointer-events: none;
        }

        .search-input {
            width: 100%;
            background: #0B0F17;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.85rem 1rem 0.85rem 2.75rem;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .search-input:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 1px var(--accent-blue);
        }

        .search-input::placeholder {
            color: var(--text-muted);
        }

        .btn-submit {
            background: var(--accent-blue);
            color: #0B0F17;
            border: none;
            border-radius: 8px;
            padding: 0.85rem 1.4rem;
            font-size: 0.9rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: background-color 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
        }

        .btn-submit:hover:not(:disabled) {
            background: var(--accent-blue-hover);
        }

        .btn-submit:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Suggestions */
        .suggestions-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
            font-size: 0.8rem;
        }

        .suggestions-label {
            color: var(--text-muted);
            margin-right: 0.25rem;
        }

        .suggestion-btn {
            background: #1F2937;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.3rem 0.65rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .suggestion-btn:hover {
            color: var(--text-primary);
            border-color: var(--border-hover);
            background: #374151;
        }

        /* Result Panel */
        .result-card {
            display: none;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        .result-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .result-heading {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-actions {
            display: flex;
            gap: 0.5rem;
        }

        .btn-action {
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.3rem 0.65rem;
            color: var(--text-secondary);
            font-size: 0.78rem;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-action:hover {
            color: var(--text-primary);
            border-color: var(--border-hover);
            background: var(--bg-subtle);
        }

        /* Answer Markdown Styles */
        .answer-content {
            color: var(--text-primary);
            font-size: 0.95rem;
            line-height: 1.65;
        }

        .answer-content p {
            margin-bottom: 0.85rem;
        }

        .answer-content p:last-child {
            margin-bottom: 0;
        }

        .answer-content ul, .answer-content ol {
            margin-left: 1.4rem;
            margin-bottom: 0.85rem;
        }

        .answer-content li {
            margin-bottom: 0.3rem;
        }

        .answer-content pre {
            background: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            overflow-x: auto;
            margin: 0.85rem 0;
        }

        .answer-content code {
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 0.85rem;
            color: #E0F2FE;
            background: #1E293B;
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
        }

        .answer-content pre code {
            background: transparent;
            padding: 0;
            color: #F3F4F6;
        }

        .answer-content strong {
            font-weight: 600;
            color: #FFFFFF;
        }

        /* Sources Section */
        .sources-block {
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .sources-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .sources-flex {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        .source-pill {
            background: #0B0F17;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.25rem 0.6rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }

        /* Status & Latency Footer */
        .meta-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            padding: 0 0.5rem;
        }

        .meta-group {
            display: flex;
            gap: 1rem;
        }

        .spinner {
            display: none;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(11, 15, 23, 0.3);
            border-top: 2px solid #0B0F17;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 600px) {
            .search-form { flex-direction: column; }
            .btn-submit { width: 100%; justify-content: center; }
            .header { flex-direction: column; align-items: flex-start; gap: 0.75rem; }
        }
    </style>
</head>
<body>
    <div class="layout">
        <!-- Header -->
        <header class="header">
            <div class="brand-group">
                <div class="brand-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                        <polyline points="2 17 12 22 22 17"></polyline>
                        <polyline points="2 12 12 17 22 12"></polyline>
                    </svg>
                </div>
                <div>
                    <h1 class="brand-title">RAGBOT</h1>
                    <p class="brand-subtitle">AWS Lambda Documentation Assistant</p>
                </div>
            </div>
            <nav class="nav-links">
                <a href="/docs" class="nav-link" target="_blank">Swagger API</a>
                <a href="/health" class="nav-link" target="_blank">Health</a>
            </nav>
        </header>

        <!-- Search Input Form -->
        <main class="search-card">
            <form id="search-form" class="search-form">
                <div class="input-wrapper">
                    <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    <input 
                        type="text" 
                        id="query-input" 
                        class="search-input" 
                        placeholder="Ask a technical question about AWS Lambda..." 
                        autocomplete="off"
                        required
                    >
                </div>
                <button type="submit" id="submit-btn" class="btn-submit">
                    <span id="submit-text">Search</span>
                    <div id="submit-spinner" class="spinner"></div>
                </button>
            </form>

            <div class="suggestions-row">
                <span class="suggestions-label">Examples:</span>
                <button type="button" class="suggestion-btn" onclick="runQuery('What is AWS Lambda?')">What is AWS Lambda?</button>
                <button type="button" class="suggestion-btn" onclick="runQuery('What is a Lambda handler?')">What is a Lambda handler?</button>
                <button type="button" class="suggestion-btn" onclick="runQuery('How does Lambda scale?')">How does Lambda scale?</button>
                <button type="button" class="suggestion-btn" onclick="runQuery('How do I deploy blank-nodejs?')">Deploy blank-nodejs</button>
            </div>
        </main>

        <!-- Result View -->
        <section id="result-card" class="result-card">
            <div class="result-top">
                <span class="result-heading">Response</span>
                <div class="result-actions">
                    <button id="copy-btn" class="btn-action" onclick="copyResponse()">Copy</button>
                    <button class="btn-action" onclick="clearResponse()">Clear</button>
                </div>
            </div>
            <div id="answer-body" class="answer-content"></div>

            <div id="sources-section" class="sources-block">
                <div class="sources-title">Referenced Sources</div>
                <div id="sources-flex" class="sources-flex"></div>
            </div>
        </section>

        <!-- Meta Footer -->
        <footer class="meta-bar">
            <div class="meta-group">
                <span>Vector Index: <strong>ChromaDB</strong></span>
                <span>Inference: <strong>Groq LPU</strong></span>
            </div>
            <div id="latency-tag">Ready</div>
        </footer>
    </div>

    <script>
        const form = document.getElementById('search-form');
        const input = document.getElementById('query-input');
        const btn = document.getElementById('submit-btn');
        const btnText = document.getElementById('submit-text');
        const spinner = document.getElementById('submit-spinner');
        const resultCard = document.getElementById('result-card');
        const answerBody = document.getElementById('answer-body');
        const sourcesFlex = document.getElementById('sources-flex');
        const latencyTag = document.getElementById('latency-tag');

        function runQuery(text) {
            input.value = text;
            executeSearch(text);
        }

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const text = input.value.trim();
            if (text) executeSearch(text);
        });

        async function executeSearch(question) {
            const startTime = performance.now();
            btn.disabled = true;
            btnText.textContent = 'Searching';
            spinner.style.display = 'inline-block';
            resultCard.style.display = 'block';
            answerBody.innerHTML = '<span style="color: var(--text-muted)">Querying vector store and generating response...</span>';
            sourcesFlex.innerHTML = '';
            latencyTag.textContent = 'Processing...';

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to get response');
                }

                const data = await response.json();
                const elapsedMs = Math.round(performance.now() - startTime);

                const rawAnswer = data.answer || 'No answer generated.';
                try {
                    answerBody.innerHTML = (typeof marked !== 'undefined' && marked.parse) ? marked.parse(rawAnswer) : rawAnswer;
                } catch(e) {
                    answerBody.textContent = rawAnswer;
                }

                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(src => {
                        const pill = document.createElement('span');
                        pill.className = 'source-pill';
                        pill.innerHTML = `
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
                            ${src}
                        `;
                        sourcesFlex.appendChild(pill);
                    });
                } else {
                    sourcesFlex.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem">No specific source files cited</span>';
                }

                latencyTag.textContent = `Completed in ${elapsedMs}ms`;
            } catch (err) {
                answerBody.innerHTML = `<span style="color: #F87171">Error: ${err.message}</span>`;
                latencyTag.textContent = 'Error';
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Search';
                spinner.style.display = 'none';
            }
        }

        function copyResponse() {
            const text = answerBody.innerText || answerBody.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const copyBtn = document.getElementById('copy-btn');
                copyBtn.textContent = 'Copied';
                setTimeout(() => copyBtn.textContent = 'Copy', 2000);
            });
        }

        function clearResponse() {
            resultCard.style.display = 'none';
            answerBody.innerHTML = '';
            sourcesFlex.innerHTML = '';
            input.value = '';
            latencyTag.textContent = 'Ready';
        }
    </script>
</body>
</html>
"""



@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    db = resources["db"]
    docs, context_text = retrieve_context(db, request.question, initial_k=6, final_k=3)
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))
    prompt = build_prompt(request.question, context_text)
    answer = get_llm_response(prompt, get_llm_backend())

    return QueryResponse(question=request.question, answer=answer, sources=sources)