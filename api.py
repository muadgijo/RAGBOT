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
    <title>RAGBOT • AWS Documentation Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>

        :root {
            --bg: #071113;
            --panel: #0d1f23;
            --panel-2: #12282d;
            --card-border: rgba(71, 240, 255, 0.18);
            --card-hover: rgba(71, 240, 255, 0.35);
            --cyan: #47f0ff;
            --cyan-glow: rgba(71, 240, 255, 0.3);
            --text-main: #e2f7f8;
            --text-muted: #8ab6bb;
            --accent-gradient: linear-gradient(135deg, #47f0ff 0%, #00b4d8 100%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(71, 240, 255, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 20%, rgba(0, 180, 216, 0.07) 0%, transparent 40%),
                linear-gradient(180deg, #071113 0%, #09171a 100%);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 860px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        /* Header */
        .hero {
            background: linear-gradient(135deg, rgba(16, 38, 42, 0.95), rgba(9, 23, 26, 0.98));
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(71, 240, 255, 0.05);
            position: relative;
            overflow: hidden;
        }

        .hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-gradient);
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.35rem 0.8rem;
            background: rgba(71, 240, 255, 0.08);
            border: 1px solid rgba(71, 240, 255, 0.3);
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--cyan);
            margin-bottom: 1rem;
        }

        .badge-dot {
            width: 7px;
            height: 7px;
            background-color: #00ffaa;
            border-radius: 50%;
            box-shadow: 0 0 8px #00ffaa;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: clamp(2rem, 4vw, 2.75rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.15;
            background: linear-gradient(135deg, #ffffff 30%, #a5f3fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.6;
            margin-top: 0.75rem;
            max-width: 650px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin-top: 1.5rem;
        }

        .stat-card {
            background: rgba(13, 31, 35, 0.75);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            display: flex;
            flex-direction: column;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-val {
            font-family: 'Outfit', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--cyan);
            margin-top: 0.2rem;
        }

        /* Search Section */
        .search-panel {
            background: var(--panel);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .search-box {
            display: flex;
            gap: 0.75rem;
        }

        .search-input {
            flex: 1;
            background: rgba(7, 17, 19, 0.85);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.9rem 1.2rem;
            color: #ffffff;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            outline: none;
            transition: all 0.2s ease;
        }

        .search-input:focus {
            border-color: var(--cyan);
            box-shadow: 0 0 18px var(--cyan-glow);
        }

        .search-input::placeholder {
            color: #557b80;
        }

        .btn-ask {
            background: var(--accent-gradient);
            border: none;
            border-radius: 14px;
            padding: 0.9rem 1.8rem;
            color: #071113;
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }

        .btn-ask:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px var(--cyan-glow);
        }

        .btn-ask:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Suggestions */
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
            align-items: center;
        }

        .suggestions-title {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-right: 0.3rem;
        }

        .chip {
            background: rgba(71, 240, 255, 0.06);
            border: 1px solid rgba(71, 240, 255, 0.2);
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            color: #b2e6ea;
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .chip:hover {
            background: rgba(71, 240, 255, 0.15);
            border-color: var(--cyan);
            color: #ffffff;
            transform: translateY(-1px);
        }

        /* Results */
        .result-panel {
            display: none;
            background: var(--panel);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.8rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(71, 240, 255, 0.1);
        }

        .result-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--cyan);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-copy {
            background: transparent;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
            color: var(--text-muted);
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-copy:hover {
            border-color: var(--cyan);
            color: #ffffff;
        }

        .answer-text {
            color: var(--text-main);
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .answer-text p {
            margin-bottom: 0.85rem;
        }

        .answer-text ul, .answer-text ol {
            margin-left: 1.5rem;
            margin-bottom: 0.85rem;
        }

        .answer-text li {
            margin-bottom: 0.35rem;
        }

        .answer-text pre {
            background: #040c0e;
            border: 1px solid rgba(71, 240, 255, 0.22);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            overflow-x: auto;
            margin: 0.9rem 0;
        }

        .answer-text code {
            font-family: Consolas, 'Courier New', monospace;
            font-size: 0.88rem;
            color: #47f0ff;
            background: rgba(71, 240, 255, 0.08);
            padding: 0.15rem 0.4rem;
            border-radius: 6px;
        }

        .answer-text pre code {
            background: transparent;
            padding: 0;
            color: #d1f7fa;
        }

        .answer-text strong {
            color: #ffffff;
            font-weight: 600;
        }


        .sources-wrapper {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(71, 240, 255, 0.1);
        }

        .sources-title {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .sources-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .source-tag {
            background: rgba(7, 17, 19, 0.7);
            border: 1px solid rgba(71, 240, 255, 0.15);
            border-radius: 8px;
            padding: 0.35rem 0.7rem;
            font-size: 0.78rem;
            color: #9fe9ef;
            font-family: monospace;
        }

        /* Spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(7, 17, 19, 0.3);
            border-top: 2px solid #071113;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #52757a;
            font-size: 0.82rem;
            margin-top: 1rem;
        }

        .footer a {
            color: var(--cyan);
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 600px) {
            .hero { padding: 1.5rem; }
            .search-box { flex-direction: column; }
            .btn-ask { justify-content: center; width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Hero Header -->
        <header class="hero">
            <div class="badge">
                <span class="badge-dot"></span>
                Live API & Vector Search
            </div>
            <h1>RAGBOT</h1>
            <p>
                Search and explore AWS Lambda documentation with sub-second citations, verified retrieval, and Groq LLM inference.
            </p>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-label">Vector Store</span>
                    <span class="stat-val">ChromaDB</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Embeddings</span>
                    <span class="stat-val">all-MiniLM-L6-v2</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">LLM Engine</span>
                    <span class="stat-val">Groq LPU</span>
                </div>
            </div>
        </header>

        <!-- Search Form -->
        <section class="search-panel">
            <form id="query-form" class="search-box">
                <input 
                    type="text" 
                    id="question-input" 
                    class="search-input" 
                    placeholder="Ask anything (e.g., What is AWS Lambda?)..." 
                    autocomplete="off"
                    required
                >
                <button type="submit" id="btn-submit" class="btn-ask">
                    <span id="btn-text">Ask Bot</span>
                    <div id="btn-spinner" class="spinner"></div>
                </button>
            </form>

            <div class="suggestions">
                <span class="suggestions-title">Try:</span>
                <button type="button" class="chip" onclick="askSuggestion('What is AWS Lambda?')">What is AWS Lambda?</button>
                <button type="button" class="chip" onclick="askSuggestion('How does Lambda scale?')">How does Lambda scale?</button>
                <button type="button" class="chip" onclick="askSuggestion('What is a Lambda handler?')">What is a Lambda handler?</button>
            </div>
        </section>

        <!-- Response Result Panel -->
        <section id="result-panel" class="result-panel">
            <div class="result-header">
                <div class="result-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    Answer
                </div>
                <button id="copy-btn" class="btn-copy" onclick="copyAnswer()">Copy</button>
            </div>
            <div id="answer-container" class="answer-text"></div>

            <div id="sources-container" class="sources-wrapper">
                <div class="sources-title">Cited Documentation Sources</div>
                <div id="sources-list" class="sources-list"></div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="footer">
            FastAPI Backend • <a href="/docs" target="_blank">Interactive Swagger Docs</a> • <a href="/health" target="_blank">Health Check</a>
        </footer>
    </div>

    <script>
        const form = document.getElementById('query-form');
        const input = document.getElementById('question-input');
        const btn = document.getElementById('btn-submit');
        const btnText = document.getElementById('btn-text');
        const spinner = document.getElementById('btn-spinner');
        const resultPanel = document.getElementById('result-panel');
        const answerContainer = document.getElementById('answer-container');
        const sourcesList = document.getElementById('sources-list');

        function askSuggestion(text) {
            input.value = text;
            handleQuery(text);
        }

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const question = input.value.trim();
            if (question) handleQuery(question);
        });

        async function handleQuery(question) {
            btn.disabled = true;
            btnText.textContent = 'Thinking...';
            spinner.style.display = 'inline-block';
            resultPanel.style.display = 'block';
            answerContainer.textContent = 'Retrieving documentation and generating answer...';
            sourcesList.innerHTML = '';

            try {
                const res = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Failed to fetch response');
                }

                const data = await res.json();
                let rawAnswer = data.answer || 'No answer returned.';
                try {
                    answerContainer.innerHTML = (typeof marked !== 'undefined' && marked.parse) ? marked.parse(rawAnswer) : rawAnswer;
                } catch(e) {
                    answerContainer.textContent = rawAnswer;
                }

                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(src => {
                        const tag = document.createElement('span');
                        tag.className = 'source-tag';
                        tag.textContent = src;
                        sourcesList.appendChild(tag);
                    });
                } else {
                    sourcesList.innerHTML = '<span class="source-tag">No sources cited</span>';
                }
            } catch (err) {
                answerContainer.textContent = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Ask Bot';
                spinner.style.display = 'none';
            }
        }

        function copyAnswer() {
            const text = answerContainer.innerText || answerContainer.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const copyBtn = document.getElementById('copy-btn');
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = 'Copy', 2000);
            });
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