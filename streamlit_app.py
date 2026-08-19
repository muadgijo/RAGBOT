from pathlib import Path
import os

import streamlit as st
from langchain_chroma import Chroma

from rag_utils import (
    build_prompt,
    get_embedding_function,
    get_llm_response,
    retrieve_context,
)

CHROMA_PATH = "chroma"



def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #071113;
            --panel: #0c1a1d;
            --panel-2: #10262a;
            --text: #e2f7f8;
            --muted: #8ab6bb;
            --cyan: #47f0ff;
            --cyan-2: #16c7db;
            --line: rgba(71, 240, 255, 0.14);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(71, 240, 255, 0.14), transparent 32%),
                radial-gradient(circle at top right, rgba(22, 199, 219, 0.12), transparent 30%),
                linear-gradient(180deg, #071113 0%, #0a1719 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081316 0%, #0a1d20 100%);
            border-right: 1px solid var(--line);
        }

        h1, h2, h3, p, span, div, label, li {
            color: var(--text);
        }

        .hero {
            padding: 2.2rem 2rem;
            border: 1px solid var(--line);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(16, 38, 42, 0.92), rgba(8, 19, 22, 0.96));
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
            margin-bottom: 1rem;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.32rem 0.7rem;
            border: 1px solid rgba(71, 240, 255, 0.34);
            border-radius: 999px;
            color: var(--cyan);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
            background: rgba(71, 240, 255, 0.06);
        }

        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4.2rem);
            line-height: 1.02;
            margin: 0;
            max-width: 12ch;
        }

        .hero p {
            max-width: 62ch;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.7;
            margin-top: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.4rem;
        }

        .stat-card, .feature-card {
            border: 1px solid var(--line);
            background: rgba(10, 23, 25, 0.88);
            border-radius: 22px;
            padding: 1rem 1.1rem;
        }

        .stat-card .value {
            display: block;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--cyan);
            margin-bottom: 0.2rem;
        }

        .stat-card .label {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.6rem 0 0.8rem;
        }

        .section-title h2 {
            margin: 0;
            font-size: 1.15rem;
        }

        .section-title .caption {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 1rem;
        }

        .feature-card h3 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
            color: var(--cyan);
        }

        .feature-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
        }

        .demo-panel {
            border: 1px solid var(--line);
            border-radius: 26px;
            background: linear-gradient(180deg, rgba(9, 20, 23, 0.98), rgba(7, 16, 18, 0.98));
            padding: 1.1rem 1rem 0.8rem;
            margin-top: 1rem;
        }

        .demo-panel .hint {
            color: var(--muted);
            margin-top: 0.35rem;
            margin-bottom: 0.9rem;
        }

        .stChatMessage {
            background: rgba(10, 23, 25, 0.95) !important;
            border: 1px solid rgba(71, 240, 255, 0.10) !important;
            border-radius: 18px !important;
        }

        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg, var(--cyan), var(--cyan-2));
            color: #051014 !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
        }

        .stTextInput input, .stTextArea textarea {
            background: rgba(9, 20, 23, 0.95) !important;
            color: var(--text) !important;
            border: 1px solid rgba(71, 240, 255, 0.18) !important;
            border-radius: 14px !important;
        }

        hr {
            border-color: rgba(71, 240, 255, 0.12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_embedding_function():
    return get_embedding_function()



@st.cache_resource
def load_vector_db() -> Chroma:
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError(
            "Chroma database not found. Run clean_docs.py and create_database.py first."
        )

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=load_embedding_function(),
    )


def ask_rag(question: str, k: int = 2):
    db = load_vector_db()

    results, context_text = retrieve_context(db, question, initial_k=max(k + 4, 6), final_k=k)
    sources = [doc.metadata.get("source", "unknown") for doc in results]

    prompt = build_prompt(question, context_text)
    answer = get_llm_response(prompt, os.getenv("LLM_BACKEND", "ollama"))

    return answer, sources


def render_sidebar() -> None:
    st.sidebar.header("Project Status")
    st.sidebar.write(f"Backend: {os.getenv('LLM_BACKEND', 'ollama')}")
    st.sidebar.write("Embeddings: all-MiniLM-L6-v2")
    st.sidebar.write("Vector DB: local Chroma")

    st.sidebar.markdown("### Pre-run checklist")
    st.sidebar.markdown("1. Run `python clean_docs.py`")
    st.sidebar.markdown("2. Run `python create_database.py`")
    if os.getenv("LLM_BACKEND", "ollama") == "groq":
        st.sidebar.markdown("3. Set `GROQ_API_KEY` and optionally `GROQ_MODEL`")
    else:
        st.sidebar.markdown("3. Start Ollama and pull `phi3`")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Local RAG portfolio demo</div>
            <h1>Search docs, cite sources, and answer locally.</h1>
            <p>
                RAGBOT is a compact showcase built for AWS Lambda documentation:
                clean the text, build a Chroma index, and answer questions through
                a local Ollama model. No API bill, no cloud lock-in, and a clean
                path to learning Docker.
            </p>
            <div class="hero-grid">
                <div class="stat-card">
                    <span class="value">Local</span>
                    <span class="label">Runs on your laptop with Ollama + Chroma</span>
                </div>
                <div class="stat-card">
                    <span class="value">Cited</span>
                    <span class="label">Shows sources from the retrieved documents</span>
                </div>
                <div class="stat-card">
                    <span class="value">Docker-ready</span>
                    <span class="label">Easy to package as a demo for your portfolio</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">
            <h2>What this project shows</h2>
            <div class="caption">A small demo with a real retrieval pipeline behind it</div>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <h3>Cleaned input</h3>
                <p>Markdown docs are stripped down before indexing so retrieval stays focused on useful text.</p>
            </div>
            <div class="feature-card">
                <h3>Vector search</h3>
                <p>Chroma finds the most relevant chunks and prints the source files for transparency.</p>
            </div>
            <div class="feature-card">
                <h3>Local generation</h3>
                <p>Ollama generates answers on your machine, so the showcase stays private and cheap.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">
            <h2>Try the demo</h2>
            <div class="caption">Ask a question and get an answer with sources</div>
        </div>
        <div class="demo-panel">
            <div class="hint">Start with questions like: What is AWS Lambda? How does Lambda scale? Why use Lambda?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="RAGBOT", page_icon="📚", layout="wide")

    apply_theme()

    render_hero()

    render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))

    question = st.chat_input("Ask a question about AWS Lambda docs")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    answer, sources = ask_rag(question)
                except Exception as exc:
                    st.error(f"Failed to answer: {exc}")
                    return

            st.markdown(answer)
            st.caption("Sources: " + ", ".join(sources))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
