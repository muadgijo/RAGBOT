from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm_backend() -> str:
    return os.getenv("LLM_BACKEND", "ollama").strip().lower()


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "phi3").strip()


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama3-8b-8192").strip()


def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()