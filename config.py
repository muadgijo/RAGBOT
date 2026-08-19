from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm_backend() -> str:
    return os.getenv("LLM_BACKEND", "ollama").strip().lower()


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "phi3").strip()


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()



def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def get_embedding_backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()


def get_hf_token() -> str:
    return os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()