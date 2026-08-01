import os
import re
from typing import List, Any, Tuple

from groq import Groq

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "for",
    "with",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "does",
    "do",
    "does",
    "can",
    "should",
    "about",
    "from",
    "into",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "be",
    "by",
    "as",
    "at",
    "use",
    "using",
    "aws",
    "lambda",
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_keywords(query: str) -> List[str]:
    tokens = normalize_text(query).split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def build_prompt(question: str, context_text: str) -> str:
    return f"""
You are a concise AWS documentation assistant. Use ONLY the Context section below to answer.

Context:
{context_text}

Question:
{question}

Instructions:
- Answer concisely and technically.
- Use only information present in Context. Do NOT hallucinate or invent details.
- If the answer cannot be found in Context, reply exactly:
  I could not find that in the documentation.
- Prefer short, direct answers with specific terms from the context.

Answer:
"""


def get_llm_response(prompt: str, backend: str) -> str:
    backend_name = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()

    if backend_name == "ollama":
        from langchain_ollama import OllamaLLM

        llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "phi3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
            stop=["\nAsk:", "\nQuestion:"],
            num_predict=256,
            num_gpu=0,
        )
        return str(llm.invoke(prompt)).strip()

    if backend_name == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_BACKEND=groq")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )

        content = response.choices[0].message.content if response.choices else ""
        return str(content or "").strip()

    raise ValueError(f"Unsupported LLM backend: {backend_name}")


def rerank_documents(query: str, docs: List[Any], top_k: int = 3) -> List[Any]:
    keywords = extract_keywords(query)
    if not keywords:
        return docs[:top_k]

    scored_docs = []
    for doc in docs:
        content = normalize_text(getattr(doc, "page_content", ""))
        score = 0

        for keyword in keywords:
            if keyword in content:
                score += 2

        if len(keywords) > 1:
            phrase = " ".join(keywords[:3])
            if phrase in content:
                score += 2

        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "")).lower()
        for keyword in keywords:
            if keyword in source:
                score += 1

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)

    if not scored_docs or all(score <= 0 for score, _ in scored_docs):
        return docs[:top_k]

    return [doc for _, doc in scored_docs[:top_k]]


def retrieve_context(db: Any, query: str, initial_k: int = 6, final_k: int = 3) -> Tuple[List[Any], str]:
    initial_docs = db.similarity_search(query, k=initial_k)
    reranked_docs = rerank_documents(query, initial_docs, top_k=final_k)
    context_text = "\n\n".join(doc.page_content for doc in reranked_docs)
    return reranked_docs, context_text
