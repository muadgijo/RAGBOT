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
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_keywords(query: str) -> List[str]:
    tokens = normalize_text(query).split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def build_prompt(question: str, context_text: str) -> str:
    return f"""You are a helpful, technically accurate AWS documentation assistant specializing in AWS Lambda, serverless architectures, and sample applications.

Use the provided Documentation Context as your primary reference to answer the question. If the question asks for fundamental concepts (such as what a service is, how it works, or best practices), provide a clear, accurate, and concise explanation and reference the documentation context where relevant.

Documentation Context:
{context_text}

User Question:
{question}

Instructions:
- Answer clearly, concisely, and technically.
- Cite specific components, sample apps, commands, or configuration details from the Context when relevant.
- Do not repeat that information is missing if you can answer the question helpfully.

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
            num_predict=1024,
            num_gpu=0,
        )
        return str(llm.invoke(prompt)).strip()

    if backend_name == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_BACKEND=groq")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )


        content = response.choices[0].message.content if response.choices else ""
        return str(content or "").strip()

    raise ValueError(f"Unsupported LLM backend: {backend_name}")


class HuggingFaceAPIEmbeddings:
    """Direct, lightweight client for HuggingFace Inference API embeddings (0 MB local memory footprint)."""

    def __init__(self, api_key: str = "", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = api_key or os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
        self.model_name = model_name
        self.endpoint_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(api_key=self.api_key or None)
            res = client.feature_extraction(texts, model=self.model_name)
            if hasattr(res, "tolist"):
                return res.tolist()
            if isinstance(res, list):
                return [item.tolist() if hasattr(item, "tolist") else list(item) for item in res]
            return [list(res)]
        except Exception:
            # Direct HTTP fallback to current router endpoint
            import requests

            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Hugging Face API error ({response.status_code}): {response.text}"
                )
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"Hugging Face API error: {data['error']}")
            return data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        res = self._embed_batch([text])
        return res[0]



def get_embedding_function(backend: str = ""):
    """Factory function returning the best available embeddings provider."""
    backend_name = (backend or os.getenv("EMBEDDING_BACKEND", "auto")).strip().lower()
    hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()

    if backend_name == "hf_api":
        if not hf_token:
            raise ValueError(
                "HF_TOKEN is required when EMBEDDING_BACKEND=hf_api. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        return HuggingFaceAPIEmbeddings(api_key=hf_token)

    if backend_name == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if backend_name == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if backend_name == "auto":
        # 1. Prefer Hugging Face Serverless API if token is provided
        if hf_token:
            return HuggingFaceAPIEmbeddings(api_key=hf_token)

        # 2. Try FastEmbed (lightweight ONNX)
        try:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

            return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            pass

        # 3. Try local PyTorch embeddings if available
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            pass

        raise ValueError(
            "No embedding provider available. Please set HF_TOKEN in your .env / cloud settings, "
            "or install fastembed (`pip install fastembed`)."
        )

    raise ValueError(f"Unsupported embedding backend: {backend_name}")




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


def retrieve_context(db: Any, query: str, initial_k: int = 8, final_k: int = 4) -> Tuple[List[Any], str]:
    initial_docs = db.similarity_search(query, k=initial_k)
    reranked_docs = rerank_documents(query, initial_docs, top_k=final_k)
    context_text = "\n\n".join(doc.page_content for doc in reranked_docs)
    return reranked_docs, context_text

