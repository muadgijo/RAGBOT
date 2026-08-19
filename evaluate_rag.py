import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from langchain_chroma import Chroma

from rag_utils import (
    build_prompt,
    get_embedding_function,
    get_llm_response,
    retrieve_context,
)

CHROMA_PATH = "chroma"

DEFAULT_EVAL_CASES = [
    {
        "query": "What is AWS Lambda?",
        "expected_terms": ["serverless", "function"],
    },
    {
        "query": "How does Lambda scale?",
        "expected_terms": ["scale", "concurrency"],
    },
    {
        "query": "What is a Lambda handler?",
        "expected_terms": ["handler", "event"],
    },
]


def load_embedding_function():
    return get_embedding_function()



def load_vector_db() -> Chroma:
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError(
            "Chroma database not found. Run clean_docs.py and create_database.py first."
        )

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=load_embedding_function(),
    )


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

Answer:
"""


def evaluate_single_query(
    query: str,
    expected_terms: Optional[List[str]] = None,
    k: int = 2,
    db: Optional[Chroma] = None,
) -> Dict[str, object]:
    if db is None:
        db = load_vector_db()

    results, context_text = retrieve_context(db, query, initial_k=max(k + 4, 6), final_k=k)
    sources = [doc.metadata.get("source", "unknown") for doc in results]

    prompt = build_prompt(query, context_text)
    answer = get_llm_response(prompt, os.getenv("LLM_BACKEND", "ollama"))

    keyword_hits: List[str] = []
    if expected_terms:
        lowered_answer = answer.lower()
        for term in expected_terms:
            if term.lower() in lowered_answer:
                keyword_hits.append(term)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context_length": len(context_text),
        "answer_length": len(answer),
        "keyword_hits": keyword_hits,
        "expected_terms": expected_terms or [],
    }


def evaluate_queries(
    cases: Optional[List[Dict[str, object]]] = None,
    k: int = 2,
) -> List[Dict[str, object]]:
    cases = cases or DEFAULT_EVAL_CASES
    db = load_vector_db()

    results: List[Dict[str, object]] = []
    for case in cases:
        query = str(case.get("query", ""))
        expected_terms = case.get("expected_terms") or []
        results.append(
            evaluate_single_query(
                query=query,
                expected_terms=list(expected_terms),
                k=k,
                db=db,
            )
        )

    return results


def print_summary(results: List[Dict[str, object]]) -> None:
    print("\nRAG Evaluation Summary")
    print("=" * 28)

    for item in results:
        print(f"\nQuery: {item['query']}")
        print(f"- Sources: {', '.join(item['sources'])}")
        print(f"- Keyword hits: {', '.join(item['keyword_hits']) if item['keyword_hits'] else 'none'}")
        print(f"- Answer preview: {str(item['answer'])[:220]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight evaluation script for the RAG pipeline")
    parser.add_argument(
        "--queries",
        nargs="*",
        help="Optional custom queries to evaluate. If omitted, a small built-in set is used.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of retrieved chunks to include for each query.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the evaluation results as JSON instead of a text summary.",
    )
    args = parser.parse_args()

    if args.queries:
        cases = [{"query": query, "expected_terms": []} for query in args.queries]
    else:
        cases = DEFAULT_EVAL_CASES

    results = evaluate_queries(cases=cases, k=args.k)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
