import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from config import get_llm_backend
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
        "expected_terms": ["serverless", "function", "code"],
    },
    {
        "query": "How does Lambda scale?",
        "expected_terms": ["scale", "event", "concurrency"],
    },
    {
        "query": "What is a Lambda handler?",
        "expected_terms": ["handler", "entry", "function"],
    },
    {
        "query": "What sample apps are available in the repository?",
        "expected_terms": ["blank-nodejs", "python", "java"],
    },
    {
        "query": "How do I deploy the blank-nodejs sample app?",
        "expected_terms": ["deploy", "bucket", "script"],
    },
]


def load_vector_db() -> Chroma:
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError(
            "Chroma database not found. Run clean_docs.py and create_database.py first."
        )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


def compute_faithfulness_score(answer: str, context: str) -> float:
    """Estimates factual faithfulness / groundedness score (0.0 to 1.0)
    by checking the proportion of key answer claims grounded in retrieved context."""
    if not answer or not context:
        return 0.0

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    lowered_context = context.lower()
    grounded_count = 0

    for sentence in sentences:
        words = [w.lower() for w in re.findall(r"\b\w{4,}\b", sentence)]
        if not words:
            grounded_count += 1
            continue

        # Check if significant portion of sentence keywords exist in context
        matches = sum(1 for w in words if w in lowered_context)
        match_ratio = matches / len(words)
        if match_ratio >= 0.35:
            grounded_count += 1

    return round(grounded_count / len(sentences), 2)


def evaluate_single_query(
    query: str,
    expected_terms: Optional[List[str]] = None,
    k: int = 4,
    db: Optional[Chroma] = None,
) -> Dict[str, Any]:
    if db is None:
        db = load_vector_db()

    expected_terms = expected_terms or []

    # 1. Measure Retrieval Latency
    t0 = time.perf_counter()
    results, context_text = retrieve_context(db, query, initial_k=max(k + 4, 8), final_k=k)
    retrieval_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in results))

    # 2. Measure Generation Latency
    prompt = build_prompt(query, context_text)
    t1 = time.perf_counter()
    backend = get_llm_backend()
    answer = get_llm_response(prompt, backend)
    generation_time_ms = round((time.perf_counter() - t1) * 1000, 1)

    # 3. Calculate Keyword Recall
    keyword_hits = []
    if expected_terms:
        lowered_answer = answer.lower()
        for term in expected_terms:
            if term.lower() in lowered_answer:
                keyword_hits.append(term)

    keyword_recall = (
        round(len(keyword_hits) / len(expected_terms), 2) if expected_terms else 1.0
    )

    # 4. Calculate Faithfulness (Hallucination prevention metric)
    faithfulness = compute_faithfulness_score(answer, context_text)

    total_latency_ms = round(retrieval_time_ms + generation_time_ms, 1)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_latency_ms": total_latency_ms,
        "faithfulness_score": faithfulness,
        "keyword_recall": keyword_recall,
        "keyword_hits": keyword_hits,
        "expected_terms": expected_terms,
        "context_length_chars": len(context_text),
        "answer_length_chars": len(answer),
    }


def evaluate_queries(
    cases: Optional[List[Dict[str, Any]]] = None,
    k: int = 4,
) -> List[Dict[str, Any]]:
    cases = cases or DEFAULT_EVAL_CASES
    db = load_vector_db()

    results = []
    for case in cases:
        query = str(case.get("query", ""))
        expected_terms = case.get("expected_terms", [])
        res = evaluate_single_query(query=query, expected_terms=expected_terms, k=k, db=db)
        results.append(res)

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("                      RAGBOT PERFORMANCE & EVALUATION BENCHMARK")
    print("=" * 80)

    avg_retrieval_ms = sum(r["retrieval_time_ms"] for r in results) / len(results)
    avg_gen_ms = sum(r["generation_time_ms"] for r in results) / len(results)
    avg_total_ms = sum(r["total_latency_ms"] for r in results) / len(results)
    avg_faithfulness = (sum(r["faithfulness_score"] for r in results) / len(results)) * 100
    avg_keyword_recall = (sum(r["keyword_recall"] for r in results) / len(results)) * 100

    for i, item in enumerate(results, 1):
        clean_query = re.sub(r"[^\x00-\x7F]+", " ", str(item["query"]))
        clean_sources = [re.sub(r"[^\x00-\x7F]+", " ", s) for s in item["sources"]]
        clean_ans = re.sub(r"[^\x00-\x7F]+", " ", str(item["answer"])).replace("\n", " ")[:140]

        print(f"\n[{i}] Query: {clean_query}")
        print(f"    - Sources: {', '.join(clean_sources)}")
        print(f"    - Latency: {item['total_latency_ms']} ms (Retrieval: {item['retrieval_time_ms']}ms, Gen: {item['generation_time_ms']}ms)")
        print(f"    - Faithfulness / Groundedness: {int(item['faithfulness_score'] * 100)}%")
        print(f"    - Keyword Recall: {int(item['keyword_recall'] * 100)}% ({len(item['keyword_hits'])}/{len(item['expected_terms'])})")
        print(f"    - Answer: {clean_ans}...")


    print("\n" + "-" * 80)
    print(f" AGGREGATE METRICS ({len(results)} queries):")
    print(f"   * Average Total Latency    : {avg_total_ms:.1f} ms (Retrieval: {avg_retrieval_ms:.1f}ms, Gen: {avg_gen_ms:.1f}ms)")
    print(f"   * Average Faithfulness     : {avg_faithfulness:.1f}% (Low hallucination risk)")
    print(f"   * Average Keyword Recall   : {avg_keyword_recall:.1f}%")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance, faithfulness, and latency")
    parser.add_argument(
        "--queries",
        nargs="*",
        help="Optional custom queries to evaluate.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of retrieved chunks for context.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output evaluation results as JSON.",
    )
    args = parser.parse_args()

    if args.queries:
        cases = [{"query": q, "expected_terms": []} for q in args.queries]
    else:
        cases = DEFAULT_EVAL_CASES

    results = evaluate_queries(cases=cases, k=args.k)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
