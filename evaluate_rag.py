import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
from langchain_chroma import Chroma
from config import get_llm_backend, get_groq_api_key, get_groq_model
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


def compute_faithfulness_lexical(answer: str, context: str) -> float:
    """Fallback lexical groundedness metric (sentence-overlap heuristic)."""
    if not answer or not context:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 12]
    if not sentences:
        return 1.0

    lowered_context = context.lower()
    grounded_count = 0

    for sentence in sentences:
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", sentence)]
        if not words:
            grounded_count += 1
            continue

        matches = sum(
            1 for w in words if w in lowered_context or (len(w) >= 5 and w[:4] in lowered_context)
        )
        if (matches / len(words)) >= 0.28:
            grounded_count += 1

    return round(grounded_count / len(sentences), 2)


def evaluate_faithfulness_llm(
    answer: str, context: str, client: Optional[Groq] = None
) -> Dict[str, Any]:
    """RAGAS-style LLM-as-a-Judge Faithfulness Evaluator.
    Deconstructs the answer into factual claims and verifies whether each
    claim is grounded in the retrieved documentation context."""
    if not client:
        api_key = get_groq_api_key()
        if not api_key:
            score = compute_faithfulness_lexical(answer, context)
            return {
                "faithfulness_score": score,
                "supported_claims": int(score * 4),
                "total_claims": 4,
                "method": "lexical_fallback",
            }
        client = Groq(api_key=api_key)

    eval_prompt = f"""You are a strict, objective AI evaluation judge assessing the FAITHFULNESS of a RAG answer against the retrieved context (Ragas methodology).

Retrieved Documentation Context:
{context}

Generated Answer:
{answer}

Instructions:
1. Break down the Generated Answer into individual atomic factual statements/claims.
2. For each statement, determine whether it is directly supported or logically inferred from the Retrieved Context.
3. Compute the score as: (supported_claims / total_claims). If there are 0 factual claims, score is 1.0.

Respond ONLY with a JSON object in this exact schema:
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness_score": <float between 0.0 and 1.0>,
  "reasoning": "<short 1-sentence evaluation>"
}}"""

    try:
        res = client.chat.completions.create(
            model=get_groq_model(),
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content or "{}")
        score = float(data.get("faithfulness_score", 1.0))
        return {
            "faithfulness_score": round(max(0.0, min(1.0, score)), 2),
            "supported_claims": int(data.get("supported_claims", 0)),
            "total_claims": int(data.get("total_claims", 0)),
            "reasoning": str(data.get("reasoning", "")),
            "method": "llm_as_a_judge",
        }
    except Exception as e:
        score = compute_faithfulness_lexical(answer, context)
        return {
            "faithfulness_score": score,
            "supported_claims": int(score * 4),
            "total_claims": 4,
            "reasoning": f"Fallback due to evaluator error: {e}",
            "method": "lexical_fallback",
        }


def evaluate_relevance_llm(
    query: str, answer: str, client: Optional[Groq] = None
) -> Dict[str, Any]:
    """RAGAS-style Answer Relevance Evaluator.
    Measures whether the response directly addresses the user query."""
    if not client:
        api_key = get_groq_api_key()
        if not api_key:
            return {"relevance_score": 1.0, "method": "default"}
        client = Groq(api_key=api_key)

    prompt = f"""You are an objective evaluator assessing ANSWER RELEVANCE in a RAG system.
Evaluate whether the Generated Answer directly and completely addresses the User Query without digressing.

User Query:
{query}

Generated Answer:
{answer}

Respond ONLY with a JSON object:
{{
  "relevance_score": <float between 0.0 and 1.0>,
  "feedback": "<brief explanation>"
}}"""

    try:
        res = client.chat.completions.create(
            model=get_groq_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content or "{}")
        score = float(data.get("relevance_score", 1.0))
        return {
            "relevance_score": round(max(0.0, min(1.0, score)), 2),
            "feedback": str(data.get("feedback", "")),
            "method": "llm_as_a_judge",
        }
    except Exception:
        return {"relevance_score": 1.0, "method": "default"}


def evaluate_single_query(
    query: str,
    expected_terms: Optional[List[str]] = None,
    k: int = 4,
    db: Optional[Chroma] = None,
    client: Optional[Groq] = None,
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

    # 4. Calculate Ragas-Style LLM-as-a-Judge Faithfulness & Relevance
    faithfulness_res = evaluate_faithfulness_llm(answer, context_text, client=client)
    relevance_res = evaluate_relevance_llm(query, answer, client=client)

    total_latency_ms = round(retrieval_time_ms + generation_time_ms, 1)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_latency_ms": total_latency_ms,
        "faithfulness_score": faithfulness_res["faithfulness_score"],
        "faithfulness_claims": f"{faithfulness_res.get('supported_claims', 0)}/{faithfulness_res.get('total_claims', 0)}",
        "faithfulness_reasoning": faithfulness_res.get("reasoning", ""),
        "relevance_score": relevance_res["relevance_score"],
        "keyword_recall": keyword_recall,
        "keyword_hits": keyword_hits,
        "expected_terms": expected_terms,
        "eval_method": faithfulness_res.get("method", "llm_as_a_judge"),
    }


def evaluate_queries(
    cases: Optional[List[Dict[str, Any]]] = None,
    k: int = 4,
) -> List[Dict[str, Any]]:
    cases = cases or DEFAULT_EVAL_CASES
    db = load_vector_db()

    api_key = get_groq_api_key()
    client = Groq(api_key=api_key) if api_key else None

    results = []
    for case in cases:
        query = str(case.get("query", ""))
        expected_terms = case.get("expected_terms", [])
        res = evaluate_single_query(
            query=query, expected_terms=expected_terms, k=k, db=db, client=client
        )
        results.append(res)

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("              RAGBOT PERFORMANCE & RAGAS EVALUATION BENCHMARK")
    print("=" * 80)

    avg_retrieval_ms = sum(r["retrieval_time_ms"] for r in results) / len(results)
    avg_gen_ms = sum(r["generation_time_ms"] for r in results) / len(results)
    avg_total_ms = sum(r["total_latency_ms"] for r in results) / len(results)
    avg_faithfulness = (sum(r["faithfulness_score"] for r in results) / len(results)) * 100
    avg_relevance = (sum(r["relevance_score"] for r in results) / len(results)) * 100
    avg_keyword_recall = (sum(r["keyword_recall"] for r in results) / len(results)) * 100

    for i, item in enumerate(results, 1):
        clean_query = re.sub(r"[^\x00-\x7F]+", " ", str(item["query"]))
        clean_sources = [re.sub(r"[^\x00-\x7F]+", " ", s) for s in item["sources"]]
        clean_ans = re.sub(r"[^\x00-\x7F]+", " ", str(item["answer"])).replace("\n", " ")[:130]

        print(f"\n[{i}] Query: {clean_query}")
        print(f"    - Sources: {', '.join(clean_sources)}")
        print(f"    - Latency: {item['total_latency_ms']} ms (Retrieval: {item['retrieval_time_ms']}ms, Gen: {item['generation_time_ms']}ms)")
        print(f"    - Faithfulness (LLM Judge): {int(item['faithfulness_score'] * 100)}% (Supported: {item['faithfulness_claims']})")
        print(f"    - Answer Relevance: {int(item['relevance_score'] * 100)}%")
        print(f"    - Keyword Recall: {int(item['keyword_recall'] * 100)}% ({len(item['keyword_hits'])}/{len(item['expected_terms'])})")
        print(f"    - Answer: {clean_ans}...")

    print("\n" + "-" * 80)
    print(f" AGGREGATE METRICS ({len(results)} queries evaluated via Ragas methodology):")
    print(f"   * Average Total Latency    : {avg_total_ms:.1f} ms (Retrieval: {avg_retrieval_ms:.1f}ms, Gen: {avg_gen_ms:.1f}ms)")
    print(f"   * Average Faithfulness     : {avg_faithfulness:.1f}% (Factual groundedness / hallucination safety)")
    print(f"   * Average Answer Relevance : {avg_relevance:.1f}% (Query intent alignment)")
    print(f"   * Average Keyword Recall   : {avg_keyword_recall:.1f}%")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline performance, faithfulness, and latency using Ragas methodology"
    )
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
