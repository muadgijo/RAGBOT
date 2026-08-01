import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_utils import build_prompt, get_llm_response, retrieve_context

print("Loading embeddings...")

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Loading database...")

db = Chroma(
    persist_directory="chroma",
    embedding_function=embedding_function
)

print("Loading model...")

print("Everything loaded successfully!")

while True:
    question = input("\nAsk: ")

    if not question.strip():
        print("Please enter a question.")
        continue

    results, context_text = retrieve_context(db, question, initial_k=6, final_k=3)

    # Print source citations (filenames) from document metadata so users
    # can see where retrieved snippets came from.
    sources = [doc.metadata.get("source", "unknown") for doc in results]
    print("\nSources:")
    for s in sources:
        print(" -", s)

    # Context injection: supply only the retrieved context.
    # Make the fallback less brittle (only when context is empty).
    prompt = build_prompt(question, context_text)

    response = get_llm_response(prompt, os.getenv("LLM_BACKEND", "ollama"))

    print("\n=== RESPONSE ===\n")
    print(response)