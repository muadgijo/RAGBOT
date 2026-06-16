from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

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

# Generation settings: lower temperature improves factual consistency
# and stop tokens prevent the model from echoing the interactive prompt.
temperature = 0.1
stop_tokens = ["\nAsk:", "\nQuestion:"]

# Set generation params on the OllamaLLM object itself. Passing these
# as top-level kwargs to `invoke()` can leak them through to the
# underlying client where they are not expected (causing the
# TypeError you saw). The `langchain_ollama` wrapper expects such
# generation parameters to be set on the LLM instance (or inside
# the `options` dict), so we provide them here.
# `num_predict` sets a safe token limit for replies to avoid truncation.
# Force Ollama to use CPU only by setting num_gpu=0. This ensures the
# Ollama runner does not attempt to initialize CUDA shared libraries
# (which can fail on systems without proper drivers/configuration).
llm = OllamaLLM(
    model="phi3",
    temperature=temperature,
    stop=stop_tokens,
    num_predict=256,
    num_gpu=0,
)

print("Everything loaded successfully!")

while True:
    question = input("\nAsk: ")

    if not question.strip():
        print("Please enter a question.")
        continue

    results = db.similarity_search(question, k=2)

    # Retrieval: gather the page content from the top results.
    # We keep the retrieval code intact to preserve the working pipeline.
    context_text = "\n\n".join([doc.page_content for doc in results])

    # Print source citations (filenames) from document metadata so users
    # can see where retrieved snippets came from.
    sources = [doc.metadata.get("source", "unknown") for doc in results]
    print("\nSources:")
    for s in sources:
        print(" -", s)

    # Context injection: supply only the retrieved context.
    # Make the fallback less brittle (only when context is empty).
    prompt = f"""
You are a concise AWS documentation assistant.
Use ONLY the Context section below to answer.

Context (each chunk with its source):
{context_text}

Question:
{question}

Instructions:
- Answer concisely and technically.
- Do NOT hallucinate or invent details.
- If the Context is empty (no retrieved text), reply exactly:
  I could not find that in the documentation.
- Otherwise, answer using the information in Context even if it doesn't exactly match the phrasing of the question.

Answer:
"""


    # Generation: call the local Ollama model. The temperature and stop
    # tokens have been set on the `llm` instance above, so we do not pass
    # them again here (passing them twice causes a ValueError).
    response = llm.invoke(prompt)

    print("\n=== RESPONSE ===\n")
    print(response)