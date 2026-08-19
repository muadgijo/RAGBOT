import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from pydantic import BaseModel

from rag_utils import (
    build_prompt,
    get_embedding_function,
    get_llm_response,
    retrieve_context,
)

load_dotenv()

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embeddings...")
    resources["embeddings"] = get_embedding_function()
    print("Loading Chroma DB...")
    resources["db"] = Chroma(
        persist_directory="chroma",
        embedding_function=resources["embeddings"],
    )
    print("Ready.")
    yield
    resources.clear()


app = FastAPI(title="RAGBOT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    db = resources["db"]
    docs, context_text = retrieve_context(db, request.question, initial_k=6, final_k=3)
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))
    prompt = build_prompt(request.question, context_text)
    answer = get_llm_response(prompt, os.getenv("LLM_BACKEND", "ollama"))

    return QueryResponse(question=request.question, answer=answer, sources=sources)