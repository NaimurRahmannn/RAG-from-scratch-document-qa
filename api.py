from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import TOP_K
from app.rag import RAGPipeline


app = FastAPI(
    title="RAG From Scratch Document Q&A API",
    description="FastAPI endpoint for asking questions against the indexed document store.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


class SourceResponse(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "RAG Document Q&A API",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        result = get_rag_pipeline().ask(
            question=question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 400 if "No indexed documents" in message else 500
        raise HTTPException(status_code=status_code, detail=message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {exc}",
        ) from exc

    return result
