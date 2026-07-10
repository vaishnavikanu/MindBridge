import json
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional
from src.pipeline.rag_pipeline import RAGPipeline

from .schemas import (
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(
    prefix="/api",
    tags=["RAG"]
)


#
# This variable will be injected by main.py
#
pipeline: Optional[RAGPipeline] = None


def set_pipeline(rag_pipeline):
    global pipeline
    pipeline = rag_pipeline


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check"
)
def health():

    if pipeline is None:
        return HealthResponse(
            status="initializing",
            pipeline_ready=False,
        )

    stats = pipeline.get_stats()

    return HealthResponse(
        status="ok",
        pipeline_ready=stats["indexes_built"],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
    description="Retrieve relevant context and generate a response using the RAG pipeline."
)
def query(request: QueryRequest):

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized.",
        )

    try:
        result = pipeline.query(
            query=request.query,
            role=request.role,
            user_id=request.user_id,
            top_k=request.top_k,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    return QueryResponse(
        response=result.response,
        latency_ms=result.latency_ms,
        retrieved_chunks=len(result.retrieved_chunks),
    )


@router.post(
        "/feedback",
        summary="Submit user feedback"
)
def feedback(request: FeedbackRequest):

    #with open("feedback.jsonl", "a") as f:

    feedback_path = Path("logs") / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                request.model_dump()
            )
            + "\n"
        )

    return {
        "status": "success",
        "message": "Feedback recorded."
    }