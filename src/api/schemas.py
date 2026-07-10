from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    role: Literal["patient", "clinician"]
    user_id: str
    top_k: int = 5


class QueryResponse(BaseModel):
    response: str
    latency_ms: float
    retrieved_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexes_built: bool


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    role: Literal["patient", "clinician"]
    user_id: str