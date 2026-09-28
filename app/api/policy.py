from fastapi import APIRouter
from pydantic import BaseModel

from app.models.domain import PolicyCitation
from app.rag.retriever import DEFAULT_K, retrieve

router = APIRouter(prefix="/policy", tags=["policy"])


class PolicySearchRequest(BaseModel):
    query: str
    domain: str
    k: int = DEFAULT_K


class PolicySearchResponse(BaseModel):
    results: list[PolicyCitation]


@router.post("/search", response_model=PolicySearchResponse)
def search_policy(payload: PolicySearchRequest) -> PolicySearchResponse:
    results = retrieve(payload.query, payload.domain, payload.k)
    return PolicySearchResponse(results=results)
