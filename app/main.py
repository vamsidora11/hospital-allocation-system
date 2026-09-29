from fastapi import FastAPI

from app.api.audit import router as audit_router
from app.api.policy import router as policy_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Hospital Resource Allocation System",
    description="Multi-agent hospital bed/staff/emergency allocation with policy-grounded RAG.",
    version="0.1.0",
)

app.include_router(policy_router)
app.include_router(audit_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
