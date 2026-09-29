from fastapi import FastAPI

from app.api.agents_status import router as agents_status_router
from app.api.audit import router as audit_router
from app.api.policy import router as policy_router
from app.api.requests import router as requests_router
from app.config import get_settings
from app.observability.logging import configure_logging
from app.observability.tracing import configure_tracing

settings = get_settings()
configure_logging()
configure_tracing()

app = FastAPI(
    title="Hospital Resource Allocation System",
    description="Multi-agent hospital bed/staff/emergency allocation with policy-grounded RAG.",
    version="0.1.0",
)

app.include_router(requests_router)
app.include_router(policy_router)
app.include_router(audit_router)
app.include_router(agents_status_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
