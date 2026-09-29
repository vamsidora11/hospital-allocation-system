from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

# Demo-only convenience: the showcase UI can be served same-origin at /demo (no CORS
# needed), but CORS is still permissive here in case it's opened from elsewhere in dev.
# TODO(pre-production): restrict this before going anywhere near real patient data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(requests_router)
app.include_router(policy_router)
app.include_router(audit_router)
app.include_router(agents_status_router)

_DEMO_DIR = Path(__file__).resolve().parents[1] / "frontend" / "demo"
if _DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(_DEMO_DIR), html=True), name="demo")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
