# Progress Log

## Phase 0 — Bootstrap

- Created repo structure per plan Section 2 (`app/`, `alembic/`, `data/policies/`, `tests/`,
  `deploy/k8s/`, `frontend/demo/`).
- FastAPI skeleton (`app/main.py`, `app/config.py`) with a working `GET /health` endpoint.
- `docker-compose.yml`, `docker-compose.test.yml`, `Dockerfile` written to spec (Postgres +
  Chroma + api services).
- `CLAUDE.md` written: pins the state model, naming conventions, folder layout, and the
  data-handling constraint, plus sandbox environment notes (see below).
- `pyproject.toml` with full planned dependency set (fastapi, sqlalchemy, alembic, langgraph,
  langchain-core/openai, chromadb, structlog, pytest) for later phases.
- Git repo initialized.

**Verification performed:**
- Installed fastapi/uvicorn/pydantic/pydantic-settings into `.venv` (`py -3 -m venv .venv`).
- Ran `uvicorn app.main:app` locally and confirmed `GET /health` → `200 {"status":"ok",...}`.

**TODO / environment constraints (see CLAUDE.md "Environment / sandbox notes" for detail):**
- Docker is **not installed** in this dev sandbox — `docker compose up` itself was not run or
  verified. Compose/Dockerfiles are written to spec but unverified end-to-end. Verify in an
  environment with Docker before relying on them.
- No live Postgres/Chroma/K8s cluster and no real Azure/OpenAI API key available here — later
  phases will default to SQLite locally and stub the LLM/embedding calls, per CLAUDE.md.
