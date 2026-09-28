# CLAUDE.md — Hospital Resource Allocation System

Context file for coding agents working in this repo. Read this fully before starting a phase.
Full spec lives in `hospital-ai-implementation-plan.md` (repo root) — this file pins the
decisions so they don't drift between sessions. `PROGRESS.md` has the phase-by-phase log.

## Data-handling constraint (applies to every phase)

This system stores *operational* patient data only: synthetic patient IDs, acuity level,
diagnosis category, isolation/monitoring flags. It never stores or displays SSN, Aadhaar, PAN,
passport numbers, driver's license numbers, insurance/payer IDs, or full medical records. Seed
data and fixtures must be synthetic (e.g. `PT-2026-00931`, never a real-looking MRN/SSN). This
is narrower than typical HIPAA-covered EHR data on purpose.

## Architecture (see plan Section 1)

FastAPI REST layer → LangGraph orchestrator (8-state workflow) → 3 domain agents (bed / staff /
emergency), each grounded via a Chroma-backed RAG retrieval layer → Postgres for state, beds,
staff, audit log.

- **LangGraph owns control flow, not the agents.** Agents are pure functions of
  `(patient_info, retrieved_policies, resource_snapshot) -> AgentResult`. They never call each
  other or the graph directly — this keeps them unit-testable in isolation.
- **Every agent output must carry ≥1 citation, or `policy_grounded: false`.** If retrieval
  returns nothing relevant (similarity below the floor, default 0.55), the agent lowers its
  confidence and sets `policy_grounded: false` rather than reasoning ungrounded.
- **The graph is resumable.** Human approval is a LangGraph interrupt with its own audit entry,
  not a bolted-on side workflow.

## The 8-state workflow

`INPUT_VALIDATION → REQUEST_CLASSIFICATION → AGENT_EXECUTION → RESULT_CONSOLIDATION →
CONFIDENCE_EVALUATION → HUMAN_APPROVAL_GATE → FINAL_DECISION → AUDIT_LOGGING`

- `AGENT_EXECUTION` fans out to 1–3 agents in parallel based on the router's `selected_agents`.
- Retries wrap only `AGENT_EXECUTION` (up to `MAX_AGENT_RETRIES`, default 3, exponential
  backoff). After exhausting retries, fall back to a conservative default (General Ward / no
  staff reassignment) and force human approval regardless of confidence.
- Confidence aggregation defaults to the **minimum** across active agents, not the average — one
  low-confidence agent should be able to pull the whole decision below threshold
  (`HUMAN_APPROVAL_THRESHOLD`, default 0.70, configurable).
- Reviewer actions: `approve`, `reject`, `override` (stored alongside, not instead of, the
  original agent recommendation), `request_info` (re-runs `AGENT_EXECUTION` with added context).

## Naming conventions

- Domain models: Pydantic, in `app/models/domain.py` — `PatientInfo`, `RouterOutput`,
  `PolicyCitation`, `AgentResult`, `ApprovalRecord`, `AuditEntry`, `WorkflowState`. Defined once
  in Phase 1; later phases must not redefine fields ad hoc.
- ORM tables: SQLAlchemy, in `app/models/orm.py`, matching plan Section 4.
- Graph nodes: one file per state under `app/graph/nodes/`, named after the state
  (`input_validation.py`, `router.py`, `agent_execution.py`, `consolidation.py`,
  `confidence_eval.py`, `human_approval.py`, `final_decision.py`, `audit_logging.py`).
- Agents: `app/agents/{bed_allocation,staff_allocation,emergency_routing}.py`.
- Request/patient IDs: synthetic, prefixed (`PT-...`, `REQ-...`) — see data-handling constraint.

## Folder layout

See plan Section 2 for the full tree. Don't add top-level dirs without updating that section.

## Environment / sandbox notes (read before assuming infra is available)

- **No Docker in this dev sandbox.** `docker`, `docker-compose` are not installed here.
  `docker-compose.yml` / `docker-compose.test.yml` / `Dockerfile` are written to spec but
  **unverified by an actual `docker compose up`** — that verification is a TODO for an
  environment that has Docker. Don't assume it silently works; check before relying on it.
- **Local/dev/test DB defaults to SQLite** (`DATABASE_URL=sqlite:///./dev.db`) so the app and
  test suite run without Postgres. `docker-compose.yml` points `DATABASE_URL` at Postgres for
  anything closer to production. Code must stay portable across both (avoid Postgres-only SQL;
  use SQLAlchemy core/ORM, not raw dialect-specific queries) except where a feature is
  Postgres-only by design (e.g. append-only grants on `audit_log` — apply those in the Alembic
  migration behind a dialect check, and skip on SQLite).
- **No real Azure/OpenAI credentials available.** `AZURE_OPENAI_API_KEY` /
  `OPENAI_API_KEY` are unset. RAG embeddings and agent LLM calls must be built behind an
  interface (`app/rag/embeddings.py`, agent LLM client) with a local/offline fallback
  (`sentence-transformers` or a deterministic stub) so tests and demos run without a live key.
  Treat wiring a real key as a pre-production TODO, not a blocker for any phase.
- **No live Kubernetes cluster available.** `deploy/k8s/` manifests are written to spec but
  applying them to kind/minikube is a TODO for an environment that has one.
- Python: use `py -3` (bare `python`/`python3` on PATH resolve to the Windows Store stub in this
  sandbox). Venv lives at `.venv/`.

## Auth

Out of scope for the first pass. `API_KEY` env var + a stub header check only —
**TODO(pre-production): replace with real auth** before this goes near real patient data.

## Testing

- Unit: router fixtures, each agent in isolation (mocked LLM + capacity data), confidence
  aggregation, retrieval ranking.
- Integration: full graph run against test Postgres + Chroma via `docker-compose.test.yml`
  (unverified here — no Docker; falls back to SQLite for the parts that don't strictly need
  Postgres/Chroma, per the sandbox note above).
- E2E regression: the three demo scenarios (`tests/e2e/test_demo_scenarios.py`) — critical
  cardiac (ICU, confidence ≥ 0.85, auto-approved), high-acuity isolation (isolation room +
  specialist staff, confidence high-0.80s), mass-casualty burst (8 concurrent patients, full
  audit trail for each).
