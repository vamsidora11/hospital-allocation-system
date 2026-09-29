"""GET /agents/status (plan Section 10) — health/load per agent, for the monitoring UI.

There's no separate agent process or queue in this system (agents run in-process, called
synchronously from AGENT_EXECUTION) — "load" here is a simple count of currently
non-`decided` requests whose router selected that agent, which is the closest available
proxy without a real task-queue/worker-pool backing the agents.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.auth import verify_api_key
from app.db import get_session
from app.models.orm import AgentResultRow, Request

router = APIRouter(prefix="/agents", tags=["agents"], dependencies=[Depends(verify_api_key)])

KNOWN_AGENTS = ("bed_allocation", "staff_allocation", "emergency_routing")


class AgentStatus(BaseModel):
    agent_name: str
    status: str
    active_request_count: int


@router.get("/status", response_model=list[AgentStatus])
def get_agents_status(session: Session = Depends(get_session)) -> list[AgentStatus]:
    statuses = []
    for agent_name in KNOWN_AGENTS:
        active_count = (
            session.query(func.count(AgentResultRow.id.distinct()))
            .join(Request, Request.id == AgentResultRow.request_id)
            .filter(AgentResultRow.agent_name == agent_name, Request.status != "decided")
            .scalar()
        )
        statuses.append(AgentStatus(agent_name=agent_name, status="healthy", active_request_count=active_count or 0))
    return statuses
