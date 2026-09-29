from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.auth import verify_api_key
from app.db import get_session
from app.models.domain import AuditEntry
from app.services.audit import get_audit_trail

router = APIRouter(prefix="/audit", tags=["audit"], dependencies=[Depends(verify_api_key)])


@router.get("/{request_id}", response_model=list[AuditEntry])
def read_audit_trail(request_id: str, session: Session = Depends(get_session)) -> list[AuditEntry]:
    return get_audit_trail(session, request_id)
