import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Request(Base):
    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_ref: Mapped[str] = mapped_column(String, ForeignKey("patients.patient_ref"))
    request_type: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="requests")
    agent_results: Mapped[list["AgentResultRow"]] = relationship(back_populates="request")
    approvals: Mapped[list["Approval"]] = relationship(back_populates="request")
    audit_entries: Mapped[list["AuditLog"]] = relationship(back_populates="request")


class Patient(Base):
    __tablename__ = "patients"

    patient_ref: Mapped[str] = mapped_column(String, primary_key=True)
    age: Mapped[int] = mapped_column(Integer)
    diagnosis_category: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    requires_isolation: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_monitoring_level: Mapped[str] = mapped_column(String, default="none")

    requests: Mapped[list["Request"]] = relationship(back_populates="patient")


class Bed(Base):
    __tablename__ = "beds"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    bed_type: Mapped[str] = mapped_column(String)  # icu | general | isolation
    status: Mapped[str] = mapped_column(String, default="available")
    ward: Mapped[str] = mapped_column(String)
    floor: Mapped[int] = mapped_column(Integer)


class Staff(Base):
    __tablename__ = "staff"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    role: Mapped[str] = mapped_column(String)
    specialty: Mapped[str | None] = mapped_column(String, nullable=True)
    shift: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="available")


class AgentResultRow(Base):
    __tablename__ = "agent_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.id"))
    agent_name: Mapped[str] = mapped_column(String)
    decision_json: Mapped[dict] = mapped_column(JSON)
    confidence_score: Mapped[float] = mapped_column(Float)
    policy_grounded: Mapped[bool] = mapped_column(Boolean, default=False)

    request: Mapped["Request"] = relationship(back_populates="agent_results")
    citations: Mapped[list["PolicyCitationRow"]] = relationship(back_populates="agent_result")


class PolicyCitationRow(Base):
    __tablename__ = "policy_citations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    agent_result_id: Mapped[str] = mapped_column(String, ForeignKey("agent_results.id"))
    policy_name: Mapped[str] = mapped_column(String)
    section: Mapped[str] = mapped_column(String)
    excerpt: Mapped[str] = mapped_column(Text)
    similarity_score: Mapped[float] = mapped_column(Float)

    agent_result: Mapped["AgentResultRow"] = relationship(back_populates="citations")


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.id"))
    reviewer_id: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)  # approve | reject | override | request_info
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    request: Mapped["Request"] = relationship(back_populates="approvals")


class AuditLog(Base):
    """Append-only audit trail.

    INSERT-only at the DB-role level in Postgres (see alembic migration — the grant
    revocation is applied behind a dialect check and skipped on SQLite, since SQLite has
    no row-level GRANT model). Application code must never UPDATE/DELETE this table.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.id"))
    state: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    input_snapshot_json: Mapped[dict] = mapped_column(JSON)
    output_snapshot_json: Mapped[dict] = mapped_column(JSON)
    actor: Mapped[str] = mapped_column(String)  # system | human

    request: Mapped["Request"] = relationship(back_populates="audit_entries")


class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String)
    source_path: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String, default="1.0")
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chunks: Mapped[list["PolicyChunk"]] = relationship(back_populates="document")


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    policy_document_id: Mapped[str] = mapped_column(String, ForeignKey("policy_documents.id"))
    chunk_text: Mapped[str] = mapped_column(Text)
    # Reference to the corresponding embedding in the Chroma collection, not a pgvector
    # column — keeps the schema portable between Postgres and the SQLite dev/test fallback.
    chroma_id: Mapped[str] = mapped_column(String)
    chunk_index: Mapped[int] = mapped_column(Integer)

    document: Mapped["PolicyDocument"] = relationship(back_populates="chunks")
