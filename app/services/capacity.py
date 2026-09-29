"""Bed/staff availability lookups — the "tool call" agents use for `resource_snapshot`."""

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.orm import Bed, Staff

BED_TYPES = ("icu", "general", "isolation")


def bed_snapshot(session: Session | None = None) -> dict[str, dict[str, int]]:
    owns_session = session is None
    session = session or SessionLocal()
    try:
        snapshot: dict[str, dict[str, int]] = {}
        for bed_type in BED_TYPES:
            total = session.query(func.count(Bed.id)).filter(Bed.bed_type == bed_type).scalar()
            available = (
                session.query(func.count(Bed.id))
                .filter(Bed.bed_type == bed_type, Bed.status == "available")
                .scalar()
            )
            snapshot[bed_type] = {"total": total, "available": available}
        return snapshot
    finally:
        if owns_session:
            session.close()


def staff_snapshot(
    specialty: str | None = None, shift: str = "day", session: Session | None = None
) -> dict[str, int]:
    """Available on-shift staff, optionally filtered to a specialty."""
    owns_session = session is None
    session = session or SessionLocal()
    try:
        query = session.query(func.count(Staff.id)).filter(
            Staff.status == "available", Staff.shift == shift
        )
        total_on_shift = query.scalar()

        specialist_available = 0
        if specialty:
            specialist_available = (
                session.query(func.count(Staff.id))
                .filter(Staff.status == "available", Staff.shift == shift, Staff.specialty == specialty)
                .scalar()
            )
        return {"available_on_shift": total_on_shift, "specialist_available": specialist_available}
    finally:
        if owns_session:
            session.close()
