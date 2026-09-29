"""Stub API-key auth (plan Section 10: "Auth is out of scope for the first pass").

TODO(pre-production): replace with real auth before this goes near real patient data — see
CLAUDE.md. This is intentionally a single shared-secret header check, nothing more.
"""

from fastapi import Header, HTTPException

from app.config import get_settings


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
