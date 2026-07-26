"""
Developer/admin console API (admin role only).

Exposes runtime configuration and observability for a single-instance dev
deployment: view & edit environment variables at runtime (the AI/OCR/config
readers use os.getenv per request, so most changes take effect immediately —
no redeploy), tail the log buffer, and read API/AI/OCR metrics.

Runtime env edits are per-instance and reset on redeploy — they are a live
config panel, not a persistent store. DATABASE_URL changes need a restart
(the SQLAlchemy engine binds at startup).
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.api.deps import RoleChecker
from app.db import models
from app.services import observability

router = APIRouter(prefix="/admin", tags=["Developer Console"])

require_admin = RoleChecker(["admin"])

# Config keys the app actually reads — surfaced first in the console.
KNOWN_KEYS = [
    # GLM_ prefix (not CATALYST_): Catalyst reserves CATALYST_* and its console
    # rejects user-defined CATALYST_* env vars.
    "GLM_AI_TOKEN",
    "GLM_REFRESH_TOKEN",
    "GLM_CLIENT_ID",
    "GLM_CLIENT_SECRET",
    "GLM_ACCOUNTS_URL",
    "GLM_AI_MODEL",
    "GLM_AI_ORG",
    "GLM_AI_URL",
    "MOCK_AI_PIPELINE",
    "FALLBACK_AI_BASE_URL",
    "FALLBACK_AI_MODEL",
    "FALLBACK_AI_API_KEYS",
    "GROQ_MODEL",
    "GEMINI_API_KEY",
    "CATALYST_AI_ENABLED",
    "ZIA_OCR_URL",
    "ZIA_CODELIB_SECRET",
    "DATABASE_URL",
    "AUTO_SEED",
    "DISABLE_APP_CORS",
    "CORS_ALLOW_ORIGINS",
]

# Changing these needs a process restart to take effect.
RESTART_KEYS = {"DATABASE_URL"}


def _is_secret(key: str) -> bool:
    upper = key.upper()
    return (
        any(tok in upper for tok in ("SECRET", "KEY", "PASSWORD", "TOKEN", "PWD"))
        or key == "DATABASE_URL"
    )


def _mask(key: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not _is_secret(key):
        return value
    if len(value) <= 4:
        return "••••"
    return f"••••••••{value[-4:]}"


@router.get("/config")
def get_config(current_user: models.User = Depends(require_admin)):
    """Known config keys (with masked current values) + all other env vars."""
    known = [
        {
            "key": k,
            "value": _mask(k, os.getenv(k)),
            "set": os.getenv(k) is not None,
            "secret": _is_secret(k),
            "needs_restart": k in RESTART_KEYS,
        }
        for k in KNOWN_KEYS
    ]
    others = {
        k: _mask(k, v)
        for k, v in sorted(os.environ.items())
        if k not in KNOWN_KEYS
    }
    return {"known": known, "others": others}


class EnvUpdate(BaseModel):
    key: str
    value: str


@router.put("/config")
def set_config(payload: EnvUpdate, current_user: models.User = Depends(require_admin)):
    """Set an environment variable at runtime (this instance)."""
    key = payload.key.strip()
    if not key:
        return {"ok": False, "message": "Key is required."}
    os.environ[key] = payload.value
    return {
        "ok": True,
        "key": key,
        "needs_restart": key in RESTART_KEYS,
        "message": (
            "Saved. Restart required for this key to take effect."
            if key in RESTART_KEYS
            else "Saved — takes effect on the next request."
        ),
    }


@router.delete("/config/{key}")
def delete_config(key: str, current_user: models.User = Depends(require_admin)):
    existed = os.environ.pop(key, None) is not None
    return {"ok": True, "removed": existed}


@router.get("/config/export", response_class=PlainTextResponse)
def export_config(current_user: models.User = Depends(require_admin)):
    """Download the currently-set known config keys as a .env file (real
    values — admin only, so you can save and re-import them)."""
    lines = ["# Vyuha Network runtime config export"]
    for k in KNOWN_KEYS:
        v = os.getenv(k)
        if v is not None:
            lines.append(f"{k}={v}")
    body = "\n".join(lines) + "\n"
    return PlainTextResponse(
        body, headers={"Content-Disposition": "attachment; filename=vyuha.env"}
    )


class ImportBody(BaseModel):
    content: str


@router.post("/config/import")
def import_config(payload: ImportBody, current_user: models.User = Depends(require_admin)):
    """Set multiple env vars from an uploaded .env-style file (KEY=VALUE lines;
    # comments and blank lines ignored). Applies at runtime like the editor."""
    applied = []
    needs_restart = []
    for raw in payload.content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ[key] = val
        applied.append(key)
        if key in RESTART_KEYS:
            needs_restart.append(key)
    return {"ok": True, "count": len(applied), "applied": applied, "needs_restart": needs_restart}


@router.get("/logs")
def get_logs(limit: int = 200, current_user: models.User = Depends(require_admin)):
    return {"logs": observability.recent_logs(limit)}


@router.get("/metrics")
def get_metrics(current_user: models.User = Depends(require_admin)):
    snap = observability.metrics.snapshot()
    # Effective AI provider as currently configured.
    from app.services.ai_service import LLMAIService

    snap["ai_configured"] = LLMAIService.is_configured()
    snap["ocr_configured"] = bool(os.getenv("ZIA_OCR_URL") and os.getenv("ZIA_CODELIB_SECRET"))
    return snap
