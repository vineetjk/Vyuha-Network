"""
Catalyst Data Store sync for RAG.

The app's relational SQLAlchemy DB stays the source of truth. This pushes a
denormalized, flat row per FIR case into a single Catalyst Data Store table so
QuickML can attach that table as a RAG dataset (RAG wants flat text per record,
not joined relational rows).

Auth reuses the QuickML refresh-token flow (_CatalystToken), so the SAME
GLM_REFRESH_TOKEN works here — but it must be minted with BOTH scopes:
    QuickML.deployment.READ,ZohoCatalyst.tables.rows.CREATE

Config (env):
    GLM_DATASTORE_TABLE  Data Store table name/id (e.g. rag_cases) — required
    GLM_DATASTORE_URL    Data Store BaaS base, default the project's BaaS URL
    GLM_AI_ORG           org id (shared with the GLM client)

NB: the exact BaaS insert URL, auth header prefix, and body/batch shape are
finalised from the table's "API Details" sample. The request builder below is
isolated in _insert_batch() so only that one function changes once confirmed.
"""

import os

from app.services.ai_service import _CatalystToken

# Catalyst BaaS (Data Store) base for this project. Overridable via env because
# the exact host differs per data center / project.
DEFAULT_BASE = "https://api.catalyst.zoho.in/baas/v1/project/46808000000019001"

# Data Store REST accepts bulk row inserts in batches (typically up to 200/call).
BATCH_SIZE = 100


def is_configured() -> bool:
    """RAG sync is available only when a target table is set and the GLM/Data
    Store refresh credentials are present."""
    return bool(os.getenv("GLM_DATASTORE_TABLE")) and _CatalystToken.configured()


def _row_for_case(c) -> dict:
    """Denormalize one CaseMaster ORM row into a flat Data Store row. Resolves
    the relational lookups (district, station, category, gravity) to plain text
    and builds a single `content` sentence for embedding/RAG."""
    district = c.unit.district.DistrictName if c.unit and c.unit.district else None
    station = c.unit.UnitName if c.unit else None
    category = c.minor_head.CrimeHeadName if c.minor_head else None
    gravity = c.gravity.LookupValue if c.gravity else None
    date = c.CrimeRegisteredDate.strftime("%Y-%m-%d") if c.CrimeRegisteredDate else None
    brief = (c.BriefFacts or "").strip()

    # One human sentence per case — this is what RAG embeds and retrieves on.
    content = (
        f"FIR {c.CrimeNo}: {category or 'Unknown offence'} "
        f"({gravity or 'unclassified'}) registered at {station or 'unknown station'}, "
        f"{district or 'unknown district'} on {date or 'unknown date'}. {brief}"
    ).strip()

    return {
        "crime_no": c.CrimeNo,
        "district": district,
        "station": station,
        "category": category,
        "gravity": gravity,
        "crime_date": date,
        "brief_facts": brief[:2000],
        "content": content[:4000],
    }


def _insert_batch(rows: list) -> int:
    """POST one batch of rows to the Data Store table. Isolated so the exact
    endpoint/auth/body can be pinned from the console API-Details sample without
    touching the sync orchestration."""
    import requests

    base = (os.getenv("GLM_DATASTORE_URL") or DEFAULT_BASE).rstrip("/")
    table = os.getenv("GLM_DATASTORE_TABLE", "rag_cases")
    org = os.getenv("GLM_AI_ORG", "60080167463")
    # AppSail runs in the Catalyst Development environment, so Data Store writes
    # must carry Environment: Development or they target the wrong environment.
    env = os.getenv("GLM_DATASTORE_ENV", "Development")
    url = f"{base}/table/{table}/row"

    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            # Catalyst BaaS uses the Zoho-oauthtoken prefix (unlike the QuickML
            # GLM API's Bearer) — confirmed in the Data Store REST docs.
            "Authorization": f"Zoho-oauthtoken {_CatalystToken.get()}",
            "CATALYST-ORG": org,
            "Environment": env,
        },
        json=rows,  # Insert Rows accepts a JSON array (up to 200 rows/call).
        timeout=45,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Data Store insert HTTP {resp.status_code}: {resp.text[:300]}")
    return len(rows)


def sync_cases(db, limit: int = 2000) -> dict:
    """Read cases (newest first), denormalize, and push to the Data Store table
    in batches. Returns a summary for the admin endpoint."""
    from app.db import models

    if not os.getenv("GLM_DATASTORE_TABLE"):
        raise RuntimeError("GLM_DATASTORE_TABLE is not set")

    cases = (
        db.query(models.CaseMaster)
        .order_by(models.CaseMaster.CrimeRegisteredDate.desc())
        .limit(limit)
        .all()
    )
    rows = [_row_for_case(c) for c in cases]

    pushed, batches, errors = 0, 0, []
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        try:
            pushed += _insert_batch(batch)
            batches += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:200])
            break  # stop on first failure so the admin sees the real error

    return {
        "ok": not errors,
        "table": os.getenv("GLM_DATASTORE_TABLE"),
        "read": len(rows),
        "pushed": pushed,
        "batches": batches,
        "errors": errors,
    }
