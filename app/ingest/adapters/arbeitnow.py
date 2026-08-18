"""Arbeitnow -- public, keyless job-board API (EU-heavy).

Included because it is the only source here that returns a genuinely different
shape: `created_at` is a unix integer, `remote` is a real boolean rather than a
string to sniff, and location is a bare city. It is a useful second source
precisely because it exercises different branches of `normalize.py`.
"""
from __future__ import annotations

from .base import Adapter, Strategy, parse_json


def _as_list(value) -> list:
    """Coerce a field that is *usually* a list.

    Observed in production data: `job_types` comes back as {"1": "manager"} on a
    small fraction of rows. Concatenating that raises TypeError and takes the
    whole batch with it, so one malformed record must not cost us 175 good ones.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    if value in (None, ""):
        return []
    return [value]

API = "https://www.arbeitnow.com/api/job-board-api"


def _parse(text: str) -> list[dict]:
    data = parse_json(text)
    if not isinstance(data, dict):
        return []
    rows = data.get("data")
    if not isinstance(rows, list):
        return []

    out: list[dict] = []
    for j in rows:
        if not isinstance(j, dict):
            continue
        out.append({
            "source_job_id": j.get("slug"),
            "title": j.get("title"),
            "company": j.get("company_name"),
            "location": j.get("location"),
            "remote": bool(j.get("remote")),
            # `url` on this feed is often the employer's own site rather than a
            # posting page. It is still the canonical link for the job, so we
            # keep it and let the reader see where it points.
            "url": j.get("url"),
            "description": j.get("description"),
            "posted_at": j.get("created_at"),
            "tags": _as_list(j.get("tags")) + _as_list(j.get("job_types")),
        })
    return out


adapter = Adapter(
    name="arbeitnow",
    kind="json-api",
    homepage="https://www.arbeitnow.com",
    cadence_s=600.0,
    licence_note="Public keyless job-board API published for third-party use.",
    strategies=(
        Strategy("api", API, "json", _parse, cost=1),
        Strategy("api-page2", f"{API}?page=2", "json", _parse, cost=2),
    ),
)
