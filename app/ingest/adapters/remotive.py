"""Remotive -- documented public JSON API.

Remotive publishes this endpoint for exactly this purpose and asks for
attribution rather than a key. That is why it is in the demo and LinkedIn is
not: the brief's scope guardrail says run the live demo against a low-risk
source, and "the operator published an API and asked to be credited" is about
as low-risk as ingestion gets.
"""
from __future__ import annotations

from .base import Adapter, Strategy, parse_json

API = "https://remotive.com/api/remote-jobs"


def _parse(text: str) -> list[dict]:
    data = parse_json(text)
    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        return []

    out: list[dict] = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        out.append({
            "source_job_id": str(j.get("id") or ""),
            "title": j.get("title"),
            "company": j.get("company_name"),
            # Remotive is remote-only, so the geography field is a *constraint*
            # ("USA", "Europe") rather than an office. Mapping it straight into
            # `location` would be a lie; prefixing it keeps the meaning.
            "location": (
                f"Remote ({j.get('candidate_required_location')})"
                if j.get("candidate_required_location") else "Remote"
            ),
            "remote": True,
            "url": j.get("url"),
            "description": j.get("description"),
            "salary": j.get("salary"),
            "posted_at": j.get("publication_date"),
            "tags": (j.get("tags") if isinstance(j.get("tags"), list) else [])
                    + ([j["category"]] if j.get("category") else []),
        })
    return out


adapter = Adapter(
    name="remotive",
    kind="json-api",
    homepage="https://remotive.com",
    cadence_s=420.0,
    licence_note="Public documented API; Remotive asks for attribution, which the UI gives.",
    strategies=(
        Strategy("api", f"{API}?limit=80", "json", _parse, cost=1),
        # Fallback rung: same endpoint, one category, smaller page. If the wide
        # query is what is being throttled, a narrower one often still answers.
        Strategy("api-narrow", f"{API}?category=software-dev&limit=40", "json", _parse, cost=2),
    ),
)
