"""Hacker News job posts, via the public Algolia search API.

This source earns its place by being *badly structured on purpose*. There are
no company, role or location fields -- there is one free-text title:

    "Tasklet (YC P26) Is Hiring a Head of Design Engineering"

Getting company and role out of that is the fuzzy-extraction case, and it is
the one where a confidence score actually means something: when the title does
not match the expected grammar we keep the row, drop `company` to null, and mark
it low-confidence rather than guessing. Guessing is how you end up with a
database full of companies called "Is Hiring".
"""
from __future__ import annotations

import re

from .base import Adapter, Strategy, parse_json

API = "https://hn.algolia.com/api/v1"

# "Acme (YC W23) Is Hiring a Senior Engineer"  ->  company / role
_HIRING = re.compile(
    r"^(?P<company>.{2,60}?)\s*(?:\((?P<batch>YC\s*[SWXFP]?\d{2})\))?\s*"
    r"(?:is\s+)?(?:hiring|looking\s+for|seeks?)\s*"
    r"(?:a|an|the)?\s*(?P<role>.{2,90})$",
    re.I,
)
_PAREN_TAIL = re.compile(r"\s*\((?!YC)[^)]{0,40}\)\s*$")


def _split(title: str) -> tuple[str | None, str, float]:
    """-> (company, role, confidence). Confidence is the honest part."""
    t = " ".join((title or "").split())
    m = _HIRING.match(t)
    if m:
        company = _PAREN_TAIL.sub("", m.group("company")).strip(" -–—,")
        role = m.group("role").strip(" -–—,.")
        if company and role:
            return company, role, 1.0
    # No recognised grammar: keep the posting, admit we do not know the split.
    return None, t, 0.45


def _parse(text: str) -> list[dict]:
    data = parse_json(text)
    if not isinstance(data, dict):
        return []
    hits = data.get("hits")
    if not isinstance(hits, list):
        return []

    out: list[dict] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        title = h.get("title") or ""
        company, role, conf = _split(title)
        obj_id = h.get("objectID")
        out.append({
            "source_job_id": obj_id,
            "title": role,
            "company": company,
            # HN job posts carry no location field at all. Empty is the correct
            # answer; the drift monitor knows this field is expected to be
            # sparse here and will not alarm on it.
            "location": "",
            "url": h.get("url") or (f"https://news.ycombinator.com/item?id={obj_id}"
                                    if obj_id else None),
            "description": h.get("story_text") or title,
            "posted_at": h.get("created_at"),
            "tags": ["hn", "yc"] if "(yc" in title.lower() else ["hn"],
            "confidence": conf,
        })
    return out


adapter = Adapter(
    name="hackernews",
    kind="json-api",
    homepage="https://news.ycombinator.com/jobs",
    cadence_s=1200.0,
    licence_note="Algolia's public HN search API, published for third-party clients.",
    strategies=(
        Strategy("api-recent", f"{API}/search_by_date?tags=job&hitsPerPage=50", "json",
                 _parse, cost=1),
        Strategy("api-ranked", f"{API}/search?tags=job&hitsPerPage=50", "json",
                 _parse, cost=2),
    ),
)
