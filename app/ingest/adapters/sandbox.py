"""Client for the hostile sandbox in `app/sandbox/server.py`.

This adapter is the only one with all three rungs -- HTML, JSON, RSS -- against
one source, which is what makes the fallback chain demonstrable: break the HTML
rung from the dashboard and watch the scheduler descend to JSON, then to RSS,
without losing a poll.

The HTML rung is also where the candidate-selector machinery earns its keep.
The sandbox can serve three completely different DOM layouts, including one
with hashed build-output class names, and the same parser handles all three
while reporting how much of the extraction ran on fallbacks.
"""
from __future__ import annotations

import os

from .base import (
    Adapter, Strategy, extraction_confidence, first_text, parse_html,
    parse_json, parse_xml_items, xml_field,
)

# The sandbox is served by this same process, so the crawler talks to itself
# over loopback. On a PaaS the platform sets PORT; locally 8000 is the default.
SELF = os.getenv("SENTINEL_SELF_URL") or f"http://127.0.0.1:{os.getenv('PORT', '8000')}"

# Candidate selectors, best-guess first. Order is the contract: index 0 is what
# we believe the page looks like, everything after it is a concession.
_ITEM_XPATHS = [
    "//li[contains(@class,'job')]",                 # v1, semantic
    "//article[@data-testid='posting']",            # v2, test-id build
    "//div[@data-title]",                           # v3, hashed classes
    "//*[self::li or self::article or self::div][.//a[contains(@href,'/jobs/')]]",
]
_TITLE_XPATHS = [
    ".//*[contains(@class,'job-title') or contains(@class,'posting__title')]",
    "./@data-title",
    ".//h1|.//h2|.//h3",
    ".//span[1]",
]
_COMPANY_XPATHS = [
    ".//*[contains(@class,'company')]",
    ".//*[@data-field='company']",
    "./@data-co",
    ".//span[2]",
]
_LOCATION_XPATHS = [
    ".//*[contains(@class,'location')]",
    ".//*[@data-field='location']",
    ".//span[3]",
]
_DESC_XPATHS = [
    ".//p",
    ".//*[contains(@class,'posting__body')]",
    ".//div[contains(@class,'desc')]",
]
_URL_XPATHS = [
    ".//a[contains(@class,'apply')]/@href",
    ".//a[@data-testid='apply-link']/@href",
    ".//a/@href",
]


def parse_sandbox_html(text: str) -> list[dict]:
    tree = parse_html(text)
    items = []
    for xp in _ITEM_XPATHS:
        items = tree.xpath(xp)
        if items:
            break
    if not items:
        return []

    out: list[dict] = []
    for node in items:
        matched: dict[str, int] = {}
        title, matched["title"] = first_text(node, _TITLE_XPATHS)
        company, matched["company"] = first_text(node, _COMPANY_XPATHS)
        location, matched["location"] = first_text(node, _LOCATION_XPATHS)
        description, matched["description"] = first_text(node, _DESC_XPATHS)
        url, matched["url"] = first_text(node, _URL_XPATHS)
        if not title:
            continue
        out.append({
            "title": title,
            "company": company,
            "location": location,
            "description": description,
            "url": url,
            "source_job_id": url.rsplit("/", 1)[-1] if url else None,
            "confidence": extraction_confidence(matched),
        })
    return out


def parse_sandbox_json(text: str) -> list[dict]:
    data = parse_json(text)
    if not isinstance(data, dict):
        return []
    return [
        {
            "source_job_id": j.get("id"),
            "title": j.get("title"),
            "company": j.get("company"),
            "location": j.get("location"),
            "remote": j.get("remote"),
            "url": j.get("url"),
            "description": j.get("description"),
            "salary": j.get("salary"),
            "posted_at": j.get("posted_at"),
            "tags": j.get("tags") or [],
        }
        for j in (data.get("jobs") or [])
        if isinstance(j, dict)
    ]


def parse_sandbox_rss(text: str) -> list[dict]:
    out: list[dict] = []
    for item in parse_xml_items(text):
        raw = xml_field(item, ["title"])
        company, _, title = raw.partition(":")
        out.append({
            "source_job_id": xml_field(item, ["guid"]),
            "title": (title or raw).strip(),
            "company": company.strip() if title else None,
            "location": xml_field(item, ["region"]),
            "url": xml_field(item, ["link"]),
            "description": xml_field(item, ["description"]),
            "posted_at": xml_field(item, ["pubDate"]),
        })
    return out


adapter = Adapter(
    name="sandbox",
    kind="hostile-sandbox",
    homepage=f"{SELF}/sandbox/jobs",
    cadence_s=45.0,          # fast, because this one exists to be watched
    licence_note="Synthetic data served by this application. Safe to hammer; nothing here is real.",
    strategies=(
        Strategy("html", f"{SELF}/sandbox/jobs", "html", parse_sandbox_html,
                 cost=1, navigation=True),
        Strategy("api", f"{SELF}/sandbox/api/jobs", "json", parse_sandbox_json, cost=2),
        Strategy("rss", f"{SELF}/sandbox/feed.rss", "xml", parse_sandbox_rss, cost=3),
    ),
)
