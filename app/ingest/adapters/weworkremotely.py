"""We Work Remotely -- public RSS.

RSS is the format this whole project argues for: a publisher-sanctioned,
cache-friendly, markup-stable contract that survives the site redesign that
would break any CSS selector. Where a board offers one, taking it instead of
scraping the HTML is the entire "ingestion strategy" answer in miniature.

The one wrinkle is that WWR packs the company into the title as
"Company: Role", so the split is a parser concern rather than a field lookup.
"""
from __future__ import annotations

from .base import Adapter, Strategy, parse_xml_items, xml_field

FEED = "https://weworkremotely.com/remote-jobs.rss"
PROGRAMMING = "https://weworkremotely.com/categories/remote-programming-jobs.rss"


def _split_title(raw: str) -> tuple[str, str]:
    """"Vonage: ServiceNow Alliance Executive" -> ("Vonage", "ServiceNow ...").

    Guarded: a colon inside a role ("Engineer: Platform") would otherwise eat
    the title. Requiring the left side to be short and the right side to be
    non-trivial keeps that from happening more often than it fixes.
    """
    if ":" in raw:
        left, right = raw.split(":", 1)
        left, right = left.strip(), right.strip()
        if 0 < len(left) <= 45 and len(right) >= 3:
            return left, right
    return "", raw.strip()


def _parse(text: str) -> list[dict]:
    out: list[dict] = []
    for item in parse_xml_items(text):
        raw_title = xml_field(item, ["title"])
        if not raw_title:
            continue
        company, title = _split_title(raw_title)
        region = xml_field(item, ["region", "country"]) or "Anywhere"
        out.append({
            "source_job_id": xml_field(item, ["guid", "link"]),
            "title": title,
            "company": company or None,
            "location": region,
            "remote": True,
            "url": xml_field(item, ["link", "id"]),
            "description": xml_field(item, ["description", "summary", "content"]),
            "posted_at": xml_field(item, ["pubDate", "published", "updated"]),
            "tags": [t for t in (xml_field(item, ["category"]),
                                 xml_field(item, ["type"])) if t],
        })
    return out


adapter = Adapter(
    name="weworkremotely",
    kind="rss",
    homepage="https://weworkremotely.com",
    cadence_s=900.0,
    licence_note="Public RSS feed published by the board for syndication.",
    strategies=(
        Strategy("rss", FEED, "xml", _parse, cost=1),
        # If the firehose feed is throttled or empty, the per-category feed is a
        # separate cache key on their CDN and frequently still serves.
        Strategy("rss-category", PROGRAMMING, "xml", _parse, cost=2),
    ),
)
