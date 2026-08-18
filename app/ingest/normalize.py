"""Normalisation and identity.

Four sources describe the same job four different ways: "Sr. Backend Engineer
(Remote - US)" at "Acme, Inc." versus "Senior Backend Engineer" at "Acme Inc"
with location in a separate field. If the canonical id is a hash of the source
URL, the same job lands four times and the counter lies.

So the id is derived from the *content* -- aggressively normalised company +
title + city -- and near-duplicates that survive that (different wording, same
posting) are caught later by cosine similarity in `app/search/index.py`.

Everything here is pure and side-effect free, which is why it is also where the
test suite spends most of its time.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

# Suffixes that carry no identity. Ordered longest-first so "Inc." is stripped
# before "Inc" leaves a stray dot behind.
_COMPANY_NOISE = (
    "private limited", "pvt ltd", "pvt. ltd.", "limited", "incorporated",
    "corporation", "holdings", "group", "technologies", "labs", "inc.", "inc",
    "llc", "ltd.", "ltd", "gmbh", "b.v.", "bv", "s.a.", "sa", "plc", "co.", "co",
)

# Seniority / mode noise that changes the string but not the job.
_TITLE_NOISE = re.compile(
    r"\b(remote|hybrid|onsite|on-site|full[- ]time|part[- ]time|contract|"
    r"w2|c2c|urgent|immediate joiner|hiring|we are hiring|new)\b",
    re.I,
)
_BRACKETS = re.compile(r"[\(\[\{][^\)\]\}]{0,60}[\)\]\}]")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

_REMOTE_HINT = re.compile(r"\b(remote|anywhere|work from home|wfh|distributed)\b", re.I)

_SALARY = re.compile(
    r"(?:[$€£₹]|\b(?:usd|eur|gbp|inr|lpa)\b)\s?\d[\d,.]*\s?(?:k|m|lpa)?"
    r"(?:\s?[-–to]+\s?(?:[$€£₹])?\s?\d[\d,.]*\s?(?:k|m|lpa)?)?",
    re.I,
)

_TAG_STRIP = re.compile(r"<[^>]+>")


def clean_text(value: Any, limit: int = 4000) -> str:
    """HTML entities out, tags out, whitespace collapsed, length bounded.

    Bounded because a handful of boards inline an entire company brochure into
    the description field and there is no reason to carry that into SQLite.
    """
    if value is None:
        return ""
    s = str(value)
    s = _TAG_STRIP.sub(" ", s)
    s = html.unescape(s)
    s = _WS.sub(" ", s).strip()
    return s[:limit]


def norm_key(value: str) -> str:
    """Lowercase, de-punctuated, de-bracketed key used only for identity."""
    s = (value or "").lower()
    s = _BRACKETS.sub(" ", s)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_company(value: str) -> str:
    s = norm_key(value)
    changed = True
    while changed:
        changed = False
        for suffix in _COMPANY_NOISE:
            if s.endswith(" " + suffix):
                s = s[: -(len(suffix) + 1)].strip()
                changed = True
    return s


def norm_title(value: str) -> str:
    s = _TITLE_NOISE.sub(" ", value or "")
    s = norm_key(s)
    # Fold the common seniority abbreviations so "Sr." and "Senior" collide.
    s = re.sub(r"\bsr\b", "senior", s)
    s = re.sub(r"\bjr\b", "junior", s)
    s = re.sub(r"\bengr?\b", "engineer", s)
    return _WS.sub(" ", s).strip()


def norm_location(value: str) -> str:
    s = norm_key(value)
    if _REMOTE_HINT.search(s):
        return "remote"
    # First comma-ish segment is the city; country granularity is too coarse to
    # distinguish jobs and too noisy to match on.
    return s.split(" ")[0] if s else ""


def canonical_id(company: str, title: str, location: str) -> str:
    basis = f"{norm_company(company)}|{norm_title(title)}|{norm_location(location)}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:20]


def payload_hash(record: dict) -> str:
    """Hash of the fields we would write. Lets the writer skip a row that has
    not changed, which keeps `last_seen` churn out of the write path."""
    basis = json.dumps(
        {k: record.get(k) for k in ("title", "company", "location", "url",
                                    "description", "salary_text", "posted_at")},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def parse_date(value: Any) -> str | None:
    """Best-effort date parsing across the formats boards actually emit.

    Returns an ISO-8601 UTC string, or None. None is a legitimate answer --
    inventing a date so the field looks populated is exactly the kind of thing
    the drift monitor exists to catch.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    s = str(value).strip()
    formats = (
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
        "%d %b %Y",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def extract_salary(*fields: Any) -> str:
    for f in fields:
        text = clean_text(f, limit=600)
        m = _SALARY.search(text)
        if m:
            return m.group(0).strip()
    return ""


def is_remote(*fields: Any) -> bool:
    return any(_REMOTE_HINT.search(clean_text(f, limit=400)) for f in fields)


REQUIRED = ("title", "url")


def normalize(raw: dict, source: str) -> dict | None:
    """Raw adapter output -> the canonical row shape.

    Returns None when the record has no title or no URL. A job you cannot name
    or link to is not a job, and letting it through would quietly inflate the
    counts the whole dashboard is judged on.
    """
    title = clean_text(raw.get("title"), 300)
    url = clean_text(raw.get("url"), 500)
    if not title or not url:
        return None

    company = clean_text(raw.get("company"), 200)
    location = clean_text(raw.get("location"), 200)
    description = clean_text(raw.get("description"), 4000)
    now = time.time()

    record = {
        "canonical_id": canonical_id(company, title, location),
        "source": source,
        "source_job_id": clean_text(raw.get("source_job_id"), 120) or None,
        "title": title,
        "company": company or None,
        "location": location or None,
        "remote": int(is_remote(location, title, raw.get("remote"))),
        "url": url,
        "description": description or None,
        "salary_text": extract_salary(raw.get("salary"), description) or None,
        "posted_at": parse_date(raw.get("posted_at")),
        "first_seen": now,
        "last_seen": now,
        "tags": json.dumps([clean_text(t, 40) for t in (raw.get("tags") or [])][:12]),
        "confidence": float(raw.get("confidence", 1.0)),
    }
    record["payload_hash"] = payload_hash(record)
    return record
