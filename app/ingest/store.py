"""Writing jobs down.

Two rules here, both of which exist because of how job boards actually behave.

First, an unchanged row must not be rewritten. Boards re-serve the same posting
for weeks; if every poll rewrites every row, `last_seen` churn drowns the write
path and it becomes impossible to answer "what actually changed today". So the
payload is hashed and a matching hash is a touch, not an update.

Second, a re-post is not a new job. The same role appears on three boards and
gets re-listed monthly with a new URL. Identity is content-derived
(`normalize.canonical_id`), and anything that slips past exact matching gets
caught by the near-duplicate pass in `app/search/index.py`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .. import db
from .normalize import normalize


@dataclass
class WriteReport:
    new: int = 0
    updated: int = 0
    touched: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.new + self.updated + self.touched

    def as_dict(self) -> dict[str, int]:
        return {"new": self.new, "updated": self.updated,
                "touched": self.touched, "rejected": self.rejected}


def upsert(records: list[dict], source: str) -> WriteReport:
    report = WriteReport()
    now = time.time()

    rows: list[dict] = []
    for raw in records:
        row = normalize(raw, source)
        if row is None:
            report.rejected += 1
            continue
        rows.append(row)

    if not rows:
        return report

    # De-duplicate within the batch itself before touching the database. A
    # single feed page routinely carries the same posting twice under different
    # URLs, and letting both through turns one INSERT into a lost update.
    by_id: dict[str, dict] = {}
    for row in rows:
        prior = by_id.get(row["canonical_id"])
        if prior is None or (row.get("confidence", 1.0) > prior.get("confidence", 1.0)):
            by_id[row["canonical_id"]] = row

    existing = {
        r["canonical_id"]: r
        for r in db.query(
            "SELECT canonical_id, payload_hash FROM jobs WHERE canonical_id IN "
            f"({','.join('?' * len(by_id))})",
            list(by_id),
        )
    } if by_id else {}

    inserts, updates, touches = [], [], []
    for cid, row in by_id.items():
        prior = existing.get(cid)
        if prior is None:
            inserts.append(row)
        elif prior["payload_hash"] != row["payload_hash"]:
            updates.append(row)
        else:
            touches.append(cid)

    if inserts:
        db.executemany(
            "INSERT OR IGNORE INTO jobs(canonical_id, source, source_job_id, title, "
            "company, location, remote, url, description, salary_text, posted_at, "
            "first_seen, last_seen, tags, confidence, payload_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (r["canonical_id"], r["source"], r["source_job_id"], r["title"],
                 r["company"], r["location"], r["remote"], r["url"], r["description"],
                 r["salary_text"], r["posted_at"], r["first_seen"], r["last_seen"],
                 r["tags"], r["confidence"], r["payload_hash"])
                for r in inserts
            ],
        )
        report.new = len(inserts)

    if updates:
        db.executemany(
            "UPDATE jobs SET title = ?, company = ?, location = ?, remote = ?, url = ?, "
            "description = ?, salary_text = ?, posted_at = ?, last_seen = ?, tags = ?, "
            "confidence = ?, payload_hash = ?, source = ? WHERE canonical_id = ?",
            [
                (r["title"], r["company"], r["location"], r["remote"], r["url"],
                 r["description"], r["salary_text"], r["posted_at"], now, r["tags"],
                 r["confidence"], r["payload_hash"], r["source"], r["canonical_id"])
                for r in updates
            ],
        )
        report.updated = len(updates)

    if touches:
        db.execute(
            f"UPDATE jobs SET last_seen = ? WHERE canonical_id IN "
            f"({','.join('?' * len(touches))})",
            [now, *touches],
        )
        report.touched = len(touches)

    return report


def stats() -> dict[str, object]:
    total = db.query_one("SELECT COUNT(*) AS n FROM jobs") or {"n": 0}
    dups = db.query_one("SELECT COUNT(*) AS n FROM jobs WHERE dup_of IS NOT NULL") or {"n": 0}
    per_source = db.query(
        "SELECT source, COUNT(*) AS n, MAX(last_seen) AS last FROM jobs GROUP BY source"
    )
    fresh = db.query_one(
        "SELECT COUNT(*) AS n FROM jobs WHERE first_seen > ?", (time.time() - 86400,)
    ) or {"n": 0}
    return {
        "total": total["n"],
        "duplicates": dups["n"],
        "new_24h": fresh["n"],
        "per_source": per_source,
    }
