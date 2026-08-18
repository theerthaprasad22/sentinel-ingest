"""Per-source circuit breaker.

The failure mode this exists to prevent is not "a request failed" -- it is a
retry loop that keeps hammering a source that has already decided to block us,
converting a soft rate-limit into a durable ban. Once the breaker opens, the
source is left alone for a cooldown, and it reopens through a single-probe
half-open state rather than resuming full rate.

State lives in SQLite so a process restart does not reset a breaker that was
open for a good reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import settings
from .. import db

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class Decision:
    allowed: bool
    state: str
    reason: str
    retry_after: float = 0.0


def _row(source: str) -> dict:
    row = db.query_one("SELECT * FROM sources WHERE name = ?", (source,))
    return row or {"breaker_state": CLOSED, "consec_fail": 0, "opened_at": None}


def state(source: str) -> str:
    return _row(source).get("breaker_state") or CLOSED


def check(source: str) -> Decision:
    row = _row(source)
    st = row.get("breaker_state") or CLOSED
    if st == CLOSED:
        return Decision(True, CLOSED, "closed")

    opened_at = row.get("opened_at") or 0.0
    # Cooldown grows with consecutive failures (capped) -- a source that has
    # blocked us five times running has earned more patience than one that
    # blipped once.
    fails = max(1, int(row.get("consec_fail") or 1))
    cooldown = min(settings.breaker_cooldown_s * (2 ** (fails - 1)), 1800.0)
    elapsed = time.time() - opened_at

    if st == OPEN:
        if elapsed >= cooldown:
            db.execute(
                "UPDATE sources SET breaker_state = ? WHERE name = ?", (HALF_OPEN, source)
            )
            db.log_event("info", "breaker", "cooldown elapsed, sending one probe", source)
            return Decision(True, HALF_OPEN, "probe")
        return Decision(False, OPEN, "cooling down", retry_after=cooldown - elapsed)

    # HALF_OPEN: exactly one request is in flight; anything else waits.
    return Decision(True, HALF_OPEN, "probe")


def record_success(source: str) -> None:
    row = _row(source)
    if (row.get("breaker_state") or CLOSED) != CLOSED:
        db.log_event("info", "breaker", "probe succeeded, circuit closed", source)
    db.execute(
        "UPDATE sources SET breaker_state = ?, consec_fail = 0, opened_at = NULL, "
        "last_ok = ?, health = MIN(1.0, COALESCE(health, 1.0) * 0.7 + 0.3) WHERE name = ?",
        (CLOSED, time.time(), source),
    )


def record_failure(source: str, reason: str) -> str:
    row = _row(source)
    fails = int(row.get("consec_fail") or 0) + 1
    if fails >= settings.breaker_threshold:
        db.execute(
            "UPDATE sources SET breaker_state = ?, consec_fail = ?, opened_at = ?, "
            "health = COALESCE(health, 1.0) * 0.6, note = ? WHERE name = ?",
            (OPEN, fails, time.time(), reason, source),
        )
        db.log_event(
            "error", "breaker", f"circuit OPEN after {fails} failures: {reason}", source
        )
        return OPEN
    db.execute(
        "UPDATE sources SET consec_fail = ?, health = COALESCE(health, 1.0) * 0.85, "
        "note = ? WHERE name = ?",
        (fails, reason, source),
    )
    return CLOSED


def force_open(source: str, reason: str) -> None:
    """Trip immediately, skipping the failure count. Used when the evidence is
    unambiguous -- an explicit 403 with a block page, or a Retry-After we are
    going to honour rather than argue with."""
    db.execute(
        "UPDATE sources SET breaker_state = ?, consec_fail = COALESCE(consec_fail,0) + 1, "
        "opened_at = ?, note = ? WHERE name = ?",
        (OPEN, time.time(), reason, source),
    )
    db.log_event("error", "breaker", f"circuit forced OPEN: {reason}", source)


def reset(source: str) -> None:
    db.execute(
        "UPDATE sources SET breaker_state = ?, consec_fail = 0, opened_at = NULL, "
        "quarantined = 0, health = 1.0, note = NULL WHERE name = ?",
        (CLOSED, source),
    )
    db.log_event("info", "breaker", "manually reset", source)
