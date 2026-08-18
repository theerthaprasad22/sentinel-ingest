"""Source registry.

Adding a source is one import and one line. That is the point: the interesting
code is the shared machinery -- pacing, identity, breakers, drift -- and a new
board should not get to reimplement any of it.
"""
from __future__ import annotations

import time

from ..config import settings
from .. import db
from .adapters import arbeitnow, hackernews, remotive, sandbox, weworkremotely
from .adapters.base import Adapter

_ALL: tuple[Adapter, ...] = (
    sandbox.adapter,
    remotive.adapter,
    arbeitnow.adapter,
    weworkremotely.adapter,
    hackernews.adapter,
)


def adapters() -> tuple[Adapter, ...]:
    if settings.sandbox_enabled:
        return _ALL
    return tuple(a for a in _ALL if a.name != "sandbox")


def get(name: str) -> Adapter | None:
    return next((a for a in adapters() if a.name == name), None)


def sync_to_db() -> None:
    """Register sources without clobbering learned state.

    The INSERT is deliberately not an upsert on the operational columns: a
    redeploy must not reset a breaker that is open, or a next_due that is
    holding a source back after a rate-limit. Only the static description
    (kind, cadence) is refreshed.
    """
    now = time.time()
    for a in adapters():
        db.execute(
            "INSERT INTO sources(name, kind, enabled, cadence_s, next_due, strategy) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET kind = excluded.kind, "
            "cadence_s = excluded.cadence_s",
            (a.name, a.kind, 1, a.cadence_s, now, a.strategies[0].name),
        )
    known = {a.name for a in adapters()}
    for row in db.query("SELECT name FROM sources"):
        if row["name"] not in known:
            db.execute("UPDATE sources SET enabled = 0 WHERE name = ?", (row["name"],))
