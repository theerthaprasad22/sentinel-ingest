"""The loop.

Sources are polled on their own cadence, not on a global timer, and each poll
walks a strategy ladder: the cheapest rung that has been working, descending
one rung per failure. Everything downstream of the fetch -- what to do about a
block, whether to descend, when to come back -- is decided here, so `fetcher.py`
stays a pure "make one request and tell me what happened".

The scheduler never raises. A source that breaks in an unanticipated way must
degrade to "that source is unhealthy", never to "the process died", because the
other four sources are still fine and a dead scheduler is the one failure mode
nobody notices until the job counter has been flat for a day.
"""
from __future__ import annotations

import asyncio
import contextlib
import time

from ..config import settings
from .. import db
from ..search import index as search_index
from ..search.tagger import tagger
from . import circuit, fetcher, registry, store
from .adapters.base import Adapter
from .pacing import pacer

# Rebuild the semantic index / retag at most this often, however much data lands.
_REINDEX_MIN_INTERVAL = 90.0

_state: dict[str, object] = {
    "running": False,
    "cycles": 0,
    "last_tick": 0.0,
    "last_reindex": 0.0,
    "started_at": 0.0,
}


def status() -> dict[str, object]:
    return dict(_state)


def _next_strategy(adapter: Adapter, current: str | None) -> str:
    """Descend one rung. At the bottom, stay there -- something is better than
    nothing, and the breaker is what stops us from grinding."""
    names = [s.name for s in adapter.strategies]
    if current not in names:
        return names[0]
    i = names.index(current)
    return names[min(i + 1, len(names) - 1)]


def _promote(adapter: Adapter, current: str | None) -> str:
    """Climb back toward the preferred rung after a clean run.

    One rung at a time rather than jumping straight to the top: if the HTML
    rung is broken because the site redesigned, going back to it immediately
    just re-breaks the pipeline every cycle.
    """
    names = [s.name for s in adapter.strategies]
    if current not in names:
        return names[0]
    return names[max(names.index(current) - 1, 0)]


async def poll_source(client, adapter: Adapter) -> dict[str, object]:
    """One source, one poll. Returns a summary for the API/UI."""
    name = adapter.name
    row = db.query_one("SELECT * FROM sources WHERE name = ?", (name,)) or {}
    strategy_name = row.get("strategy") or adapter.strategies[0].name
    strategy = adapter.strategy(strategy_name) or adapter.strategies[0]

    gate = circuit.check(name)
    if not gate.allowed:
        return {"source": name, "skipped": "breaker " + gate.state,
                "retry_in": round(gate.retry_after, 1)}

    tier = pacer.bandit.choose(name)
    db.execute("UPDATE sources SET last_attempt = ? WHERE name = ?", (time.time(), name))

    result = await fetcher.fetch(client, adapter, strategy)
    summary: dict[str, object] = {
        "source": name, "strategy": strategy.name, "tier": tier,
        "verdict": result.verdict, "status": result.status,
        "items": len(result.records), "block_prob": round(result.block_prob, 3),
        "latency_ms": round(result.latency_ms, 1), "identity": result.identity,
        "extraction_confidence": result.extraction_confidence,
    }

    if result.verdict == "deferred":
        # Owed budget to this host; not a failure, just come back shortly.
        db.execute("UPDATE sources SET next_due = ? WHERE name = ?",
                   (time.time() + 20.0, name))
        return summary

    if result.ok:
        pacer.bandit.update(name, tier, success=True)
        circuit.record_success(name)
        if result.unchanged:
            summary["written"] = {"unchanged": True}
        else:
            report = store.upsert(result.records, name)
            summary["written"] = report.as_dict()
            if report.new:
                db.log_event(
                    "info", "ingest",
                    f"{report.new} new, {report.updated} updated, "
                    f"{report.touched} unchanged via {strategy.name}",
                    name,
                )
        promoted = _promote(adapter, strategy.name)
        gap = pacer.next_gap(name, adapter.cadence_s, tier)
        db.execute(
            "UPDATE sources SET strategy = ?, next_due = ? WHERE name = ?",
            (promoted, time.time() + gap, name),
        )
        summary["next_in_s"] = round(gap, 1)
        return summary

    # --- failure paths ---------------------------------------------------
    pacer.bandit.update(name, tier, success=False)

    if result.verdict == "skipped":
        # robots said no. That is a permanent answer for this URL, so descend
        # to a rung that might be permitted rather than retrying the same path.
        nxt = _next_strategy(adapter, strategy.name)
        db.execute("UPDATE sources SET strategy = ?, next_due = ? WHERE name = ?",
                   (nxt, time.time() + adapter.cadence_s, name))
        summary["next_strategy"] = nxt
        return summary

    if result.verdict == "drift":
        # Extraction is producing junk. Descending is the right move -- another
        # rung is likely to have a stable contract (an API does not redesign
        # its markup) -- but the breaker stays closed because the source is not
        # blocking us, it just changed shape.
        nxt = _next_strategy(adapter, strategy.name)
        db.log_event("warn", "drift",
                     f"quarantined batch; falling back {strategy.name} -> {nxt}", name)
        db.execute("UPDATE sources SET strategy = ?, next_due = ? WHERE name = ?",
                   (nxt, time.time() + 60.0, name))
        summary["next_strategy"] = nxt
        return summary

    reason = f"{result.verdict}: {result.note[:160]}"
    if result.verdict == "blocked" or result.status in (403, 429):
        circuit.force_open(name, reason)
    else:
        circuit.record_failure(name, reason)

    nxt = _next_strategy(adapter, strategy.name)
    if nxt != strategy.name:
        db.log_event("warn", "fallback",
                     f"{strategy.name} -> {nxt} after {result.verdict}", name)
    # Back off hard on the source itself, independently of the breaker.
    db.execute(
        "UPDATE sources SET strategy = ?, next_due = ? WHERE name = ?",
        (nxt, time.time() + max(adapter.cadence_s, 60.0), name),
    )
    summary["next_strategy"] = nxt
    return summary


async def _maybe_reindex(force: bool = False) -> None:
    """Rebuild the semantic index and retag, off the fetch path.

    Both are CPU-bound scikit-learn calls, so they run in a thread -- blocking
    the event loop here would stall every in-flight fetch and skew the very
    latency features the block classifier reads.
    """
    now = time.time()
    if not force and now - float(_state["last_reindex"]) < _REINDEX_MIN_INTERVAL:
        return
    _state["last_reindex"] = now
    try:
        built = await asyncio.to_thread(search_index.index.build)
        if built.get("built"):
            dups = await asyncio.to_thread(search_index.index.mark_duplicates)
            await asyncio.to_thread(tagger.fit_and_apply)
            if dups:
                db.log_event("info", "dedupe", f"flagged {dups} near-duplicate postings")
    except Exception as exc:                          # noqa: BLE001
        db.log_event("error", "index", f"rebuild failed: {type(exc).__name__}: {exc}")


async def run_forever() -> None:
    _state.update(running=True, started_at=time.time())
    sem = asyncio.Semaphore(settings.max_concurrency)
    client = fetcher.make_client()
    db.log_event("info", "scheduler", "started")

    async def guarded(adapter: Adapter) -> None:
        async with sem:
            try:
                await poll_source(client, adapter)
            except Exception as exc:                  # noqa: BLE001
                # Last line of defence. A source that fails in a way nobody
                # anticipated must not take the loop with it.
                db.log_event("error", "scheduler",
                             f"unhandled {type(exc).__name__}: {exc}", adapter.name)
                circuit.record_failure(adapter.name, f"unhandled {type(exc).__name__}")
                db.execute("UPDATE sources SET next_due = ? WHERE name = ?",
                           (time.time() + 120.0, adapter.name))

    try:
        while True:
            now = time.time()
            _state["last_tick"] = now
            _state["cycles"] = int(_state["cycles"]) + 1

            ready = {
                r["name"] for r in db.query(
                    "SELECT name FROM sources WHERE enabled = 1 AND next_due <= ?", (now,)
                )
            }
            due = [a for a in registry.adapters() if a.name in ready]
            if due:
                await asyncio.gather(*(guarded(a) for a in due))
                await _maybe_reindex()

            if int(_state["cycles"]) % 60 == 0:
                db.prune()

            await asyncio.sleep(settings.tick_seconds)
    except asyncio.CancelledError:
        db.log_event("info", "scheduler", "stopped")
        raise
    finally:
        _state["running"] = False
        with contextlib.suppress(Exception):
            await client.aclose()
