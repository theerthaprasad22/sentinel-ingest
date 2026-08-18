"""FastAPI application: dashboard, JSON API, and the scheduler's host process.

The API surface is small and deliberately operational. Anyone can build a jobs
endpoint; the endpoints worth having here are the ones that let a stranger
answer "is this pipeline actually working, and how would it know if it were
not" -- /api/health, /api/sources, /api/events, /api/ml.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db
from .config import settings
from .ingest import blockdetect, circuit, fetcher, registry, scheduler
from .ingest.identity import pool
from .ingest.drift import monitor
from .ingest.pacing import pacer
from .ingest.store import stats as job_stats
from .sandbox.server import router as sandbox_router
from .search.index import index
from .search.tagger import tagger

BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE / "web" / "templates"))

_tasks: set[asyncio.Task] = set()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    registry.sync_to_db()

    # Train on boot. It takes well under a second and it means a cold container
    # is never serving with a disarmed block detector -- which is exactly when
    # a source is most likely to greet you with a challenge page.
    metrics = await asyncio.to_thread(blockdetect.classifier.train)
    db.log_event(
        "info", "model",
        f"block classifier trained: precision={metrics['precision']} "
        f"recall={metrics['recall']} on {metrics['n_holdout']} held-out samples",
    )
    await asyncio.to_thread(index.build)
    await asyncio.to_thread(tagger.fit_and_apply)

    if settings.scheduler_enabled:
        task = asyncio.create_task(scheduler.run_forever(), name="scheduler")
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)

    try:
        yield
    finally:
        for task in list(_tasks):
            task.cancel()
        for task in list(_tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(
    title="Sentinel Ingest",
    description=(
        "Resilient job-listing ingestion: adaptive pacing, coherent identities, "
        "ML soft-block detection, schema-drift quarantine, and a hostile sandbox "
        "you can break on purpose."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

if settings.sandbox_enabled:
    app.include_router(sandbox_router)

static_dir = BASE / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------------------------------------------------
# Operational API
# --------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    rows = db.query("SELECT * FROM sources ORDER BY name")
    recent = db.query(
        "SELECT verdict, COUNT(*) AS n FROM fetch_log WHERE ts > ? GROUP BY verdict",
        (time.time() - 3600,),
    )
    by_verdict = {r["verdict"]: r["n"] for r in recent}
    total = sum(by_verdict.values()) or 1
    healthy = sum(1 for r in rows if (r["breaker_state"] or "closed") == "closed"
                  and r["enabled"])
    return {
        "status": "ok" if healthy else "degraded",
        "sources_total": len(rows),
        "sources_healthy": healthy,
        "scheduler": scheduler.status(),
        "last_hour": {
            "requests": sum(by_verdict.values()),
            "clean_pct": round(
                100.0 * (by_verdict.get("clean", 0) + by_verdict.get("unchanged", 0))
                / total, 1
            ),
            "by_verdict": by_verdict,
        },
        "jobs": job_stats(),
        "uptime_s": round(time.time() - float(scheduler.status()["started_at"]), 1)
        if scheduler.status()["started_at"] else 0,
    }


@app.get("/api/sources")
async def sources():
    out = []
    identities = pool.snapshot()
    for a in registry.adapters():
        row = db.query_one("SELECT * FROM sources WHERE name = ?", (a.name,)) or {}
        recent = db.query(
            "SELECT ts, strategy, status, latency_ms, items, block_prob, verdict, note "
            "FROM fetch_log WHERE source = ? ORDER BY id DESC LIMIT 12",
            (a.name,),
        )
        n_clean = sum(1 for r in recent if r["verdict"] in ("clean", "unchanged"))
        out.append({
            "name": a.name,
            "kind": a.kind,
            "homepage": a.homepage,
            "licence_note": a.licence_note,
            "synthetic": a.name == "sandbox",
            "cadence_s": a.cadence_s,
            "strategies": [
                {"name": s.name, "expected": s.expected, "cost": s.cost, "url": s.url}
                for s in a.strategies
            ],
            "current_strategy": row.get("strategy"),
            "breaker": row.get("breaker_state") or "closed",
            "consec_fail": row.get("consec_fail") or 0,
            "quarantined": bool(row.get("quarantined")),
            "next_due_in_s": round(max(0.0, (row.get("next_due") or 0) - time.time()), 1),
            "last_ok": row.get("last_ok"),
            "note": row.get("note"),
            "success_rate_recent": round(n_clean / len(recent), 2) if recent else None,
            # A source we are declining to fetch is not a source that is
            # failing. Collapsing the two would make the dashboard read as
            # "broken" when it is actually working exactly as designed.
            "excluded_by_robots": bool(
                recent and recent[0]["verdict"] == "skipped"
                and "robots" in (recent[0]["note"] or "")
            ),
            "identity": identities.get(a.name, {}),
            "pace_posterior": pacer.bandit.snapshot(a.name),
            "drift_baseline": monitor.baseline(a.name),
            "recent": recent,
            "job_count": (db.query_one(
                "SELECT COUNT(*) AS n FROM jobs WHERE source = ?", (a.name,)
            ) or {"n": 0})["n"],
        })
    return out


@app.get("/api/jobs")
async def jobs(
    q: str = Query("", description="semantic query; blank returns most recent"),
    source: str = Query(""),
    role: str = Query(""),
    seniority: str = Query(""),
    include_duplicates: bool = Query(False),
    limit: int = Query(40, ge=1, le=200),
):
    where, params = [], []
    if source:
        where.append("source = ?")
        params.append(source)
    if role:
        where.append("role_family = ?")
        params.append(role)
    if seniority:
        where.append("seniority = ?")
        params.append(seniority)
    if not include_duplicates:
        where.append("dup_of IS NULL")

    if q.strip():
        hits = index.search(q, k=limit * 3)
        if not hits:
            return {"query": q, "mode": "semantic", "count": 0, "results": []}
        ranks = {cid: score for cid, score in hits}
        where.append(f"canonical_id IN ({','.join('?' * len(ranks))})")
        params.extend(ranks)
        rows = db.query(
            f"SELECT * FROM jobs WHERE {' AND '.join(where)}", params
        )
        rows.sort(key=lambda r: -ranks.get(r["canonical_id"], 0.0))
        rows = rows[:limit]
        for r in rows:
            r["score"] = round(ranks.get(r["canonical_id"], 0.0), 4)
        mode = "semantic"
    else:
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = db.query(
            f"SELECT * FROM jobs {clause} ORDER BY last_seen DESC LIMIT ?",
            [*params, limit],
        )
        mode = "recent"

    for r in rows:
        r["tags"] = json.loads(r.get("tags") or "[]")
        r["synthetic"] = r["source"] == "sandbox"
    return {"query": q, "mode": mode, "count": len(rows), "results": rows}


@app.get("/api/jobs/{canonical_id}/similar")
async def similar(canonical_id: str, k: int = Query(6, ge=1, le=20)):
    hits = index.similar(canonical_id, k=k)
    if not hits:
        return {"canonical_id": canonical_id, "results": []}
    ids = [h[0] for h in hits]
    scores = dict(hits)
    rows = db.query(
        f"SELECT canonical_id, title, company, location, url, source FROM jobs "
        f"WHERE canonical_id IN ({','.join('?' * len(ids))})",
        ids,
    )
    for r in rows:
        r["score"] = round(scores.get(r["canonical_id"], 0.0), 4)
    rows.sort(key=lambda r: -r["score"])
    return {"canonical_id": canonical_id, "results": rows}


@app.get("/api/events")
async def events(limit: int = Query(60, ge=1, le=300)):
    return db.query(
        "SELECT ts, level, source, kind, message FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    )


@app.get("/api/events/stream")
async def events_stream(request: Request):
    """Server-sent events. Chosen over a websocket because the traffic is one
    way and SSE reconnects by itself, which matters on a free host that puts
    the container to sleep."""

    async def gen():
        last_id = (db.query_one("SELECT MAX(id) AS m FROM events") or {}).get("m") or 0
        while True:
            if await request.is_disconnected():
                break
            rows = db.query(
                "SELECT id, ts, level, source, kind, message FROM events "
                "WHERE id > ? ORDER BY id LIMIT 40",
                (last_id,),
            )
            for row in rows:
                last_id = row["id"]
                yield f"data: {json.dumps(row)}\n\n"
            if not rows:
                yield ": keepalive\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


def _live_confusion() -> dict[str, object]:
    """Score the block classifier against ground truth on live traffic.

    Every sandbox response carries an `x-sandbox-truth` header saying whether it
    was really a block. That makes this the only number on the page measured
    outside the model's own training distribution -- the holdout metrics score
    the model against data it was designed to fit, which is exactly the trap
    this endpoint exists to avoid falling into quietly.

    It is still a partial picture: the sandbox is the only labelled source, so
    this measures the model on synthetic blocks, not on a real vendor's
    challenge page. Stated plainly rather than papered over.
    """
    rows = db.query(
        "SELECT truth, verdict, block_prob FROM fetch_log "
        "WHERE truth IS NOT NULL ORDER BY id DESC LIMIT 500"
    )
    if not rows:
        return {"n": 0, "note": "no labelled traffic yet -- poll the sandbox"}

    tp = fp = tn = fn = 0
    for r in rows:
        actually_blocked = r["truth"] == "blocked"
        called_blocked = r["verdict"] in ("blocked", "suspect", "empty")
        if actually_blocked and called_blocked:
            tp += 1
        elif actually_blocked:
            fn += 1
        elif called_blocked:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n": len(rows),
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "note": (
            "Measured against the sandbox's own ground-truth header on real "
            "requests. Labelled traffic comes from the sandbox only, so this "
            "scores synthetic blocks -- no real vendor challenge page is in here."
        ),
    }


@app.get("/api/ml")
async def ml_status():
    """Everything the ML layer knows about itself, including provenance. The
    provenance line is not decoration -- a metric without it is a claim."""
    return {
        "block_classifier": blockdetect.classifier.metrics
        or db.kv_get("blockdetect_metrics", {}),
        "block_classifier_live": _live_confusion(),
        "semantic_index": index.status(),
        "tagger": tagger.status(),
        "feature_names": list(blockdetect.FEATURE_NAMES),
    }


@app.get("/api/drift")
async def drift_log(limit: int = Query(40, ge=1, le=200)):
    return db.query(
        "SELECT ts, source, field, fill_rate, baseline, z, action FROM drift_log "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )


@app.post("/api/sources/{name}/poll")
async def poll_now(name: str):
    """Force one poll, bypassing the schedule but not the breaker or robots."""
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(404, f"unknown source {name!r}")
    client = fetcher.make_client()
    try:
        return await scheduler.poll_source(client, adapter)
    finally:
        await client.aclose()


@app.post("/api/sources/{name}/reset")
async def reset_source(name: str):
    if registry.get(name) is None:
        raise HTTPException(404, f"unknown source {name!r}")
    circuit.reset(name)
    db.execute("UPDATE sources SET next_due = ?, strategy = ? WHERE name = ?",
               (time.time(), registry.get(name).strategies[0].name, name))
    return {"ok": True, "source": name}


@app.post("/api/reindex")
async def reindex():
    built = await asyncio.to_thread(index.build)
    dups = await asyncio.to_thread(index.mark_duplicates) if built.get("built") else 0
    tags = await asyncio.to_thread(tagger.fit_and_apply)
    return {"index": built, "duplicates_marked": dups, "tagger": tags}


@app.get("/api/robots-check")
async def robots_check(url: str = Query(..., description="URL to test the gate against")):
    """Show the robots decision for any URL without fetching it.

    Exposed because "we respect robots.txt" is a claim, and a claim a reviewer
    can execute is worth more than a paragraph in a design document.
    """
    from .ingest import robots as robots_mod
    client = fetcher.make_client()
    try:
        rules = await robots_mod.rules_for(url, client)
        permitted, why = robots_mod.allowed(url, rules)
        return {
            "url": url,
            "allowed": permitted,
            "reason": why,
            "crawl_delay": rules.crawl_delay,
            "robots_reachable": rules.reachable,
            "disallow_rules": rules.disallow[:20],
        }
    finally:
        await client.aclose()


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    """Serve our own robots.txt from disk.

    The crawler fetches the sandbox source over loopback against this same
    origin, so this file is what its robots gate actually reads before every
    sandbox poll -- the policy is enforced against ourselves first.
    """
    return FileResponse(BASE / "web" / "robots.txt", media_type="text/plain")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return JSONResponse({"ok": True})
