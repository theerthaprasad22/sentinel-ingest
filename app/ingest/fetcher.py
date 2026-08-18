"""The fetch path.

One function does the whole outbound trip, in this order, and the order is the
design:

  1. robots gate       -- a disallowed path is never requested
  2. pacing gate       -- token bucket, per host
  3. identity lease    -- sticky coherent browser profile + egress slot
  4. conditional GET   -- ETag / If-Modified-Since; 304 is a *success*
  5. request           -- HTTP/2, bounded retries, decorrelated jitter backoff
  6. parse             -- adapter's parser for this rung
  7. block scoring     -- ML classifier over the response, not just the status
  8. drift check       -- per-field fill rates against learned baselines

Steps 7 and 8 are the ones that separate this from a `requests.get` in a loop.
A response can be 200, well-formed, and still be a block or a silent breakage,
and the pipeline has to know the difference before it writes anything.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

from ..config import settings
from .. import db
from . import blockdetect, circuit, drift, robots
from .adapters.base import Adapter, Strategy
from .identity import pool
from .pacing import pacer

_RETRYABLE_STATUS = {408, 425, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BODY_CAP = 3_000_000        # 3MB; a job feed larger than this is a bug or a trap


@dataclass
class FetchResult:
    source: str
    strategy: str
    ok: bool
    records: list[dict] = field(default_factory=list)
    status: int = 0
    latency_ms: float = 0.0
    bytes: int = 0
    block_prob: float = 0.0
    verdict: str = "clean"
    note: str = ""
    identity: str = ""
    unchanged: bool = False
    truth: str | None = None          # ground truth, when the source supplies one
    extraction_confidence: float = 1.0
    drift_alarms: list[drift.Alarm] = field(default_factory=list)
    explain: list[tuple[str, float]] = field(default_factory=list)


def _rolling(source: str) -> tuple[float, float]:
    """Median-ish item count and latency for this source, from its own history.

    Used as the denominator of the ratio features. Falls back to neutral values
    on a cold start so a fresh deployment does not read its first response as
    an anomaly against a baseline it has not built yet.
    """
    rows = db.query(
        "SELECT items, latency_ms FROM fetch_log WHERE source = ? AND verdict = 'clean' "
        "ORDER BY id DESC LIMIT 25",
        (source,),
    )
    if len(rows) < 3:
        return 20.0, 500.0
    items = sorted(r["items"] or 0 for r in rows)
    lats = sorted(r["latency_ms"] or 0.0 for r in rows)
    mid = len(rows) // 2
    return max(float(items[mid]), 1.0), max(float(lats[mid]), 50.0)


def _backoff(attempt: int, previous: float) -> float:
    """Decorrelated jitter (AWS architecture blog's variant).

    Plain exponential backoff synchronises every client that failed at the same
    moment into the same retry instant, which is how a rate-limit becomes an
    outage. This spreads them.
    """
    return min(30.0, random.uniform(1.0, max(previous, 1.0) * 3.0))


async def fetch(
    client: httpx.AsyncClient,
    adapter: Adapter,
    strategy: Strategy,
) -> FetchResult:
    source = adapter.name
    lease = pool.lease(source)
    result = FetchResult(source=source, strategy=strategy.name, ok=False,
                         identity=lease.label)

    # --- 1. robots -------------------------------------------------------
    if settings.respect_robots:
        rules = await robots.rules_for(strategy.url, client)
        permitted, why = robots.allowed(strategy.url, rules)
        if not permitted:
            result.note = f"robots: {why}"
            result.verdict = "skipped"
            db.log_event("warn", "robots", f"{strategy.url} not fetched -- {why}", source)
            _log(result, strategy.url)
            return result
        if rules.crawl_delay:
            # A declared Crawl-delay is a stated preference. Honour it rather
            # than the (faster) bucket rate.
            await asyncio.sleep(min(rules.crawl_delay, 10.0))

    # --- 2. pacing -------------------------------------------------------
    host = httpx.URL(strategy.url).host or "unknown"
    wait = pacer.gate(host)
    if wait > 0:
        if wait > 20.0:
            result.note = f"pacing: {wait:.0f}s of budget owed to {host}"
            result.verdict = "deferred"
            return result
        await asyncio.sleep(wait)

    # --- 3/4. identity + conditional GET ---------------------------------
    headers = dict(lease.profile.headers(
        navigation=strategy.navigation, contact=settings.contact
    ))
    headers = {k: v for k, v in headers.items() if v}
    if strategy.expected == "json":
        headers["accept"] = "application/json, text/plain, */*"
    elif strategy.expected == "xml":
        headers["accept"] = "application/rss+xml, application/xml;q=0.9, */*;q=0.8"

    row = db.query_one("SELECT etag, last_modified FROM sources WHERE name = ?", (source,))
    if row and row.get("etag"):
        headers["if-none-match"] = row["etag"]
    if row and row.get("last_modified"):
        headers["if-modified-since"] = row["last_modified"]

    # --- 5. request ------------------------------------------------------
    started = time.monotonic()
    resp: httpx.Response | None = None
    delay = 1.0
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = await client.get(strategy.url, headers=headers, follow_redirects=True)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            result.note = f"{type(exc).__name__}: {exc}"
            if attempt == _MAX_ATTEMPTS:
                result.verdict = "error"
                result.latency_ms = (time.monotonic() - started) * 1000
                _log(result, strategy.url)
                return result
            delay = _backoff(attempt, delay)
            await asyncio.sleep(delay)
            continue

        if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS:
            delay = _backoff(attempt, delay)
            await asyncio.sleep(delay)
            continue
        break

    assert resp is not None
    result.latency_ms = (time.monotonic() - started) * 1000
    result.status = resp.status_code

    # 304: the source is telling us nothing changed. This is the *best* outcome
    # -- no bandwidth, no parsing, no risk -- and a pipeline that scores it as
    # "zero items, must be blocked" would trip its own breaker on good news.
    if resp.status_code == 304:
        result.ok = True
        result.unchanged = True
        result.verdict = "unchanged"
        result.note = "304 Not Modified"
        _log(result, strategy.url)
        return result

    body = resp.text[:_BODY_CAP]
    result.bytes = len(resp.content)
    # Only the sandbox sends this. It is what lets /api/ml report a confusion
    # matrix measured on live traffic instead of on the model's own training
    # distribution -- the difference between a metric and a claim.
    result.truth = resp.headers.get("x-sandbox-truth")

    # --- 6. parse --------------------------------------------------------
    records: list[dict] = []
    if 200 <= resp.status_code < 300:
        try:
            records = strategy.parse(body) or []
        except Exception as exc:                      # noqa: BLE001
            # A parser blowing up is itself a drift signal, not a crash. Record
            # it, keep the pipeline alive, let the breaker decide.
            result.note = f"parser raised {type(exc).__name__}: {exc}"
            db.log_event("error", "parser", result.note, source)
            records = []

    # --- 7. block scoring ------------------------------------------------
    med_items, med_lat = _rolling(source)
    features = blockdetect.ResponseFeatures(
        status=resp.status_code,
        body=body,
        content_type=resp.headers.get("content-type", ""),
        expected_content=strategy.expected,
        items=len(records),
        latency_ms=result.latency_ms,
        n_redirects=len(resp.history),
        set_cookie=";".join(resp.headers.get_list("set-cookie")),
        retry_after=resp.headers.get("retry-after", ""),
        median_items=med_items,
        median_latency_ms=med_lat,
    )
    result.block_prob = blockdetect.classifier.score(features)
    result.verdict = blockdetect.verdict(result.block_prob, settings.block_prob_threshold)
    result.explain = blockdetect.classifier.explain(features)

    if result.verdict != "clean":
        # Burn the identity before the next attempt. Walking back in with the
        # same fingerprint after being challenged is how a soft block becomes a
        # durable one.
        pool.burn(source)
        reasons = ", ".join(f"{n}{v:+.2f}" for n, v in result.explain[:3])
        result.note = f"block_prob={result.block_prob:.2f} [{reasons}]"
        if resp.headers.get("retry-after"):
            result.note += f" retry-after={resp.headers['retry-after']}"
        _log(result, strategy.url)
        return result

    if not records:
        result.note = "parsed zero records from a clean-looking response"
        result.verdict = "empty"
        _log(result, strategy.url)
        return result

    # Mean fraction of fields that matched their *first* candidate selector.
    # This is the early warning that a markup change is coming: the batch is
    # still complete, but it is being held together by fallbacks. Waiting for
    # the fill rate to collapse means waiting until data is already lost.
    confidences = [float(r.get("confidence", 1.0)) for r in records]
    result.extraction_confidence = round(sum(confidences) / len(confidences), 3)
    if result.extraction_confidence < 0.8:
        db.log_event(
            "warn", "extraction",
            f"selectors degraded: {result.extraction_confidence:.0%} of fields matched "
            f"their primary candidate on {strategy.name} -- refresh them before the "
            f"fallbacks go too",
            source,
        )

    # --- 8. drift --------------------------------------------------------
    result.drift_alarms = drift.monitor.check(source, records)
    if result.drift_alarms:
        # Quarantine rather than write. The last good rows keep serving; a
        # human gets an alarm that names the field and the magnitude.
        result.verdict = "drift"
        result.note = "; ".join(a.message for a in result.drift_alarms)
        db.execute("UPDATE sources SET quarantined = 1 WHERE name = ?", (source,))
        _log(result, strategy.url)
        return result

    # Store validators only after a batch we actually trust, so a poisoned
    # response cannot install an ETag that suppresses the next real fetch.
    db.execute(
        "UPDATE sources SET etag = ?, last_modified = ?, quarantined = 0 WHERE name = ?",
        (resp.headers.get("etag"), resp.headers.get("last-modified"), source),
    )

    result.ok = True
    result.records = records
    _log(result, strategy.url)
    return result


def _log(result: FetchResult, url: str) -> None:
    db.execute(
        "INSERT INTO fetch_log(ts, source, strategy, url, status, latency_ms, bytes, "
        "items, block_prob, verdict, identity, truth, note) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (time.time(), result.source, result.strategy, url, result.status,
         round(result.latency_ms, 1), result.bytes, len(result.records),
         round(result.block_prob, 4), result.verdict, result.identity,
         result.truth,
         (f"conf={result.extraction_confidence} " if result.records else "")
         + result.note[:360]),
    )


def make_client(egress: str | None = None) -> httpx.AsyncClient:
    """HTTP/2 on purpose.

    An HTTP/1.1-only client is a fingerprint all by itself now that every real
    browser negotiates h2 -- the ALPN list and the HTTP/2 SETTINGS frame are
    both things an edge can hash. Matching the transport is table stakes before
    any of the header work above means anything.
    """
    return httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        proxy=egress,
    )
