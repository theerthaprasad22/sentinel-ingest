"""SQLite persistence.

Single-writer, WAL mode, short-lived connections. Everything the pipeline
learns at runtime (source health, fetch traces, drift alarms) is durable, so a
restart resumes with its breakers and baselines intact instead of re-learning
them by hammering a source that was already blocking us.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from .config import settings

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    canonical_id      TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    source_job_id     TEXT,
    title             TEXT NOT NULL,
    company           TEXT,
    location          TEXT,
    remote            INTEGER DEFAULT 0,
    url               TEXT,
    description       TEXT,
    salary_text       TEXT,
    posted_at         TEXT,
    first_seen        REAL NOT NULL,
    last_seen         REAL NOT NULL,
    tags              TEXT DEFAULT '[]',
    role_family       TEXT,
    seniority         TEXT,
    confidence        REAL DEFAULT 1.0,
    dup_of            TEXT,
    payload_hash      TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_source   ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_lastseen ON jobs(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_dup      ON jobs(dup_of);

CREATE TABLE IF NOT EXISTS sources (
    name           TEXT PRIMARY KEY,
    kind           TEXT,
    enabled        INTEGER DEFAULT 1,
    cadence_s      REAL DEFAULT 300,
    breaker_state  TEXT DEFAULT 'closed',
    consec_fail    INTEGER DEFAULT 0,
    opened_at      REAL,
    next_due       REAL DEFAULT 0,
    last_ok        REAL,
    last_attempt   REAL,
    etag           TEXT,
    last_modified  TEXT,
    strategy       TEXT,
    health         REAL DEFAULT 1.0,
    quarantined    INTEGER DEFAULT 0,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    source      TEXT NOT NULL,
    strategy    TEXT,
    url         TEXT,
    status      INTEGER,
    latency_ms  REAL,
    bytes       INTEGER,
    items       INTEGER,
    block_prob  REAL,
    verdict     TEXT,
    identity    TEXT,
    pace_tier   TEXT,
    truth       TEXT,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_ts ON fetch_log(ts DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_src ON fetch_log(source, ts DESC);

CREATE TABLE IF NOT EXISTS drift_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    source    TEXT NOT NULL,
    field     TEXT NOT NULL,
    fill_rate REAL,
    baseline  REAL,
    z         REAL,
    action    TEXT
);
CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_log(ts DESC);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    source  TEXT,
    kind    TEXT,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

-- Key/value scratch for baselines, bandit posteriors and model metadata that
-- must survive a restart.
CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


_schema_ready = False


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    global _schema_ready
    _ensure_dir(settings.db_path)
    conn = sqlite3.connect(settings.db_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        if not _schema_ready:
            # Self-healing schema. The module-level singletons (drift baselines,
            # pacing posteriors) read their state at import time, which is before
            # any explicit init would have run -- and an ephemeral host can hand
            # us a blank disk on any restart. Creating the schema on first
            # connection makes both cases a non-event instead of a crash loop.
            conn.executescript(SCHEMA)
            _schema_ready = True
        yield conn
        conn.commit()
    finally:
        conn.close()


# (column, DDL) pairs applied to an existing database that predates them.
# CREATE TABLE IF NOT EXISTS silently does nothing when the table already
# exists, so new columns need an explicit ALTER or a redeploy onto a persisted
# volume comes up with a schema one version behind the code.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("fetch_log", "truth", "ALTER TABLE fetch_log ADD COLUMN truth TEXT"),
)


def init_db() -> None:
    """Explicit init for startup and tests; `connect()` also self-heals."""
    with _lock, connect() as conn:
        conn.executescript(SCHEMA)
        for table, column, ddl in _MIGRATIONS:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(ddl)


def execute(sql: str, params: Iterable[Any] = ()) -> None:
    with _lock, connect() as conn:
        conn.execute(sql, tuple(params))


def executemany(sql: str, rows: Iterable[Iterable[Any]]) -> None:
    with _lock, connect() as conn:
        conn.executemany(sql, [tuple(r) for r in rows])


def query(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(sql, tuple(params))
        return [dict(r) for r in cur.fetchall()]


def query_one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


# --- kv helpers -----------------------------------------------------------

def kv_get(key: str, default: Any = None) -> Any:
    row = query_one("SELECT v FROM kv WHERE k = ?", (key,))
    if not row:
        return default
    try:
        return json.loads(row["v"])
    except json.JSONDecodeError:
        return default


def kv_set(key: str, value: Any) -> None:
    execute(
        "INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, json.dumps(value)),
    )


# --- event feed -----------------------------------------------------------

def log_event(level: str, kind: str, message: str, source: str | None = None) -> None:
    """Append to the operator-visible event feed backing the dashboard's SSE
    stream. Deliberately cheap: no fan-out, the UI tails the table."""
    execute(
        "INSERT INTO events(ts, level, source, kind, message) VALUES(?,?,?,?,?)",
        (time.time(), level, source, kind, message),
    )


def prune(max_fetch_rows: int = 4000, max_events: int = 1500) -> None:
    """Ephemeral disks are small. Keep the operational tables bounded; the jobs
    table is the only thing we never truncate on a schedule."""
    execute(
        "DELETE FROM fetch_log WHERE id NOT IN "
        "(SELECT id FROM fetch_log ORDER BY id DESC LIMIT ?)",
        (max_fetch_rows,),
    )
    execute(
        "DELETE FROM events WHERE id NOT IN "
        "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
        (max_events,),
    )
