"""Schema-drift detection.

The scenario the brief asks about -- "the source changes its markup overnight"
-- almost never announces itself as an error. The fetch is 200, the parser
finds *something*, and the pipeline happily writes 500 rows where `company` is
now null and `salary` is the posting date. Silent corruption is worse than an
outage, because an outage pages someone.

The guard is per-field fill rate tracked as an EWMA with an EWMA of variance,
so each field carries its own baseline and its own tolerance. A field that has
always been 40% populated does not alarm at 40%; a field that has been 99%
populated for a month alarms hard at 60%.

When a field trips, the batch is quarantined rather than dropped: the last good
rows keep serving, the source is marked degraded, and the raw payload is kept
so a human can look at what actually changed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..config import settings
from .. import db

# Fields worth watching. `title` and `url` are structural -- if those drop the
# extraction is broken outright. The rest degrade more gracefully.
TRACKED_FIELDS: tuple[str, ...] = (
    "title", "company", "location", "url", "description", "posted_at",
)

_ALPHA = 0.2          # EWMA weight -- ~10-sample memory
_MIN_OBS = 5          # do not alarm before a baseline exists
_KV = "drift_baselines"


@dataclass
class Alarm:
    source: str
    field: str
    fill_rate: float
    baseline: float
    z: float

    @property
    def message(self) -> str:
        return (
            f"{self.field} fill-rate {self.fill_rate:.0%} vs baseline "
            f"{self.baseline:.0%} (z={self.z:.1f})"
        )


class DriftMonitor:
    """State: {source: {field: [mean, var, n]}}, persisted to SQLite."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, list[float]]] = db.kv_get(_KV, {}) or {}

    def _slot(self, source: str, field: str) -> list[float]:
        s = self._state.setdefault(source, {})
        return s.setdefault(field, [0.0, 0.0, 0.0])   # mean, var, n

    @staticmethod
    def fill_rates(records: list[dict]) -> dict[str, float]:
        if not records:
            return {f: 0.0 for f in TRACKED_FIELDS}
        n = len(records)
        return {
            f: sum(1 for r in records if str(r.get(f) or "").strip()) / n
            for f in TRACKED_FIELDS
        }

    def check(self, source: str, records: list[dict]) -> list[Alarm]:
        """Score a freshly parsed batch. Returns alarms; does not itself decide
        what to do about them -- that is the caller's policy call."""
        alarms: list[Alarm] = []
        rates = self.fill_rates(records)

        for field, rate in rates.items():
            mean, var, n = self._slot(source, field)
            if n >= _MIN_OBS:
                sd = math.sqrt(max(var, 1e-6))
                z = (rate - mean) / max(sd, 0.05)   # floor the sd: a field that
                                                    # has been perfectly stable
                                                    # must not have infinite
                                                    # sensitivity
                if z <= -settings.drift_z_threshold and rate < mean - 0.15:
                    alarms.append(Alarm(source, field, rate, mean, z))

            # Update after scoring, so a batch is judged against its own past.
            # Skip the update when the batch already alarmed -- otherwise the
            # baseline chases the breakage and the alarm silences itself.
            if not any(a.field == field for a in alarms):
                if n == 0:
                    # Seed from the first observation rather than from zero.
                    # Starting at zero makes the EWMA climb toward the true rate
                    # over ~10 batches, and the variance term accumulates that
                    # climb as if it were noise -- which inflates the standard
                    # deviation enough to swallow a real collapse. The first
                    # sample is the best prior we have.
                    mean, var, n = rate, 0.0, 1
                else:
                    delta = rate - mean
                    mean += _ALPHA * delta
                    var = (1 - _ALPHA) * (var + _ALPHA * delta * delta)
                    n += 1
                self._state[source][field] = [mean, var, n]

        if alarms:
            for a in alarms:
                db.execute(
                    "INSERT INTO drift_log(ts, source, field, fill_rate, baseline, z, action) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (time.time(), a.source, a.field, a.fill_rate, a.baseline, a.z,
                     "quarantine"),
                )
            db.log_event(
                "warn", "drift",
                "; ".join(a.message for a in alarms),
                source,
            )
        db.kv_set(_KV, self._state)
        return alarms

    def baseline(self, source: str) -> dict[str, dict[str, float]]:
        return {
            f: {"mean": round(v[0], 3), "sd": round(math.sqrt(max(v[1], 0.0)), 3),
                "n": int(v[2])}
            for f, v in self._state.get(source, {}).items()
        }


monitor = DriftMonitor()
