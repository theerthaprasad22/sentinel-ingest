"""Request pacing.

Two things get you caught on timing. The obvious one is volume, which a token
bucket fixes. The subtle one is *regularity*: a request every 300.0s forever
has zero variance, and no human produces zero variance. So on top of the bucket
there is lognormal jitter and a circadian weight, and the choice of how
aggressive to be is not hard-coded -- it is a Thompson-sampling bandit that
learns, per source, which pacing tier gets data without drawing a block.

The bandit is the honest answer to "what happens when the source changes its
rules next week": nobody has to notice and retune a constant.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from .. import db

# Pacing tiers the bandit chooses between. mean_gap is a multiplier on the
# source's configured cadence.
TIERS: dict[str, float] = {
    "aggressive": 0.55,
    "normal": 1.0,
    "cautious": 1.9,
    "stealth": 3.6,
}
TIER_NAMES = tuple(TIERS)


@dataclass
class TokenBucket:
    """Classic bucket, per host. Capacity is deliberately small: bursts are what
    trip rate limiters, and a large bucket just buys you the right to burst."""

    rate_per_s: float
    capacity: float
    tokens: float = 0.0
    updated: float = field(default_factory=time.monotonic)

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate_per_s)
        self.updated = now

    def take(self, n: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def wait_time(self, n: float = 1.0) -> float:
        self._refill()
        if self.tokens >= n:
            return 0.0
        return (n - self.tokens) / max(self.rate_per_s, 1e-9)


def circadian_weight(ts: float | None = None) -> float:
    """Multiplier on the gap between requests, shaped like human traffic.

    Cheap and defensible: a raised cosine peaking at ~14:00 UTC. At 03:00 the
    gap roughly doubles. A crawler that pulls the same volume at 4am as at 2pm
    stands out in any hourly-volume dashboard the target happens to keep.
    """
    hour = time.gmtime(ts if ts is not None else time.time()).tm_hour
    # 1.0 at the daytime peak, 2.0 at the nighttime trough.
    return 1.5 - 0.5 * math.cos((hour - 14) / 24 * 2 * math.pi)


class PaceBandit:
    """Beta-Bernoulli Thompson sampling over pacing tiers, per source.

    Reward = the fetch came back with usable data and the block classifier did
    not fire. Posteriors are persisted so a restart does not re-explore into a
    ban. Priors are asymmetric on purpose: `aggressive` starts pessimistic
    because the cost of a false positive there is an IP burn, while the cost of
    being too slow is only latency.
    """

    PRIOR: dict[str, tuple[float, float]] = {
        "aggressive": (1.0, 3.0),
        "normal": (2.0, 1.0),
        "cautious": (1.5, 1.0),
        "stealth": (1.0, 1.0),
    }

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)
        self._state: dict[str, dict[str, list[float]]] = db.kv_get("pace_bandit", {}) or {}

    def _post(self, source: str) -> dict[str, list[float]]:
        if source not in self._state:
            self._state[source] = {t: list(self.PRIOR[t]) for t in TIER_NAMES}
        return self._state[source]

    def choose(self, source: str) -> str:
        post = self._post(source)
        draws = {t: float(self._rng.beta(post[t][0], post[t][1])) for t in TIER_NAMES}
        return max(draws, key=draws.__getitem__)

    def update(self, source: str, tier: str, success: bool) -> None:
        post = self._post(source)
        if tier not in post:
            post[tier] = list(self.PRIOR.get(tier, (1.0, 1.0)))
        idx = 0 if success else 1
        post[tier][idx] += 1.0
        # Bounded memory: decay so a source that changes its policy is not
        # outvoted forever by six months of stale evidence.
        total = post[tier][0] + post[tier][1]
        if total > 60:
            post[tier][0] *= 0.7
            post[tier][1] *= 0.7
        db.kv_set("pace_bandit", self._state)

    def snapshot(self, source: str) -> dict[str, dict[str, float]]:
        post = self._post(source)
        out = {}
        for t in TIER_NAMES:
            a, b = post[t]
            out[t] = {
                "alpha": round(a, 2),
                "beta": round(b, 2),
                "mean": round(a / (a + b), 3),
                "n": round(a + b - sum(self.PRIOR[t]), 1),
            }
        return out


class Pacer:
    """Owns the buckets and the bandit; answers "how long before I may hit this
    source again, and how careful should I be about it"."""

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._rng = random.Random(99)
        self.bandit = PaceBandit()

    def bucket(self, host: str) -> TokenBucket:
        if host not in self._buckets:
            rate = settings.host_rpm_cap / 60.0
            self._buckets[host] = TokenBucket(rate_per_s=rate, capacity=max(2.0, rate * 20))
        return self._buckets[host]

    def gate(self, host: str) -> float:
        """Seconds to wait before this host may be touched. 0 == go now."""
        b = self.bucket(host)
        if b.take():
            return 0.0
        return b.wait_time()

    def next_gap(self, source: str, base_cadence_s: float, tier: str) -> float:
        """Gap until this source's next poll.

        base * tier * circadian, then lognormal jitter (sigma 0.35). Lognormal
        rather than uniform because real inter-arrival gaps are right-skewed:
        mostly short, occasionally long. Uniform jitter is still a flat
        distribution and still reads as synthetic.
        """
        mean = base_cadence_s * TIERS.get(tier, 1.0) * circadian_weight()
        jitter = self._rng.lognormvariate(0.0, 0.35)
        return max(5.0, mean * jitter)


pacer = Pacer()
