"""Identity management.

The naive version of "rotate your user agent" is worse than doing nothing.
Real fingerprinting joins the UA string against the headers that a browser of
that exact build would have sent -- Sec-CH-UA brand list, Accept ordering,
Sec-Fetch-* triplet, Accept-Language, and header *order*. A Chrome-131 UA
arriving with a Firefox Accept header and no client hints is a stronger bot
signal than a plain honest UA would have been.

So identities here are whole coherent profiles, they are pinned to a source for
a TTL (a session that changes browser mid-conversation is itself an anomaly),
and each one is bound to an egress slot so the (IP, fingerprint) pair stays
stable the way a real user's does.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ..config import settings


@dataclass(frozen=True)
class Profile:
    """A self-consistent browser identity. Values are taken from real shipped
    builds; nothing here is invented to look impressive."""

    key: str
    user_agent: str
    sec_ch_ua: str
    sec_ch_ua_platform: str
    accept_language: str
    accept: str
    accept_encoding: str = "gzip, deflate, br, zstd"
    platform_mobile: str = "?0"

    def headers(self, *, navigation: bool, contact: str) -> dict[str, str]:
        """Header set in the order Chromium actually emits it. dict preserves
        insertion order and httpx honours it, which matters because header
        order is part of the fingerprint."""
        h: dict[str, str] = {
            "sec-ch-ua": self.sec_ch_ua,
            "sec-ch-ua-mobile": self.platform_mobile,
            "sec-ch-ua-platform": self.sec_ch_ua_platform,
            "upgrade-insecure-requests": "1",
            "user-agent": self.user_agent,
            "accept": self.accept,
            "sec-fetch-site": "none" if navigation else "same-origin",
            "sec-fetch-mode": "navigate" if navigation else "cors",
            "sec-fetch-user": "?1",
            "sec-fetch-dest": "document" if navigation else "empty",
            "accept-encoding": self.accept_encoding,
            "accept-language": self.accept_language,
        }
        if not navigation:
            # A fetch() from page JS never carries these; sending them on an
            # XHR-shaped request is exactly the kind of incoherence that gets
            # flagged.
            for k in ("upgrade-insecure-requests", "sec-fetch-user"):
                h.pop(k, None)
        # Identify ourselves honestly on top of the browser profile. On the
        # sources this project actually ships against we lead with this: a
        # contactable crawler is the difference between "rate limited" and
        # "IP banned".
        h["from"] = contact
        return h


PROFILES: tuple[Profile, ...] = (
    Profile(
        key="chrome-win",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        sec_ch_ua_platform='"Windows"',
        accept_language="en-US,en;q=0.9",
        accept=("text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"),
    ),
    Profile(
        key="chrome-mac",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        sec_ch_ua_platform='"macOS"',
        accept_language="en-GB,en;q=0.9",
        accept=("text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"),
    ),
    Profile(
        key="firefox-win",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
            "Gecko/20100101 Firefox/133.0"
        ),
        # Firefox ships no client hints at all -- emitting them here would be
        # the tell. Empty strings are stripped before the request goes out.
        sec_ch_ua="",
        sec_ch_ua_platform="",
        accept_language="en-US,en;q=0.5",
        accept=("text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"),
        accept_encoding="gzip, deflate, br",
    ),
    Profile(
        key="safari-mac",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/18.1 Safari/605.1.15"
        ),
        sec_ch_ua="",
        sec_ch_ua_platform="",
        accept_language="en-US,en;q=0.9",
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        accept_encoding="gzip, deflate, br",
    ),
)


@dataclass
class Lease:
    """An identity bound to a source for a bounded lifetime."""

    profile: Profile
    egress: str | None
    issued_at: float
    ttl: float
    requests: int = 0

    @property
    def expired(self) -> bool:
        return (time.time() - self.issued_at) > self.ttl

    @property
    def label(self) -> str:
        egress = self.egress.split("@")[-1] if self.egress else "direct"
        return f"{self.profile.key}@{egress}"


@dataclass
class IdentityPool:
    """Hands out sticky leases per source and burns one on demand.

    `egress` is a list of proxy URLs. The demo runs with an empty list -- one
    IP, direct -- because the sources it is pointed at are public APIs that do
    not need evasion. The pool exists because the *design* needs an answer for
    what happens when one does, and that answer should be a config change, not
    a rewrite.
    """

    egress: list[str] = field(default_factory=lambda: list(settings.proxies))
    ttl: float = settings.identity_ttl_s
    _leases: dict[str, Lease] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=lambda: random.Random(1337))

    def _mint(self, source: str) -> Lease:
        profile = self._rng.choice(PROFILES)
        egress = self._rng.choice(self.egress) if self.egress else None
        # Jitter the TTL so identities from a restart do not all expire in
        # lockstep -- synchronised rotation is a pattern too.
        ttl = self.ttl * self._rng.uniform(0.7, 1.3)
        return Lease(profile=profile, egress=egress, issued_at=time.time(), ttl=ttl)

    def lease(self, source: str) -> Lease:
        current = self._leases.get(source)
        if current is None or current.expired:
            current = self._mint(source)
            self._leases[source] = current
        current.requests += 1
        return current

    def burn(self, source: str) -> Lease:
        """Discard the identity for a source after a block. Called by the
        fetcher when the block classifier fires, so the next attempt does not
        walk back in wearing the same face."""
        self._leases.pop(source, None)
        return self.lease(source)

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            src: {
                "identity": lease.label,
                "age_s": round(time.time() - lease.issued_at, 1),
                "requests": lease.requests,
            }
            for src, lease in self._leases.items()
        }


pool = IdentityPool()
