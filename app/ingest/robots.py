"""robots.txt gate.

This is the technical half of the "where would you stop" answer. It is not
decoration: every outbound request in this system passes through `allowed()`,
and a disallowed path is not fetched, full stop. There is no override flag,
because a flag is what turns a policy into a suggestion.

Python ships `urllib.robotparser`, but it is synchronous and it has no notion
of `Crawl-delay`, which is the directive that actually matters for pacing. This
is a small async parser that handles the subset that is real: User-agent
grouping, Allow/Disallow with `*` and `$`, Crawl-delay, and longest-match-wins
precedence per RFC 9309.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from .. import db

_TTL = 3600.0            # re-read robots.txt hourly
_FETCH_TIMEOUT = 8.0


@dataclass
class Rules:
    allow: list[str] = field(default_factory=list)
    disallow: list[str] = field(default_factory=list)
    crawl_delay: float | None = None
    fetched_at: float = 0.0
    # A robots.txt we could not fetch is not a licence. 4xx means "no rules"
    # per the RFC; 5xx or a network error means "unknown", and unknown is
    # treated as disallowed for anything outside a small safe list.
    reachable: bool = True

    @property
    def stale(self) -> bool:
        return (time.time() - self.fetched_at) > _TTL


_cache: dict[str, Rules] = {}


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    out = []
    for ch in pattern:
        if ch == "*":
            out.append(".*")
        elif ch == "$":
            out.append("$")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out))


def parse(text: str, agent_token: str = "*") -> Rules:
    """Parse robots.txt, collecting the group for our token and falling back to
    the wildcard group. Directives outside any User-agent block are ignored, as
    the RFC says they should be."""
    groups: dict[str, Rules] = {}
    current: list[str] = []
    starting_group = False

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "user-agent":
            if not starting_group:
                current = []
                starting_group = True
            current.append(value.lower())
            groups.setdefault(value.lower(), Rules())
            continue

        starting_group = False
        for token in current:
            r = groups.setdefault(token, Rules())
            if key == "disallow" and value:
                r.disallow.append(value)
            elif key == "disallow":
                # "Disallow:" with an empty value means allow everything.
                r.allow.append("/")
            elif key == "allow" and value:
                r.allow.append(value)
            elif key == "crawl-delay":
                try:
                    r.crawl_delay = float(value)
                except ValueError:
                    pass

    chosen = groups.get(agent_token.lower()) or groups.get("*") or Rules()
    chosen.fetched_at = time.time()
    return chosen


def _match_len(path: str, patterns: list[str]) -> int:
    """Length of the longest matching pattern, or -1 for no match."""
    best = -1
    for p in patterns:
        if _pattern_to_regex(p).match(path):
            best = max(best, len(p))
    return best


async def rules_for(url: str, client: httpx.AsyncClient, agent_token: str = "*") -> Rules:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    cached = _cache.get(origin)
    if cached and not cached.stale:
        return cached

    try:
        resp = await client.get(
            f"{origin}/robots.txt", timeout=_FETCH_TIMEOUT,
            follow_redirects=True, headers={"accept": "text/plain,*/*"},
        )
        if resp.status_code >= 500:
            rules = Rules(fetched_at=time.time(), reachable=False)
        elif resp.status_code >= 400:
            # No robots.txt published -> no restrictions declared.
            rules = Rules(fetched_at=time.time(), reachable=True)
        else:
            rules = parse(resp.text, agent_token)
    except (httpx.HTTPError, UnicodeDecodeError) as exc:
        db.log_event("warn", "robots", f"could not read {origin}/robots.txt: {exc}")
        rules = Rules(fetched_at=time.time(), reachable=False)

    _cache[origin] = rules
    return rules


def allowed(url: str, rules: Rules) -> tuple[bool, str]:
    path = urlparse(url).path or "/"
    if not rules.reachable:
        return False, "robots.txt unreachable -- treating as disallowed"

    d = _match_len(path, rules.disallow)
    a = _match_len(path, rules.allow)
    if d < 0:
        return True, "no matching Disallow"
    if a >= d:
        # Longest match wins; ties go to Allow, per RFC 9309.
        return True, "Allow overrides Disallow (longer or equal match)"
    return False, f"Disallow matches {path}"


def clear_cache() -> None:
    _cache.clear()
