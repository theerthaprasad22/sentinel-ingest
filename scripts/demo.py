"""Scripted walkthrough of every failure mode, against the live service.

Run it against a local server or the deployed URL:

    python scripts/demo.py                       # http://127.0.0.1:8000
    python scripts/demo.py https://your-app.onrender.com

It flips one sandbox defence at a time and prints what the pipeline did about
it. Nothing here is staged -- it drives the same public endpoints the dashboard
uses, so every line of output is a real request and a real verdict.
"""
from __future__ import annotations

import json
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
client = httpx.Client(base_url=BASE, timeout=90.0)


def call(method: str, path: str, **kw) -> dict:
    """One request, retried through free-tier blips.

    A free instance occasionally answers with the platform's own 502 while it
    is being rescheduled. That is not the pipeline failing, and a walkthrough
    script that dies on it tells you nothing -- so transient non-JSON responses
    are retried rather than raised.
    """
    last = ""
    for attempt in range(5):
        try:
            r = client.request(method, path, **kw)
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(3 * (attempt + 1))
    print(f"  (gave up on {method} {path}: {last})")
    return {}


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * max(len(title), 40))


def control(**flags) -> dict:
    return call("POST", "/sandbox/control", json=flags).get("state", {})


def reset() -> None:
    control(reset=True)
    call("POST", "/api/sources/sandbox/reset")


def poll() -> str:
    r = call("POST", "/api/sources/sandbox/poll")
    if not r:
        return "  (no response -- instance busy, skipping this probe)"
    if "skipped" in r:
        return f"  SKIPPED    {r['skipped']} (retry in {r.get('retry_in')}s)"
    conf = r.get("extraction_confidence")
    return (
        f"  {str(r.get('strategy')):9} verdict={str(r.get('verdict')):9} "
        f"http={r.get('status')} items={r.get('items')} "
        f"p_block={r.get('block_prob')}"
        + (f" extraction={conf:.0%}" if isinstance(conf, (int, float)) else "")
    )


def main() -> None:
    health = call("GET", "/api/health")
    print(f"Connected to {BASE} -- {health['sources_healthy']}/{health['sources_total']} "
          f"sources healthy, {health['jobs']['total']} jobs stored")

    rule("0. Baseline: the sandbox behaving itself")
    reset()
    print(poll())

    rule("1. CAPTCHA wall -- HTTP 200 with a challenge body")
    print("   A status-code-only pipeline records this as a success.")
    reset()
    control(captcha_wall=True)
    for _ in range(3):
        print(poll())
    print("   -> classifier scores the body, identity is burned, circuit opens.")

    rule("2. Silent empty -- 200, right content-type, zero rows")
    reset()
    control(silent_empty=True)
    for _ in range(2):
        print(poll())

    rule("3. Hard block 403 -- the strategy ladder descends")
    reset()
    control(hard_block=True)
    for _ in range(3):
        print(poll())
    src = next(s for s in call("GET", "/api/sources") if s["name"] == "sandbox")
    print(f"   -> now on rung '{src['current_strategy']}', circuit {src['breaker']}")

    rule("4. Markup v3 -- hashed class names, every stored selector dead")
    reset()
    control(markup_version=3)
    print(poll())
    print("   -> the batch is still complete, but extraction confidence falls from")
    print("      100% to 20%: every field is now coming from a fallback selector.")
    print("      That is the warning you want *before* the last candidate dies.")

    rule("5. Rate limit -- 429 with Retry-After")
    reset()
    control(rate_limit=True, rate_limit_rpm=1)
    for _ in range(3):
        print(poll())

    rule("6. Fingerprint check -- can our own client still get in?")
    reset()
    control(fingerprint_check=True)
    print(poll())
    print("\n   The same endpoint, hit by less careful clients:")
    probes = [
        ("httpx defaults", {}),
        ("Chrome UA, no client hints", {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "accept-language": "en-US,en;q=0.9",
            "accept": "text/html,application/xhtml+xml",
        }),
        ("Firefox UA *with* client hints", {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
                          "Gecko/20100101 Firefox/133.0",
            "sec-ch-ua": '"Google Chrome";v="131"',
            "accept-language": "en-US,en;q=0.5",
            "accept": "text/html,application/xhtml+xml",
        }),
    ]
    for label, headers in probes:
        r = httpx.get(f"{BASE}/sandbox/jobs", headers=headers, timeout=30.0)
        reason = r.headers.get("x-sandbox-reason", "allowed")
        print(f"   {label:32} -> {r.status_code}  {reason}")

    rule("7. robots.txt is enforced, not merely claimed")
    for path in ("/sandbox/private/jobs", "/sandbox/jobs"):
        r = call("GET", "/api/robots-check", params={"url": BASE + path})
        print(f"   {path:24} allowed={str(r['allowed']):5} -- {r['reason']}")

    rule("8. Block classifier scored on ground truth, not on its training set")
    reset()
    for _ in range(2):
        poll()
    control(captcha_wall=True)
    for _ in range(2):
        poll()
    reset()
    time.sleep(1)
    print(json.dumps(call("GET", "/api/ml").get("block_classifier_live", {}), indent=2))

    reset()
    print("\nDefences reset. Dashboard: " + BASE)


if __name__ == "__main__":
    main()
