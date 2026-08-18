# Sentinel Ingest

**A job-listing pipeline that expects to be blocked.**

Acdyon Technologies frontend challenge — **Part 1: getting data out of a
platform that doesn't want you to**.

Five sources, five ways in, and a hostile job board served by the application
itself so you can break the demo on purpose and watch it recover.

- 📊 **Live demo** → *(deployed URL)*
- 📐 **Design document** → [DESIGN.md](DESIGN.md) — detection surface, ingestion
  strategy, resilience, and where I'd stop
- 🧭 **Decisions** → [DECISIONS.md](DECISIONS.md) — one page, three questions

---

## The 60-second tour

Open the dashboard and do this:

1. Scroll to **Break it on purpose**.
2. Flip **captcha wall** and hit **poll sandbox now**.

The sandbox returns **HTTP 200** with a challenge body. A pipeline that checks
status codes records a success and ingests nothing. Here, the classifier scores
the body at `p_block ≈ 0.99`, the identity is burned, the circuit opens, and the
strategy ladder steps down to the next rung — all visible in the live event
feed.

Then try **markup version → v3** (hashed build-output class names, every stored
selector dead). The batch still comes back complete, but extraction confidence
falls from 100% to 20% and a warning fires — the signal you want *before* the
last fallback selector dies.

Or skip the clicking:

```bash
python scripts/demo.py https://your-deployed-url
```

That walks every failure mode in turn and prints what the pipeline did. Nothing
is staged — it drives the same public endpoints the dashboard uses.

---

## What's actually here

| | |
|---|---|
| **Coherent identities** | Whole browser profiles — UA × `Sec-CH-UA` × `Accept` × `Accept-Encoding`, in Chromium's header order. Firefox profiles ship *no* client hints, because Firefox doesn't. Rotating the UA alone is worse than doing nothing. |
| **Adaptive pacing** | Token bucket + lognormal jitter + circadian shaping, with a **Thompson-sampling bandit** learning each source's tolerable pace. Nobody has to retune a constant when a source tightens up. |
| **Soft-block detection** | Logistic regression over 21 response features. Catches the HTTP-200 CAPTCHA wall and the silent empty. Every prediction is attributable to the features that caused it. |
| **Schema-drift quarantine** | Per-field fill-rate EWMA with per-field variance. Alarms *and quarantines* rather than writing 500 rows where `company` is now null. |
| **Strategy ladder** | Each source is an ordered list of ways in: API → RSS → HTML. Descends one rung per failure, climbs back one per clean run. |
| **Circuit breaker** | Three states, persisted, exponential cooldown, single-probe recovery. Stops a soft rate-limit becoming a durable ban. |
| **Semantic layer** | TF-IDF (word 1-2gram ∪ char 3-5gram) → TruncatedSVD → cosine. Search by meaning, near-duplicate detection across boards, weak-supervision role/seniority tagging. |
| **robots.txt as a gate** | Every request passes through it. No override flag. Unreachable robots.txt = disallowed. Check it yourself: `GET /api/robots-check?url=…` |

**Stack:** Python 3.12 · FastAPI · httpx (HTTP/2) · lxml · scikit-learn ·
SQLite (WAL) · vanilla JS dashboard, no build step.

---

## Sources

Four live public feeds, one synthetic sandbox. **No LinkedIn, Indeed or Naukri
account was touched** — per the brief's scope guardrail.

| Source | Kind | Basis for access |
|---|---|---|
| [Remotive](https://remotive.com) | JSON API | Documented public API; asks for attribution, which the UI gives |
| [Arbeitnow](https://www.arbeitnow.com) | JSON API | Public keyless API published for third-party use |
| [We Work Remotely](https://weworkremotely.com) | RSS | Public feed published for syndication |
| [Hacker News](https://news.ycombinator.com/jobs) | JSON API | Algolia's public HN search API |
| `sandbox` | hostile HTML/JSON/RSS | Served by this app. Synthetic data, badged as such in the UI, the API and the database |

Every source states its basis for access on its dashboard card, and the test
suite fails the build if one doesn't.

---

## Running it

### Docker

```bash
docker build -t sentinel-ingest . && docker run -p 8000:8000 sentinel-ingest
```

### Local

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000>. The first poll lands within a minute; the
sandbox polls every ~45 seconds.

### Tests

```bash
pytest -q
```

98 tests. They cover identity coherence, robots precedence (including
longest-match-wins and fail-closed), classifier behaviour in *both* directions,
drift baselines, dedupe, and every sandbox defence end to end.

---

## API

Small and operational on purpose. Anyone can build a jobs endpoint; these are
the ones that let a stranger answer *"is this working, and how would it know if
it weren't?"*

| Endpoint | What it tells you |
|---|---|
| `GET /api/health` | Source health, clean-fetch rate, job counts |
| `GET /api/sources` | Per source: circuit state, current rung, identity, bandit posteriors, drift baselines |
| `GET /api/jobs?q=…` | Semantic search over stored postings |
| `GET /api/ml` | Model metrics — holdout **and** live confusion matrix, both with provenance |
| `GET /api/events/stream` | Server-sent scheduler events |
| `GET /api/robots-check?url=…` | The robots decision for any URL, without fetching it |
| `POST /sandbox/control` | Flip the sandbox's defences |
| `GET /docs` | OpenAPI |

---

## Deploying

`render.yaml` is a complete Render blueprint — Docker runtime, free plan, health
check wired up. Push the repo, point Render at it, and set `SENTINEL_CONTACT` to
a real address.

**One honest caveat about free tiers:** the instance sleeps after ~15 minutes of
inactivity, which stops the scheduler. On wake it cold-starts in ~40 seconds
(training the classifier on boot) and repopulates within one poll cycle. If you
want continuous ingestion, ping `/healthz` every 10 minutes or run it somewhere
that doesn't sleep.

---

## Layout

```
app/
  main.py              FastAPI app, dashboard, operational API
  config.py            env-driven settings
  db.py                SQLite schema, migrations, event log
  ingest/
    scheduler.py       the loop: cadence, ladder, failure policy
    fetcher.py         one request, start to finish
    identity.py        coherent browser profiles + egress leases
    pacing.py          token bucket, circadian shaping, Thompson bandit
    circuit.py         three-state breaker
    robots.py          async RFC 9309 parser + gate
    blockdetect.py     soft-block classifier
    drift.py           per-field fill-rate EWMA
    normalize.py       canonical identity, cleaning, date parsing
    store.py           upsert with touch/update/insert distinction
    adapters/          one file per source
  sandbox/server.py    the job board that fights back
  search/
    index.py           TF-IDF → SVD → cosine: search, dedupe, similarity
    tagger.py          weak-supervision role & seniority tagging
  web/                 dashboard (no build step)
tests/                 98 tests
scripts/demo.py        scripted walkthrough of every failure mode
```

---

*There is something hidden on the dashboard. It is not important.*
