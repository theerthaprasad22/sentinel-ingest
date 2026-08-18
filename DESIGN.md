# Sentinel Ingest — design document

Acdyon Technologies frontend challenge, **Part 1: getting data out of a platform
that doesn't want you to**.

The brief asks four questions. This document answers them in order, and every
claim it makes is executable against the running service — the endpoint or
command to check it is given inline.

---

## 0. What this is, in one paragraph

A job-listing ingestion service in Python/FastAPI that polls five sources, five
different ways, and keeps working when they start pushing back. It has a small
ML layer where ML actually earns its place: a supervised classifier that detects
*soft* blocks (HTTP 200 responses that are really CAPTCHA walls), a statistical
drift monitor that catches markup changes before they corrupt the database, a
Thompson-sampling bandit that learns each source's tolerable request pace, and a
TF-IDF→SVD semantic index for search and near-duplicate detection.

The fifth source is a **hostile sandbox served by the application itself**, with
fingerprinting, rate-limiting, CAPTCHA walls and three different DOM layouts on
switches you can flip from the dashboard. That is how the resilience claims in
this document get demonstrated instead of asserted.

```
                         ┌──────────────────────────────────────────┐
                         │            scheduler (asyncio)           │
                         │  per-source cadence · strategy ladder    │
                         └───────────────────┬──────────────────────┘
                                             │
   ┌──────────────┬──────────────┬───────────┴──────┬─────────────────┐
   ▼              ▼              ▼                  ▼                 ▼
 robots        pacing        identity          conditional         circuit
 gate          bucket +      lease             GET (ETag)          breaker
 (RFC 9309)    bandit        (coherent FP)     304 = success       3-state
   │              │              │                  │                 │
   └──────────────┴──────────────┴────────┬─────────┴─────────────────┘
                                          ▼
                                    HTTP/2 request
                                          │
                            ┌─────────────┴──────────────┐
                            ▼                            ▼
                     block classifier              adapter parser
                     (logistic regression)         (API │ RSS │ HTML)
                            │                            │
                            └─────────────┬──────────────┘
                                          ▼
                                   drift monitor
                                 (per-field EWMA)
                                          │
                            ┌─────────────┴──────────────┐
                            ▼                            ▼
                      quarantine batch            normalise → dedupe
                      keep last good rows         → SQLite → semantic index
```

---

## 1. Detection surface

### What actually gives an automated client away

Ordered roughly by how cheap they are for the defender to check, which is also
the order they get used:

| Layer | Signal | What a naive scraper does wrong |
|---|---|---|
| **Transport** | TLS JA3/JA4 hash, ALPN list, HTTP/2 SETTINGS frame + header-table size | Speaks HTTP/1.1 only, or presents a TLS fingerprint no browser produces |
| **Headers** | `User-Agent` × `Sec-CH-UA` coherence, header **order**, `Accept` specificity, `Sec-Fetch-*` triplet, `Accept-Encoding` | Rotates the UA string and nothing else — a Chrome-131 UA with no client hints is a *stronger* signal than an honest one |
| **Session** | Cookie continuity, whether an identity persists or changes per request, whether the client ever fetched the assets a browser would | Fresh session every request; requests the JSON endpoint the page calls, but never the page |
| **Timing** | Inter-arrival regularity, requests-per-minute, hour-of-day distribution | Perfectly even 300.0s gaps; same volume at 04:00 as at 14:00 |
| **Network** | ASN reputation (datacentre vs residential), IP request volume, rDNS | One cloud IP doing 10k requests/day |
| **Behaviour** | Depth-first crawling of every page in order, no referer chain, no mouse/JS execution | Enumerates `?page=1..500` in sequence |

### What this design accounts for

Everything above the network line, honestly and completely — that is what a
single application process can control:

- **Coherent identity profiles** ([`app/ingest/identity.py`](app/ingest/identity.py)).
  A profile is a whole browser: UA + `Sec-CH-UA` brand list + platform +
  `Accept` + `Accept-Language` + `Accept-Encoding`, emitted **in Chromium's
  header order** (dicts preserve insertion order and httpx honours it). Firefox
  and Safari profiles ship *no* client hints, because those browsers don't —
  sending them would be the tell. A test pins this invariant:
  `tests/test_defences.py::TestIdentity::test_every_profile_is_internally_coherent`.
- **HTTP/2** ([`fetcher.make_client`](app/ingest/fetcher.py)). Every real
  browser negotiates h2. An HTTP/1.1-only client is a fingerprint by itself.
- **Truthful `Accept-Encoding`**. This was a live bug during development:
  advertising `br` without a brotli decoder made Remotive's responses arrive as
  binary garbage, parse to zero rows, and look like a block. Claiming a
  capability you don't have is worse than not claiming it.
- **Sticky sessions.** An identity is leased *per source* for a jittered TTL
  (~15 min), not rotated per request. A client that changes browser mid-session
  is anomalous in a way a stable one is not.
- **Pacing that looks human** ([`app/ingest/pacing.py`](app/ingest/pacing.py)).
  Token bucket for volume, **lognormal** jitter for shape (real inter-arrival
  gaps are right-skewed; uniform jitter is still a flat distribution and still
  reads as synthetic), and a circadian weight that roughly doubles the gap at
  03:00 UTC.

### What it does *not* account for, and why

- **TLS fingerprint.** Python's `ssl` module produces a JA3 that is not
  Chrome's, and no amount of header work changes that. Fixing it properly means
  `curl-impersonate` or a real browser under Playwright — a different deployment
  shape and 400MB of image. It is the first thing I would add for a source that
  checks it, and the honest position is that this demo would not survive
  Cloudflare's bot-management tier.
- **IP diversity.** The pool abstraction exists
  ([`IdentityPool.egress`](app/ingest/identity.py)) and takes proxies from
  `SENTINEL_PROXIES`, but the live demo runs on one datacentre IP, direct. The
  sources it points at are public APIs that do not need evasion, and paying for
  residential proxies to demo against an endpoint that welcomes traffic would be
  theatre.
- **JS execution.** No headless browser. Everything here targets JSON, RSS and
  server-rendered HTML on purpose — see §2.

---

## 2. Ingestion strategy

### The core decision: climb *down* the stack, not up

The obvious approach to "get data off a job board" is Playwright with stealth
plugins. I rejected it as the primary strategy, and this is the trade I would
defend hardest:

|  | Headless browser | This design |
|---|---|---|
| Resource cost | ~400MB RAM per instance | ~90MB total |
| Detection surface | Largest — CDP artefacts, `navigator.webdriver`, canvas/WebGL, font metrics, timing | Smallest — no JS runtime to fingerprint |
| Fragility | Breaks on any DOM change | RSS/API contracts survive redesigns |
| Free-tier viability | No | Yes |

A headless browser is a *last* resort — the rung below HTML parsing — not the
first. Most job data is reachable without one: boards publish RSS for
syndication, ship JSON APIs their own frontends call, and server-render their
listing pages for SEO. Reaching for a browser first means paying the maximum
detection cost to solve a problem you may not have.

### The strategy ladder

Every source is an **ordered list of ways to get the same data**
([`Adapter.strategies`](app/ingest/adapters/base.py)). The scheduler starts at
the cheapest rung, descends one rung per failure, and climbs back one rung at a
time after a clean run:

| Source | Rung 1 | Rung 2 | Rung 3 |
|---|---|---|---|
| `remotive` | JSON API | narrowed API query | — |
| `arbeitnow` | JSON API | page 2 | — |
| `weworkremotely` | firehose RSS | per-category RSS (separate CDN cache key) | — |
| `hackernews` | Algolia by date | Algolia by relevance | — |
| `sandbox` | HTML scrape | JSON API | RSS |

Climbing back **one rung at a time** matters: if HTML broke because the site
redesigned, jumping straight back to it re-breaks the pipeline every cycle.

### Adaptive pacing (Thompson sampling)

Rather than hard-coding "one request every 5 minutes", each source has a
Beta-Bernoulli posterior over four pacing tiers — `aggressive` (0.55× cadence)
through `stealth` (3.6×). Reward is "got usable data and the block classifier
did not fire". Priors are asymmetric: `aggressive` starts pessimistic (α=1, β=3)
because the cost of being wrong there is an IP burn, while the cost of being too
slow is only latency.

Posteriors are persisted, so a restart doesn't re-explore into a ban, and they
**decay** past 60 observations so a source that changes its policy isn't
outvoted forever by stale evidence. Visible per source on the dashboard, and at
`GET /api/sources` → `pace_posterior`.

**This is the plan-B answer.** When a source tightens its rules next week,
nobody has to notice and retune a constant — the posterior shifts within a few
polls.

### Bandwidth and courtesy

- **Conditional GET**: `ETag`/`If-Modified-Since` stored per source. A **304 is
  scored as a success**, not as "zero items, must be blocked" — a subtle bug
  that would trip a breaker on the best possible outcome.
- Validators are only stored after a batch we *trust*, so a poisoned response
  cannot install an ETag that suppresses the next real fetch.
- A `From:` header carrying a contact address rides on top of the browser
  profile. A contactable crawler gets rate-limited; an anonymous one gets banned.

### Plan B when the primary approach dies

1. **Same source, next rung** — automatic, one poll's latency.
2. **Different pacing tier** — automatic, the bandit reallocates within a few polls.
3. **Fresh identity** — automatic; the identity is burned the moment the classifier fires.
4. **Egress rotation** — config change (`SENTINEL_PROXIES`), no code change.
5. **New source** — one adapter file and one line in `registry.py`. Coverage is
   a portfolio problem: five sources means no single block is an outage. This is
   the real answer to "what if it gets shut down in a week", and it is why the
   adapter interface is the smallest thing in the codebase.
6. **Publisher relationship** — for anything that matters commercially, an
   email asking for API access is cheaper than an arms race. See §4.

---

## 3. Resilience

Three distinct failure modes, three distinct mechanisms. The point of separating
them is that they need *different* responses: a block means back off, a drift
means fall back, and an outage means wait.

### 3a. Soft blocks — the ML classifier

Hard blocks (403, 429, connection reset) need no model. The expensive failure is
the **soft block**: HTTP 200, correct content-type, plausible length, body is a
challenge page or a listing that silently returned zero rows. A pipeline
checking status codes reports 100% success while ingesting nothing.

[`app/ingest/blockdetect.py`](app/ingest/blockdetect.py) is a **logistic
regression over 21 response features** — body length, content-type match,
vendor-phrase counts, challenge cookies, tag density, redirect depth, latency
relative to the source's own median, and item count relative to the source's own
median.

Logistic regression on purpose: the coefficients are readable and every
prediction is attributable. `classifier.explain()` returns
standardised-value × coefficient per feature, which is what turns
`block_prob=0.91` into *"because `has_captcha_token` and `items_vs_median`"* —
an operator cannot act on a bare number.

**The most interesting learned weight is `log_median_items`**, and it encodes
the insight that makes the whole thing work: *an empty response is only
suspicious relative to the source's own baseline*. A niche board whose median
poll is two jobs returning none is a slow week. The identical body from a source
whose median is 25 is a block. Nothing inside the response separates them.

**On the metrics, honestly.** Holdout precision/recall sit near 1.0. That is not
a brag — it means the synthetic corpus is close to separable by construction,
and it is the number I trust *least*. The corpus deliberately includes
confusable classes (security-job listings stuffed with block vocabulary,
tiny-but-valid API responses, terse unbranded block pages, legitimately-quiet
feeds) and the score barely moves, which tells you the generator, not the model.

So the service also reports a **live confusion matrix**: every sandbox response
carries an `x-sandbox-truth` header, and `GET /api/ml` →
`block_classifier_live` scores predictions against it on real traffic. That is
still synthetic blocks — no real vendor challenge page is in there, and the
endpoint says so in its own payload.

### 3b. Markup drift — statistical, not exception-based

A site redesign doesn't raise an exception. The parser finds *something*, and
500 rows land where `company` is now null.

Two layers:

- **Candidate selectors.** Each field has an ordered list of XPaths.
  `first_text()` returns which candidate matched, and
  `extraction_confidence` = fraction of fields that matched their *first*
  candidate. When the sandbox switches from markup v1 to v3 (hashed build-output
  class names), the batch stays complete but confidence falls from **100% to
  20%** and a warning fires. That is the signal you want *before* the last
  candidate dies — not after.
- **Per-field fill-rate EWMA** ([`app/ingest/drift.py`](app/ingest/drift.py)).
  Each `(source, field)` carries its own mean and variance, so a field that has
  always been 40% populated doesn't alarm at 40%, and a field that has been 99%
  populated for a month alarms hard at 60%. HN job posts have no location field
  at all; the monitor learns that and stays quiet.

  Two details that took a bug each to get right: the baseline is **seeded from
  the first observation** rather than from zero (starting at zero makes the EWMA
  climb toward the truth over ~10 batches, and the variance term absorbs that
  climb as noise, inflating the standard deviation enough to swallow a real
  collapse), and the baseline is **not updated on a batch that alarmed** —
  otherwise it chases the breakage and silences its own alarm.

On alarm the batch is **quarantined, not written**. Last-good rows keep serving,
the source is marked degraded, and the scheduler descends a rung — because
another rung usually has a stable contract, and an API does not redesign its
markup.

### 3c. Blocks and outages — circuit breaker

Three-state breaker per source, persisted to SQLite so a restart doesn't reset a
breaker that was open for a good reason. Cooldown grows with consecutive
failures (capped at 30 min); recovery is a **single half-open probe**, not a
resumption of full rate. Unambiguous evidence (403, or a `Retry-After` we intend
to honour) trips it immediately rather than after N strikes.

The failure this prevents isn't "a request failed" — it's the retry loop that
converts a soft rate-limit into a durable ban.

### 3c. Why the semantic layer is not a transformer

Moved here from DECISIONS.md, which the brief caps at one page.

A MiniLM checkpoint plus torch is ~900MB of image for a corpus of a few thousand
short job postings, on a host with 512MB of RAM. TF-IDF into a truncated SVD
gives vectors that behave the same way *for this corpus*, trains in under a
second on every restart, needs no model download, and can be explained end to
end. Two vectorisers are unioned: word 1-2 grams for the semantics ("machine
learning engineer" vs "ml engineer") and char 3-5 grams for the robustness
(typos, "K8s"/"Kubernetes", "Sr."/"Senior").

At ten times the corpus I would revisit it. The honest limit is that this
embedding has no world knowledge: it finds "deep learning" from a PyTorch
description because those words co-occur in *this* corpus, not because it knows
what PyTorch is.

### 3c-bis. Near-duplicate detection, and the version of it that was wrong

Worth documenting because the first implementation *looked* like it worked.

Postings repeat: the same role appears on three boards, and gets re-listed
monthly with a new URL. Exact identity is content-derived
(`normalize.canonical_id`), and the leftovers were supposed to be caught by
cosine similarity in the SVD space.

That version flagged **143 pairs out of 388 postings**, and the pairs were
nonsense — "Site Reliability Engineer" at Acme against "Site Reliability
Engineer" at Globex. The document being compared included the description, and
across a corpus of job ads the descriptions are mostly the same words in a
different order. Blocking by employer fixed the cross-company case and made the
comparison closer to linear rather than O(n²) — but *within* one employer the
boilerplate is near-identical, and it scored "Head of Marketing" against "Head
of Design" at **0.99**.

The fix is that search and dedupe want opposite things from a document:

- **Search** wants context. The description is what makes the query "deep
  learning" find a PyTorch role that never uses the phrase.
- **Dedupe** wants only the fields that distinguish postings *at the same
  employer* — title and location, nothing else.

So there are two vector spaces. Search uses TF-IDF → SVD over the full
document; dedupe uses a separate sparse title-space, blocked by normalised
company, with a guard that refuses to collapse different seniority levels
("Software Engineer" and "Senior Software Engineer" at one company are two
vacancies, not one). Threshold calibrated by inspection against the live
corpus: 0.75 keeps three pairs a human agrees with; 0.70 starts pairing
"Software Engineer" with "Data Engineer".

Known limitation, stated rather than hidden: a re-listing whose title was
*substantially rewritten* is not caught. Between missing duplicates and
inventing them, missing is the cheaper error — a wrongly-flagged duplicate
hides a real vacancy from the reader.

### 3d. Everything else

- Retries use **decorrelated jitter** (`min(30, uniform(1, prev*3))`). Plain
  exponential backoff synchronises every client that failed at the same instant
  into the same retry instant.
- The scheduler **never raises**: `poll_source` is wrapped, unhandled exceptions
  become "that source is unhealthy". A dead scheduler is the failure nobody
  notices until the counter has been flat for a day.
- A parser exception is treated as a **drift signal**, not a crash.
- One malformed record must not cost a good batch — `arbeitnow` returns
  `job_types` as a dict on ~2 of 176 rows, which took down the entire batch until
  it was coerced.
- Identity is **burned** on any non-clean verdict. Walking back in with the same
  fingerprint after a challenge is how a soft block becomes a durable one.

**Try it:** `python scripts/demo.py <url>` flips every defence in turn and prints
what the pipeline did. Nothing is staged — it drives the same public endpoints
the dashboard uses.

---

## 4. Where I'd stop

### The technical line, enforced in code

- **robots.txt is a gate, not a suggestion.** Every outbound request passes
  through `robots.allowed()`. There is no override flag, because a flag turns a
  policy into a preference. `Crawl-delay` is honoured over our own faster bucket
  rate. An **unreachable** robots.txt is treated as *disallowed*: an unreadable
  policy is not permission.

  Verifiable without reading any code:
  `GET /api/robots-check?url=<origin>/sandbox/private/jobs` → `allowed: false`.
- **No authentication is ever forged.** Nothing here logs in, holds a session
  cookie for a logged-in user, or touches content behind a paywall. Every source
  is public and unauthenticated.
- **No CAPTCHA solving.** When the classifier says "challenge", the response is
  to back off, not to route the image to a solving service. Solving a CAPTCHA is
  unambiguously circumventing a stated access-control decision.
- **Every source states its basis for access** in the adapter and on its
  dashboard card. `tests/test_pipeline.py` fails the build if a source has no
  `licence_note`. If you can't say why you're allowed to be there, you aren't.

### The personal line

The brief is right that every major platform has ToS against scraping, and I'm
not going to pretend those documents are ambiguous. Where I draw it:

- **Public, unauthenticated, published-for-syndication** — RSS feeds, documented
  APIs, `jobs.json`. This is what the demo does. Attribution and a contact
  header are the price, and both are paid.
- **Public but not offered** — server-rendered listing pages with no feed. I'll
  read them at a rate no human would notice, honour robots.txt, and never
  behind auth. This is the grey zone and I'd want it to be a considered
  decision, not a default.
- **Behind a login, or explicitly ToS-prohibited with enforcement** — LinkedIn,
  Indeed, Naukri. I would not run this against them, which is why **the live
  demo doesn't**, per the brief's own scope guardrail. Not because it is
  technically impossible, but because the downside lands on a *person* — the
  account holder gets banned, or the company gets a letter — and "the scraper
  worked" is no defence.
- **Personal data** — I ingest company, role, location, link. Not applicant
  data, not recruiter contact details, not anything that would make this a
  GDPR conversation.

The design respects that line while still doing the job, because coverage comes
from **breadth over depth**: five permitted sources with good uptime beat one
prohibited source you have to fight. The moment the answer requires forging a
login, it has stopped being an engineering problem and become a legal one, and
the cheaper move is an email to the publisher.

---

## Appendix: what a reviewer can check in two minutes

| Claim | Command |
|---|---|
| Every defence works | `python scripts/demo.py <url>` |
| robots is enforced | `GET /api/robots-check?url=<url>/sandbox/private/jobs` |
| Blocks are detected at HTTP 200 | Dashboard → toggle **captcha wall** → **poll sandbox now** |
| Markup drift is survived | Dashboard → markup **v3** → **poll sandbox now** |
| Model metrics are real | `GET /api/ml` — holdout *and* live confusion, both with provenance |
| Fingerprint coherence matters | `curl <url>/sandbox/jobs` with fingerprint check on → 403 |
| It's tested | `pytest -q` → 98 tests |
