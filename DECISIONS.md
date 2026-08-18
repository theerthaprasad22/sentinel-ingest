# DECISIONS.md

**Sentinel Ingest** — Acdyon frontend challenge, Part 1. Live demo · repo ·
full design notes in [DESIGN.md](DESIGN.md).

---

### 1. Why this ingestion strategy over the obvious alternative I rejected?

The obvious alternative is **Playwright with stealth plugins**. I rejected it as
the *primary* strategy and made it the bottom rung instead.

A headless browser is the largest possible detection surface — CDP artefacts,
`navigator.webdriver`, canvas and WebGL hashes, font metrics, timing
irregularities — and it costs ~400MB of RAM per instance, which does not fit on
free-tier hosting. Reaching for it first means paying the maximum detection cost
to solve a problem you may not have. Most job data never needed a browser: boards
publish RSS for syndication, ship JSON APIs their own frontends call, and
server-render listings for SEO.

So each source is an **ordered ladder** — API → RSS → HTML → (browser, if it ever
came to that) — and the scheduler descends one rung per failure and climbs back
one rung per clean run. On top of it, pacing is not a hard-coded constant but a
**Thompson-sampling bandit** over four tiers, so when a source tightens its rules
next week nobody has to notice and retune anything; the posterior shifts within a
few polls. Priors are asymmetric because the cost of being too aggressive (an IP
burn) is not symmetric with the cost of being too slow (latency).

The second rejected alternative was checking **status codes only**. The failure
that actually costs you is HTTP 200 with a CAPTCHA body — the pipeline reports
100% success and ingests nothing. That is why there is a classifier at all.

---

### 2. One trade-off I made under the time limit, and what I'd do with a real week

**The trade-off: I did not solve the TLS fingerprint.**

Python's `ssl` module produces a JA3/JA4 hash that is not Chrome's, and no amount
of header work changes that. I did everything above the transport layer properly
— coherent identity profiles where a Firefox UA correctly ships *no* client
hints, Chromium header ordering, HTTP/2, truthful `Accept-Encoding` — and then
stopped, because fixing TLS properly means `curl-impersonate` or a real browser,
which is a different deployment shape. **This demo would not survive
Cloudflare's bot-management tier**, and I would rather say that than imply
otherwise.

With a real week, in order:

1. `curl-impersonate` behind the same `Strategy` interface — it slots in as one
   more rung, no other code changes.
2. Move the block classifier from a synthetic corpus to **replayed real
   responses**. The holdout precision/recall are near 1.0 and that number tells
   you about the generator, not the model. I mitigated it by adding a live
   confusion matrix against the sandbox's own ground-truth header, but that is
   still synthetic blocks.
3. Postgres instead of SQLite the moment there is more than one writer — one
   file changes (`app/db.py`).
4. An approximate nearest-neighbour index. Dedupe is blocked by employer so it
   is already close to linear, but *search* still scores the query against every
   stored vector. Fine at a few thousand postings; the first thing to break at
   fifty thousand.

**The smaller trade:** TF-IDF → TruncatedSVD instead of sentence-transformers for
the semantic layer. MiniLM plus torch is ~900MB of image for a corpus of a few
thousand short job postings on a host with 512MB of RAM. For this corpus the
cheap embedding behaves the same way, trains in under a second on every restart,
and needs no model download. At ten times the corpus I would revisit it.

---

### 3. Where did I use AI tools, and what did I personally verify or change?

I used Claude heavily — for scaffolding the FastAPI/adapter boilerplate, drafting
the dashboard CSS, and generating the first pass of the synthetic block corpus.
Roughly the shape of the code is AI-assisted; the decisions in it are not.

What I changed or caught afterwards, all of which are real bugs the first draft
shipped with:

- **The brotli bug.** The generated code advertised `Accept-Encoding: gzip,
  deflate, br` without a brotli decoder installed. Remotive serves brotli, so
  responses arrived as binary garbage, parsed to zero rows, and looked exactly
  like a soft block. I found it by reading the fetch log, not by it throwing.
  Claiming a capability you don't have is worse than not claiming it — and the
  fix made the header *truthful*, which is also the point of the whole identity
  module.
- **The EWMA warm-up bug.** The drift monitor initialised its baseline at zero,
  so the mean climbed toward the true fill rate over ~10 batches while the
  variance term absorbed that climb as if it were noise. The inflated standard
  deviation was large enough to swallow a real field collapse. A test caught it;
  seeding from the first observation fixed it.
- **304 scored as a block.** Conditional GET returns zero items, and the first
  version treated that as "empty, therefore suspicious" — tripping a breaker on
  the best possible outcome.
- **A dict where a list belonged.** `arbeitnow` returns `job_types` as a dict on
  about 2 of 176 rows. One malformed record took down the entire batch.
- **The classifier's most important feature is one I added by hand.**
  `log_median_items` — an empty response is only suspicious *relative to the
  source's own baseline*. A niche board whose median poll is two jobs returning
  none is a slow week; the identical body from a source whose median is 25 is a
  block. Nothing inside the response separates them. It is now the highest-weight
  feature in the model.
- **The dedupe pass was confidently wrong.** The generated version compared
  full document vectors and flagged 143 near-duplicate pairs out of 388
  postings — "Site Reliability Engineer" at one company against the same title
  at another, because job descriptions are mostly the same words in a different
  order. It took two rewrites: block by employer (which also turns an O(n²) scan
  into something near-linear), then give dedupe its *own* narrow title-space,
  because within one employer the boilerplate is identical and full documents
  scored "Head of Marketing" against "Head of Design" at 0.99. The threshold is
  calibrated by inspection against the live corpus, not guessed.
- **I rewrote the metrics story.** The first version reported holdout scores as
  though they meant something. They mostly measure the corpus generator, so I
  added the live confusion matrix, and both numbers now ship with a provenance
  string in the API payload itself.

On the honesty constraint: every listing on the dashboard is either from a real
public feed (Remotive, Arbeitnow, We Work Remotely, Hacker News) or comes from
the sandbox and is **badged synthetic** in the UI, the API and the database. No
LinkedIn, Indeed or Naukri account was touched.
