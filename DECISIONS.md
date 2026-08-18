# DECISIONS.md

**Sentinel Ingest** — Acdyon challenge, Part 1.
Live demo · [Repo](https://github.com/theerthaprasad22/sentinel-ingest) · full
design notes in [DESIGN.md](DESIGN.md).

### 1. Why this ingestion strategy over the obvious alternative I rejected?

The obvious alternative is **Playwright with stealth plugins**. I made it the
*bottom* rung rather than the first.

A headless browser is the largest detection surface available — CDP artefacts,
`navigator.webdriver`, canvas and WebGL hashes — and costs ~400MB RAM, which
does not fit free-tier hosting. Reaching for it first pays the maximum detection
cost to solve a problem you may not have. Most job data never needed one: boards
publish RSS for syndication and ship JSON APIs their own frontends call.

So each source is an ordered ladder — API → RSS → HTML → (browser, if it ever
came to that) — descending one rung per failure and climbing back one per clean
run. Pacing is not a hard-coded constant but a **Thompson-sampling bandit** over
four tiers, so when a source tightens its rules next week nobody has to notice
and retune anything.

I also rejected checking **status codes only**. The failure that actually costs
you is HTTP 200 with a CAPTCHA body: 100% "success", nothing ingested. That is
why there is a classifier at all.

### 2. One trade-off under the time limit, and what I'd do with a real week

**I did not solve the TLS fingerprint.** Python's `ssl` produces a JA3 that is
not Chrome's, and no amount of header work changes that. I did everything above
the transport layer properly — coherent identity profiles, Chromium header
order, HTTP/2, truthful `Accept-Encoding` — then stopped, because fixing TLS
means `curl-impersonate` or a real browser, which is a different deployment
shape. **This would not survive Cloudflare's bot-management tier**, and I would
rather say so than imply otherwise.

With a week: `curl-impersonate` as one more rung behind the same interface;
retrain the block classifier on **replayed real responses** instead of a
synthetic corpus (holdout precision near 1.0 tells you about the generator, not
the model — which is why `/api/ml` also reports a live confusion matrix against
ground truth); Postgres the moment there is more than one writer.

### 3. Where did I use AI tools, and what did I verify or change?

Claude wrote much of the scaffolding, the dashboard CSS, and the first synthetic
corpus. The shape of the code is AI-assisted; the decisions in it are not. Real
bugs the first draft shipped with:

- **Brotli.** It advertised `Accept-Encoding: br` with no decoder installed.
  Remotive serves brotli, so responses arrived as binary garbage, parsed to zero
  rows, and looked exactly like a soft block. Found by reading the fetch log,
  not by anything throwing.
- **Dedupe was confidently wrong.** Comparing full document vectors flagged 143
  near-duplicate pairs out of 388 — job descriptions are mostly the same words
  in a different order. Two rewrites: block by employer, then give dedupe its
  own narrow title-space.
- **304 scored as a block.** Conditional GET returns zero items, and the first
  version tripped a breaker on the best possible outcome.

The classifier's highest-weight feature, `log_median_items`, is one I added by
hand: an empty response is only suspicious *relative to that source's own
baseline*.

Every listing is either from a real public feed or badged **synthetic**. No
LinkedIn, Indeed or Naukri account was touched.

And the robots gate cost me a source in production: Remotive publishes a
documented public API, but their `robots.txt` says `Disallow: /api/*`, so the
gate refuses it and that source sits at zero rows on the live dashboard. I kept
the gate rather than adding an override — a policy with an exception list is not
a policy.
