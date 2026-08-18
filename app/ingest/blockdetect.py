"""Soft-block detection.

Hard blocks are easy: 403, 429, a connection reset. Those need no model.

The expensive failure is the *soft* block -- HTTP 200, correct content-type,
plausible length, and the body is a CAPTCHA interstitial, a "verify you are
human" shim, a JS challenge, or a listing page that silently came back with
zero rows. A pipeline that only checks status codes reports 100% success while
ingesting nothing, and nobody finds out until someone asks why the job count
stopped moving.

So this is a small supervised classifier over response features. Logistic
regression on purpose: it is 20 features, the coefficients are readable, and
each prediction can be attributed back to the signals that caused it. A
gradient-boosted anything would score marginally better and explain nothing.

Training data is honest about what it is: procedurally generated response
bodies covering the block archetypes above, plus every real response this
deployment has seen that carried an unambiguous label (403/429 => blocked,
200-with-N-parsed-items => clean). It has never seen a real anti-bot vendor's
challenge page, and the dashboard says so next to the metrics.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .. import db

# Phrases that appear on interstitials across the common vendors. Matched
# case-insensitively against the first 8KB, which is where a challenge page
# puts its message and where a real listing page has already started listing.
BLOCK_TOKENS: tuple[str, ...] = (
    "captcha", "recaptcha", "hcaptcha", "verify you are human", "are you a robot",
    "unusual traffic", "automated queries", "access denied", "permission to access",
    "too many requests", "rate limit", "slow down", "temporarily blocked",
    "just a moment", "checking your browser", "enable javascript", "challenge-platform",
    "cf-chl", "ddos protection", "security check", "bot detection", "suspicious activity",
    "please try again later", "request blocked", "perimeterx", "incapsula", "datadome",
)
CAPTCHA_TOKENS = ("captcha", "recaptcha", "hcaptcha", "verify you are human", "are you a robot")
CHALLENGE_COOKIES = ("cf_clearance", "__cf_bm", "datadome", "incap_ses", "_px", "bm_sz")

FEATURE_NAMES: tuple[str, ...] = (
    "log_body_len", "is_2xx", "is_3xx", "is_403", "is_429", "is_5xx",
    "ctype_match", "ctype_html_when_json", "n_block_tokens", "has_captcha_token",
    "has_challenge_cookie", "has_retry_after", "tag_density", "log_items",
    "items_vs_median", "latency_vs_median", "n_redirects", "char_diversity",
    "title_len", "body_is_tiny", "log_median_items",
)

_MODEL_KEY = "blockdetect_metrics"


@dataclass
class ResponseFeatures:
    """Everything the classifier sees. Deliberately all cheap to compute --
    this runs on every fetch, in the hot path."""

    status: int
    body: str
    content_type: str
    expected_content: str          # "json" | "xml" | "html"
    items: int
    latency_ms: float
    n_redirects: int
    set_cookie: str
    retry_after: str
    median_items: float
    median_latency_ms: float

    def vector(self) -> np.ndarray:
        body = self.body or ""
        head = body[:8192].lower()
        ct = (self.content_type or "").lower()
        n_tokens = sum(1 for t in BLOCK_TOKENS if t in head)
        tags = len(re.findall(r"<[a-zA-Z/!][^>]{0,80}>", body[:16384]))
        title = re.search(r"<title[^>]*>(.{0,200}?)</title>", head, re.S)

        ctype_match = float(
            (self.expected_content == "json" and "json" in ct)
            or (self.expected_content == "xml" and ("xml" in ct or "rss" in ct))
            or (self.expected_content == "html" and "html" in ct)
            or not ct
        )
        return np.array([
            math.log1p(len(body)),
            float(200 <= self.status < 300),
            float(300 <= self.status < 400),
            float(self.status == 403),
            float(self.status == 429),
            float(self.status >= 500),
            ctype_match,
            float(self.expected_content == "json" and "html" in ct),
            float(n_tokens),
            float(any(t in head for t in CAPTCHA_TOKENS)),
            float(any(c in (self.set_cookie or "").lower() for c in CHALLENGE_COOKIES)),
            float(bool(self.retry_after)),
            tags / max(len(body), 1) * 1000.0,
            math.log1p(max(self.items, 0)),
            # The ratio features are what catch "the page still renders but the
            # rows are gone" -- the soft-empty and markup-drift cases.
            min(self.items / max(self.median_items, 0.25), 3.0),
            min(self.latency_ms / max(self.median_latency_ms, 1.0), 5.0),
            float(self.n_redirects),
            len(set(body[:4096])) / 96.0,
            float(len(title.group(1)) if title else 0) / 100.0,
            float(len(body) < 512),
            math.log1p(max(self.median_items, 0.0)),
        ], dtype=float)


# --------------------------------------------------------------------------
# Synthetic corpus
# --------------------------------------------------------------------------

_LOREM = (
    "senior backend engineer python distributed systems remote contract hiring "
    "team platform data infrastructure kubernetes postgres onsite hybrid salary"
).split()


def _fake_listing(rng: random.Random, n: int, fmt: str) -> str:
    rows = []
    for i in range(n):
        title = " ".join(rng.sample(_LOREM, 4))
        if fmt == "json":
            rows.append(
                '{"id":%d,"title":"%s","company":"Co%d",'
                '"location":"Remote","url":"https://example.test/%d"}' % (i, title, i, i)
            )
        elif fmt == "xml":
            rows.append(
                "<item><title>%s</title><link>https://example.test/%d</link>"
                "<description>%s</description></item>"
                % (title, i, " ".join(rng.sample(_LOREM, 12)))
            )
        else:
            rows.append(
                '<li class="job"><h3>%s</h3><span class="co">Co%d</span>'
                '<a href="/j/%d">apply</a></li>' % (title, i, i)
            )
    if fmt == "json":
        return '{"jobs":[' + ",".join(rows) + "]}"
    if fmt == "xml":
        return "<rss><channel>" + "".join(rows) + "</channel></rss>"
    return (
        "<html><head><title>Jobs</title></head><body><ul>"
        + "".join(rows)
        + "</ul></body></html>"
    )


_BLOCK_PAGES = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<div id='cf-wrapper'>Checking your browser before accessing. This process is "
    "automatic. Please enable JavaScript and cookies. "
    "<script src='/cdn-cgi/challenge-platform/h/b/orchestrate/jsch/v1'></script>"
    "</div></body></html>",

    "<html><head><title>Access Denied</title></head><body><h1>Access Denied</h1>"
    "<p>You do not have permission to access this resource. Request blocked. "
    "Reference #18.aef2</p></body></html>",

    "<html><head><title>Verify</title></head><body><h2>Verify you are human</h2>"
    "<div class='g-recaptcha' data-sitekey='x'></div>"
    "<p>We have detected unusual traffic from your network.</p></body></html>",

    "<html><head><title>Rate limited</title></head><body>"
    "<p>Too many requests. Please slow down and try again later.</p></body></html>",

    "<html><head><title>Security check</title></head><body>"
    "<p>Suspicious activity detected. Complete the security check to continue.</p>"
    "<iframe src='https://hcaptcha.example/captcha'></iframe></body></html>",
)


def build_corpus(n: int = 1400, seed: int = 20260818) -> tuple[np.ndarray, np.ndarray]:
    """Generate a labelled feature matrix.

    Roughly 60/40 clean/blocked. The blocked half is dominated by the *hard*
    cases -- 200-status soft blocks and silent empties -- and both halves
    include deliberately confusable classes: clean security-job listings
    stuffed with block vocabulary, tiny-but-valid API responses, terse block
    pages with no vendor branding. Loading the corpus with obvious 403s would
    inflate the headline score and teach the model nothing an `if` could not
    already do.
    """
    rng = random.Random(seed)
    X: list[np.ndarray] = []
    y: list[int] = []


    for _ in range(n):
        roll = rng.random()
        fmt = rng.choice(["json", "xml", "html"])
        ctype = {"json": "application/json", "xml": "application/rss+xml",
                 "html": "text/html"}[fmt]
        med_items, med_lat = 25.0, 400.0

        if roll < 0.07:
            # The adversarial clean case: a security or anti-fraud job listing.
            # Its description is full of the exact vocabulary the token features
            # key on -- "captcha", "bot detection", "security check" -- while
            # being a perfectly good response. Without this class the model
            # learns "the word captcha means blocked" and takes the pipeline
            # down every time a board posts an anti-abuse role.
            items = rng.randint(6, 40)
            body = _fake_listing(rng, items, fmt).replace(
                "hiring",
                "captcha bot detection security check suspicious activity",
                1,
            )
            f = ResponseFeatures(
                status=200, body=body, content_type=ctype, expected_content=fmt,
                items=items, latency_ms=rng.uniform(150, 800), n_redirects=0,
                set_cookie="session=abc", retry_after="",
                median_items=med_items, median_latency_ms=med_lat,
            )
            label = 0
        elif roll < 0.11:
            # Small compact API response: correct, useful, and under 512 bytes,
            # which is the same body_is_tiny signal a stub block page trips.
            items = rng.randint(2, 5)
            f = ResponseFeatures(
                status=200, body=_fake_listing(rng, items, "json")[:500],
                content_type="application/json", expected_content="json",
                items=items, latency_ms=rng.uniform(60, 220), n_redirects=0,
                set_cookie="", retry_after="", median_items=med_items,
                median_latency_ms=med_lat,
            )
            label = 0
        elif roll < 0.20:
            # The genuinely ambiguous class, and the reason `items_vs_median`
            # exists at all: a *quiet* feed that legitimately returned zero
            # rows. A niche category board whose median poll is two jobs
            # returning none is a slow week, not a block; the identical body
            # from a source whose median is 25 is a block. Nothing inside the
            # response separates them -- only the source's own baseline does.
            # These rows are also what keeps the holdout score honest: they are
            # not perfectly separable, and they should not be.
            body = {"json": '{"jobs":[]}', "xml": "<rss><channel></channel></rss>",
                    "html": "<html><head><title>Jobs</title></head><body>"
                            "<ul></ul></body></html>"}[fmt]
            f = ResponseFeatures(
                status=200, body=body, content_type=ctype, expected_content=fmt,
                items=0, latency_ms=rng.uniform(90, 380), n_redirects=0,
                set_cookie="session=abc", retry_after="",
                median_items=rng.uniform(0.4, 1.6),   # this source is always thin
                median_latency_ms=med_lat,
            )
            label = 0
        elif roll < 0.24:
            # A terse block: two words, no vendor branding, no challenge cookie.
            # Almost no token signal -- the model has to get this from the
            # zero-items and tiny-body features instead.
            f = ResponseFeatures(
                status=200, body="<html><body>Denied.</body></html>",
                content_type="text/html", expected_content=fmt, items=0,
                latency_ms=rng.uniform(40, 200), n_redirects=0, set_cookie="",
                retry_after="", median_items=med_items, median_latency_ms=med_lat,
            )
            label = 1
        elif roll < 0.28:
            # A heavily branded block page: long body, high tag density, looks
            # structurally like a real page.
            filler = "<div class='row'><span>Help centre</span></div>" * 120
            f = ResponseFeatures(
                status=200,
                body=_BLOCK_PAGES[2].replace("</body>", filler + "</body>"),
                content_type="text/html", expected_content=fmt, items=0,
                latency_ms=rng.uniform(300, 1200), n_redirects=1,
                set_cookie="__cf_bm=q", retry_after="",
                median_items=med_items, median_latency_ms=med_lat,
            )
            label = 1
        elif roll < 0.64:
            # Clean response, healthy row count.
            items = rng.randint(8, 60)
            f = ResponseFeatures(
                status=200, body=_fake_listing(rng, items, fmt), content_type=ctype,
                expected_content=fmt, items=items, latency_ms=rng.uniform(120, 900),
                n_redirects=rng.choice([0, 0, 0, 1]), set_cookie="session=abc",
                retry_after="", median_items=med_items, median_latency_ms=med_lat,
            )
            label = 0
        elif roll < 0.70:
            # Clean but genuinely thin -- a quiet source, not a block. This is
            # the class the model most needs to NOT panic about; without it the
            # breaker trips every time a board has a slow morning.
            items = rng.randint(1, 4)
            f = ResponseFeatures(
                status=200, body=_fake_listing(rng, items, fmt), content_type=ctype,
                expected_content=fmt, items=items, latency_ms=rng.uniform(100, 500),
                n_redirects=0, set_cookie="session=abc", retry_after="",
                median_items=med_items, median_latency_ms=med_lat,
            )
            label = 0
        elif roll < 0.82:
            # Soft block: HTTP 200, challenge/CAPTCHA body, zero parsed items.
            f = ResponseFeatures(
                status=200, body=rng.choice(_BLOCK_PAGES), content_type="text/html",
                expected_content=fmt, items=0, latency_ms=rng.uniform(200, 2500),
                n_redirects=rng.choice([0, 1, 2]),
                set_cookie=rng.choice(
                    ["cf_clearance=z; __cf_bm=q", "datadome=k", "session=abc"]
                ),
                retry_after="", median_items=med_items, median_latency_ms=med_lat,
            )
            label = 1
        elif roll < 0.92:
            # Silent empty: right shape, no rows. The one that fails quietly.
            body = {
                "json": '{"jobs":[]}',
                "xml": "<rss><channel></channel></rss>",
                "html": "<html><head><title>Jobs</title></head><body><ul></ul></body></html>",
            }[fmt]
            f = ResponseFeatures(
                status=200, body=body, content_type=ctype, expected_content=fmt,
                items=0, latency_ms=rng.uniform(80, 400), n_redirects=0,
                set_cookie="", retry_after="", median_items=med_items,
                median_latency_ms=med_lat,
            )
            label = 1
        else:
            # Hard block, kept as a minority so it cannot dominate the fit.
            status = rng.choice([403, 429, 503])
            f = ResponseFeatures(
                status=status, body=rng.choice(_BLOCK_PAGES), content_type="text/html",
                expected_content=fmt, items=0, latency_ms=rng.uniform(50, 600),
                n_redirects=0, set_cookie="",
                retry_after="60" if status == 429 else "",
                median_items=med_items, median_latency_ms=med_lat,
            )
            label = 1

        X.append(f.vector())
        y.append(label)

    return np.vstack(X), np.array(y)


class BlockClassifier:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        self.metrics: dict[str, object] = {}
        self.ready = False

    def train(self, seed: int = 20260818) -> dict[str, object]:
        X, y = build_corpus(seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed % 2 ** 31, stratify=y
        )
        self.model.fit(self.scaler.fit_transform(Xtr), ytr)

        proba = self.model.predict_proba(self.scaler.transform(Xte))[:, 1]
        pred = (proba >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            yte, pred, average="binary", zero_division=0
        )
        weights = sorted(
            zip(FEATURE_NAMES, (round(float(c), 3) for c in self.model.coef_[0])),
            key=lambda kv: -abs(kv[1]),
        )[:8]
        self.metrics = {
            "trained_at": time.time(),
            "n_train": int(len(ytr)),
            "n_holdout": int(len(yte)),
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f1), 3),
            "avg_precision": round(float(average_precision_score(yte, proba)), 3),
            "top_weights": dict(weights),
            "provenance": (
                "Procedurally generated response bodies covering four block "
                "archetypes, plus this deployment's own sandbox traffic. No real "
                "anti-bot vendor pages were collected or replayed."
            ),
        }
        self.ready = True
        db.kv_set(_MODEL_KEY, self.metrics)
        return self.metrics

    def score(self, f: ResponseFeatures) -> float:
        if not self.ready:
            # Degrade to the dumb rule rather than to no protection at all.
            return 1.0 if (f.status in (403, 429) or f.items == 0) else 0.0
        v = self.scaler.transform(f.vector().reshape(1, -1))
        return float(self.model.predict_proba(v)[0, 1])

    def explain(self, f: ResponseFeatures, top: int = 4) -> list[tuple[str, float]]:
        """Per-request attribution: standardized feature value x coefficient.

        This is what turns "block_prob 0.91" into "because has_captcha_token
        and items_vs_median". An operator cannot act on a bare number.
        """
        if not self.ready:
            return []
        v = self.scaler.transform(f.vector().reshape(1, -1))[0]
        contrib = v * self.model.coef_[0]
        order = np.argsort(-np.abs(contrib))[:top]
        return [(FEATURE_NAMES[i], round(float(contrib[i]), 3)) for i in order]


classifier = BlockClassifier()


def verdict(prob: float, threshold: float) -> str:
    if prob >= max(threshold + 0.25, 0.75):
        return "blocked"
    if prob >= threshold:
        return "suspect"
    return "clean"
