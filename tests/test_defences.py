"""The parts that decide whether we are being blocked, drifting, or fine.

A false negative here silently ingests nothing; a false positive trips a
breaker on a healthy source. Both are expensive, so both directions are tested.
"""
from __future__ import annotations

import pytest

from app.ingest import robots
from app.ingest.blockdetect import ResponseFeatures, classifier, verdict
from app.ingest.drift import DriftMonitor
from app.ingest.identity import PROFILES, IdentityPool
from app.ingest.pacing import TokenBucket, circadian_weight, PaceBandit

CAPTCHA_BODY = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<h2>Verify you are human</h2><p>We have detected unusual traffic from your "
    "network. Please enable JavaScript.</p>"
    "<div class='g-recaptcha'></div></body></html>"
)
HEALTHY_BODY = '{"jobs":[' + ",".join(
    '{"id":%d,"title":"Backend Engineer","company":"Co%d"}' % (i, i) for i in range(30)
) + "]}"


@pytest.fixture(scope="module", autouse=True)
def trained():
    classifier.train()


def _features(**kw) -> ResponseFeatures:
    base = dict(
        status=200, body=HEALTHY_BODY, content_type="application/json",
        expected_content="json", items=30, latency_ms=400.0, n_redirects=0,
        set_cookie="", retry_after="", median_items=25.0, median_latency_ms=400.0,
    )
    base.update(kw)
    return ResponseFeatures(**base)


class TestBlockClassifier:
    def test_healthy_response_scores_low(self):
        assert classifier.score(_features()) < 0.3

    def test_captcha_interstitial_at_http_200_scores_high(self):
        # The whole reason the model exists: status 200, and it is a block.
        score = classifier.score(_features(
            body=CAPTCHA_BODY, content_type="text/html", items=0,
            set_cookie="__cf_bm=x",
        ))
        assert score > 0.7

    def test_hard_403_scores_high(self):
        assert classifier.score(_features(
            status=403, body=CAPTCHA_BODY, content_type="text/html", items=0
        )) > 0.7

    def test_empty_from_a_busy_source_is_suspicious(self):
        score = classifier.score(_features(
            body='{"jobs":[]}', items=0, median_items=25.0
        ))
        assert score > 0.5

    def test_empty_from_a_habitually_thin_source_is_not(self):
        # A niche board whose median poll is one job returning none is a slow
        # week. Alarming here would trip a breaker on a perfectly healthy feed.
        score = classifier.score(_features(
            body='{"jobs":[]}', items=0, median_items=1.0
        ))
        assert score < classifier.score(_features(
            body='{"jobs":[]}', items=0, median_items=25.0
        ))

    def test_a_security_job_listing_is_not_a_block(self):
        # Full of block vocabulary and completely legitimate. Without this
        # behaviour the pipeline dies whenever a board posts an anti-abuse role.
        body = HEALTHY_BODY.replace(
            "Backend Engineer",
            "Engineer, CAPTCHA and bot detection - security check systems", 1)
        assert classifier.score(_features(body=body, items=30)) < 0.5

    def test_explanations_name_the_features_that_drove_it(self):
        f = _features(body=CAPTCHA_BODY, content_type="text/html", items=0)
        names = [n for n, _ in classifier.explain(f)]
        assert names and all(isinstance(n, str) for n in names)

    @pytest.mark.parametrize("prob,expected", [
        (0.05, "clean"), (0.55, "suspect"), (0.95, "blocked"),
    ])
    def test_verdict_thresholds(self, prob, expected):
        assert verdict(prob, 0.5) == expected


class TestDriftMonitor:
    def _feed(self, monitor, source, batches):
        alarms = []
        for batch in batches:
            alarms = monitor.check(source, batch)
        return alarms

    def test_no_alarm_while_the_source_is_stable(self):
        m = DriftMonitor()
        good = [{"title": "t", "company": "c", "location": "l", "url": "u",
                 "description": "d", "posted_at": "2026-01-01"}] * 20
        alarms = self._feed(m, "stable", [good] * 8)
        assert alarms == []

    def test_alarm_when_a_reliable_field_collapses(self):
        m = DriftMonitor()
        good = [{"title": "t", "company": "c", "location": "l", "url": "u",
                 "description": "d", "posted_at": "2026-01-01"}] * 20
        self._feed(m, "breaks", [good] * 8)
        broken = [dict(r, company="") for r in good]      # markup changed
        alarms = m.check("breaks", broken)
        assert any(a.field == "company" for a in alarms)

    def test_a_field_that_was_always_sparse_does_not_alarm(self):
        # HN job posts never carry a location. A monitor that alarms on a field
        # which has been empty since day one is a monitor nobody reads.
        m = DriftMonitor()
        sparse = [{"title": "t", "company": "c", "location": "", "url": "u",
                   "description": "d", "posted_at": ""}] * 20
        alarms = self._feed(m, "sparse", [sparse] * 10)
        assert alarms == []

    def test_baselines_are_reported_per_field(self):
        m = DriftMonitor()
        rows = [{"title": "t", "url": "u"}] * 10
        m.check("reported", rows)
        base = m.baseline("reported")
        assert base["title"]["mean"] > 0
        assert base["company"]["mean"] == 0


class TestRobots:
    SAMPLE = """
    User-agent: *
    Disallow: /private
    Allow: /private/public-bit
    Crawl-delay: 3

    User-agent: BadBot
    Disallow: /
    """

    def test_crawl_delay_is_read(self):
        assert robots.parse(self.SAMPLE).crawl_delay == 3.0

    def test_disallowed_path_is_refused(self):
        rules = robots.parse(self.SAMPLE)
        allowed, _ = robots.allowed("https://x/private/thing", rules)
        assert allowed is False

    def test_longer_allow_beats_shorter_disallow(self):
        rules = robots.parse(self.SAMPLE)
        allowed, _ = robots.allowed("https://x/private/public-bit", rules)
        assert allowed is True

    def test_unlisted_path_is_permitted(self):
        rules = robots.parse(self.SAMPLE)
        allowed, _ = robots.allowed("https://x/jobs", rules)
        assert allowed is True

    def test_unreachable_robots_means_no(self):
        # Fail closed. An unreadable policy is not permission.
        rules = robots.Rules(reachable=False)
        allowed, reason = robots.allowed("https://x/jobs", rules)
        assert allowed is False and "unreachable" in reason

    def test_wildcards_and_end_anchors(self):
        rules = robots.parse("User-agent: *\nDisallow: /*.pdf$\n")
        assert robots.allowed("https://x/a/b/report.pdf", rules)[0] is False
        assert robots.allowed("https://x/a/b/report.pdf.html", rules)[0] is True


class TestIdentity:
    def test_every_profile_is_internally_coherent(self):
        # A Chrome UA must carry client hints; a non-Chromium UA must not. This
        # mismatch is the single cheapest bot signal there is, and the sandbox
        # rejects on exactly this.
        for p in PROFILES:
            is_chrome = "Chrome/" in p.user_agent and "Edg/" not in p.user_agent
            assert bool(p.sec_ch_ua) == is_chrome, p.key

    def test_leases_are_sticky_within_their_ttl(self):
        pool = IdentityPool(ttl=3600)
        assert pool.lease("src").profile.key == pool.lease("src").profile.key

    def test_burning_an_identity_replaces_the_lease(self):
        pool = IdentityPool(ttl=3600)
        first = pool.lease("src")
        pool.burn("src")
        assert pool.lease("src") is not first

    def test_headers_are_emitted_in_browser_order(self):
        headers = list(PROFILES[0].headers(navigation=True, contact="x"))
        assert headers.index("user-agent") < headers.index("accept")
        assert headers.index("accept") < headers.index("accept-language")

    def test_xhr_requests_drop_navigation_only_headers(self):
        h = PROFILES[0].headers(navigation=False, contact="x")
        assert "upgrade-insecure-requests" not in h
        assert h["sec-fetch-dest"] == "empty"


class TestPacing:
    def test_bucket_refuses_once_drained(self):
        b = TokenBucket(rate_per_s=0.0, capacity=2.0, tokens=2.0)
        assert b.take() and b.take()
        assert not b.take()

    def test_wait_time_is_finite_and_positive_when_drained(self):
        b = TokenBucket(rate_per_s=0.5, capacity=1.0, tokens=0.0)
        assert 0 < b.wait_time() < 10

    def test_circadian_weight_stays_in_a_sane_band(self):
        weights = [circadian_weight(h * 3600) for h in range(24)]
        assert all(0.9 <= w <= 2.1 for w in weights)
        assert max(weights) > min(weights) * 1.5      # it actually varies

    def test_bandit_prefers_a_tier_that_keeps_working(self):
        bandit = PaceBandit(seed=1)
        for _ in range(40):
            bandit.update("s", "cautious", success=True)
            bandit.update("s", "aggressive", success=False)
        post = bandit.snapshot("s")
        assert post["cautious"]["mean"] > post["aggressive"]["mean"]

    def test_posteriors_decay_so_a_source_can_change_its_mind(self):
        bandit = PaceBandit(seed=2)
        for _ in range(200):
            bandit.update("s", "normal", success=True)
        a, b = bandit._post("s")["normal"]
        assert a + b < 200        # bounded memory, not an ever-growing count
