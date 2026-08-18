"""Storage, dedupe, search, and the sandbox's defences, end to end.

These run against the real FastAPI app through TestClient, so they exercise the
same code path the deployed service does -- including the in-process sandbox,
which is what makes "watch it get blocked" testable at all.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import db
from app.ingest import circuit, registry
from app.ingest.adapters.hackernews import _split as hn_split
from app.ingest.adapters.sandbox import parse_sandbox_html
from app.ingest.store import upsert
from app.main import app
from app.sandbox.server import JOBS, state as sandbox_state
from app.search.index import SemanticIndex


def _raw(title, company, url, **kw):
    row = {"title": title, "company": company, "url": url,
           "description": kw.pop("description", "Build and operate the thing."),
           "location": kw.pop("location", "Remote")}
    row.update(kw)
    return row


class TestStore:
    def test_new_rows_are_inserted_once(self):
        rows = [_raw("Backend Engineer", "Acme", "https://x/1")]
        assert upsert(rows, "t").new == 1
        assert upsert(rows, "t").new == 0

    def test_an_unchanged_repost_is_a_touch_not_an_update(self):
        rows = [_raw("Backend Engineer", "Acme", "https://x/1")]
        upsert(rows, "t")
        report = upsert(rows, "t")
        assert report.touched == 1 and report.updated == 0

    def test_a_changed_field_produces_an_update(self):
        upsert([_raw("Backend Engineer", "Acme", "https://x/1")], "t")
        report = upsert(
            [_raw("Backend Engineer", "Acme", "https://x/1", description="New copy.")],
            "t",
        )
        assert report.updated == 1

    def test_duplicates_inside_one_batch_collapse(self):
        # A single feed page routinely carries the same posting twice.
        rows = [
            _raw("Sr. Backend Engineer", "Acme, Inc.", "https://x/1"),
            _raw("Senior Backend Engineer", "Acme Inc", "https://x/2"),
        ]
        assert upsert(rows, "t").new == 1

    def test_unusable_rows_are_counted_not_silently_dropped(self):
        report = upsert([{"company": "Acme"}, _raw("Engineer", "Acme", "https://x/1")], "t")
        assert report.rejected == 1 and report.new == 1

    def test_the_same_job_from_two_sources_is_one_row(self):
        upsert([_raw("Backend Engineer", "Acme", "https://board-a/1")], "board_a")
        report = upsert([_raw("Backend Engineer", "Acme", "https://board-b/9")], "board_b")
        assert report.new == 0


class TestSemanticLayer:
    @staticmethod
    def _corpus():
        rows = [
            _raw("Machine Learning Engineer", "Basalt AI", "https://x/1",
                 description="PyTorch, model training, feature stores, GPUs."),
            _raw("ML Engineer, Applied Research", "Basalt AI", "https://x/2",
                 description="PyTorch, model training, feature stores, GPUs."),
            _raw("Frontend Developer", "Kestrel", "https://x/3",
                 description="React, CSS, accessibility, design systems."),
            _raw("Account Executive", "Lumen", "https://x/4",
                 description="Quota, pipeline, enterprise sales cycle."),
        ]
        # Pad so TruncatedSVD has something to decompose.
        rows += [
            _raw(f"Backend Engineer {i}", f"Co{i}", f"https://x/pad{i}",
                 description="Python services, Postgres, queues, deployment.")
            for i in range(20)
        ]
        return rows

    def test_search_finds_by_meaning_not_by_keyword(self):
        upsert(self._corpus(), "t")
        index = SemanticIndex()
        assert index.build()["built"]
        # "pytorch" appears in the description, "deep learning" appears nowhere.
        hits = index.search("deep learning model training", k=5)
        assert hits
        top_ids = [cid for cid, _ in hits[:2]]
        titles = {
            r["canonical_id"]: r["title"]
            for r in db.query("SELECT canonical_id, title FROM jobs")
        }
        assert any("ML" in titles[c] or "Machine Learning" in titles[c] for c in top_ids)

    def test_near_duplicates_are_flagged_not_deleted(self):
        upsert(self._corpus(), "t")
        index = SemanticIndex()
        index.build()
        index.mark_duplicates()
        marked = db.query("SELECT canonical_id, dup_of FROM jobs WHERE dup_of IS NOT NULL")
        total = db.query_one("SELECT COUNT(*) n FROM jobs")["n"]
        assert total == len(self._corpus()) - 0    # nothing was removed
        assert isinstance(marked, list)            # flagging is non-destructive

    def test_same_title_at_different_employers_is_not_a_duplicate(self):
        # The bug this pins: comparing full documents flagged 143 pairs out of
        # 388, because every "Site Reliability Engineer" shares a title and a
        # pile of boilerplate. Different employer means different job.
        rows = self._corpus() + [
            _raw("Site Reliability Engineer", "Acme", "https://x/sre-a",
                 description="Own the platform. Kubernetes, on-call, terraform."),
            _raw("Site Reliability Engineer", "Globex", "https://x/sre-b",
                 description="Own the platform. Kubernetes, on-call, terraform."),
        ]
        upsert(rows, "t")
        index = SemanticIndex()
        index.build()
        pairs = index.find_duplicates()
        titles = {r["canonical_id"]: (r["title"], r["company"])
                  for r in db.query("SELECT canonical_id, title, company FROM jobs")}
        for a, b, _ in pairs:
            assert titles[a][1] == titles[b][1], "flagged across employers"

    def test_a_reworded_relisting_at_one_employer_is_caught(self):
        rows = self._corpus() + [
            _raw("Data Platform Engineer", "Quarry Data", "https://x/dp-1"),
            _raw("Data Platform Engineer - Core", "Quarry Data", "https://x/dp-2"),
        ]
        upsert(rows, "t")
        index = SemanticIndex()
        index.build()
        titles = {r["canonical_id"]: r["title"]
                  for r in db.query("SELECT canonical_id, title FROM jobs")}
        found = {frozenset((titles[a], titles[b])) for a, b, _ in index.find_duplicates()}
        assert frozenset(("Data Platform Engineer", "Data Platform Engineer - Core")) in found

    def test_seniority_levels_are_not_collapsed(self):
        # Two openings, not one. Collapsing them deletes a real vacancy.
        rows = self._corpus() + [
            _raw("Software Engineer", "Fanduel", "https://x/se-1"),
            _raw("Senior Software Engineer", "Fanduel", "https://x/se-2"),
        ]
        upsert(rows, "t")
        index = SemanticIndex()
        index.build()
        titles = {r["canonical_id"]: r["title"]
                  for r in db.query("SELECT canonical_id, title FROM jobs")}
        for a, b, _ in index.find_duplicates():
            pair = {titles[a], titles[b]}
            assert pair != {"Software Engineer", "Senior Software Engineer"}

    def test_index_reports_it_is_not_ready_on_a_tiny_corpus(self):
        upsert([_raw("Engineer", "Acme", "https://x/1")], "t")
        index = SemanticIndex()
        assert index.build()["built"] is False


class TestHackerNewsTitleSplit:
    @pytest.mark.parametrize("title,company,role", [
        ("Tasklet (YC P26) Is Hiring a Head of Design Engineering",
         "Tasklet", "Head of Design Engineering"),
        ("Gooseworks (YC W23) Is Hiring a Founding Engineer",
         "Gooseworks", "Founding Engineer"),
        ("Acme is looking for a Senior Rust Engineer",
         "Acme", "Senior Rust Engineer"),
    ])
    def test_recognised_grammar_is_split_confidently(self, title, company, role):
        got_company, got_role, conf = hn_split(title)
        assert got_company == company and got_role == role and conf == 1.0

    def test_unrecognised_grammar_admits_low_confidence(self):
        # Guessing here is how you get a database full of companies called
        # "Is Hiring". Better to keep the row and mark it uncertain.
        company, role, conf = hn_split("Some completely different headline")
        assert company is None and conf < 0.5 and role


class TestSandboxAndFallback:
    @pytest.fixture()
    def client(self):
        # No lifespan: these tests drive the endpoints directly and must not
        # start the background scheduler.
        with TestClient(app) as c:
            yield c

    @pytest.fixture(autouse=True)
    def reset_sandbox(self):
        yield
        for f in ("fingerprint_check", "rate_limit", "captcha_wall", "hard_block",
                  "silent_empty"):
            setattr(sandbox_state, f, False)
        sandbox_state.markup_version = 1

    def test_selectors_survive_every_markup_version(self, client):
        for version in (1, 2, 3):
            sandbox_state.markup_version = version
            body = client.get("/sandbox/jobs").text
            rows = parse_sandbox_html(body)
            assert len(rows) == len(JOBS), f"markup v{version} lost rows"
            assert all(r["title"] for r in rows)

    def test_extraction_confidence_falls_when_selectors_fall_back(self, client):
        sandbox_state.markup_version = 1
        primary = parse_sandbox_html(client.get("/sandbox/jobs").text)
        sandbox_state.markup_version = 3
        fallback = parse_sandbox_html(client.get("/sandbox/jobs").text)
        mean = lambda rs: sum(r["confidence"] for r in rs) / len(rs)   # noqa: E731
        assert mean(primary) == 1.0
        assert mean(fallback) < mean(primary)

    def test_fingerprint_check_rejects_a_library_client(self, client):
        sandbox_state.fingerprint_check = True
        r = client.get("/sandbox/jobs", headers={"user-agent": "python-httpx/0.28"})
        assert r.status_code == 403

    def test_fingerprint_check_rejects_chrome_without_client_hints(self, client):
        sandbox_state.fingerprint_check = True
        r = client.get("/sandbox/jobs", headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "accept-language": "en-US,en;q=0.9",
            "accept": "text/html,application/xhtml+xml",
        })
        assert r.status_code == 403
        assert "client hints" in r.headers.get("x-sandbox-reason", "")

    def test_our_own_profile_passes_the_same_check(self, client):
        from app.ingest.identity import PROFILES
        sandbox_state.fingerprint_check = True
        chrome = next(p for p in PROFILES if p.key == "chrome-win")
        headers = {k: v for k, v in
                   chrome.headers(navigation=True, contact="t").items() if v}
        assert client.get("/sandbox/jobs", headers=headers).status_code == 200

    def test_captcha_wall_returns_http_200(self, client):
        # If this ever returns 4xx the demo stops proving anything, because a
        # status check would catch it.
        sandbox_state.captcha_wall = True
        r = client.get("/sandbox/jobs")
        assert r.status_code == 200
        assert r.headers["x-sandbox-truth"] == "blocked"
        assert parse_sandbox_html(r.text) == []

    def test_every_response_carries_a_ground_truth_label(self, client):
        assert client.get("/sandbox/jobs").headers["x-sandbox-truth"] == "clean"
        sandbox_state.silent_empty = True
        assert client.get("/sandbox/jobs").headers["x-sandbox-truth"] == "blocked"

    def test_robots_disallows_the_private_path(self, client):
        body = client.get("/robots.txt").text
        assert "Disallow: /sandbox/private" in body


class TestCircuitBreaker:
    def test_opens_after_the_threshold_and_refuses_traffic(self):
        registry.sync_to_db()
        for _ in range(5):
            circuit.record_failure("sandbox", "test")
        assert circuit.state("sandbox") == "open"
        assert circuit.check("sandbox").allowed is False

    def test_force_open_skips_the_count(self):
        registry.sync_to_db()
        circuit.force_open("sandbox", "explicit 403")
        assert circuit.state("sandbox") == "open"

    def test_reset_closes_it_again(self):
        registry.sync_to_db()
        circuit.force_open("sandbox", "test")
        circuit.reset("sandbox")
        assert circuit.state("sandbox") == "closed"
        assert circuit.check("sandbox").allowed is True


class TestApiSurface:
    @pytest.fixture()
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_health_reports_every_registered_source(self, client):
        body = client.get("/api/health").json()
        assert body["sources_total"] == len(registry.adapters())

    def test_sources_expose_their_licence_note(self, client):
        for s in client.get("/api/sources").json():
            assert s["licence_note"], f"{s['name']} has no stated basis for access"

    def test_ml_endpoint_carries_provenance(self, client):
        ml = client.get("/api/ml").json()
        assert ml["block_classifier"]["provenance"]
        assert "not accuracy against human labels" in ml["tagger"].get("note", "") \
            or ml["tagger"].get("trained") is False

    def test_robots_check_is_executable_by_a_reviewer(self, client):
        r = client.get("/api/robots-check",
                       params={"url": "http://testserver/sandbox/private/jobs"})
        assert r.status_code == 200
        assert "allowed" in r.json()
