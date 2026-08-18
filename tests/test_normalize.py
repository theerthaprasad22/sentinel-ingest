"""Identity and normalisation.

These are the tests that matter most, because everything downstream inherits
their answers: a wrong canonical id shows up as an inflated job count, and a
silently-dropped row shows up as nothing at all.
"""
from __future__ import annotations

import json

import pytest

from app.ingest.normalize import (
    canonical_id, extract_salary, is_remote, norm_company, norm_title,
    normalize, parse_date, payload_hash,
)


class TestCompanyNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("Acme, Inc.", "acme"),
        ("Acme Inc", "acme"),
        ("ACME LLC", "acme"),
        ("Acme Technologies Pvt Ltd", "acme"),
        ("Acme GmbH", "acme"),
        ("Acme  Labs", "acme"),
    ])
    def test_suffixes_are_stripped(self, raw, expected):
        assert norm_company(raw) == expected

    def test_a_suffix_that_is_the_whole_name_survives(self):
        # "Group" as a company name is not noise if there is nothing else left;
        # stripping to the empty string would collide every such company.
        assert norm_company("Group") == "group"


class TestTitleNormalisation:
    @pytest.mark.parametrize("a,b", [
        ("Sr. Backend Engineer", "Senior Backend Engineer"),
        ("Backend Engineer (Remote)", "Backend Engineer"),
        ("Backend Engineer - Full Time", "Backend Engineer"),
        ("Jr Data Scientist", "Junior Data Scientist"),
    ])
    def test_equivalent_titles_collapse(self, a, b):
        assert norm_title(a) == norm_title(b)

    def test_genuinely_different_titles_do_not_collapse(self):
        assert norm_title("Backend Engineer") != norm_title("Frontend Engineer")
        assert norm_title("Senior Engineer") != norm_title("Junior Engineer")


class TestCanonicalId:
    def test_same_job_from_two_boards_gets_one_id(self):
        a = canonical_id("Acme, Inc.", "Sr. Backend Engineer", "Remote - US")
        b = canonical_id("Acme Inc", "Senior Backend Engineer (Remote)", "Anywhere")
        assert a == b

    def test_different_companies_do_not_collide(self):
        a = canonical_id("Acme", "Backend Engineer", "Berlin")
        b = canonical_id("Globex", "Backend Engineer", "Berlin")
        assert a != b

    def test_id_is_stable_across_calls(self):
        args = ("Acme", "Backend Engineer", "Berlin, DE")
        assert canonical_id(*args) == canonical_id(*args)


class TestDateParsing:
    @pytest.mark.parametrize("raw", [
        "2026-08-16T14:14:11",
        "2026-08-16T14:14:11Z",
        "2026-08-16",
        "Sat, 16 Aug 2026 14:14:11 +0000",
        1786516800,
    ])
    def test_formats_boards_actually_emit(self, raw):
        assert parse_date(raw) is not None

    @pytest.mark.parametrize("raw", ["", None, "not a date", "yesterday"])
    def test_unparseable_returns_none_rather_than_inventing_one(self, raw):
        # A fabricated date would satisfy the drift monitor's fill-rate check
        # while being a lie. None is the honest answer and the monitor is built
        # to tolerate it.
        assert parse_date(raw) is None


class TestNormalizeRecord:
    def test_rejects_a_row_with_no_title(self):
        assert normalize({"url": "https://x/1"}, "test") is None

    def test_rejects_a_row_with_no_url(self):
        assert normalize({"title": "Engineer"}, "test") is None

    def test_strips_html_from_descriptions(self):
        row = normalize(
            {"title": "Engineer", "url": "https://x/1",
             "description": "<p>Build <b>things</b> &amp; ship them</p>"},
            "test",
        )
        assert "<" not in row["description"]
        assert "&amp;" not in row["description"]
        assert "Build things & ship them" in row["description"]

    def test_remote_is_inferred_from_any_field(self):
        row = normalize(
            {"title": "Engineer", "url": "https://x/1", "location": "Work from home"},
            "test",
        )
        assert row["remote"] == 1

    def test_tags_are_stored_as_json(self):
        row = normalize(
            {"title": "Engineer", "url": "https://x/1", "tags": ["python", "ml"]},
            "test",
        )
        assert json.loads(row["tags"]) == ["python", "ml"]

    def test_payload_hash_ignores_timestamps(self):
        base = {"title": "Engineer", "url": "https://x/1", "company": "Acme"}
        a = normalize(dict(base), "test")
        b = normalize(dict(base), "test")
        # first_seen/last_seen differ between the two calls; the hash must not,
        # or every poll would rewrite every row.
        assert a["payload_hash"] == b["payload_hash"]

    def test_payload_hash_changes_when_content_changes(self):
        a = normalize({"title": "Engineer", "url": "https://x/1", "company": "Acme"}, "t")
        b = normalize({"title": "Engineer", "url": "https://x/1", "company": "Globex"}, "t")
        assert a["payload_hash"] != b["payload_hash"]


class TestSalary:
    @pytest.mark.parametrize("text", ["$120k - $160k", "€70k", "₹28 LPA", "USD 95,000"])
    def test_recognised_formats(self, text):
        assert extract_salary(text) != ""

    def test_absent_salary_is_empty_not_guessed(self):
        assert extract_salary("Competitive compensation package") == ""


def test_is_remote_across_phrasings():
    assert is_remote("Remote (EU)")
    assert is_remote("Anywhere in the World")
    assert is_remote("WFH")
    assert not is_remote("Berlin, DE")
