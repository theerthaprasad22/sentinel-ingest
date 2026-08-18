"""Test bootstrap.

`app.config.Settings` reads the environment at import time, so the database
path has to be redirected *before* anything under `app.` is imported. Doing it
here, in the root conftest, is the only place that is guaranteed to run first.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="sentinel-tests-"))
os.environ["SENTINEL_DB"] = str(_TMP / "test.db")
os.environ["SENTINEL_SCHEDULER"] = "0"       # no background loop during tests
os.environ["SENTINEL_RESPECT_ROBOTS"] = "1"

import pytest                                                    # noqa: E402

from app import db                                               # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    """Every test gets an empty database.

    Cheaper and less surprising than sharing one: several of these tests train
    models off table contents, and cross-test contamination there produces
    failures that look like model bugs.
    """
    db.init_db()
    for table in ("jobs", "sources", "fetch_log", "drift_log", "events", "kv"):
        db.execute(f"DELETE FROM {table}")
    yield
