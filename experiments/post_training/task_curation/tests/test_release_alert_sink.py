# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release-stage alert sink never resolves to the production feed in tests.

Coordinator ruling (2026-11-23): fixture dry-runs and tests emitted STAGE_FAIL rows
for fixture sources into the real job-watchdog feed. The conftest autouse fixture
repoints the module default for every test; this test pins that behavior.
"""

import json
from pathlib import Path

import pytest

from experiments.post_training.task_curation.release_stages import admission_check, alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

PRODUCTION_FEED = Path.home() / "Documents/experiments/active/job-watchdog/alerts.jsonl"


def test_default_sink_is_never_the_production_feed(tmp_path):
    """The conftest fixture repoints the default; an omitted feeds argument lands there."""
    assert alerts.WATCHDOG_FEED != PRODUCTION_FEED
    assert alerts.WATCHDOG_FEED.is_relative_to(tmp_path)
    alerts.stage_started("sink-proof", "alert_sink", {"mode": "test"}, feeds=None)
    alerts.stage_failed("sink-proof", "alert_sink", {"error": "fixture"}, feeds=None)
    rows = [json.loads(line) for line in alerts.WATCHDOG_FEED.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [alerts.KIND_START, alerts.KIND_FAIL]
    assert all(row["key"] == "atlas-release:sink-proof:alert_sink" for row in rows)
    if PRODUCTION_FEED.exists():
        assert "sink-proof" not in PRODUCTION_FEED.read_text()


def test_stage_run_without_feeds_lands_in_the_test_sink(tmp_path):
    """A stage run that forgets its feeds argument writes to the per-test sink."""
    decoy = tmp_path / "declarations.json"
    decoy.write_text(json.dumps({}))
    backend = tmp_path / "backend.json"
    backend.write_text(json.dumps({}))
    inputs = admission_check.AdmissionCheckInputs(
        release_id="sink-proof",
        source_revision="a" * 40,
        declarations=ArtifactRef(uri=str(decoy), sha256="0" * 64),
        backend=ArtifactRef(uri=str(backend), sha256="0" * 64),
        snapshot_id="assembly-test-v1",
        catalog_root="test:catalog",
        prior_load_id="b" * 32,
        output_root=str(tmp_path / "release" / "admission"),
        expected_source_count=1,
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        admission_check.run_admission_check(inputs, dry_run=True, query=lambda sql, params: {"rows": []})
    rows = [json.loads(line) for line in alerts.WATCHDOG_FEED.read_text().splitlines()]
    assert rows and rows[0]["key"] == "atlas-release:sink-proof:admission_check"
    assert rows[0]["kind"] == alerts.KIND_START
    if PRODUCTION_FEED.exists():
        assert "sink-proof" not in PRODUCTION_FEED.read_text()
