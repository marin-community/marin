# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the main-source sync stage (release pipeline)."""

import hashlib
import json
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import main_sources
from experiments.post_training.task_curation.release_stages.graph import (
    release_graph,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan
from infra.marina.apps.rl_data_catalog.server import license_registry, licensing

MAIN_REVISION = "b" * 40
DECLARED = "Marin:declared"


def _new_source_row(source_id: str = "Marin:sync-new") -> dict:
    return {
        "id": source_id,
        "dataset_repo": "example/sync-new",
        "dataset_revision": "c" * 40,
        "url": None,
        "num_tasks": 42,
        "task_format": "taskspec",
        "data_restriction": "open",
        "license_ids": [],
        "intended_use": "train",
    }


def _extract_rows() -> list[dict]:
    return [
        _new_source_row(),
        # A historical ID resolves through the alias table onto a source the Atlas
        # stages pending, so the alias join decides it away from the diff.
        {
            **_new_source_row("Task Trove:laion__nemotron-gym-structured-outputs-v4"),
            "dataset_repo": "open-thoughts/TaskTrove",
        },
        # Eval-side declarations are counted, never queued.
        {**_new_source_row("lm-eval:fixture-eval"), "intended_use": "eval"},
        # A source the ban registry excludes never re-queues.
        _new_source_row("HuggingFaceH4:aime24"),
    ]


def _declarations_ref(tmp_path: Path, document: dict) -> ArtifactRef:
    path = tmp_path / "declarations.json"
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _queue(tmp_path: Path) -> Path:
    queue = tmp_path / "CANDIDATES.md"
    queue.write_text(
        "# RL ingest queue\n"
        "\n"
        "| Dataset | Link | Status |\n"
        "|---|---|---|\n"
        "| Existing dataset | https://example.org | Admitted |\n"
        "\n"
        "Rows tagged SFT-candidate are ingested as RL datasets now.\n"
    )
    return queue


def _inputs(tmp_path: Path, queue: Path | None) -> main_sources.MainSourceSyncInputs:
    return main_sources.MainSourceSyncInputs(
        release_id="testrelease",
        source_revision="a" * 40,
        main_source_revision=MAIN_REVISION,
        declarations=_declarations_ref(tmp_path, {DECLARED: {"id": DECLARED}}),
        report_root=str(tmp_path / "main-source-sync"),
        queue_path=str(queue) if queue else None,
    )


@pytest.fixture
def sync_env(monkeypatch, tmp_path):
    """Offline decision records and a fixture extraction boundary."""
    monkeypatch.setattr(main_sources, "extract_main_sources", lambda _root, _rev: _extract_rows())
    # The autouse evaluation fixture blanks the ban registry; the diff pins the real
    # registry's decisions, so restore it (and its cache) for the run.
    monkeypatch.setattr(licensing, "LICENSE_BANS", license_registry.LICENSE_BANS)
    licensing.license_bans.cache_clear()
    yield
    licensing.license_bans.cache_clear()


def test_new_main_source_becomes_exactly_one_candidate_row(tmp_path, sync_env):
    """One new main training source appends exactly one unclaimed queue row."""
    queue = _queue(tmp_path)
    proof = main_sources.run_main_source_sync(_inputs(tmp_path, queue))
    assert [row["source_id"] for row in proof["new_sources"]] == ["Marin:sync-new"]
    assert proof["appended"] == ["Marin:sync-new"]
    assert proof["already_queued"] == []
    lines = queue.read_text().splitlines()
    assert "| Marin:sync-new |" in [line[:18] for line in lines]
    row = next(line for line in lines if line.startswith("| Marin:sync-new"))
    assert (
        row.startswith("| Marin:sync-new | https://huggingface.co/datasets/example/sync-new/tree/")
        and "cccccccccccccccccccccccccccccccccccccccc" in row
    )
    assert "| P1 —" in row
    # The row lands inside the table, before the trailing prose.
    assert lines.index(row) < next(index for index, line in enumerate(lines) if line.startswith("Rows tagged"))
    assert lines[lines.index(row) - 1].startswith("| Existing dataset |")


def test_aliased_pending_eval_and_banned_sources_make_no_rows(tmp_path, sync_env):
    """The alias join, the pending staging and the ban registry each decide a source away."""
    queue = _queue(tmp_path)
    proof = main_sources.run_main_source_sync(_inputs(tmp_path, queue))
    queued = queue.read_text()
    assert "TaskTrove:laion__nemotron-gym-structured-outputs-v4" not in queued
    assert "HuggingFaceH4:aime24" not in queued
    assert "lm-eval:fixture-eval" not in queued
    assert proof["main_source_count"] == 4
    assert proof["eval_source_count"] == 1


def test_repeat_runs_make_no_duplicate_rows(tmp_path, sync_env):
    """A rerun finds the row present, appends nothing and reuses the sealed report."""
    queue = _queue(tmp_path)
    first = main_sources.run_main_source_sync(_inputs(tmp_path, queue))
    second = main_sources.run_main_source_sync(_inputs(tmp_path, queue))
    assert first["appended"] == ["Marin:sync-new"]
    assert second["appended"] == []
    assert second["already_queued"] == ["Marin:sync-new"]
    assert second["report"] == first["report"]
    assert queue.read_text().count("| Marin:sync-new |") == 1


def test_report_seals_write_once_into_the_release(tmp_path, sync_env):
    """The sealed report carries the typed rows and lives under the stage's root."""
    proof = main_sources.run_main_source_sync(_inputs(tmp_path, None))
    ref = proof["report"]
    document = json.loads(StoragePath(ref["uri"]).read_bytes())
    assert document["kind"] == "atlas-main-source-sync-report"
    assert document["release_id"] == "testrelease"
    assert document["main_source_revision"] == MAIN_REVISION
    assert [row["source_id"] for row in document["new_sources"]] == ["Marin:sync-new"]
    row = document["new_sources"][0]
    assert row["dataset_revision"] == "c" * 40 and row["num_tasks"] == 42
    assert row["task_format"] == "taskspec" and row["license_ids"] == []
    digest = hashlib.sha256(StoragePath(ref["uri"]).read_bytes()).hexdigest()
    assert ref["sha256"] == digest


def test_dry_run_reports_without_sealing_or_queueing(tmp_path, sync_env):
    """A dry run touches neither the release prefix nor the queue."""
    queue = _queue(tmp_path)
    proof = main_sources.run_main_source_sync(_inputs(tmp_path, queue), dry_run=True)
    assert "report" not in proof
    assert proof["appended"] == []
    assert "Marin:sync-new" not in queue.read_text()


def test_digest_mismatch_refuses(tmp_path, sync_env):
    """A declarations document whose bytes drift from its pin refuses."""
    inputs = _inputs(tmp_path, None)
    drifted = main_sources.MainSourceSyncInputs(
        release_id=inputs.release_id,
        source_revision=inputs.source_revision,
        main_source_revision=MAIN_REVISION,
        declarations=ArtifactRef(uri=inputs.declarations.uri, sha256="f" * 64),
        report_root=inputs.report_root,
        queue_path=None,
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        main_sources.run_main_source_sync(drifted)


def test_sync_node_is_plan_gated_and_never_moves_another_fingerprint():
    """The node exists for every plan, but only an opt-in plan can run it.

    Opting in changes the sync node's own fingerprint alone: every other node's
    fingerprint is identical with and without the opt-in, so the wiring folds into
    no in-flight bundle.
    """
    without = release_graph(fixture_plan())
    with_sync = release_graph(fixture_plan(main_source_sync={"main_source_revision": MAIN_REVISION}))
    assert set(without.stages) | {"main_source_sync"} == set(with_sync.stages)
    assert with_sync.stages["main_source_sync"].deps[0].name == "atlas-release-testrelease-declarations"
    # An opted-in plan legitimately carries a different plan document; the byte
    # stability of an unconfigured plan is pinned by the plan-boundary test above.
    for node, handle in without.stages.items():
        if node == "main_source_sync":
            continue
        assert handle.fingerprint() == with_sync.stages[node].fingerprint(), node
    assert with_sync.stages["main_source_sync"].fingerprint() != without.stages["main_source_sync"].fingerprint()
