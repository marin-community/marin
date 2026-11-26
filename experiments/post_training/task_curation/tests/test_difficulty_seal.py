# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.ledger import append_rows
from experiments.post_training.task_curation.difficulty.seal import (
    capture_snapshot_inventory,
    seal_ledger,
    snapshot_active_ledger,
    snapshot_cells,
    verify_snapshot,
)
from experiments.post_training.task_curation.difficulty.state import CatalogSource, GapReason, LedgerRow, Outcome


def attempt(identity, task, outcome=Outcome.SCORED):
    return LedgerRow(
        "population",
        "source",
        task * 64,
        Solver.SMALL,
        identity,
        outcome,
        None if outcome == Outcome.SCORED else GapReason.GRADER_INFRASTRUCTURE_FAILURE,
        "run",
        0.0,
        "2026-11-24T18:00:00-05:00",
        task_id=task,
        reward=1.0 if outcome == Outcome.SCORED else None,
    )


def test_seal_keeps_partial_cells_unmeasured_and_preserves_parked_repairs(tmp_path):
    ledger = str(tmp_path / "ledger")
    append_rows(ledger, [attempt("one", "a"), attempt("retry", "a"), attempt("failed", "b", Outcome.UNSCORED)])
    repair = replace(attempt("repair", "b", Outcome.REPAIRED), task_sha256=None)
    append_rows(ledger, [repair])
    parked = StoragePath(ledger) / "v1/repair-verifications-parked/hold.json"
    parked.write_text('{"source":"facet_v2","state":"held"}', auto_mkdir=True)
    source = CatalogSource("source", "population", (("a", "a" * 64), ("b", "b" * 64)), "queue", "digest")
    ref = seal_ledger(ledger, [source], str(tmp_path / "sealed"), {}, "2026-11-24T18:01:00-05:00")
    manifest = verify_snapshot(ref)
    root = StoragePath(ref.uri).parent
    cells = [json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()]
    small = next(cell for cell in cells if cell["solver"] == "small")
    assert (small["scored_attempts"], small["scored_tasks"]) == (2, 1)
    assert not small["measured"] and small["difficulty"] is None and small["pass_rate"] is None
    assert small["observed_pass_rate"] == 1.0
    assert manifest["summary"]["covered_cells"] == 0
    provenance = json.loads((root / "provenance.json").read_text())
    saved = next(row for row in provenance["side_objects"] if "parked" in row["source_uri"])
    assert (root / saved["file"]).read_bytes() == parked.read_bytes()
    assert json.loads((root / "repairs.jsonl").read_text())["attempt_id"] == "repair"
    with pytest.raises(ValueError, match="ledger is sealed"):
        append_rows(ledger, [attempt("new", "b")])
    assert verify_snapshot(ref) == manifest
    assert manifest["final_sequence_number"] == 4


def test_seal_covers_only_full_current_population_and_respects_holds(tmp_path):
    ledger = str(tmp_path / "ledger")
    append_rows(
        ledger, [attempt("a", "a"), attempt("b", "b"), replace(attempt("stale", "c"), registry_revision="stale")]
    )
    source = CatalogSource("source", "population", (("a", "a" * 64), ("b", "b" * 64)), "queue", "digest")
    ref = seal_ledger(ledger, [source], str(tmp_path / "sealed"), {}, "2026-11-24T18:01:00-05:00")
    manifest = verify_snapshot(ref)
    assert manifest["summary"]["covered_cells"] == 1
    cell = json.loads((StoragePath(ref.uri).parent / "scored-cells.jsonl").read_text())
    assert (cell["scored_attempts"], cell["difficulty"], cell["pass_rate"]) == (2, 0.0, 1.0)
    held_ledger = str(tmp_path / "held-ledger")
    append_rows(held_ledger, [attempt("a", "a"), attempt("b", "b")])
    held = seal_ledger(
        held_ledger,
        [source],
        str(tmp_path / "held"),
        {"holds": {"source": {"reason": "COVERAGE_SHORTFALL", "detail": "repair verification parked"}}},
        "2026-11-24T18:01:00-05:00",
    )
    assert verify_snapshot(held)["summary"]["covered_cells"] == 0


def test_seal_rejects_corrupted_partition_and_snapshot(tmp_path):
    ledger = str(tmp_path / "ledger")
    partition = append_rows(ledger, [attempt("a", "a")])
    assert partition is not None
    path = StoragePath(partition)
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"reward": 1.0', b'"reward": 0.0'))
    with pytest.raises(ValueError, match="content hash differs"):
        seal_ledger(ledger, [], str(tmp_path / "sealed"), {}, "2026-11-24T18:01:00-05:00")
    path.write_bytes(original)
    ref = seal_ledger(ledger, [], str(tmp_path / "sealed"), {}, "2026-11-24T18:01:00-05:00")
    (StoragePath(ref.uri).parent / "repairs.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file differs"):
        verify_snapshot(ref)


def test_partial_credit_is_not_a_full_pass_and_unknown_reward_scale_fails_closed(tmp_path):
    source = CatalogSource("source", "population", (("a", "a" * 64), ("b", "b" * 64)), "queue", "digest")
    partial = replace(attempt("partial", "a"), reward=0.5)
    full = attempt("full", "b")
    small = snapshot_cells([partial, full], [source], {})[0]
    assert small["pass_rate"] == 0.5 and small["mean_reward"] == 0.75 and small["difficulty"] == 0.5
    bad = snapshot_cells([replace(partial, reward=1.25), full], [source], {})[0]
    assert not bad["measured"] and bad["pass_rate"] is None and bad["difficulty"] is None
    assert bad["unqualified_reward_attempts"] == ["partial"]


def test_active_snapshot_freezes_cutoff_preserves_repairs_and_keeps_scoring(tmp_path):
    ledger = str(tmp_path / "active")
    before = attempt("before", "a")
    after = replace(attempt("after", "b"), recorded_at="2026-11-24T18:02:00-05:00")
    append_rows(ledger, [before, after])
    parked = StoragePath(ledger) / "v1/repair-verifications-parked/nested/hold.json"
    parked.write_bytes(b'{"state":"held"}\n', auto_mkdir=True)
    source = CatalogSource("source", "population", (("a", "a" * 64), ("b", "b" * 64)), "queue", "digest")
    destination = str(tmp_path / "cut")
    capture_snapshot_inventory(ledger, destination, {}, "2026-11-24T18:01:00-05:00")
    append_rows(ledger, [replace(before, attempt_id="late-before-materialization")])
    reference = snapshot_active_ledger(ledger, [source], destination, {}, "2026-11-24T18:01:00-05:00")
    manifest = verify_snapshot(reference)
    assert manifest["final_sequence_number"] == 1
    assert manifest["summary"]["covered_cells"] == 0
    root = StoragePath(reference.uri).parent
    assert json.loads((root / "ledger.jsonl").read_text())["attempt_id"] == "before"
    provenance = json.loads((root / "provenance.json").read_text())
    saved = provenance["side_objects"][0]
    assert (root / saved["file"]).read_bytes() == parked.read_bytes()
    # A late-arriving pre-cutoff row must not alter the selected release snapshot.
    append_rows(ledger, [replace(before, attempt_id="late-arrival")])
    append_rows(ledger, [replace(after, attempt_id="still-scoring")])
    assert not (StoragePath(ledger) / "v1/closed.json").exists()
    assert snapshot_active_ledger(ledger, [], destination, {"new": "ignored"}, "2026-11-24T18:01:00-05:00") == reference
    assert verify_snapshot(reference) == manifest
