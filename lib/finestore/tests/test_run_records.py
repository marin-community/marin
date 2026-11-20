# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.rl.run_records import (
    ATTEMPTS_TABLE,
    EVALUATIONS_TABLE,
    METRICS_TABLE,
    AttemptRecord,
    AttemptRow,
    CheckpointReference,
    EvaluationTaskRow,
    LaunchResult,
    LaunchState,
    MetricRow,
    read_attempt,
    read_attempt_metrics,
    read_evaluations,
    register_run_tables,
    run_records_root,
    update_attempt,
    write_evaluations,
)
from finestore.store import DataStore


def _write(export_root, attempt_id, metrics=None, **fields):
    with DataStore.open(run_records_root(str(export_root))) as store:
        update_attempt(store, AttemptRecord(run_id="history", attempt_id=attempt_id, **fields))
        table = register_run_tables(store)[METRICS_TABLE]
        for step, values in (metrics or {}).items():
            table.extend(
                [
                    MetricRow(attempt_id=attempt_id, step=step, name=name, value=value).model_dump()
                    for name, value in values.items()
                ]
            )


def test_evaluations_read_legacy_strata_alongside_current_rows(tmp_path):
    export = str(tmp_path)
    answer = {
        "ordinal": 0,
        "native_uid": "answer",
        "optimization_reward": 0.0,
        "verifier_score": None,
        "raw_verifier_score": None,
        "score_min": None,
        "score_max": None,
        "verification_status": None,
        "passed": False,
        "outcome": "unavailable",
        "error_type": None,
        "error_treatment": None,
    }
    legacy = {
        "attempt_id": "attempt",
        "step": 1,
        "evaluation": "eval/",
        "panel_name": "eval-lite",
        "panel_version": "old",
        "task_id": "legacy",
        "data_source": "fixture",
        "strata": {"family": "agents"},
        "agentic": True,
        "score": None,
        "outcome": "unavailable",
        "error_type": None,
        "answers": [answer],
    }
    with DataStore.open(run_records_root(export)) as store:
        store.table(
            EVALUATIONS_TABLE,
            primary_key=("attempt_id", "step", "evaluation", "task_id"),
            on_conflict=OnConflict.SUPERSEDE,
        ).extend([legacy, {**legacy, "task_id": "unknown", "agentic": None}])
    current = EvaluationTaskRow.model_validate(
        {
            **legacy,
            "task_id": "current",
            "task_digest": "digest",
            "strata": {"family": "agents", "agentic": False},
            "answers": [answer | {"response_token_count": 12}],
        }
    )
    write_evaluations(export, [current])
    rows = {row.task_id: row for row in read_evaluations(export, attempt_id="attempt", evaluation="eval/")}
    assert rows["legacy"].strata.agentic is True and rows["legacy"].task_digest is None
    assert rows["legacy"].answers[0].response_token_count is None
    assert rows["unknown"].strata.agentic is False and rows["unknown"].task_digest is None
    assert rows["current"] == current and rows["current"].strata.agentic is False


def test_metric_history_bounds_checkpoint_ancestors_and_detects_damaged_lineage(tmp_path):
    ancestor = str(tmp_path / "ancestor")
    root = str(tmp_path / "exports")
    _write(ancestor, "original", {step: {"loss": step / 10} for step in (0, 1, 2, 3, 8)})
    _write(ancestor, "original", {0: {"ancestor_only": 77}})
    parent = CheckpointReference(export_root=ancestor, attempt_id="original", step=3)
    _write(root, "middle", {4: {"loss": 0.4}, 5: {"loss": 0.5}, 9: {"loss": 0.9}}, resumed_step=3, parent=parent)
    with DataStore.open(run_records_root(root)) as store:
        register_run_tables(store)[ATTEMPTS_TABLE].extend(
            [AttemptRow(attempt_id="middle", field="issue", value_json='"opted_out"').model_dump()]
        )
    assert read_attempt(root, "middle").issue is None
    _write(root, "sibling", {4: {"loss": 9.9}}, resumed_step=3, parent=parent)
    assert [row["loss"] for row in read_attempt_metrics(root, "middle").values()] == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    _write(root, "rewind", {2: {"loss": 2.5}}, resumed_step=1, parent=parent.model_copy(update={"step": 1}))
    rewind = read_attempt_metrics(root, "rewind")
    assert list(rewind) == [0, 1, 2] and rewind[2]["loss"] == 2.5
    _write(
        root,
        "rewind-middle",
        {4: {"loss": 9.3}},
        resumed_step=2,
        parent=CheckpointReference(export_root=root, attempt_id="middle", step=2),
    )
    middle_rewind = read_attempt_metrics(root, "rewind-middle")
    assert list(middle_rewind) == [0, 1, 2, 4] and middle_rewind[4]["loss"] == 9.3
    _write(
        root,
        "stage",
        {0: {"loss": 9}, 1: {"loss": 8}},
        resumed_step=0,
        parent=CheckpointReference(export_root=root, attempt_id="middle", step=5),
    )
    stage = read_attempt_metrics(root, "stage")
    assert list(stage) == [0, 1] and [row["loss"] for row in stage.values()] == [9, 8]
    assert "ancestor_only" not in stage[0] and read_attempt(root, "stage").parent.step == 5
    _write(
        root,
        "missing-parent",
        {3: {"loss": 3}},
        resumed_step=2,
        parent=CheckpointReference(export_root=ancestor, attempt_id="unrecorded", step=2),
    )
    assert read_attempt_metrics(root, "missing-parent")[3]["loss"] == 3
    _write(root, "main-resume", {9: {"loss": 0.9}}, resumed_step=8)
    assert read_attempt_metrics(root, "main-resume")[9]["loss"] == 0.9
    assert read_attempt(root, "main-resume").resumed_step == 8 and read_attempt(root, "main-resume").parent is None
    absent = str(tmp_path / "absent")
    _write(
        root,
        "missing-store",
        {3: {"loss": 0.3}},
        resumed_step=2,
        parent=CheckpointReference(export_root=absent, attempt_id="unknown", step=2),
    )
    _write(
        root,
        "grandchild",
        {4: {"loss": 0.4}},
        resumed_step=3,
        parent=CheckpointReference(export_root=root + "/", attempt_id="missing-store", step=3),
    )
    grandchild = read_attempt_metrics(root, "grandchild")
    assert list(grandchild) == [3, 4] and grandchild[3]["loss"] == 0.3 and grandchild[4]["loss"] == 0.4
    _write(absent, "unknown")
    manifest = Path(ReadView(run_records_root(absent)).token.manifest_path)
    contents = manifest.read_bytes()
    manifest.unlink()
    try:
        with pytest.raises(FileNotFoundError):
            read_attempt_metrics(root, "missing-store")
    finally:
        manifest.write_bytes(contents)
    cycle = str(tmp_path / "cycle")
    for attempt, parent_id in (("left", "right"), ("right", "left")):
        _write(
            cycle,
            attempt,
            resumed_step=1,
            parent=CheckpointReference(export_root=cycle + "/", attempt_id=parent_id, step=1),
        )
    with pytest.raises(ValueError, match="cycle"):
        read_attempt_metrics(cycle, "left")


@pytest.mark.parametrize("first", ["metrics", "result"])
def test_attempt_updates_preserve_other_writers_and_explicit_null_clears(tmp_path, first):
    root = str(tmp_path / first)
    attempt_id = "race-" + first
    parent = CheckpointReference(export_root="ancestor", attempt_id="original", step=3)
    _write(root, attempt_id, started_at="2026-11-11T01:00:00+00:00", vllm_version="first", iris_job_id="/tests/worker")
    with DataStore.open(run_records_root(root)) as left, DataStore.open(run_records_root(root)) as right:
        metrics = AttemptRecord(
            run_id="history",
            attempt_id=attempt_id,
            metrics_started_at="2026-11-11T01:01:00+00:00",
            resumed_step=3,
            parent=parent,
        )
        result = AttemptRecord(
            run_id="history",
            attempt_id=attempt_id,
            result=LaunchResult(state=LaunchState.FAILED, failure="interleaved terminal"),
        )
        updates = ((left, metrics), (right, result))
        for store, update in updates if first == "metrics" else reversed(updates):
            update_attempt(store, update)
    stored = read_attempt(root, attempt_id)
    assert stored.started_at == "2026-11-11T01:00:00+00:00" and stored.vllm_version == "first"
    assert stored.iris_job_id == "/tests/worker"
    assert stored.parent == parent and stored.resumed_step == 3
    assert stored.metrics_started_at == "2026-11-11T01:01:00+00:00"
    assert stored.result.state == LaunchState.FAILED and stored.result.failure == "interleaved terminal"
    with DataStore.open(run_records_root(root)) as clear:
        update_attempt(clear, AttemptRecord(run_id="history", attempt_id=attempt_id, vllm_version=None))
    cleared = read_attempt(root, attempt_id)
    assert cleared.vllm_version is None and cleared.parent == parent and cleared.result == stored.result
