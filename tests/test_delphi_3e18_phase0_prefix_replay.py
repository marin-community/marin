# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import io

import pytest
from marin.evaluation.olmo_base_eval.components import scored_tasks

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase0_table9_20260820 as materialize,
)


def test_phase_0_boundary_matches_original_delphi_horizon() -> None:
    prefix_steps, hf_step = replay.phase_0_boundary(
        replay.EXPECTED_FULL_TRAIN_STEPS,
        replay.base.TARGET_BATCH_SIZE,
    )

    assert prefix_steps == replay.EXPECTED_PREFIX_TRAIN_STEPS
    assert hf_step == replay.EXPECTED_PREFIX_HF_STEP
    assert prefix_steps * replay.base.TARGET_BATCH_SIZE * replay.base.SEQ_LEN_DELPHI == (
        replay.EXPECTED_PREFIX_TRAIN_TOKENS
    )


def test_replay_commit_accepts_explicit_sha_without_git_metadata() -> None:
    requested_commit = "a" * 40

    assert replay.validate_replay_code_commit(requested_commit, None) == requested_commit


def test_replay_commit_rejects_local_head_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match the local workspace HEAD"):
        replay.validate_replay_code_commit("a" * 40, "b" * 40)


@pytest.mark.parametrize("requested_commit", ["", "a" * 39, "A" * 40, "g" * 40])
def test_replay_commit_requires_full_lowercase_sha(requested_commit: str) -> None:
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        replay.validate_replay_code_commit(requested_commit, None)


def _source_spec(run_order: int) -> dict:
    phase_0 = {f"bucket_{index:02d}": (index + 1) / 780 for index in range(39)}
    return {
        "run_order": run_order,
        "run_name": f"fit_{run_order:03d}_run_{run_order:05d}",
        "source_run_name": f"run_{run_order:05d}",
        "source_experiment": "source",
        "panel_source": "qsplit_signal",
        "data_seed": 1000 + run_order,
        "trainer_seed": 0,
        "phase_0_fraction": 0.8,
        "phase_1_fraction": 0.2,
        "phase_weights": {"phase_0": phase_0, "phase_1": dict(phase_0)},
    }


def _result(source_spec: dict, value: float) -> dict:
    tasks = {task: value + index / 1000 for index, task in enumerate(scored_tasks())}
    components = materialize.assemble_table9(
        {task: tasks[task] for task in materialize.leaf_components()},
        {subject: tasks[subject] for subject in materialize.mmlu_subjects()},
    )
    return {
        "name": f"t9_boundary_{source_spec['run_name']}",
        "checkpoint_path": f"gs://bucket/{source_spec['run_name']}-abc123/hf/step-2399",
        "task_bpb": tasks,
        "table9_components": components,
        "table9_macro_bpb": sum(components.values()) / len(components),
        "provenance": {
            "panel": materialize.EXPECTED_PANEL,
            "scale": materialize.EXPECTED_SCALE,
            "temporal_position": materialize.EXPECTED_TEMPORAL_POSITION,
            "source_run_name": source_spec["source_run_name"],
            "swarm_run_name": source_spec["run_name"],
            "panel_source": source_spec["panel_source"],
        },
    }


def test_materialize_phase0_table9_emits_fit_compatible_component_targets() -> None:
    specs = [_source_spec(0), _source_spec(1)]
    results = [_result(specs[0], 1.0), _result(specs[1], 2.0)]

    tables = materialize.materialize_tables(
        specs,
        results,
        ["result-0.json", "result-1.json"],
        expected_rows=2,
        allow_incomplete=False,
        source_manifest_path="manifest.json",
        source_manifest_sha256="abc",
    )

    assert len(tables.fit_matrix) == 2
    assert len(tables.components_long) == 2 * 51
    assert len(tables.tasks_long) == 2 * len(scored_tasks())
    assert "phase_0_bucket_00" in tables.fit_matrix[0]
    assert "planned_phase_1_bucket_00" not in tables.fit_matrix[0]
    assert "planned_phase_1_bucket_00" in tables.policy_registry[0]
    assert "olmo_base_eval/easy_bpb/arc_easy/bpb" in tables.fit_matrix[0]
    assert "mmlu_stem" in tables.fit_matrix[0]
    assert materialize.table9_component_key("arc_easy") in tables.metrics_wide[0]

    payload = materialize._csv_bytes(tables.fit_matrix)
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    assert [row["source_run_name"] for row in rows] == ["run_00000", "run_00001"]


def test_materialize_phase0_table9_rejects_missing_or_duplicate_results() -> None:
    specs = [_source_spec(0), _source_spec(1)]
    first = _result(specs[0], 1.0)

    with pytest.raises(ValueError, match="Missing 1/2 evaluator results"):
        materialize.materialize_tables(
            specs,
            [first],
            ["result-0.json"],
            expected_rows=2,
            allow_incomplete=False,
            source_manifest_path="manifest.json",
            source_manifest_sha256="abc",
        )

    with pytest.raises(ValueError, match="Duplicate evaluator results"):
        materialize.materialize_tables(
            specs,
            [first, first],
            ["result-0.json", "result-0-copy.json"],
            expected_rows=2,
            allow_incomplete=True,
            source_manifest_path="manifest.json",
            source_manifest_sha256="abc",
        )
