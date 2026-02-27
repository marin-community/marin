# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import asdict

import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.contracts import (
    Candidate,
    LoopConfig,
    LoopState,
    RunRecord,
    ValidationRecord,
)
from experiments.domain_phase_mix.nextgen.design import plan_new_runs
from experiments.domain_phase_mix.nextgen.fit_propose import (
    CANDIDATE_ASSIGNMENTS_JSON,
    CANDIDATES_JSON,
)
from experiments.domain_phase_mix.nextgen.import_sources import (
    THREE_PHASE_EXPERIMENT,
    THREE_PHASE_STARCODER_EXPERIMENT,
    TWO_PHASE_STARCODER_EXPERIMENT,
    default_legacy_sources,
)
from experiments.domain_phase_mix.nextgen.merge_export import (
    MERGED_RUNS_JSON,
    MERGED_TRAJ_PARQUET,
    RESULTS_CSV,
    RUNS_PARQUET,
    TRAJ_CSV,
    ExportDatasetConfig,
    MergeDatasetConfig,
    export_dataset,
    merge_dataset,
)
from experiments.domain_phase_mix.nextgen.collect import (
    IMPORTED_RUNS_FILE,
    IMPORTED_TRAJ_FILE,
    NEW_RUNS_FILE,
    NEW_TRAJ_FILE,
)
from experiments.domain_phase_mix.nextgen.state_store import write_loop_state
from experiments.domain_phase_mix.nextgen.validation import (
    CollectValidationConfig,
    PENDING_CANDIDATES_JSON,
    PlanValidationConfig,
    SLOT_ASSIGNMENTS_JSON,
    VALIDATION_RESULTS_JSON,
    VALIDATION_PLAN_JSON,
    collect_validation_results,
    plan_validation,
    run_validation_slot,
    ValidationSlotConfig,
)


class _FakeSampler:
    def __init__(self, configs):
        self._configs = configs

    def sample_n_configs(self, n, deduplicate=True, existing_configs=None):
        assert n == len(self._configs)
        return self._configs


class _FakeExperiment:
    def __init__(self, configs):
        self._sampler = _FakeSampler(configs)

    def create_weight_sampler(self, seed=42):
        return self._sampler


def _write_json(path: str | os.PathLike[str], payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)



def test_append_runs_planning_uses_next_local_run_id():
    existing_state = LoopState(
        loop_name="loop",
        objective_metric="eval/loss",
        next_local_run_id=8,
        runs=[
            RunRecord(
                wandb_run_id="abc",
                source_experiment="loop",
                local_run_id=7,
                run_name="run_00007",
                phase_weights={"phase_0": {"a": 0.7, "b": 0.3}},
                status="completed",
            )
        ],
    )

    new_configs = [
        WeightConfig(run_id=0, phase_weights={"phase_0": {"a": 0.2, "b": 0.8}}),
        WeightConfig(run_id=1, phase_weights={"phase_0": {"a": 0.1, "b": 0.9}}),
    ]
    experiment = _FakeExperiment(new_configs)

    loop = LoopConfig(name="loop", objective_metric="eval/loss", model_names=("Linear",), n_new_runs=2)
    planned = plan_new_runs(loop, experiment, existing_state)

    assert [item.local_run_id for item in planned] == [8, 9]
    assert [item.run_name for item in planned] == ["run_00008", "run_00009"]


def test_default_legacy_sources_cover_core_experiments(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://unit-test-prefix")
    sources = default_legacy_sources()
    source_names = {source.source_experiment for source in sources}
    assert TWO_PHASE_STARCODER_EXPERIMENT in source_names
    assert THREE_PHASE_EXPERIMENT in source_names
    assert THREE_PHASE_STARCODER_EXPERIMENT in source_names



def test_export_csv_contains_phase_columns_and_objective(tmp_path):
    merged_dir = tmp_path / "merged"
    export_dir = tmp_path / "export"
    merged_dir.mkdir()

    runs = [
        RunRecord(
            wandb_run_id="run-a",
            source_experiment="legacy",
            local_run_id=42,
            run_name="run_00042",
            phase_weights={
                "phase_0": {"nemotron_full": 0.9, "starcoder": 0.1},
                "phase_1": {"nemotron_full": 0.5, "starcoder": 0.5},
            },
            status="completed",
            metrics={"eval/paloma/dolma_100_programing_languages/bpb": 0.8123},
        )
    ]

    _write_json(str(merged_dir / MERGED_RUNS_JSON), [asdict(r) for r in runs])

    traj = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "legacy",
                "local_run_id": 42,
                "run_name": "run_00042",
                "step": 1000,
                "total_tokens": 1.0e8,
                "metric_key": "eval/paloma/dolma_100_programing_languages/bpb",
                "metric_value": 0.95,
            },
            {
                "wandb_run_id": "run-a",
                "source_experiment": "legacy",
                "local_run_id": 42,
                "run_name": "run_00042",
                "step": 2000,
                "total_tokens": 2.0e8,
                "metric_key": "eval/paloma/dolma_100_programing_languages/bpb",
                "metric_value": 0.81,
            },
        ]
    )
    traj.to_parquet(merged_dir / MERGED_TRAJ_PARQUET, index=False)

    export_dataset(
        ExportDatasetConfig(
            output_path=str(export_dir),
            merged_output_path=str(merged_dir),
        )
    )

    result_df = pd.read_csv(export_dir / RESULTS_CSV)
    assert "phase_0_nemotron_full" in result_df.columns
    assert "phase_1_starcoder" in result_df.columns
    assert "eval/paloma/dolma_100_programing_languages/bpb" in result_df.columns

    traj_df = pd.read_csv(export_dir / TRAJ_CSV)
    assert len(traj_df) == 2

    # Parquet exports must exist for downstream model fitting.
    assert (export_dir / RUNS_PARQUET).exists()



def test_model_resubmit_validation_plan_reuses_prior_candidate(tmp_path):
    state_dir = tmp_path / "state"
    fit_dir = tmp_path / "fit"
    output_dir = tmp_path / "plan"
    state_dir.mkdir()
    fit_dir.mkdir()

    state = LoopState(
        loop_name="loop",
        objective_metric="eval/loss",
        validated_candidates={
            "cand-1": ValidationRecord(
                candidate_id="cand-1",
                model_name="Linear",
                status="planned",
                metric_value=0.81,
            )
        },
    )
    write_loop_state(os.path.join(state_dir, "loop_state.json"), state)

    _write_json(
        os.path.join(fit_dir, CANDIDATES_JSON),
        [
            asdict(
                Candidate(
                    candidate_id="cand-1",
                    model_name="Linear",
                    kind="schedule",
                    phase_weights={"phase_0": {"a": 0.8, "b": 0.2}},
                    policy_ref=None,
                    predicted_objective=0.79,
                )
            )
        ],
    )
    _write_json(os.path.join(fit_dir, CANDIDATE_ASSIGNMENTS_JSON), {"Linear": "cand-1"})

    plan_validation(
        PlanValidationConfig(
            output_path=str(output_dir),
            fit_output_path=str(fit_dir),
            state_output_path=str(state_dir),
            model_names_json=json.dumps(["Linear"]),
        )
    )

    pending = json.load(open(output_dir / PENDING_CANDIDATES_JSON))
    assert pending == []

    slot_assignments = json.load(open(output_dir / SLOT_ASSIGNMENTS_JSON))
    assert slot_assignments["Linear"] is None

    validation_plan = json.load(open(output_dir / VALIDATION_PLAN_JSON))
    assert validation_plan[0]["status"] == "reused"


def test_legacy_merge_dedup_by_wandb_preserves_local_run_id(tmp_path):
    state_dir = tmp_path / "state"
    import_dir = tmp_path / "imported"
    new_dir = tmp_path / "new"
    merge_dir = tmp_path / "merge"
    state_dir.mkdir()
    import_dir.mkdir()
    new_dir.mkdir()
    merge_dir.mkdir()

    state = LoopState(
        loop_name="loop",
        objective_metric="eval/loss",
        runs=[
            RunRecord(
                wandb_run_id="run-1",
                source_experiment="legacy",
                local_run_id=11,
                run_name="run_00011",
                phase_weights={"phase_0": {"a": 0.6, "b": 0.4}},
                status="completed",
                metrics={"eval/loss": 1.2},
            )
        ],
    )
    write_loop_state(os.path.join(state_dir, "loop_state.json"), state)

    imported_runs = [
        asdict(
            RunRecord(
                wandb_run_id="run-1",
                source_experiment="legacy",
                local_run_id=11,
                run_name="run_00011",
                phase_weights={"phase_0": {"a": 0.7, "b": 0.3}},
                status="completed",
                metrics={"eval/loss": 0.9},
            )
        )
    ]
    _write_json(import_dir / IMPORTED_RUNS_FILE, imported_runs)

    new_runs = [
        asdict(
            RunRecord(
                wandb_run_id="run-1",
                source_experiment="loop",
                local_run_id=None,
                run_name=None,
                phase_weights={},
                status="completed",
                metrics={"eval/loss": 0.8, "eval/aux": 0.3},
            )
        )
    ]
    _write_json(new_dir / NEW_RUNS_FILE, new_runs)

    pd.DataFrame(
        [
            {
                "wandb_run_id": "run-1",
                "source_experiment": "legacy",
                "local_run_id": 11,
                "run_name": "run_00011",
                "step": 1000,
                "total_tokens": 1.0e8,
                "metric_key": "eval/loss",
                "metric_value": 0.9,
            }
        ]
    ).to_parquet(import_dir / IMPORTED_TRAJ_FILE, index=False)
    pd.DataFrame(
        [
            {
                "wandb_run_id": "run-1",
                "source_experiment": "loop",
                "local_run_id": None,
                "run_name": None,
                "step": 1000,
                "total_tokens": 1.0e8,
                "metric_key": "eval/loss",
                "metric_value": 0.8,
            }
        ]
    ).to_parquet(new_dir / NEW_TRAJ_FILE, index=False)

    merge_dataset(
        MergeDatasetConfig(
            output_path=str(merge_dir),
            loop_name="loop",
            objective_metric="eval/loss",
            state_output_path=str(state_dir),
            imported_output_path=str(import_dir),
            new_output_path=str(new_dir),
        )
    )

    merged_runs = json.load(open(merge_dir / MERGED_RUNS_JSON))
    assert len(merged_runs) == 1
    merged = merged_runs[0]
    assert merged["wandb_run_id"] == "run-1"
    assert merged["local_run_id"] == 11
    assert merged["metrics"]["eval/loss"] == 0.8
    assert merged["metrics"]["eval/aux"] == 0.3


def test_policy_and_schedule_candidates_share_validation_flow(tmp_path):
    state_dir = tmp_path / "state"
    fit_dir = tmp_path / "fit"
    plan_dir = tmp_path / "plan"
    slot_schedule_dir = tmp_path / "slot_schedule"
    slot_policy_dir = tmp_path / "slot_policy"
    collect_dir = tmp_path / "collect"
    state_dir.mkdir()
    fit_dir.mkdir()

    write_loop_state(
        os.path.join(state_dir, "loop_state.json"),
        LoopState(loop_name="loop", objective_metric="eval/loss"),
    )

    _write_json(
        fit_dir / CANDIDATES_JSON,
        [
            asdict(
                Candidate(
                    candidate_id="cand-schedule",
                    model_name="Linear",
                    kind="schedule",
                    phase_weights={"phase_0": {"a": 0.9, "b": 0.1}},
                    policy_ref=None,
                    predicted_objective=0.8,
                )
            ),
            {
                "candidate_id": "cand-policy",
                "model_name": "OfflineRL",
                "kind": "policy",
                "phase_weights": None,
                "policy_ref": {"uri": "gs://policy/artifact.json", "format": "json"},
                "predicted_objective": 0.79,
            },
        ],
    )
    _write_json(
        fit_dir / CANDIDATE_ASSIGNMENTS_JSON,
        {"Linear": "cand-schedule", "OfflineRL": "cand-policy"},
    )

    plan_validation(
        PlanValidationConfig(
            output_path=str(plan_dir),
            fit_output_path=str(fit_dir),
            state_output_path=str(state_dir),
            model_names_json=json.dumps(["Linear", "OfflineRL"]),
        )
    )

    pending = json.load(open(plan_dir / PENDING_CANDIDATES_JSON))
    assert {row["candidate_id"] for row in pending} == {"cand-schedule", "cand-policy"}

    run_validation_slot(
        ValidationSlotConfig(
            output_path=str(slot_schedule_dir),
            plan_output_path=str(plan_dir),
            fit_output_path=str(fit_dir),
            model_name="Linear",
            execute_slot=False,
        )
    )
    run_validation_slot(
        ValidationSlotConfig(
            output_path=str(slot_policy_dir),
            plan_output_path=str(plan_dir),
            fit_output_path=str(fit_dir),
            model_name="OfflineRL",
            execute_slot=False,
        )
    )

    collect_validation_results(
        CollectValidationConfig(
            output_path=str(collect_dir),
            slot_output_paths=(str(slot_schedule_dir), str(slot_policy_dir)),
            slot_model_names_json=json.dumps(["Linear", "OfflineRL"]),
        )
    )

    results = json.load(open(collect_dir / VALIDATION_RESULTS_JSON))
    assert len(results) == 2
    by_model = {row["model_name"]: row for row in results}
    assert by_model["Linear"]["candidate_id"] == "cand-schedule"
    assert by_model["OfflineRL"]["candidate_id"] == "cand-policy"
    assert by_model["Linear"]["status"] == "planned"
    assert by_model["OfflineRL"]["status"] == "planned"
