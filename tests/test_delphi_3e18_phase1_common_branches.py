# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import cast

import pytest
from marin.execution.executor import collect_dependencies_and_version
from marin.execution.types import versioned

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_common_branches as branches
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_common_branches_20260824 as materialize,
)

CANDIDATE_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
CONTINUATION_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
SELECTED_CANDIDATES = (
    "observed_cap10_best",
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
)


@dataclass(frozen=True)
class _PrefixSpec:
    phase_weights: dict[str, dict[str, float]]
    data_seed: int
    trainer_seed: int


def test_frozen_continuation_design_obeys_runtime_contract() -> None:
    buckets, continuations = branches.load_continuations(
        branches.DEFAULT_CONTINUATION_WEIGHTS,
        CONTINUATION_SHA256,
        branches.DEFAULT_CANDIDATE_WEIGHTS,
        CANDIDATE_SHA256,
    )

    assert len(buckets) == 39
    assert len(continuations) == branches.COMMON_CONTINUATION_COUNT
    assert sum(bool(row["fit_budget"]) for row in continuations) == branches.COMMON_FIT_CONTINUATION_COUNT
    assert max(float(row["max_phase_1_materialized_epoch"]) for row in continuations) <= (
        branches.HISTORICAL_PHASE_1_EPOCH_CAP
    )
    assert max(float(row["max_total_materialized_epoch_across_candidate_prefixes"]) for row in continuations) <= (
        branches.HISTORICAL_TOTAL_EPOCH_CAP
    )


def test_branch_panel_crosses_common_fit_rows_and_keeps_controls_outside_budget() -> None:
    buckets, continuations = branches.load_continuations(
        branches.DEFAULT_CONTINUATION_WEIGHTS,
        CONTINUATION_SHA256,
        branches.DEFAULT_CANDIDATE_WEIGHTS,
        CANDIDATE_SHA256,
    )
    uniform = {bucket: 1.0 / len(buckets) for bucket in buckets}
    prefixes = []
    prefix_specs = {}
    for candidate_id in SELECTED_CANDIDATES:
        for repeat_seed in (branches.PRIMARY_BRANCH_SEED, branches.STABILITY_BRANCH_SEED):
            prefix = branches.PrefixCheckpoint(
                candidate_id=candidate_id,
                repeat_seed=repeat_seed,
                checkpoint_uri=f"gs://marin-us-east5/{candidate_id}/step-2399",
                provenance_sha256=f"provenance-{candidate_id}-{repeat_seed}",
            )
            prefixes.append(prefix)
            prefix_specs[(candidate_id, repeat_seed)] = cast(
                base.DelphiSwarmRunSpec,
                _PrefixSpec(
                    phase_weights={"phase_0": uniform, "phase_1": uniform},
                    data_seed=930_000 + repeat_seed,
                    trainer_seed=repeat_seed,
                ),
            )

    rows = branches.enrich_branch_rows(
        branches.branch_rows(prefixes=prefixes, prefix_specs=prefix_specs, continuations=continuations),
        prefix_specs,
    )

    assert len(rows) == branches.TOTAL_BRANCH_ROWS == 232
    assert sum(bool(row["fit_budget"]) for row in rows) == 200
    assert sum(row["branch_role"] == "primary_cross" for row in rows) == 212
    assert sum(row["branch_role"] == "prefix_tied_control" for row in rows) == 4
    assert sum(row["branch_role"] == "prefix_seed_stability_sentinel" for row in rows) == 12
    assert sum(row["branch_role"] == "same_prefix_branch_noise" for row in rows) == 4
    assert len({row["run_name"] for row in rows}) == len(rows)

    fit_rows = [row for row in rows if row["fit_budget"]]
    fit_by_prefix = {
        candidate_id: {row["continuation_id"] for row in fit_rows if row["prefix"].candidate_id == candidate_id}
        for candidate_id in SELECTED_CANDIDATES
    }
    assert all(len(continuation_ids) == 50 for continuation_ids in fit_by_prefix.values())
    assert len({frozenset(continuation_ids) for continuation_ids in fit_by_prefix.values()}) == 1

    noise_rows = [row for row in rows if row["branch_role"] == "same_prefix_branch_noise"]
    assert {row["prefix"].candidate_id for row in noise_rows} == {branches.BRANCH_NOISE_PREFIX_CANDIDATE}
    assert len({row["data_seed"] for row in noise_rows}) == branches.BRANCH_NOISE_REPEAT_COUNT
    assert len({row["trainer_seed"] for row in noise_rows}) == 1
    assert len({branches.phase_weights_sha256(row["phase_weights"]) for row in noise_rows}) == 1


def test_terminal_metric_record_accepts_identical_retry_rows(tmp_path) -> None:
    run_name = "branch_retry"
    metric_dir = tmp_path / f"{run_name}-deadbeef" / "checkpoints"
    metric_dir.mkdir(parents=True)
    record = {
        "step": materialize.EXPECTED_TERMINAL_STEP,
        materialize.PRIMARY_METRIC: 1.0,
        materialize.DIAGNOSTIC_METRIC: 0.8,
    }
    metric_path = metric_dir / "eval_metrics.jsonl"
    metric_path.write_text("\n".join([json.dumps(record), json.dumps(record)]) + "\n")

    fs, root = materialize.fsspec.core.url_to_fs(str(tmp_path))
    _, observed = materialize.metric_record(fs, root, run_name)
    assert observed == record

    conflicting = {**record, materialize.PRIMARY_METRIC: 1.1}
    metric_path.write_text("\n".join([json.dumps(record), json.dumps(conflicting)]) + "\n")
    with pytest.raises(ValueError, match="Conflicting step-3006 metric rows"):
        materialize.metric_record(fs, root, run_name)


def test_manifest_step_versions_the_selected_run_orders() -> None:
    one_row = branches.SaveBranchManifestConfig(
        output_path="unused",
        selected_prefixes_json="[]",
        selected_prefixes_sha256="selected",
        candidate_weights_sha256=CANDIDATE_SHA256,
        continuation_weights_sha256=CONTINUATION_SHA256,
        prefix_replay_code_commit="prefix",
        code_commit="branch",
        branch_rows_json="[]",
        selected_run_orders=versioned((0,)),
    )
    full_panel = branches.SaveBranchManifestConfig(
        output_path="unused",
        selected_prefixes_json="[]",
        selected_prefixes_sha256="selected",
        candidate_weights_sha256=CANDIDATE_SHA256,
        continuation_weights_sha256=CONTINUATION_SHA256,
        prefix_replay_code_commit="prefix",
        code_commit="branch",
        branch_rows_json="[]",
        selected_run_orders=versioned(tuple(range(branches.TOTAL_BRANCH_ROWS))),
    )

    one_row_version = collect_dependencies_and_version(one_row).version
    full_panel_version = collect_dependencies_and_version(full_panel).version

    assert one_row_version == {"selected_run_orders": (0,)}
    assert full_panel_version == {"selected_run_orders": tuple(range(branches.TOTAL_BRANCH_ROWS))}
