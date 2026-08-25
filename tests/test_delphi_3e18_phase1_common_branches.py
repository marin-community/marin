# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class _Device:
    platform: str
    device_kind: str


def _full_run_spec(
    *,
    tpu_type: str = "v5p-8",
    tpu_region: str = "us-east5",
    tpu_zone: str = "us-east5-a",
) -> base.DelphiSwarmRunSpec:
    return base.DelphiSwarmRunSpec(
        run_order=0,
        run_id=1,
        run_name="source",
        source_run_name="source",
        source_experiment="source",
        panel_source="test",
        target_flops=3e18,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        batch_size=128,
        train_steps=3_007,
        realized_train_tokens=1_576_534_016,
        expected_checkpoint_step=3_006,
        model_hidden_dim=896,
        model_layers=10,
        non_embedding_params=128_469_376,
        total_trainable_params=358_304_128,
        tensor_parallel_size=1,
        data_seed=930_000,
        trainer_seed=0,
        phase_boundary=0.8,
        phase_0_fraction=0.8,
        phase_1_fraction=0.2,
        simulated_epoch_target_budget=1_576_534_016,
        available_top_level_tokens=1_576_534_016,
        max_simulated_epoch=1.0,
        q95_simulated_epoch=1.0,
        mean_phase_tv_to_proportional=0.1,
        phase_weights={"phase_0": {"bucket": 1.0}, "phase_1": {"bucket": 1.0}},
    )


def _branch_training_config() -> branches.BranchTrainingConfig:
    return branches.BranchTrainingConfig(
        experiment_name=branches.V6E_EXPERIMENT_NAME,
        analysis_output_path="analysis",
        output_path="output",
        run_spec=branches.move_run_spec_to_branch_hardware(_full_run_spec(), branches.V6E_DEPLOYMENT),
        validation_configs=None,
        prefix_checkpoint=branches.PrefixCheckpoint(
            candidate_id="shared_bounded_ensemble_kl0p05",
            repeat_seed=0,
            checkpoint_uri="gs://marin-us-east5/prefix/step-2399",
            provenance_sha256="a" * 64,
        ),
        prefix_replay_code_commit="b" * 40,
        candidate_weights_sha256="c" * 64,
        continuation_weights_sha256="d" * 64,
        continuation_id="fit_maximin_00",
        code_commit="e" * 40,
        prefix_hardware=branches.PREFIX_HARDWARE,
        continuation_hardware=branches.V6E_DEPLOYMENT.hardware,
        continuation_hardware_version=versioned(branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)),
    )


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
    assert tuple(row["run_order"] for row in noise_rows) == branches.hardware_canary_gate().noise_run_orders


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


def test_materializer_filters_manifest_by_observed_and_declared_hardware(tmp_path) -> None:
    valid_observation = {
        "platform": "tpu",
        "device_kind": "TPU v6e",
        "global_device_count": 8,
        "local_device_count": 8,
    }
    assert (
        materialize.validate_observed_hardware(
            valid_observation, materialize.TpuHardware("v6e-8", "us-east5", "us-east5-b")
        )
        == valid_observation
    )
    with pytest.raises(ValueError, match="device count"):
        materialize.validate_observed_hardware(
            {**valid_observation, "global_device_count": 4},
            materialize.TpuHardware("v6e-8", "us-east5", "us-east5-b"),
        )

    common = {
        "experiment_name": branches.V6E_EXPERIMENT_NAME,
        "prefix_hardware": asdict(materialize.PREFIX_HARDWARE),
        "candidate_weights_sha256": CANDIDATE_SHA256,
        "continuation_weights_sha256": CONTINUATION_SHA256,
        "selected_prefixes_sha256": "selected",
        "prefix_replay_code_commit": "prefix",
        "code_commit": "branch",
        "expected_full_design_rows": materialize.EXPECTED_FULL_ROWS,
        "selected_design_rows": materialize.EXPECTED_FULL_ROWS,
        "branch_rows": [{}] * materialize.EXPECTED_FULL_ROWS,
        "hardware_canary_gate": materialize.hardware_canary_gate_payload(),
    }
    for name, hardware in (
        ("manifest-v5p", branches.V5P_DEPLOYMENT.hardware),
        ("manifest-v6e", branches.V6E_DEPLOYMENT.hardware),
    ):
        manifest_dir = tmp_path / name
        manifest_dir.mkdir()
        (manifest_dir / "manifest.json").write_text(
            json.dumps(
                {
                    **common,
                    "continuation_hardware": asdict(hardware),
                    "panel_hardware_status": branches.panel_hardware_status(hardware),
                }
            )
        )
    fs, root = materialize.fsspec.core.url_to_fs(str(tmp_path))
    manifest_path, _ = materialize.matching_full_manifest(
        fs,
        root,
        candidate_sha256=CANDIDATE_SHA256,
        continuation_sha256=CONTINUATION_SHA256,
        selected_prefixes_sha256="selected",
        prefix_replay_code_commit="prefix",
        branch_code_commit="branch",
        expected_experiment_name=branches.V6E_EXPERIMENT_NAME,
        continuation_hardware=materialize.TpuHardware("v6e-8", "us-east5", "us-east5-b"),
    )

    assert "manifest-v6e" in manifest_path


def test_branch_hardware_migration_preserves_scientific_run_spec() -> None:
    source = _full_run_spec()

    migrated = branches.move_run_spec_to_branch_hardware(source, branches.V6E_DEPLOYMENT)
    expected = asdict(source)
    expected.update(
        tpu_type="v6e-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-b",
        tensor_parallel_size=1,
    )

    assert asdict(migrated) == expected
    assert asdict(materialize.PREFIX_HARDWARE) == asdict(branches.PREFIX_HARDWARE)
    assert asdict(materialize.HardwareCanaryGate()) == asdict(branches.hardware_canary_gate())
    with pytest.raises(ValueError, match="Unsupported branch TPU deployment"):
        branches.resolve_branch_deployment("v6e-8", "us-east5", "us-east5-a")
    assert branches.resolve_branch_deployment("v5p-8", "us-east5", "us-east5-a") == branches.V5P_DEPLOYMENT
    with pytest.raises(ValueError, match="Prefix run spec hardware changed"):
        branches.move_run_spec_to_branch_hardware(
            _full_run_spec(tpu_type="v6e-8", tpu_zone="us-east5-b"), branches.V6E_DEPLOYMENT
        )


def test_observed_hardware_and_worker_guards(monkeypatch) -> None:
    monkeypatch.setattr(branches.jax, "devices", lambda: [_Device("tpu", "TPU v6e") for _ in range(8)])
    monkeypatch.setattr(branches.jax, "local_device_count", lambda: 8)

    observed = branches.observe_tpu_hardware(branches.V6E_DEPLOYMENT.hardware)

    assert observed == branches.ObservedTpuHardware(
        platform="tpu",
        device_kind="TPU v6e",
        global_device_count=8,
        local_device_count=8,
    )
    config = _branch_training_config()
    with pytest.raises(ValueError, match="Prefix hardware changed"):
        branches.verify_prefix_checkpoint_on_worker(replace(config, prefix_hardware=branches.V6E_DEPLOYMENT.hardware))
    with pytest.raises(ValueError, match="run-spec hardware"):
        branches.verify_prefix_checkpoint_on_worker(
            replace(config, continuation_hardware=branches.V5P_DEPLOYMENT.hardware)
        )


def test_manifest_step_versions_the_selected_run_orders() -> None:
    one_row = branches.SaveBranchManifestConfig(
        experiment_name=branches.V6E_EXPERIMENT_NAME,
        output_path="unused",
        selected_prefixes_json="[]",
        selected_prefixes_sha256="selected",
        candidate_weights_sha256=CANDIDATE_SHA256,
        continuation_weights_sha256=CONTINUATION_SHA256,
        prefix_replay_code_commit="prefix",
        code_commit="branch",
        branch_rows_json="[]",
        selected_run_orders=versioned((0,)),
        prefix_hardware=branches.PREFIX_HARDWARE,
        continuation_hardware=branches.V6E_DEPLOYMENT.hardware,
        continuation_hardware_version=versioned(branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)),
    )
    full_panel = branches.SaveBranchManifestConfig(
        experiment_name=branches.V6E_EXPERIMENT_NAME,
        output_path="unused",
        selected_prefixes_json="[]",
        selected_prefixes_sha256="selected",
        candidate_weights_sha256=CANDIDATE_SHA256,
        continuation_weights_sha256=CONTINUATION_SHA256,
        prefix_replay_code_commit="prefix",
        code_commit="branch",
        branch_rows_json="[]",
        selected_run_orders=versioned(tuple(range(branches.TOTAL_BRANCH_ROWS))),
        prefix_hardware=branches.PREFIX_HARDWARE,
        continuation_hardware=branches.V6E_DEPLOYMENT.hardware,
        continuation_hardware_version=versioned(branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)),
    )

    one_row_version = collect_dependencies_and_version(one_row).version
    full_panel_version = collect_dependencies_and_version(full_panel).version

    expected_hardware_version = branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)
    assert one_row_version == {
        "selected_run_orders": (0,),
        "continuation_hardware_version": expected_hardware_version,
    }
    assert full_panel_version == {
        "selected_run_orders": tuple(range(branches.TOTAL_BRANCH_ROWS)),
        "continuation_hardware_version": expected_hardware_version,
    }


def test_branch_wandb_tags_fit_wandb_limit() -> None:
    config = _branch_training_config()

    tags = branches.branch_wandb_tags(config)
    version = collect_dependencies_and_version(config).version

    assert max(map(len, tags)) <= branches.WANDB_TAG_MAX_LENGTH
    assert "prefix_replay_commit=" + "b" * branches.WANDB_HASH_TAG_LENGTH in tags
    assert "continuation_sha=" + "d" * branches.WANDB_HASH_TAG_LENGTH in tags
    assert "prefix_tpu=v5p-8" in tags
    assert "continuation_tpu=v6e-8" in tags
    assert "continuation_zone=us-east5-b" in tags
    assert version == {"continuation_hardware_version": branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)}
