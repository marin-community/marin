# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import cast

<<<<<<< HEAD
=======
import numpy as np
import pandas as pd
>>>>>>> 0dd17851fd (Freeze Delphi KL0.05 Wave 2 acquisition)
import pytest
from marin.execution.executor import collect_dependencies_and_version
from marin.execution.types import versioned

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_common_branches as branches
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_kl0p05_wave2 as wave2_launch
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_common_branches_20260824 as materialize,
)
<<<<<<< HEAD
=======
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_kl0p05_noise_controls_20260825 as noise_materialize,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_kl0p05_wave1_20260825 as wave1_materialize,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    select_delphi_phase1_kl0p05_wave2_20260825 as wave2_select,
)
>>>>>>> 0dd17851fd (Freeze Delphi KL0.05 Wave 2 acquisition)

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


<<<<<<< HEAD
=======
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
        selection_manifest_sha256=None,
        selection_contract_sha256=None,
    )


>>>>>>> 0dd17851fd (Freeze Delphi KL0.05 Wave 2 acquisition)
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
<<<<<<< HEAD
=======
        prefix_hardware=branches.PREFIX_HARDWARE,
        continuation_hardware=branches.V6E_DEPLOYMENT.hardware,
        continuation_hardware_version=versioned(branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)),
        selection_manifest_sha256=None,
        selection_contract_sha256=None,
>>>>>>> 0dd17851fd (Freeze Delphi KL0.05 Wave 2 acquisition)
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
<<<<<<< HEAD
=======
        prefix_hardware=branches.PREFIX_HARDWARE,
        continuation_hardware=branches.V6E_DEPLOYMENT.hardware,
        continuation_hardware_version=versioned(branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)),
        selection_manifest_sha256=None,
        selection_contract_sha256=None,
>>>>>>> 0dd17851fd (Freeze Delphi KL0.05 Wave 2 acquisition)
    )

    one_row_version = collect_dependencies_and_version(one_row).version
    full_panel_version = collect_dependencies_and_version(full_panel).version

    assert one_row_version == {"selected_run_orders": (0,)}
    assert full_panel_version == {"selected_run_orders": tuple(range(branches.TOTAL_BRANCH_ROWS))}


def test_branch_wandb_tags_fit_wandb_limit() -> None:
    uniform = {"bucket": 1.0}
    config = branches.BranchTrainingConfig(
        analysis_output_path="analysis",
        output_path="output",
        run_spec=cast(
            base.DelphiSwarmRunSpec,
            _PrefixSpec(phase_weights={"phase_0": uniform, "phase_1": uniform}, data_seed=1, trainer_seed=2),
        ),
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
    )

    tags = branches.branch_wandb_tags(config)

    assert max(map(len, tags)) <= branches.WANDB_TAG_MAX_LENGTH
    assert "prefix_replay_commit=" + "b" * branches.WANDB_HASH_TAG_LENGTH in tags
    assert "continuation_sha=" + "d" * branches.WANDB_HASH_TAG_LENGTH in tags
<<<<<<< HEAD
=======
    assert "prefix_tpu=v5p-8" in tags
    assert "continuation_tpu=v6e-8" in tags
    assert "continuation_zone=us-east5-b" in tags
    assert version == {"continuation_hardware_version": branches.hardware_identity(branches.V6E_DEPLOYMENT.hardware)}


def _noise_manifest_payload() -> dict[str, object]:
    rows = []
    for position, run_order in enumerate(noise_materialize.RUN_ORDERS):
        continuation = "control_proportional" if position < 4 else "fit_maximin_26"
        repeat_index = position % 4 + 1
        rows.append(
            {
                "run_order": run_order,
                "run_id": noise_materialize.RUN_ID_BASE + run_order,
                "run_name": f"branch_{noise_materialize.TARGET_PREFIX}_seed0_{continuation}_noise{repeat_index}",
                "data_seed": noise_materialize.DATA_SEEDS[position],
                "trainer_seed": 0,
                "fit_budget": False,
                "branch_role": "same_prefix_branch_noise",
                "continuation_id": f"{continuation}_noise{repeat_index}",
                "noise_group_id": f"{noise_materialize.TARGET_PREFIX}/{continuation}",
                "branch_noise_repeat_index": repeat_index,
                "prefix": {"candidate_id": noise_materialize.TARGET_PREFIX, "repeat_seed": 0},
            }
        )
    return {
        "experiment_name": noise_materialize.EXPERIMENT_NAME,
        "candidate_weights_sha256": noise_materialize.CANDIDATE_SHA256,
        "continuation_weights_sha256": noise_materialize.CONTINUATION_SHA256,
        "selected_prefixes_sha256": noise_materialize.SELECTED_PREFIXES_SHA256,
        "prefix_replay_code_commit": noise_materialize.PREFIX_REPLAY_CODE_COMMIT,
        "code_commit": noise_materialize.BRANCH_CODE_COMMIT,
        "branch_noise_design_sha256": noise_materialize.NOISE_DESIGN_SHA256,
        "continuation_hardware": asdict(noise_materialize.CONTINUATION_HARDWARE),
        "hardware_canary_gate": noise_materialize.expected_hardware_gate(),
        "branch_run_id_base": noise_materialize.RUN_ID_BASE,
        "expected_full_design_rows": noise_materialize.EXPECTED_FULL_DESIGN_ROWS,
        "selected_design_rows": noise_materialize.EXPECTED_FRESH_ROWS,
        "fit_budget_rows": 0,
        "control_rows": noise_materialize.EXPECTED_FRESH_ROWS,
        "same_prefix_branch_noise_rows": noise_materialize.EXPECTED_FRESH_ROWS,
        "selected_run_orders": list(noise_materialize.RUN_ORDERS),
        "branch_rows": rows,
    }


def test_noise_materializer_selects_only_the_exact_eight_row_contract(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest-noise"
    manifest_dir.mkdir()
    payload = _noise_manifest_payload()
    (manifest_dir / "manifest.json").write_text(json.dumps(payload))
    distractor_dir = tmp_path / "manifest-distractor"
    distractor_dir.mkdir()
    distractor = {**payload, "code_commit": "different"}
    (distractor_dir / "manifest.json").write_text(json.dumps(distractor))

    fs, root = materialize.fsspec.core.url_to_fs(str(tmp_path))
    match = noise_materialize.matching_noise_manifest(fs, root)

    assert match is not None
    path, observed = match
    assert "manifest-noise" in path
    assert observed["selected_run_orders"] == list(noise_materialize.RUN_ORDERS)


def test_materializer_accepts_duplicate_terminal_metrics_with_only_operational_timing_differences(tmp_path) -> None:
    run_name = "branch_test"
    metric_dir = tmp_path / f"{run_name}-abc" / "checkpoints"
    metric_dir.mkdir(parents=True)
    records = [
        {
            "step": materialize.EXPECTED_TERMINAL_STEP,
            materialize.PRIMARY_METRIC: 1.0,
            materialize.DIAGNOSTIC_METRIC: 0.8,
            "eval/loading_time": 1.0,
            "eval/total_time": 2.0,
        },
        {
            "step": materialize.EXPECTED_TERMINAL_STEP,
            materialize.PRIMARY_METRIC: 1.0,
            materialize.DIAGNOSTIC_METRIC: 0.8,
            "eval/loading_time": 3.0,
            "eval/total_time": 4.0,
        },
    ]
    (metric_dir / "eval_metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    fs, root = materialize.fsspec.core.url_to_fs(str(tmp_path))

    _, observed = materialize.metric_record(fs, root, run_name)

    assert observed == records[0]


def test_materializer_rejects_duplicate_terminal_metrics_with_scientific_disagreement(tmp_path) -> None:
    run_name = "branch_test"
    metric_dir = tmp_path / f"{run_name}-abc" / "checkpoints"
    metric_dir.mkdir(parents=True)
    records = [
        {
            "step": materialize.EXPECTED_TERMINAL_STEP,
            materialize.PRIMARY_METRIC: 1.0,
            materialize.DIAGNOSTIC_METRIC: 0.8,
        },
        {
            "step": materialize.EXPECTED_TERMINAL_STEP,
            materialize.PRIMARY_METRIC: 1.001,
            materialize.DIAGNOSTIC_METRIC: 0.8,
        },
    ]
    (metric_dir / "eval_metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    fs, root = materialize.fsspec.core.url_to_fs(str(tmp_path))

    with pytest.raises(ValueError, match="Conflicting step-3006 metric rows"):
        materialize.metric_record(fs, root, run_name)


def test_noise_summary_uses_five_rows_per_fixed_action() -> None:
    rows = []
    for group_position, group in enumerate(("prefix/control", "prefix/maximin")):
        for repeat in range(5):
            rows.append(
                {
                    "noise_group_id": group,
                    "metric": noise_materialize.base.PRIMARY_METRIC,
                    "value": 1.0 + 0.01 * group_position + 0.001 * repeat,
                }
            )

    summary = noise_materialize.summarize_metrics(pd.DataFrame(rows))

    assert len(summary) == 2
    assert summary.n.eq(5).all()
    assert summary.sample_sd.gt(0).all()


def test_wave2_contract_is_label_blind_and_caps_model_width() -> None:
    contract = wave2_select.validate_frozen_contract()
    models = cast(dict[str, object], contract["models"])
    fit_budget = cast(dict[str, object], contract["fit_budget"])

    assert contract["endpoint_outcomes_used_to_define_contract"] is False
    assert models["semantic_bucket_partitions"] is False
    assert models["maximum_columns_excluding_intercept"] == 14
    assert fit_budget["total_wave2_rows"] == 80
    assert fit_budget["confirmations_and_noise_controls_included"] is False


def test_wave2_spatial_folds_are_balanced_and_cover_each_row_once() -> None:
    weights = np.random.default_rng(20260825).dirichlet(np.ones(39), size=100)

    folds = wave2_select.spatial_folds(weights, wave2_select.OUTER_FOLDS, seed=0)

    test_rows = np.concatenate([test for _, test in folds])
    assert [len(test) for _, test in folds] == [20] * 5
    assert [len(train) for train, _ in folds] == [80] * 5
    assert np.array_equal(np.sort(test_rows), np.arange(100))


def test_wave2_candidate_instability_uses_fifteen_spatial_subfits() -> None:
    generator = np.random.default_rng(20260825)
    pool = generator.dirichlet(np.ones(39), size=160)
    train = pool[:100]
    response = 1.0 + 0.05 * train[:, 0] - 0.03 * np.sqrt(train[:, 1])
    bank = wave2_select.build_feature_bank(
        pool,
        np.full(39, 1.0 / 39),
        np.linspace(1.0, 20.0, 39),
        np.linspace(0.5, 10.0, 39),
    )

    predictions, alphas = wave2_select.spatial_subfit_prediction_ensemble(
        ["hellinger_linear_14"],
        bank,
        train,
        response,
        pool,
    )
    stacked = np.stack(list(predictions.values()))

    assert stacked.shape == (15, 160)
    assert len(alphas["hellinger_linear_14"]) == 15
    assert float(stacked.std(axis=0, ddof=1).max()) > wave2_select.MINIMUM_PREDICTION_SPREAD


def test_wave2_real_frozen_geometry_supports_guided_and_fail_closed_panels() -> None:
    pool = np.load(wave2_select.POOL_DIR / "candidate_pool_counts.npy", allow_pickle=False).astype(float)
    pool /= wave2_select.branch_design.MIXTURE_BLOCK_SIZE
    metadata = pd.read_csv(wave2_select.POOL_DIR / "candidate_pool_metadata.csv")
    coverage = pd.read_csv(wave2_select.POOL_DIR / "coverage_summary.csv")
    coverage_weights = pd.read_csv(wave2_select.POOL_DIR / "coverage_weights.csv")
    panel = wave2_select.branch_design.load_canonical_panel_geometry()
    wave1 = wave2_select.frozen_wave1_design_weights(panel.buckets)
    index = np.arange(len(pool), dtype=float)
    center = metadata.hellinger_to_proportional.to_numpy(dtype=float)
    winner_predictions = np.stack([center, center + 0.001 * np.sin(index)])
    all_predictions = np.vstack([winner_predictions, center + 0.001 * np.cos(index)])

    guided, guided_tranches, _, guided_mode = wave2_select.select_guided_indices(
        pool,
        wave1,
        coverage,
        winner_predictions,
        all_predictions,
    )
    fallback, fallback_tranches, fallback_diagnostics, fallback_mode = wave2_select.select_guided_indices(
        pool,
        wave1,
        coverage,
        np.zeros((2, len(pool))),
        np.zeros((2, len(pool))),
    )
    summary, weights = wave2_select.build_wave2_weights(
        panel.buckets,
        pool,
        metadata,
        fallback,
        fallback_tranches,
        fallback_diagnostics,
        coverage,
        coverage_weights,
        panel,
    )

    assert guided_mode == "model_guided"
    assert len(guided) == 40
    assert guided_tranches.count("guided_lcb") == 32
    assert guided_tranches.count("guided_disagreement") == 8
    assert fallback_mode == "outcome_blind_fallback_degenerate_instability"
    assert len(fallback) == 40
    assert len(summary) == 80
    assert len(weights) == 80 * len(panel.buckets)
    assert (
        not summary[["tv_to_proportional", "hellinger_to_proportional", "max_phase_1_materialized_epoch"]]
        .isna()
        .any()
        .any()
    )
    assert np.isclose(
        wave2_select.median_nearest_neighbor_hellinger(wave1),
        wave2_select.GUIDED_SUPPORT_RADIUS,
        rtol=0.0,
        atol=1e-15,
    )


def test_wave2_feature_ladder_stays_within_frozen_width() -> None:
    generator = np.random.default_rng(20260825)
    pool = generator.dirichlet(np.ones(39), size=80)
    prefix = np.full(39, 1.0 / 39)
    bank = wave2_select.build_feature_bank(
        pool,
        prefix,
        np.linspace(1.0, 20.0, 39),
        np.linspace(0.5, 10.0, 39),
    )

    for model_name in wave2_select.MODEL_NAMES:
        features = bank.transform(pool[:20], model_name)
        assert features.shape == (20, wave2_select.MAX_MODEL_COLUMNS)
        assert np.isfinite(features).all()


def test_wave2_model_gate_requires_rank_fit_and_top3_selection() -> None:
    assert wave2_select.model_is_eligible(0.5, 0.008, 0.01, 0.004, 0.005)
    assert not wave2_select.model_is_eligible(0.39, 0.008, 0.01, 0.004, 0.005)
    assert not wave2_select.model_is_eligible(0.5, 0.0091, 0.01, 0.004, 0.005)
    assert not wave2_select.model_is_eligible(0.5, 0.008, 0.01, 0.006, 0.005)


def test_wave2_ranked_acquisition_enforces_pairwise_diversity() -> None:
    coordinates = np.column_stack([np.arange(12, dtype=float) / 10.0, np.zeros(12)])
    references = np.asarray([[-1.0, 0.0]])
    ascending = wave2_select.greedy_ranked_selection(
        coordinates,
        references,
        np.ones(12, dtype=bool),
        np.arange(12, dtype=float),
        4,
        descending=False,
    )
    descending = wave2_select.greedy_ranked_selection(
        coordinates,
        references,
        np.ones(12, dtype=bool),
        np.arange(12, dtype=float),
        4,
        descending=True,
    )

    assert np.array_equal(ascending, [0, 3, 6, 9])
    assert np.array_equal(descending, [11, 8, 5, 2])
    with pytest.raises(ValueError, match="satisfy the guided diversity contract"):
        wave2_select.greedy_ranked_selection(
            coordinates[:6],
            references,
            np.ones(6, dtype=bool),
            np.arange(6, dtype=float),
            4,
            descending=False,
        )


def test_wave2_materialization_manifest_rejects_changed_artifact(tmp_path) -> None:
    artifact = tmp_path / "branch_fit_matrix.csv"
    artifact.write_text("value\n1\n")
    coverage = tmp_path / wave1_materialize.MATERIALIZATION_COVERAGE
    coverage.write_text('{"complete": true}\n')
    manifest = wave1_materialize.write_materialization_manifest(
        tmp_path,
        {
            artifact.name: wave1_materialize.artifact_record(artifact, 1),
            coverage.name: wave1_materialize.artifact_record(coverage, 1),
        },
        {"test": True},
    )
    expected_manifest_sha256 = wave1_materialize.local_file_sha256(manifest)

    loaded = wave2_select.load_materialization_manifest(
        tmp_path,
        expected_manifest_sha256,
        {artifact.name, coverage.name},
    )
    assert loaded["complete"] is True

    artifact.write_text("value\n2\n")
    with pytest.raises(ValueError, match=r"materialized artifact branch_fit_matrix\.csv changed"):
        wave2_select.load_materialization_manifest(
            tmp_path,
            expected_manifest_sha256,
            {artifact.name, coverage.name},
        )


def test_wave2_selector_requires_noise_controls_from_the_same_wave1_materialization() -> None:
    wave1 = {
        "artifacts": {
            "branch_results.csv": {"sha256": "results"},
            "uncheatable_metrics_long.csv": {"sha256": "metrics"},
        }
    }
    matching_noise = {
        "provenance": {
            "wave1_results_sha256": "results",
            "wave1_metrics_sha256": "metrics",
        }
    }

    wave2_select.validate_noise_wave1_crosslink(wave1, matching_noise)

    mismatched_noise = {
        "provenance": {
            "wave1_results_sha256": "other-results",
            "wave1_metrics_sha256": "metrics",
        }
    }
    with pytest.raises(ValueError, match=r"frozen Wave-1 branch_results\.csv"):
        wave2_select.validate_noise_wave1_crosslink(wave1, mismatched_noise)


def _write_wave2_fixture(tmp_path: Path) -> tuple[Path, str, Path, str]:
    buckets = ("a", "b")
    candidate_rows = []
    for candidate_id in (
        "shared_bounded_ensemble_kl0p05",
        "shared_bounded_ensemble_kl0p2",
        "shared_bounded_ensemble_kl0p5",
        "observed_cap10_best",
        "proportional_control",
    ):
        for bucket in buckets:
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "bucket": bucket,
                    "phase_0_weight": 0.5,
                    "phase_0_materialized_epochs": 1.0,
                }
            )
    candidate_path = tmp_path / "candidate.csv"
    pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
    continuation_rows = []
    for run_order in range(wave2_launch.EXPECTED_CONTINUATIONS):
        first_count = 1_024 + run_order
        tranche = "guided_lcb" if run_order < 40 else "fixed_near_coverage"
        referee = 40 <= run_order < 48
        for bucket, count in zip(buckets, (first_count, 2_048 - first_count), strict=True):
            weight = count / 2_048
            continuation_rows.append(
                {
                    "continuation_id": f"fit_wave2_{run_order:02d}",
                    "role": f"wave2_{tranche}",
                    "selection_tranche": tranche,
                    "fit_budget": True,
                    "referee_holdout": referee,
                    "bucket": bucket,
                    "phase_1_count": count,
                    "phase_1_weight": weight,
                    "phase_1_materialized_epochs": 2.0 * weight,
                    "historical_phase_1_bucket_epoch_cap": 10.0,
                    "historical_total_bucket_epoch_cap": 20.0,
                }
            )
    continuation_path = tmp_path / "wave2.csv"
    pd.DataFrame(continuation_rows).to_csv(continuation_path, index=False)
    return (
        candidate_path,
        branches.file_sha256(candidate_path),
        continuation_path,
        branches.file_sha256(continuation_path),
    )


def test_wave2_launcher_loads_exact_eighty_row_guided_coverage_design(tmp_path) -> None:
    candidate_path, candidate_sha256, continuation_path, continuation_sha256 = _write_wave2_fixture(tmp_path)

    buckets, continuations = wave2_launch.load_wave2_continuations(
        continuation_path,
        continuation_sha256,
        candidate_path,
        candidate_sha256,
    )

    assert buckets == ("a", "b")
    assert len(continuations) == 80
    assert sum(row["selection_tranche"].startswith("guided_") for row in continuations) == 40
    assert sum(row["selection_tranche"].startswith("fixed_") for row in continuations) == 40
    assert sum(bool(row["referee_holdout"]) for row in continuations) == 8


def test_wave2_launcher_rejects_a_continuation_outside_frozen_support(tmp_path) -> None:
    candidate_path, candidate_sha256, continuation_path, _ = _write_wave2_fixture(tmp_path)
    frame = pd.read_csv(continuation_path)
    frame.loc[0, "historical_phase_1_bucket_epoch_cap"] = 0.0
    frame.to_csv(continuation_path, index=False)

    with pytest.raises(ValueError, match="Per-bucket phase-1 support exceeded"):
        wave2_launch.load_wave2_continuations(
            continuation_path,
            branches.file_sha256(continuation_path),
            candidate_path,
            candidate_sha256,
        )


def test_wave2_launcher_requires_the_acknowledged_selection_mode(tmp_path) -> None:
    continuation = tmp_path / "continuation.csv"
    continuation.write_text("continuation_id\nfit\n")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "target_prefix": wave2_launch.TARGET_PREFIX,
                "fit_budget": {"total_wave2_rows": wave2_launch.EXPECTED_CONTINUATIONS},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_prefix": wave2_launch.TARGET_PREFIX,
                "total_wave2_rows": wave2_launch.EXPECTED_CONTINUATIONS,
                "selection_mode": "model_guided",
                "continuation_weights_sha256": branches.file_sha256(continuation),
                "contract_sha256": branches.file_sha256(contract),
            }
        )
    )

    with pytest.raises(ValueError, match="Selection mode changed"):
        wave2_launch.verify_selection_artifacts(
            continuation,
            branches.file_sha256(continuation),
            manifest,
            branches.file_sha256(manifest),
            contract,
            branches.file_sha256(contract),
            "outcome_blind_fallback_no_eligible_model",
        )


def test_wave2_target_prefix_checkpoint_rejects_ambiguous_identity() -> None:
    prefix = branches.PrefixCheckpoint(
        candidate_id=wave2_launch.TARGET_PREFIX,
        repeat_seed=0,
        checkpoint_uri="gs://marin-us-east5/prefix/step-2399",
        provenance_sha256="provenance",
    )

    with pytest.raises(ValueError, match="Expected one exact"):
        wave2_launch.target_prefix_checkpoint([prefix, prefix])


def test_wave2_rows_use_one_prefix_and_disjoint_run_namespace() -> None:
    prefix = branches.PrefixCheckpoint(
        candidate_id=wave2_launch.TARGET_PREFIX,
        repeat_seed=0,
        checkpoint_uri="gs://marin-us-east5/prefix/step-2399",
        provenance_sha256="provenance",
    )
    uniform = {bucket: 1.0 / len(base.DOMAIN_NAMES) for bucket in base.DOMAIN_NAMES}
    prefix_spec = cast(
        base.DelphiSwarmRunSpec,
        _PrefixSpec(
            phase_weights={"phase_0": uniform, "phase_1": uniform},
            data_seed=930_000,
            trainer_seed=0,
        ),
    )
    continuations = [
        {
            "continuation_id": f"fit_wave2_{index:02d}",
            "role": "wave2_guided_lcb" if index < 40 else "wave2_fixed_near_coverage",
            "selection_tranche": "guided_lcb" if index < 40 else "fixed_near_coverage",
            "fit_budget": True,
            "referee_holdout": 40 <= index < 48,
            "weights": uniform,
        }
        for index in range(80)
    ]

    rows = wave2_launch.wave2_rows(prefix, prefix_spec, continuations, wave2_launch.BRANCH_RUN_ID_BASE)

    assert len(rows) == 80
    assert rows[0]["run_id"] == 953_000
    assert rows[-1]["run_id"] == 953_079
    assert {row["trainer_seed"] for row in rows} == {0}
    assert {row["data_seed"] for row in rows} == {930_000}
>>>>>>> 0dd17851fd (Freeze Delphi KL0.05 Wave 2 acquisition)
