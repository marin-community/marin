# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pytest
from marin.execution.executor import instantiate_config

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_harsh_cap_candidates as prefix_launch
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as launch
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_delphi_phase1_harsh_cap_branch_response as fit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase0_harsh_cap_validation_20260825 as prefix_materialize,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_harsh_cap_branches as materialize,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    score_delphi_phase1_harsh_cap_referees as referee_score,
)

FROZEN_PROPORTIONAL = np.asarray(
    (
        0.0040417726214116,
        0.0163417000243851,
        0.0066329102360266,
        0.0219961630999220,
        0.0098808079720213,
        0.0359641216118575,
        0.0176160649879605,
        0.0189756893623004,
        0.0089192571342428,
        0.0606049913397553,
        0.0220549117371660,
        0.0778600460936579,
        0.0370996616466542,
        0.0244530271563001,
        0.0120362107030710,
        0.0323401875928955,
        0.0121249705566055,
        0.0623305910391983,
        0.0265653574462228,
        0.0193391600820501,
        0.0053220543675932,
        0.0116901604313954,
        0.0061895911544051,
        0.0369774877910664,
        0.0098593275782407,
        0.0347870492656606,
        0.0158797880980409,
        0.0048668414974546,
        0.0191902049354434,
        0.0005251805879418,
        0.1885921649496403,
        0.0294717527825882,
        0.0191632974893215,
        0.0007462684143601,
        0.0026996340748741,
        0.0025736223490378,
        0.0031233246969685,
        0.0754538759902185,
        0.0057107711020423,
    )
)


def _frozen_panel_geometry() -> design.common_design.CanonicalPanelGeometry:
    frame = pd.read_csv(design.DEFAULT_CANDIDATE_WEIGHTS)
    rows = frame[frame.candidate_id.eq("cap4_shared_bounded_ensemble_kl0")]
    buckets = tuple(rows.bucket)
    alpha = design.common_design.EXPECTED_PREFIX_TRAIN_STEPS / design.common_design.EXPECTED_FULL_TRAIN_STEPS
    c0 = design.common_design.PROPORTIONAL_POLICY_EPOCHS * alpha / FROZEN_PROPORTIONAL
    c1 = design.common_design.PROPORTIONAL_POLICY_EPOCHS * (1.0 - alpha) / FROZEN_PROPORTIONAL
    return design.common_design.CanonicalPanelGeometry(
        buckets=buckets,
        phase0=np.empty((0, len(buckets))),
        phase1=np.empty((0, len(buckets))),
        row_id=(),
        c0=c0,
        c1=c1,
        proportional=FROZEN_PROPORTIONAL,
    )


def _stub_historical_frontier(monkeypatch, panel: design.common_design.CanonicalPanelGeometry) -> None:
    payload = (
        pd.DataFrame(
            {
                "domain": panel.buckets,
                "phase_1_weight": np.roll(panel.proportional, 1),
            }
        )
        .to_csv(index=False)
        .encode()
    )
    monkeypatch.setattr(design, "read_uri_bytes", lambda _: payload)
    monkeypatch.setattr(design, "HISTORICAL_FRONTIER_SHA256", hashlib.sha256(payload).hexdigest())


def _boundary_design(candidate_id: str, monkeypatch) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = _frozen_panel_geometry()
    _stub_historical_frontier(monkeypatch, panel)
    centers = design.candidate_centers(design.DEFAULT_CANDIDATE_WEIGHTS, (candidate_id,), panel.buckets)
    center = centers[candidate_id]
    anchors = design.anchor_mixtures(panel.buckets, design.runtime_weights(panel.proportional))
    assert set(anchors) == {"historical_frontier", "proportional", "uniform"}
    pool, anchor_keys = design.generate_pool(center, anchors, center * panel.c0, panel.c1)
    selected, diagnostics = design.select_fit_points(pool, anchor_keys, center)
    referees = design.select_referee_points(pool, selected, center)
    summary_rows, weight_rows = design.design_rows(
        candidate_id,
        center,
        selected,
        referees,
        panel.buckets,
        panel.c0,
        panel.c1,
    )
    manifest = {
        "selected_candidate_ids": [candidate_id],
        "rows": {
            "controls_per_prefix": design.CONTROL_ROWS_PER_PREFIX,
            "fit_per_prefix": design.FIT_ROWS_PER_PREFIX,
            "sealed_referees_per_prefix": design.REFEREE_ROWS_PER_PREFIX,
            "total": design.FIT_ROWS_PER_PREFIX + design.REFEREE_ROWS_PER_PREFIX + design.CONTROL_ROWS_PER_PREFIX,
        },
        "diagnostics": {candidate_id: diagnostics},
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(weight_rows), manifest


def _write_design(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    summary, weights, manifest = _boundary_design("cap4_shared_bounded_ensemble_kl0", monkeypatch)
    summary_path = tmp_path / "continuation_summary.csv"
    weights_path = tmp_path / "continuation_weights.csv"
    manifest_path = tmp_path / "manifest.json"
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return summary_path, weights_path, manifest_path


def test_boundary_design_is_full_rank_and_keeps_referees_sealed(monkeypatch) -> None:
    for candidate_id in ("cap4_shared_bounded_ensemble_kl0", "cap6_shared_bounded_ensemble_kl0"):
        summary, weights, manifest = _boundary_design(candidate_id, monkeypatch)

        assert len(summary) == launch.ROWS_PER_PREFIX
        assert summary.fit_budget.sum() == launch.FIT_ROWS_PER_PREFIX
        assert summary.role.value_counts().to_dict() == {
            "fixed_prefix_response_fit": launch.FIT_ROWS_PER_PREFIX,
            "sealed_geometry_referee": launch.REFEREE_ROWS_PER_PREFIX,
            "prefix_state_tied_control": 4,
            "fresh_tied_control": 3,
            "common_random_tied_control": 1,
        }
        fit_ids = set(summary.loc[summary.fit_budget, "continuation_id"])
        referee_ids = set(summary.loc[summary.role.eq("sealed_geometry_referee"), "continuation_id"])
        assert fit_ids.isdisjoint(referee_ids)
        assert weights.total_materialized_epochs.max() <= design.TOTAL_MATERIALIZED_EPOCH_CAP
        assert manifest["diagnostics"][candidate_id]["direct_feature_rank"] == 38
        assert manifest["diagnostics"][candidate_id]["sqrt_feature_rank"] == 39


def test_load_design_preserves_common_random_number_identity(tmp_path: Path, monkeypatch) -> None:
    summary_path, weights_path, manifest_path = _write_design(tmp_path, monkeypatch)

    rows = launch.load_design(
        summary_path,
        launch.file_sha256(summary_path),
        weights_path,
        launch.file_sha256(weights_path),
        manifest_path,
        launch.file_sha256(manifest_path),
        ("cap4_shared_bounded_ensemble_kl0",),
    )

    fit_rows = [row for row in rows if row["fit_budget"]]
    referees = [row for row in rows if row["role"] == "sealed_geometry_referee"]
    matched = [row for row in rows if row["role"] == "common_random_tied_control"]
    assert {row["data_seed"] for row in [*fit_rows, *referees, *matched]} == {design.FIT_DATA_SEED}
    assert len(fit_rows) == launch.FIT_ROWS_PER_PREFIX
    assert len(referees) == launch.REFEREE_ROWS_PER_PREFIX
    assert len(matched) == 1


def test_selected_prefix_loader_accepts_one_cap(tmp_path: Path) -> None:
    payload = {
        "candidate_weights_sha256": "candidate-hash",
        "prefix_replay_code_commit": "prefix-commit",
        "selected_aliases": [{"canonical_candidate_id": "cap4_shared_bounded_ensemble_kl0"}],
        "prefixes": [
            {
                "canonical_candidate_id": "cap4_shared_bounded_ensemble_kl0",
                "repeat_seed": seed,
                "checkpoint_uri": f"gs://marin-us-east5/prefix-seed-{seed}/checkpoints/step-2399",
                "provenance_sha256": f"provenance-{seed}",
            }
            for seed in (0, 1)
        ],
    }
    path = tmp_path / "selected.json"
    encoded = (json.dumps(payload) + "\n").encode()
    path.write_bytes(encoded)

    rows, loaded = launch.selected_prefixes(
        str(path), hashlib.sha256(encoded).hexdigest(), "candidate-hash", "prefix-commit"
    )

    assert loaded == payload
    assert [asdict(row) for row in rows] == [
        {
            "candidate_id": "cap4_shared_bounded_ensemble_kl0",
            "repeat_seed": seed,
            "checkpoint_uri": f"gs://marin-us-east5/prefix-seed-{seed}/checkpoints/step-2399",
            "provenance_sha256": f"provenance-{seed}",
        }
        for seed in (0, 1)
    ]


def test_cap6_prefix_provenance_uses_full_candidate_order() -> None:
    assert (
        prefix_materialize.candidate_run_order("cap6_shared_bounded_ensemble_kl0", 0, prefix_launch.CANDIDATE_IDS) == 18
    )


def test_manifest_write_is_idempotent(tmp_path: Path) -> None:
    config = launch.SaveManifestConfig(
        experiment_name="experiment",
        output_path=str(tmp_path),
        selected_prefixes_json="[]",
        selected_prefixes_sha256="selected",
        candidate_weights_sha256="candidates",
        candidate_aliases_sha256="aliases",
        continuation_summary_sha256="summary",
        continuation_weights_sha256="weights",
        design_manifest_sha256="design",
        prefix_replay_code_commit="prefix-commit",
        code_commit="branch-commit",
        branch_run_id_base=973_000,
        full_design_rows=1,
        branch_rows_json=json.dumps([{"fit_budget": True, "role": "fixed_prefix_response_fit", "run_order": 0}]),
        manifest_identity=launch.versioned("test-manifest"),
    )

    runtime_config = instantiate_config(config, output_path=None, output_paths={}, prefix="")
    launch.save_manifest(runtime_config)
    first = (tmp_path / "manifest.json").read_bytes()
    launch.save_manifest(runtime_config)

    assert (tmp_path / "manifest.json").read_bytes() == first


def test_hardware_observation_initializes_distributed_jax_before_device_discovery(monkeypatch) -> None:
    events = []

    class Device:
        platform = "tpu"
        device_kind = "TPU v6 lite"

    def initialize() -> None:
        events.append("initialize")

    def devices() -> list[Device]:
        assert events == ["initialize"]
        events.append("devices")
        return [Device() for _ in range(launch.EXPECTED_TPU_DEVICE_COUNT)]

    monkeypatch.setattr(launch, "initialize_jax", initialize)
    monkeypatch.setattr(launch.jax, "devices", devices)
    monkeypatch.setattr(launch.jax, "local_device_count", lambda: launch.EXPECTED_TPU_DEVICE_COUNT)

    observed = launch.observe_tpu_hardware()

    assert events == ["initialize", "devices"]
    assert observed.global_device_count == launch.EXPECTED_TPU_DEVICE_COUNT


def test_default_materialization_does_not_open_referee_metrics() -> None:
    fs = fsspec.filesystem("memory")
    experiment_root = "memory://harsh-branches"
    base_row = {
        "run_id": 973_000,
        "prefix_candidate_id": "cap4_shared_bounded_ensemble_kl0",
        "prefix_repeat_seed": 0,
        "continuation_id": "fit_000",
        "fit_budget": True,
        "data_seed": design.FIT_DATA_SEED,
        "trainer_seed": 0,
        "source": "test",
        "prefix": {
            "candidate_id": "cap4_shared_bounded_ensemble_kl0",
            "repeat_seed": 0,
            "checkpoint_uri": "gs://marin-us-east5/prefix/checkpoints/step-2399",
            "provenance_sha256": "prefix-provenance",
        },
        "phase_weights": {"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
    }
    rows = [
        {**base_row, "run_order": 0, "run_name": "fit", "role": "fixed_prefix_response_fit"},
        {
            **base_row,
            "run_order": 1,
            "run_id": 973_001,
            "run_name": "referee",
            "continuation_id": "referee_000",
            "role": "sealed_geometry_referee",
            "fit_budget": False,
        },
    ]
    manifest = {
        "experiment_name": "experiment",
        "prefix_replay_code_commit": "prefix-commit",
        "candidate_weights_sha256": "candidates",
        "candidate_aliases_sha256": "aliases",
        "continuation_weights_sha256": "weights",
        "design_manifest_sha256": "design",
        "code_commit": "branch-commit",
        "prefix_hardware": asdict(launch.TPU_HARDWARE),
        "continuation_hardware": asdict(launch.TPU_HARDWARE),
        "branch_rows": rows,
    }
    for row in rows:
        output_path = f"/harsh-branches/{row['run_name']}"
        provenance = {
            **materialize.expected_provenance(row, manifest),
            "observed_continuation_hardware": {
                "platform": "tpu",
                "device_kind": "TPU v6 lite",
                "global_device_count": 8,
                "local_device_count": 8,
            },
            "terminal_checkpoint_uri": f"memory://{output_path}/checkpoints/step-{materialize.TERMINAL_STEP}",
        }
        fs.makedirs(output_path, exist_ok=True)
        with fs.open(f"{output_path}/{launch.BRANCH_PROVENANCE_FILENAME}", "w") as handle:
            json.dump(provenance, handle)
        fs.makedirs(f"{output_path}/checkpoints", exist_ok=True)
        with fs.open(f"{output_path}/checkpoints/eval_metrics.jsonl", "w") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": materialize.TERMINAL_STEP,
                        "eval/uncheatable_eval/bpb": 0.99,
                    }
                )
                + "\n"
            )

    results, _, referees, missing = materialize.materialize(
        experiment_root,
        manifest,
        open_referee=False,
    )

    assert results.run_name.tolist() == ["fit"]
    assert referees[["run_name", "outcome_opened", "terminal_metrics_verified"]].to_dict("records") == [
        {"run_name": "referee", "outcome_opened": False, "terminal_metrics_verified": True}
    ]
    assert missing == []


def test_manifest_payload_finds_executor_hashed_manifest() -> None:
    fs = fsspec.filesystem("memory")
    root = "memory://harsh-manifest"
    payload = (json.dumps({"branch_rows": []}, sort_keys=True) + "\n").encode()
    fs.makedirs("/harsh-manifest/manifest-a1b2c3", exist_ok=True)
    with fs.open("/harsh-manifest/manifest-a1b2c3/manifest.json", "wb") as handle:
        handle.write(payload)

    manifest, observed = materialize.manifest_payload(root, hashlib.sha256(payload).hexdigest())

    assert manifest == {"branch_rows": []}
    assert observed == payload


def test_terminal_metrics_accepts_identical_retries_and_rejects_conflicts() -> None:
    fs = fsspec.filesystem("memory")
    output_path = "/retry-metrics"
    fs.makedirs(f"{output_path}/checkpoints", exist_ok=True)
    row = {"step": materialize.TERMINAL_STEP, "eval/uncheatable_eval/bpb": 0.99}
    with fs.open(f"{output_path}/checkpoints/eval_metrics.jsonl", "w") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps(row) + "\n")

    assert materialize.terminal_metrics(fs, output_path)["bpb"] == 0.99

    with fs.open(f"{output_path}/checkpoints/eval_metrics.jsonl", "a") as handle:
        handle.write(json.dumps({**row, "eval/uncheatable_eval/bpb": 1.01}) + "\n")
    with pytest.raises(ValueError, match="Conflicting"):
        materialize.terminal_metrics(fs, output_path)


def test_model_candidate_pool_excludes_sealed_referee_coordinates(tmp_path: Path, monkeypatch) -> None:
    candidate_id = "cap4_shared_bounded_ensemble_kl0"
    summary, weights, _ = _boundary_design(candidate_id, monkeypatch)
    panel = _frozen_panel_geometry()
    monkeypatch.setattr(fit.design.common_design, "load_canonical_panel_geometry", lambda: panel)
    buckets = panel.buckets
    center = fit.tied_center(design.DEFAULT_CANDIDATE_WEIGHTS, candidate_id, buckets)
    fit_ids = tuple(summary[summary.fit_budget].continuation_id)
    referee_ids = tuple(summary[summary.role.eq("sealed_geometry_referee")].continuation_id)
    weights_path = tmp_path / "weights.csv"
    weights.to_csv(weights_path, index=False)
    _, measured = fit.load_weights(weights_path, candidate_id, fit_ids)
    _, referees = fit.load_weights(weights_path, candidate_id, referee_ids)
    excluded = {tuple(fit.design.common_design.runtime_counts(row).tolist()) for row in referees}
    pool, _ = fit.candidate_pool(center, buckets, measured, excluded)

    pool_keys = {tuple(fit.design.common_design.runtime_counts(row).tolist()) for row in pool}
    assert pool_keys.isdisjoint(excluded)


def test_opened_referees_are_scored_without_changing_the_frozen_optimum(tmp_path: Path, monkeypatch) -> None:
    candidate_id = "cap4_shared_bounded_ensemble_kl0"
    summary, weights, _ = _boundary_design(candidate_id, monkeypatch)
    referee = summary[summary.role.eq("sealed_geometry_referee")].copy()
    referee["run_order"] = np.arange(len(referee))
    referee["run_name"] = [f"referee_{index:03d}" for index in range(len(referee))]
    referee[fit.TARGET] = np.linspace(0.985, 0.992, len(referee))
    weights_path = tmp_path / "weights.csv"
    weights.to_csv(weights_path, index=False)
    panel = _frozen_panel_geometry()
    center = fit.tied_center(design.DEFAULT_CANDIDATE_WEIGHTS, candidate_id, panel.buckets)
    frozen = {
        "feature_kind": "direct",
        "ridge_alpha": 1.0,
        "damage_coefficient": 0.0,
        "coefficients": dict.fromkeys(panel.buckets, 0.0),
        "baselines": {"matched_tied_bpb": 0.990},
        "weights": dict(zip(panel.buckets, center, strict=True)),
    }

    scores, report = referee_score.score_candidate(
        referee,
        frozen,
        weights_path,
        design.DEFAULT_CANDIDATE_WEIGHTS,
        candidate_id,
    )

    assert len(scores) == design.REFEREE_ROWS_PER_PREFIX
    assert report["predicted_optimum_excluded_from_referees"] is True
    assert report["referee_rows"] == design.REFEREE_ROWS_PER_PREFIX


def test_local_response_is_tied_anchored_and_recovers_nonnegative_damage() -> None:
    generator = np.random.default_rng(20260825)
    center = np.asarray([0.45, 0.35, 0.20])
    weights = generator.dirichlet(50.0 * center, size=80)
    coefficients = np.asarray([-0.010, 0.006, 0.004])
    center_root = np.sqrt(center)
    coefficients -= center_root * (center_root @ coefficients)
    effects = fit.feature_map(weights, center, "sqrt") @ coefficients + 0.08 * fit.hellinger(weights, center) ** 2

    model = fit.fit_model(weights, effects, center, "sqrt", 1e-6)
    predictions = fit.predict(model, weights, center)

    np.testing.assert_allclose(predictions, effects, atol=1e-6, rtol=0)
    np.testing.assert_allclose(fit.predict(model, center[None, :], center), [0.0], atol=1e-14, rtol=0)
    assert model.damage >= 0.0
    design_matrix = np.column_stack([fit.feature_map(weights, center, "sqrt"), fit.hellinger(weights, center) ** 2])
    assert np.linalg.matrix_rank(design_matrix) == len(center)
    folds = fit.geometric_fold_ids(weights, center, folds=5, seed=20260825)
    assert set(folds) == set(range(5))
    assert np.bincount(folds).tolist() == [16] * 5
    fold_predictions, selections = fit.fold_ensemble_predictions(weights, effects, center, weights[:4])
    assert fold_predictions.shape == (fit.OUTER_FOLDS, 4)
    assert len(selections) == fit.OUTER_FOLDS


def test_control_baselines_use_matched_seed_for_effects_and_four_seed_mean_for_level() -> None:
    results = pd.DataFrame(
        [
            {
                "role": "common_random_tied_control",
                "prefix_repeat_seed": 0,
                "bpb": 0.990,
            },
            *[{"role": "fresh_tied_control", "prefix_repeat_seed": 0, "bpb": value} for value in (0.982, 0.984, 0.986)],
            *[
                {"role": "prefix_state_tied_control", "prefix_repeat_seed": 1, "bpb": value}
                for value in (0.985, 0.986, 0.987, 0.988)
            ],
        ]
    )

    baselines = fit.control_baselines(results)

    assert baselines["matched_tied_bpb"] == 0.990
    assert baselines["expected_tied_bpb"] == np.mean([0.990, 0.982, 0.984, 0.986])
