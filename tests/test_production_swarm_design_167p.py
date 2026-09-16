# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import design_production_swarm_167p as design
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    generate_production_swarm_167p_baseline_perturbations as baselines,
)


def _toy_weights() -> np.ndarray:
    return np.array(
        [
            [[0.80, 0.15, 0.05], [0.70, 0.20, 0.10]],
            [[0.60, 0.30, 0.10], [0.55, 0.35, 0.10]],
            [[0.40, 0.50, 0.10], [0.45, 0.40, 0.15]],
            [[0.20, 0.70, 0.10], [0.25, 0.60, 0.15]],
            [[0.70, 0.05, 0.25], [0.65, 0.10, 0.25]],
            [[0.50, 0.10, 0.40], [0.45, 0.15, 0.40]],
            [[0.30, 0.20, 0.50], [0.35, 0.15, 0.50]],
            [[0.10, 0.30, 0.60], [0.15, 0.25, 0.60]],
        ],
        dtype=float,
    )


def test_fisher_dsp_aligned_downweights_tiny_bucket_after_standardization():
    weights = _toy_weights()
    tokens = np.array([90.0, 9.0, 1.0])
    proportional = tokens / tokens.sum()
    epochs = weights * np.array([[10.0, 100.0, 900.0], [2.5, 25.0, 225.0]])[None, :, :]

    bundle = design.fisher_dsp_aligned_features(weights, epochs, proportional)

    assert bundle.name == "fisher_dsp_aligned"
    assert not any(name.startswith("weight_") for name in bundle.feature_names)

    log_tilt_stds = bundle.matrix[:, :6].std(axis=0)
    large_bucket_phase0 = log_tilt_stds[0]
    tiny_bucket_phase0 = log_tilt_stds[2]
    large_bucket_phase1 = log_tilt_stds[3]
    tiny_bucket_phase1 = log_tilt_stds[5]

    assert tiny_bucket_phase0 < large_bucket_phase0
    assert tiny_bucket_phase1 < large_bucket_phase1


def test_fisher_dsp_aligned_with_risk_adds_only_global_risk_features():
    weights = _toy_weights()
    tokens = np.array([90.0, 9.0, 1.0])
    proportional = tokens / tokens.sum()
    epochs = weights * np.array([[10.0, 100.0, 900.0], [2.5, 25.0, 225.0]])[None, :, :]

    base = design.fisher_dsp_aligned_features(weights, epochs, proportional)
    risk = design.fisher_dsp_aligned_with_risk_features(weights, epochs, proportional)

    assert risk.name == "fisher_dsp_aligned_with_risk"
    assert risk.matrix.shape[1] == base.matrix.shape[1] + 7
    assert risk.feature_names[: len(base.feature_names)] == base.feature_names
    assert risk.feature_names[-7:] == (
        "risk_log_max_epoch_0",
        "risk_log_max_epoch_1",
        "risk_log_q95_epoch_0",
        "risk_log_q95_epoch_1",
        "risk_log_total_epoch_max_0",
        "risk_log_phase_epoch_imbalance_0",
        "risk_small_bucket_mass_0",
    )


def test_min_average_phase_tv_filter_keeps_ordered_representatives():
    weights = np.array(
        [
            [[0.70, 0.20, 0.10], [0.60, 0.30, 0.10]],
            [[0.7002, 0.1998, 0.10], [0.5999, 0.3001, 0.10]],
            [[0.20, 0.70, 0.10], [0.25, 0.65, 0.10]],
        ],
        dtype=float,
    )

    filtered, kept_indices = design.filter_min_average_phase_tv(weights, min_distance=0.001)

    assert kept_indices.tolist() == [0, 2]
    np.testing.assert_allclose(filtered, weights[[0, 2]])


def test_proportional_logit_pool_keeps_simplex_and_sigma_controls_distance():
    proportional = np.array([0.5, 0.3, 0.2])

    sigma0 = design.proportional_logit_pool(
        n_points=8,
        n_phases=2,
        proportional=proportional,
        seed=123,
        sigma_values=(0.0,),
    )
    sigma1 = design.proportional_logit_pool(
        n_points=8,
        n_phases=2,
        proportional=proportional,
        seed=123,
        sigma_values=(1.0,),
    )

    np.testing.assert_allclose(sigma0, np.broadcast_to(proportional, sigma0.shape))
    np.testing.assert_allclose(sigma1.sum(axis=2), 1.0)
    assert np.all(sigma1 > 0)
    assert np.abs(sigma1 - proportional[None, None, :]).sum(axis=(1, 2)).mean() > 0.1


def test_max_epoch_cap_filter_keeps_feasible_candidates_and_source_indices():
    weights = _toy_weights()
    source_indices = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)
    epochs = np.array(
        [
            [[10.0, 20.0, 30.0], [5.0, 10.0, 15.0]],
            [[10.0, 20.0, 30.1], [5.0, 10.0, 15.0]],
            [[8.0, 12.0, 25.0], [4.0, 9.0, 13.0]],
            [[31.0, 12.0, 20.0], [4.0, 9.0, 13.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[35.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[12.0, 15.0, 29.0], [7.0, 8.0, 9.0]],
            [[12.0, 15.0, 30.0], [7.0, 8.0, 9.0]],
        ],
        dtype=float,
    )

    filtered_weights, filtered_epochs, kept_indices = design.filter_max_epoch_cap(
        weights,
        epochs,
        source_indices,
        max_epoch_cap=30.0,
    )

    assert kept_indices.tolist() == [10, 12, 14, 16, 17]
    np.testing.assert_allclose(filtered_weights, weights[[0, 2, 4, 6, 7]])
    np.testing.assert_allclose(filtered_epochs, epochs[[0, 2, 4, 6, 7]])


def test_epoch_band_quotas_sum_to_requested_count():
    quotas = design.epoch_band_quotas(
        n_select=11,
        quota_fractions=np.array([0.1, 0.3, 0.6]),
    )

    assert quotas.tolist() == [1, 3, 7]
    assert quotas.sum() == 11


def test_banded_qr_selection_respects_epoch_band_quotas():
    max_epoch = np.array([10.0, 20.0, 35.0, 45.0, 65.0, 70.0, 80.0, 95.0])
    features = design.FeatureBundle(
        name="toy",
        matrix=np.column_stack(
            [
                np.linspace(-1.0, 1.0, len(max_epoch)),
                np.cos(np.arange(len(max_epoch))),
            ]
        ),
        feature_names=("linear", "cosine"),
    )

    selected = design.select_banded_by_qr(
        features,
        n_select=6,
        max_epoch=max_epoch,
        band_edges=np.array([0.0, 30.0, 60.0, 75.0, 100.0]),
        quota_fractions=np.array([1.0, 2.0, 2.0, 1.0]),
    )

    selected_epochs = max_epoch[selected]
    assert len(selected) == 6
    assert np.sum((selected_epochs >= 0.0) & (selected_epochs < 30.0)) == 1
    assert np.sum((selected_epochs >= 30.0) & (selected_epochs < 60.0)) == 2
    assert np.sum((selected_epochs >= 60.0) & (selected_epochs < 75.0)) == 2
    assert np.sum((selected_epochs >= 75.0) & (selected_epochs <= 100.0)) == 1


def test_epoch_penalty_row_scale_downweights_cap_boundary_rows():
    epochs = np.array(
        [
            [[10.0, 20.0, 25.0], [5.0, 10.0, 15.0]],
            [[70.0, 90.0, 100.0], [5.0, 10.0, 15.0]],
        ],
        dtype=float,
    )

    scale = design.epoch_penalty_row_scale(epochs, max_epoch_cap=100.0, penalty_lambda=2.0)

    assert scale[0] > scale[1]
    assert np.isclose(scale[1], np.exp(-2.0))


def test_epoch_scale_penalty_row_scale_supports_power_long_tail():
    epochs = np.array(
        [
            [[10.0, 20.0, 25.0], [5.0, 10.0, 15.0]],
            [[70.0, 90.0, 100.0], [5.0, 10.0, 15.0]],
        ],
        dtype=float,
    )

    scale = design.epoch_scale_penalty_row_scale(
        epochs,
        epoch_tau=25.0,
        penalty_lambda=2.0,
        power=1.0,
        penalty_kind="power",
    )

    assert scale[0] > scale[1]
    assert np.isclose(scale[0], (1.0 + 25.0 / 25.0) ** -2.0)
    assert np.isclose(scale[1], (1.0 + 100.0 / 25.0) ** -2.0)


def test_default_production_design_args_use_cap100_and_large_pool():
    parser = design.build_arg_parser()

    args = parser.parse_args([])

    assert args.max_epoch_cap == 100.0
    assert args.pool_size == 262_144


def test_integer_simplex_counts_uses_largest_remainder():
    counts = design.integer_simplex_counts(np.array([0.51, 0.26, 0.23]), denominator=10)

    assert counts.tolist() == [5, 3, 2]
    assert counts.sum() == 10


def test_partition_ablation_deletes_target_and_renormalizes_on_lattice():
    proportional = np.array([0.5, 0.3, 0.2])

    counts = baselines.partition_ablation_counts(proportional, target_index=1, denominator=10)

    assert counts.tolist() == [7, 0, 3]
    assert counts.sum() == 10


def test_candidate_diagnostics_frame_computes_scope_metrics_and_lattice_errors():
    buckets = ("large", "middle", "small")
    tokens = np.array([50.0, 30.0, 20.0])
    proportional = np.array([0.5, 0.3, 0.2])
    frame = pd.DataFrame(
        [
            {
                "candidate_name": "baseline_proportional",
                "candidate_type": "baseline_proportional",
                "phase_0/large": 0.5,
                "phase_0/middle": 0.3,
                "phase_0/small": 0.2,
                "phase_1/large": 0.5,
                "phase_1/middle": 0.3,
                "phase_1/small": 0.2,
            },
            {
                "candidate_name": "fisher_dsp_aligned_000000",
                "candidate_type": "fisher_dsp_aligned",
                "phase_0/large": 0.7,
                "phase_0/middle": 0.2,
                "phase_0/small": 0.1,
                "phase_1/large": 0.4,
                "phase_1/middle": 0.4,
                "phase_1/small": 0.2,
            },
        ]
    )

    diagnostics = design.candidate_diagnostics_frame(
        frame,
        buckets=buckets,
        tokens=tokens,
        proportional=proportional,
        denominator=10,
        target_budget=100.0,
    )

    assert diagnostics["scope_is_d_optimal"].tolist() == [False, True]
    assert diagnostics["support_min"].tolist() == [3, 3]
    assert diagnostics["lattice_max_abs_error"].max() == 0.0
    assert diagnostics.loc[1, "phase_tv"] == 0.3
    assert np.isclose(diagnostics.loc[1, "max_epoch"], 1.12)


def test_write_interactive_sanity_dashboard_includes_scope_toggle(tmp_path):
    buckets = ("large", "middle", "small")
    tokens = np.array([50.0, 30.0, 20.0])
    proportional = np.array([0.5, 0.3, 0.2])
    frame = pd.DataFrame(
        [
            {
                "candidate_name": "baseline_proportional",
                "candidate_type": "baseline_proportional",
                "phase_0/large": 0.5,
                "phase_0/middle": 0.3,
                "phase_0/small": 0.2,
                "phase_1/large": 0.5,
                "phase_1/middle": 0.3,
                "phase_1/small": 0.2,
            },
            {
                "candidate_name": "fisher_dsp_aligned_000000",
                "candidate_type": "fisher_dsp_aligned",
                "phase_0/large": 0.7,
                "phase_0/middle": 0.2,
                "phase_0/small": 0.1,
                "phase_1/large": 0.4,
                "phase_1/middle": 0.4,
                "phase_1/small": 0.2,
            },
        ]
    )
    output_path = tmp_path / "dashboard.html"

    design.write_interactive_sanity_dashboard(
        output_path,
        scopes={"Full swarm": frame, "D-optimal only": frame.iloc[[1]].copy()},
        buckets=buckets,
        tokens=tokens,
        proportional=proportional,
        denominator=10,
        target_budget=100.0,
        title="Toy Dashboard",
        metadata={"pool_size": 8, "max_epoch_cap": 100.0},
    )

    html = output_path.read_text()
    assert "Toy Dashboard" in html
    assert "Full swarm" in html
    assert "D-optimal only" in html
    assert "showScope" in html
    assert "Candidate type counts" in html


def test_baseline_candidates_are_first_and_include_requested_unimax_caps():
    proportional = np.array([0.5, 0.3, 0.2])
    buckets = ("large", "middle", "small")
    tokens = np.array([50.0, 30.0, 20.0])

    candidates, directions = baselines.build_candidates(
        buckets=buckets,
        tokens=tokens,
        proportional=proportional,
        denominator=100,
        random_direction_count=2,
        direction_seed=123,
        alpha=0.10,
    )

    assert directions.shape == (2, 3)
    assert [candidate.name for candidate in candidates[:6]] == [
        "baseline_proportional",
        "baseline_uniform",
        "baseline_unimax_epoch_cap_1",
        "baseline_unimax_epoch_cap_4",
        "baseline_unimax_epoch_cap_8",
        "baseline_unimax_epoch_cap_16",
    ]
    assert [candidate.candidate_type for candidate in candidates[:6]] == [
        "baseline_proportional",
        "baseline_uniform",
        "baseline_unimax",
        "baseline_unimax",
        "baseline_unimax",
        "baseline_unimax",
    ]
    assert [candidate.unimax_epoch_cap for candidate in candidates[:6]] == [None, None, 1.0, 4.0, 8.0, 16.0]
    np.testing.assert_array_equal(candidates[0].counts[0], np.array([50, 30, 20]))
    np.testing.assert_array_equal(candidates[1].counts[0], np.array([34, 33, 33]))
    assert all(candidate.intervention_type == "baseline" for candidate in candidates[:6])


def test_log_odds_tilt_materializes_symmetric_target_count_changes():
    proportional = np.array([0.5, 0.3, 0.2])

    plus = baselines.log_odds_tilt_counts(proportional, 2, log_odds_alpha=np.log(2.0), sign=1, denominator=100)
    minus = baselines.log_odds_tilt_counts(proportional, 2, log_odds_alpha=np.log(2.0), sign=-1, denominator=100)

    assert plus.sum() == 100
    assert minus.sum() == 100
    assert plus[2] > 20
    assert minus[2] < 20


def test_random_fisher_directions_are_centered_unit_norm_and_deterministic():
    proportional = np.array([0.5, 0.3, 0.2])

    directions_a = baselines.sample_centered_fisher_sphere_directions(proportional, n_directions=4, seed=123)
    directions_b = baselines.sample_centered_fisher_sphere_directions(proportional, n_directions=4, seed=123)

    np.testing.assert_allclose(directions_a, directions_b)
    np.testing.assert_allclose(directions_a @ proportional, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(np.sqrt((directions_a * directions_a * proportional[None, :]).sum(axis=1)), 1.0)


def test_random_direction_candidates_are_paired_and_label_sign_in_type():
    proportional = np.array([0.5, 0.3, 0.2])
    buckets = ("large", "middle", "small")
    tokens = np.array([50.0, 30.0, 20.0])

    candidates, directions = baselines.build_candidates(
        buckets=buckets,
        tokens=tokens,
        proportional=proportional,
        denominator=100,
        random_direction_count=2,
        direction_seed=123,
        alpha=0.10,
    )

    assert len(candidates) == 6 + 3 + 2 * 2
    assert directions.shape == (2, 3)
    assert [candidate.candidate_type for candidate in candidates[6:9]] == ["partition_ablation"] * 3
    assert [candidate.candidate_type for candidate in candidates[9:]] == [
        "projected_controllability_plus",
        "projected_controllability_minus",
        "projected_controllability_plus",
        "projected_controllability_minus",
    ]
    assert [candidate.direction_id for candidate in candidates[9:]] == [
        "pcdir_000",
        "pcdir_000",
        "pcdir_001",
        "pcdir_001",
    ]


def test_random_direction_pair_realized_cosine_stays_close_after_quantization():
    proportional = np.array([0.5, 0.3, 0.2])
    direction = baselines.sample_centered_fisher_sphere_directions(proportional, n_directions=1, seed=123)[0]

    plus = baselines.central_logit_tilt_counts(proportional, direction, alpha=0.10, sign=1, denominator=10_000)
    minus = baselines.central_logit_tilt_counts(proportional, direction, alpha=0.10, sign=-1, denominator=10_000)
    cosine = baselines.realized_logit_direction_cosine(
        proportional=proportional,
        intended_direction=direction,
        plus_counts=plus,
        minus_counts=minus,
        denominator=10_000,
        alpha=0.10,
    )

    assert cosine > 0.99


def test_central_logit_tilt_counts_preserve_support_for_projected_controllability():
    proportional = np.array([2.0, 3.0, 9995.0]) / 10_000
    direction = baselines.sample_centered_fisher_sphere_directions(proportional, n_directions=1, seed=123)[0]

    plus = baselines.central_logit_tilt_counts(proportional, direction, alpha=0.10, sign=1, denominator=10_000)
    minus = baselines.central_logit_tilt_counts(proportional, direction, alpha=0.10, sign=-1, denominator=10_000)

    assert plus.sum() == 10_000
    assert minus.sum() == 10_000
    assert plus.min() >= 1
    assert minus.min() >= 1


def test_final_candidate_frame_orders_baselines_perturbations_then_quantized_d_optimal():
    buckets = ("large", "middle", "small")
    generated = pd.DataFrame(
        [
            {
                "candidate_name": "baseline_proportional",
                "candidate_type": "baseline_proportional",
                "phase_0/large": 0.5,
                "phase_0/middle": 0.3,
                "phase_0/small": 0.2,
                "phase_1/large": 0.5,
                "phase_1/middle": 0.3,
                "phase_1/small": 0.2,
            },
            {
                "candidate_name": "abl_del_large",
                "candidate_type": "partition_ablation",
                "phase_0/large": 0.0,
                "phase_0/middle": 0.6,
                "phase_0/small": 0.4,
                "phase_1/large": 0.0,
                "phase_1/middle": 0.6,
                "phase_1/small": 0.4,
            },
        ]
    )
    d_optimal = pd.DataFrame(
        [
            {
                "candidate_name": "fisher_dsp_aligned_000000",
                "candidate_type": "fisher_dsp_aligned",
                "phase_0/large": 0.8,
                "phase_0/middle": 0.1,
                "phase_0/small": 0.1,
                "phase_1/large": 0.401,
                "phase_1/middle": 0.2,
                "phase_1/small": 0.399,
            }
        ]
    )

    final = baselines.final_candidate_frame(
        generated,
        d_optimal,
        buckets=buckets,
        denominator=100,
        tokens=np.array([50.0, 30.0, 20.0]),
        max_epoch_cap=1.0,
        target_budget=100.0,
    )

    assert final["candidate_name"].tolist() == [
        "baseline_proportional",
        "abl_del_large",
        "fisher_dsp_aligned_000000",
    ]
    for phase in ("phase_0", "phase_1"):
        values = final[[f"{phase}/{bucket}" for bucket in buckets]].to_numpy(dtype=float)
        np.testing.assert_allclose(values.sum(axis=1), 1.0)
        np.testing.assert_allclose(values * 100, np.round(values * 100))
    assert final.loc[2, "phase_0/large"] <= 0.62


def test_final_dashboard_scopes_use_quantized_d_optimal_rows():
    frame = pd.DataFrame(
        [
            {
                "candidate_name": "baseline_proportional",
                "candidate_type": "baseline_proportional",
                "phase_0/large": 0.5,
                "phase_0/small": 0.5,
                "phase_1/large": 0.5,
                "phase_1/small": 0.5,
            },
            {
                "candidate_name": "fisher_dsp_aligned_000000",
                "candidate_type": "fisher_dsp_aligned",
                "phase_0/large": 0.61,
                "phase_0/small": 0.39,
                "phase_1/large": 0.40,
                "phase_1/small": 0.60,
            },
        ]
    )

    scopes = baselines.final_dashboard_scopes(frame, generated_row_count=1)

    assert scopes["Full swarm"] is frame
    d_optimal_scope = scopes["D-optimal only"]
    assert d_optimal_scope["candidate_name"].tolist() == ["fisher_dsp_aligned_000000"]
    assert d_optimal_scope.loc[1, "phase_0/large"] == 0.61
