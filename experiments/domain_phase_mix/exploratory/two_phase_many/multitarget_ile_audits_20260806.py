# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-fit audits for the multi-target interference-evidence round.

Four questions, all answerable before a single parameter is fitted:

1. Do the published macro metrics equal the component aggregates they claim to be? If not, any joint
   fit that treats a macro and its own components as separate labels is counting the same measurement
   twice.
2. Can several metrics on the same checkpoint carry more information about a shared nonlinear state
   than one metric alone? If their residuals are near-perfectly correlated and they respond to the
   state in near-parallel directions, the answer is no and the round stops here.
3. Does the state have the algebraic properties it claims -- exact phase-blindness at zero
   interference, boundedness, a clean tied restriction?
4. Under measured noise, does fitting recover a known transition, and does it return zero interference
   when the truth is phase-blind?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    interference_evidence_model_20260806 as ile,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import starcoder_wsd80_panel_20260728 as wsd80

NUM_TABLE9_COMPONENTS = 51
NUM_MMLU_BUCKETS = 4
SYNTHETIC_DRAWS = 40
SYNTHETIC_SEED = 20260806


def _wsd80_metrics() -> tuple[wsd80.Panel, pd.DataFrame, tuple[str, ...]]:
    panel = wsd80.load_surface()
    metrics = pd.read_csv(wsd80.SURFACE_DIR / "wsd80_all_bpb_metrics.csv").drop_duplicates("wandb_run_id")
    joined = panel.frame[["wandb_run_id"]].merge(metrics, on="wandb_run_id", how="left", validate="one_to_one")
    columns = tuple(c for c in metrics.columns if c != "wandb_run_id" and joined[c].notna().all())
    return panel, joined, columns


def aggregation_audit() -> dict:
    """Check that each published macro is the aggregate it advertises, and find the missing components."""
    frame = expanded.load_300m("uncheatable").frame
    unc = [
        c
        for c in frame.columns
        if c.startswith("eval_uncheatable_eval_")
        and c.endswith("_bpb")
        and "macro" not in c
        and c != "eval_uncheatable_eval_bpb"
    ]
    with_components = frame[unc].notna().all(axis=1)
    unc_mean = frame.loc[with_components, unc].mean(axis=1)
    unc_macro_error = float(np.abs(unc_mean - frame.loc[with_components, "eval_uncheatable_eval_macro_bpb"]).max())

    leaves = [c for c in frame.columns if c.startswith("olmo_base_eval/")]
    leaf_sum = frame[leaves].sum(axis=1).to_numpy()
    macro = frame["table9_macro_bpb"].to_numpy()
    flat_error = float(np.abs(frame[leaves].mean(axis=1).to_numpy() - macro).max())
    implied = (NUM_TABLE9_COMPONENTS * macro - leaf_sum) / NUM_MMLU_BUCKETS

    return {
        "uncheatable_component_rows": int(with_components.sum()),
        "uncheatable_total_rows": len(frame),
        "uncheatable_macro_is_flat_mean_max_error": unc_macro_error,
        "table9_leaves_present": len(leaves),
        "table9_declared_components": NUM_TABLE9_COMPONENTS,
        "table9_flat_mean_of_present_leaves_max_error": flat_error,
        "derived_mmlu_bucket_mean_min": float(implied.min()),
        "derived_mmlu_bucket_mean_max": float(implied.max()),
        "derived_mmlu_bucket_mean_mean": float(implied.mean()),
        "leaf_component_bpb_min": float(frame[leaves].to_numpy().min()),
        "leaf_component_bpb_max": float(frame[leaves].to_numpy().max()),
    }


def cross_target_information() -> dict:
    """Is there room for several metrics to identify a state better than one?

    Two measurements. First, how correlated are the metrics' residuals once a common aggregate response
    is removed -- perfectly correlated residuals mean the second metric is a copy of the first for
    fitting purposes. Second, how parallel are the metrics' sensitivities to the shared nonlinear
    parameters -- parallel sensitivities mean they all pull the state in the same direction and only one
    of them was ever needed.
    """
    panel, joined, columns = _wsd80_metrics()
    geometry = _wsd80_geometry()
    values = joined[list(columns)].to_numpy(dtype=float)

    # Residual after a common phase-blind aggregate response: total epochs of each bucket, plus
    # over-exposure. This is the null every candidate has to beat, so residual structure relative to it
    # is exactly the signal a phase model can use.
    epochs_0, epochs_1 = ile.phase_epochs(panel.weights, geometry)
    basis = np.column_stack([np.ones(len(panel.y)), epochs_0 + epochs_1, ile.overexposure(panel.weights, geometry)])
    projector = basis @ np.linalg.pinv(basis)
    residual = values - projector @ values
    residual /= np.linalg.norm(residual, axis=0, keepdims=True)
    correlation = residual.T @ residual
    off_diagonal = correlation[~np.eye(len(columns), dtype=bool)]

    # Sensitivity direction: how each metric's best-fit response changes when the shared state changes.
    probes = (ile.Shape(0.5, 2.0), ile.Shape(0.75, 2.0), ile.Shape(0.5, 4.0))
    base = ile.evidence_state(panel.weights, geometry, probes[0])
    sensitivities = []
    for shape in probes[1:]:
        moved = ile.evidence_state(panel.weights, geometry, shape)
        direction = (moved - base).sum(axis=1)
        direction = direction - direction.mean()
        loadings = residual.T @ (direction / np.linalg.norm(direction))
        sensitivities.append(loadings)
    sensitivity = np.column_stack(sensitivities)
    normalized = sensitivity / np.linalg.norm(sensitivity, axis=1, keepdims=True)
    angles = normalized @ normalized.T
    angle_off = angles[~np.eye(len(columns), dtype=bool)]

    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    shares = eigenvalues / eigenvalues.sum()
    count = len(columns)
    mean_off_diagonal = float((correlation.sum() - count) / (count * (count - 1)))
    # Two summaries of how many independent labels these metrics are worth for a shared parameter. The
    # exact generalized-least-squares ratio is reported too, but it is not usable: the correlation
    # matrix is numerically singular, so inverting it credits differences between near-identical
    # metrics with almost noiseless information they do not actually carry.
    ones = np.ones(count)
    return {
        "n_metrics": count,
        "residual_correlation_median": float(np.median(np.abs(off_diagonal))),
        "residual_correlation_q10": float(np.quantile(np.abs(off_diagonal), 0.10)),
        "residual_correlation_q90": float(np.quantile(np.abs(off_diagonal), 0.90)),
        "residual_correlation_mean_off_diagonal": mean_off_diagonal,
        "residual_effective_rank": float(np.exp(-np.sum(shares * np.log(shares + 1e-300)))),
        "equicorrelation_effective_labels": float(count / (1.0 + (count - 1) * mean_off_diagonal)),
        "residual_top_eigenvalue_share": float(shares[0]),
        "residual_second_eigenvalue_share": float(shares[1]),
        "residual_correlation_condition_number": float(eigenvalues[0] / max(eigenvalues[-1], 1e-300)),
        "unusable_gls_effective_labels": float(ones @ np.linalg.pinv(correlation) @ ones),
        "sensitivity_alignment_median": float(np.median(np.abs(angle_off))),
        "sensitivity_alignment_q10": float(np.quantile(np.abs(angle_off), 0.10)),
        "sensitivity_same_sign_fraction": float(max(np.mean(sensitivity[:, 0] > 0), np.mean(sensitivity[:, 0] < 0))),
    }


def _wsd80_geometry() -> ile.Geometry:
    c0, c1 = wsd80.epoch_multipliers()
    return ile.Geometry(
        c0=c0,
        c1=c1,
        phase_1_fraction=wsd80.REALIZED_PHASE_1_FRACTION,
        family_index=np.arange(len(wsd80.DOMAIN_NAMES)),
    )


def algebraic_audit() -> dict:
    """Exact phase-blindness at zero interference, boundedness, tied restriction, refinement behaviour."""
    geometry = _wsd80_geometry()
    rng = np.random.default_rng(11)
    p0 = rng.uniform(0.0, 1.0, 400)
    p1 = rng.uniform(0.0, 1.0, 400)
    weights = np.stack(
        [np.column_stack([1 - p0, p0]), np.column_stack([1 - p1, p1])],
        axis=1,
    )

    # Zero interference must make the state a function of total epochs alone. Constructing pairs with
    # exactly equal total epochs is the sharpest form of that check.
    epochs_0, epochs_1 = ile.phase_epochs(weights, geometry)
    total = epochs_0 + epochs_1
    blind = ile.evidence_state(weights, geometry, ile.Shape(0.6, 0.0))
    predicted_blind = 1.0 - np.exp(-0.6 * total)
    blind_error = float(np.abs(blind - predicted_blind).max())

    bounded = ile.evidence_state(weights, geometry, ile.Shape(3.0, 8.0))
    tied = ile.evidence_state(ile.tied_weights(np.column_stack([1 - p0, p0])), geometry, ile.Shape(0.6, 4.0))

    # Monotone in rho at fixed policy: more efficient acquisition can never retain less evidence.
    low = ile.evidence_state(weights, geometry, ile.Shape(0.3, 4.0))
    high = ile.evidence_state(weights, geometry, ile.Shape(0.9, 4.0))

    # Monotone in interference: more interference can never retain more evidence.
    quiet = ile.evidence_state(weights, geometry, ile.Shape(0.6, 1.0))
    loud = ile.evidence_state(weights, geometry, ile.Shape(0.6, 8.0))

    # Refinement: split the broad bucket into two halves with the same pool density and weights.
    split_geometry = ile.Geometry(
        c0=np.array([geometry.c0[0] * 2, geometry.c0[0] * 2, geometry.c0[1]]),
        c1=np.array([geometry.c1[0] * 2, geometry.c1[0] * 2, geometry.c1[1]]),
        phase_1_fraction=geometry.phase_1_fraction,
        family_index=np.array([0, 0, 1]),
    )
    split_weights = np.stack(
        [
            np.column_stack([(1 - p0) / 2, (1 - p0) / 2, p0]),
            np.column_stack([(1 - p1) / 2, (1 - p1) / 2, p1]),
        ],
        axis=1,
    )
    split_state = ile.evidence_state(split_weights, split_geometry, ile.Shape(0.6, 4.0))
    coarse_state = ile.evidence_state(weights, geometry, ile.Shape(0.6, 4.0))
    refinement_gap = float(np.abs(split_state[:, 0] - coarse_state[:, 0]).max())

    return {
        "zero_interference_total_epoch_max_error": blind_error,
        "state_min": float(bounded.min()),
        "state_max": float(bounded.max()),
        "tied_state_finite": bool(np.all(np.isfinite(tied))),
        "monotone_in_rho_violations": int(np.sum(high < low - 1e-12)),
        "monotone_in_interference_violations": int(np.sum(loud > quiet + 1e-12)),
        "broad_bucket_refinement_max_gap": refinement_gap,
    }


def synthetic_recovery(law: ile.InterferenceLaw = ile.InterferenceLaw.ABSOLUTE) -> pd.DataFrame:
    """Fit simulated panels and check the transition comes back, including when the truth is phase-blind."""
    panel, _joined, _columns = _wsd80_metrics()
    geometry = _wsd80_geometry()
    shapes = ile.shape_grid(law=law)
    designs = {shape: ile.design_matrix(panel.weights, geometry, shape) for shape in shapes}
    rng = np.random.default_rng(SYNTHETIC_SEED)

    truths = (
        ile.Shape(0.5, 4.0, law),
        ile.Shape(0.75, 2.0, law),
        ile.Shape(0.5, 0.0, law),
    )
    # Training-seed noise on this panel, measured from its replicated coordinates.
    sigma = wsd80.training_seed_sigma(wsd80.load_fiber_replicates())

    rows = []
    for truth in truths:
        state = ile.evidence_state(panel.weights, geometry, truth)
        clean = 1.4 - 0.35 * state[:, 0] - 0.55 * state[:, 1]
        for draw in range(SYNTHETIC_DRAWS):
            observed = clean + rng.normal(0.0, sigma, len(clean))
            scored = []
            for shape, design in designs.items():
                head = ile.solve_head(design, observed, geometry, 1e-3)
                residual = design @ ile.coefficient_vector(head) - observed
                scored.append((float(residual @ residual), shape))
            best = min(scored)[1]
            rows.append(
                {
                    "true_rho": truth.rho,
                    "true_interference": truth.interference,
                    "draw": draw,
                    "fitted_rho": best.rho,
                    "fitted_interference": best.interference,
                    "noise_sigma": sigma,
                }
            )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> None:
    # Deferred to break the cycle with the harness module; see its `main` for the same note.
    from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
        benchmark_multitarget_interference_evidence_20260806 as harness,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregation = aggregation_audit()
    information = cross_target_information()
    algebra = algebraic_audit()
    recovery = synthetic_recovery()

    recovery.to_csv(output_dir / "synthetic_recovery.csv", index=False)
    recovery["rho_exact"] = recovery["fitted_rho"] == recovery["true_rho"]
    recovery["interference_exact"] = recovery["fitted_interference"] == recovery["true_interference"]
    summary = (
        recovery.groupby(["true_rho", "true_interference"])
        .agg(
            median_fitted_rho=("fitted_rho", "median"),
            median_fitted_interference=("fitted_interference", "median"),
            rho_exact_rate=("rho_exact", "mean"),
            interference_exact_rate=("interference_exact", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "synthetic_recovery_summary.csv", index=False)

    payload = {
        "protocol": harness.protocol_hash({"stage": "audit"}),
        "aggregation": aggregation,
        "cross_target_information": information,
        "algebra": algebra,
        "synthetic_recovery": summary.to_dict(orient="records"),
    }
    harness.write_json(output_dir / "audit.json", payload)

    print("=== aggregation ===")
    for key, value in aggregation.items():
        print(f"  {key}: {value}")
    print("=== cross-target information ===")
    for key, value in information.items():
        print(f"  {key}: {value}")
    print("=== algebra ===")
    for key, value in algebra.items():
        print(f"  {key}: {value}")
    print("=== synthetic recovery ===")
    print(summary.to_string(index=False))
