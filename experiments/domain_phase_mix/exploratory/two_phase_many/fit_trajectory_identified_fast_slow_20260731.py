# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test a trajectory-identified fast/slow consolidation state.

For bucket ``i`` and normalized training progress ``u``, the latent states obey

    dz_fast / du = alpha * q_i(u) * (1 - z_fast) - lambda * z_fast
    dz_slow / du = kappa * z_fast * (1 - z_slow).

The fast state acquires and forgets. The slow state consolidates accumulated
fast mastery and does not forget in this first falsification. Both remain in
``[0, 1]`` and have an exact transition under a constant exposure rate.

Only paired 300M Uncheatable increments ending by step 21,000 select nonlinear
parameters and response ridge. The 21,000-to-22,000 increment and final
step-22,887 endpoint are held out. Table-9 fits only response amplitudes after
the transition and ridge are frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_trajectory_identified_acquisition_forgetting_20260731 as one_state,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "trajectory_identified_fast_slow_20260731"

ACQUISITION_GRID = (0.3, 1.0, 3.0, 10.0)
FORGETTING_GRID = (0.3, 1.0, 3.0, 10.0)
CONSOLIDATION_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RIDGE_GRID = one_state.RIDGE_GRID


@dataclass(frozen=True)
class Candidate:
    """Nonlinear transition parameters and response regularization."""

    acquisition_rate: float
    forgetting_rate: float
    consolidation_rate: float
    ridge: float


@dataclass(frozen=True)
class Evaluation:
    """Predictions and response parameters for one state ablation."""

    metrics: tuple[dict[str, float | str], ...]
    interval_prediction: np.ndarray
    predicted_trajectory: np.ndarray
    endpoint_prediction: np.ndarray
    table9_prediction: np.ndarray
    full_coefficients: np.ndarray
    fold_coefficients: np.ndarray
    table9_coefficients: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def advance_fast_slow(
    fast: np.ndarray,
    slow: np.ndarray,
    exposure_rate: np.ndarray,
    duration: float,
    acquisition_rate: float,
    forgetting_rate: float,
    consolidation_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact constant-rate fast/slow transition."""
    total_rate = acquisition_rate * exposure_rate + forgetting_rate
    equilibrium = acquisition_rate * exposure_rate / total_rate
    decay = np.exp(-total_rate * duration)
    next_fast = equilibrium + (fast - equilibrium) * decay
    integrated_fast = equilibrium * duration + (fast - equilibrium) * (1.0 - decay) / total_rate
    next_slow = 1.0 - (1.0 - slow) * np.exp(-consolidation_rate * integrated_fast)
    return next_fast, next_slow


def states_at_progress(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    progress: np.ndarray,
    acquisition_rate: float,
    forgetting_rate: float,
    consolidation_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate fast and slow states under the exact two-phase schedule."""
    phase_fraction = float(np.median(c0 / (c0 + c1)))
    phase0_rate = c0[None, :] * weights[:, 0, :] / phase_fraction
    phase1_rate = c1[None, :] * weights[:, 1, :] / (1.0 - phase_fraction)
    initial_fast = np.zeros_like(phase0_rate)
    initial_slow = np.zeros_like(phase0_rate)
    boundary_fast, boundary_slow = advance_fast_slow(
        initial_fast,
        initial_slow,
        phase0_rate,
        phase_fraction,
        acquisition_rate,
        forgetting_rate,
        consolidation_rate,
    )

    fast_states = []
    slow_states = []
    for time in progress:
        if time <= phase_fraction:
            fast, slow = advance_fast_slow(
                initial_fast,
                initial_slow,
                phase0_rate,
                float(time),
                acquisition_rate,
                forgetting_rate,
                consolidation_rate,
            )
        else:
            fast, slow = advance_fast_slow(
                boundary_fast,
                boundary_slow,
                phase1_rate,
                float(time - phase_fraction),
                acquisition_rate,
                forgetting_rate,
                consolidation_rate,
            )
        fast_states.append(fast)
        slow_states.append(slow)
    return np.stack(fast_states, axis=1), np.stack(slow_states, axis=1)


def state_features(
    data: one_state.PairData,
    acquisition_rate: float,
    forgetting_rate: float,
    consolidation_rate: float,
) -> np.ndarray:
    """Return family-pooled fast and slow asymmetric-minus-tied states."""
    asymmetric_fast, asymmetric_slow = states_at_progress(
        data.asymmetric_weights,
        data.c0,
        data.c1,
        data.progress,
        acquisition_rate,
        forgetting_rate,
        consolidation_rate,
    )
    tied_fast, tied_slow = states_at_progress(
        data.tied_weights,
        data.c0,
        data.c1,
        data.progress,
        acquisition_rate,
        forgetting_rate,
        consolidation_rate,
    )
    fast_delta = asymmetric_fast - tied_fast
    slow_delta = asymmetric_slow - tied_slow
    fast_features = np.stack(
        [fast_delta[:, :, members].mean(axis=2) for members in data.family_members],
        axis=2,
    )
    slow_features = np.stack(
        [slow_delta[:, :, members].mean(axis=2) for members in data.family_members],
        axis=2,
    )
    return np.concatenate([fast_features, slow_features], axis=2)


def transition_grid() -> tuple[tuple[float, float, float], ...]:
    """Return the frozen nonlinear transition grid."""
    return tuple(
        (acquisition, forgetting, consolidation)
        for acquisition in ACQUISITION_GRID
        for forgetting in FORGETTING_GRID
        for consolidation in CONSOLIDATION_GRID
    )


def feature_cache(data: one_state.PairData) -> dict[tuple[float, float, float], np.ndarray]:
    """Materialize all frozen transition features once."""
    return {transition: state_features(data, *transition) for transition in transition_grid()}


def select_candidate(
    data: one_state.PairData,
    features_by_transition: dict[tuple[float, float, float], np.ndarray],
    pair_indices: np.ndarray | None = None,
    seed: int = one_state.SPLIT_SEED,
) -> tuple[Candidate, pd.DataFrame]:
    """Select the transition only from pre-final paired increments."""
    local = np.arange(len(data.keys)) if pair_indices is None else np.asarray(pair_indices, dtype=int)
    local_splits = one_state.pair_splits(len(local), seed)
    splits = tuple((local[train], local[test]) for train, test in local_splits)
    local_mask = np.zeros(len(data.keys), dtype=bool)
    local_mask[local] = True
    rows = []
    best_key: tuple[float, float, float, float, float, float] | None = None
    best_candidate: Candidate | None = None

    for transition, features in features_by_transition.items():
        for ridge in RIDGE_GRID:
            prediction, target, phase1, _coefficients = one_state.candidate_oof(
                data,
                features,
                ridge,
                splits,
            )
            phase1_rmse = one_state.rmse(target[local_mask][:, phase1], prediction[local_mask][:, phase1])
            all_rmse = one_state.rmse(target[local_mask], prediction[local_mask])
            candidate = Candidate(*transition, ridge)
            rows.append(
                {
                    **asdict(candidate),
                    "phase1_interval_oof_rmse": phase1_rmse,
                    "all_interval_oof_rmse": all_rmse,
                }
            )
            key = (
                phase1_rmse,
                all_rmse,
                candidate.acquisition_rate,
                candidate.forgetting_rate,
                candidate.consolidation_rate,
                candidate.ridge,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No fast/slow candidate was scored")
    return best_candidate, pd.DataFrame(rows)


def evaluate_state_variant(
    variant: str,
    data: one_state.PairData,
    features: np.ndarray,
    ridge: float,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> Evaluation:
    """Evaluate one fixed-state response ablation."""
    design, interval_target, phase1, _end_steps = one_state.interval_arrays(data, features)
    interval_prediction, _target, _phase1, fold_coefficients = one_state.candidate_oof(
        data,
        features,
        ridge,
        splits,
    )
    flat_design = design.reshape(-1, design.shape[2])
    flat_target = interval_target.reshape(-1)
    finite = np.isfinite(flat_target)
    full_coefficients = one_state.fit_nonnegative_response(
        flat_design[finite],
        flat_target[finite],
        ridge,
    )
    predicted_trajectory = -features @ full_coefficients

    heldout_start = int(np.flatnonzero(data.steps == one_state.TRANSITION_TRAIN_END_STEP)[0])
    heldout_end = int(np.flatnonzero(data.steps == one_state.TRANSITION_HOLDOUT_END_STEP)[0])
    heldout_design = -(features[:, heldout_end, :] - features[:, heldout_start, :])
    heldout_observed = data.observed_delta[:, heldout_end] - data.observed_delta[:, heldout_start]
    heldout_prediction = heldout_design @ full_coefficients

    endpoint_index = int(np.flatnonzero(data.steps == one_state.FINAL_STEP)[0])
    endpoint_prediction = predicted_trajectory[:, endpoint_index]
    endpoint_design = -features[:, endpoint_index, :]
    table9_prediction, table9_coefficients = one_state.endpoint_oof(
        endpoint_design,
        data.table9_delta,
        ridge,
        splits,
    )

    metric_rows = (
        {
            "variant": variant,
            "evaluation": "uncheatable_pre_final_interval_oof_all",
            **one_state.prediction_metrics(interval_target.ravel(), interval_prediction.ravel()),
            "hpr_reference_rmse": float("nan"),
        },
        {
            "variant": variant,
            "evaluation": "uncheatable_pre_final_interval_oof_phase1",
            **one_state.prediction_metrics(
                interval_target[:, phase1].ravel(),
                interval_prediction[:, phase1].ravel(),
            ),
            "hpr_reference_rmse": float("nan"),
        },
        {
            "variant": variant,
            "evaluation": "uncheatable_21000_to_22000_holdout",
            **one_state.prediction_metrics(heldout_observed, heldout_prediction),
            "hpr_reference_rmse": float("nan"),
        },
        {
            "variant": variant,
            "evaluation": "uncheatable_final_endpoint_strict_holdout",
            **one_state.prediction_metrics(data.endpoint_delta, endpoint_prediction),
            "hpr_reference_rmse": one_state.HPR_PAIR_RMSE["uncheatable"],
        },
        {
            "variant": variant,
            "evaluation": "table9_final_endpoint_frozen_state_oof",
            **one_state.prediction_metrics(data.table9_delta, table9_prediction),
            "hpr_reference_rmse": one_state.HPR_PAIR_RMSE["table9"],
        },
    )
    return Evaluation(
        metrics=metric_rows,
        interval_prediction=interval_prediction,
        predicted_trajectory=predicted_trajectory,
        endpoint_prediction=endpoint_prediction,
        table9_prediction=table9_prediction,
        full_coefficients=full_coefficients,
        fold_coefficients=fold_coefficients,
        table9_coefficients=table9_coefficients,
    )


def feature_labels(data: one_state.PairData, variant: str) -> tuple[str, ...]:
    """Return response-feature labels for one ablation."""
    if variant == "fast_only":
        return tuple(f"fast:{family}" for family in data.family_names)
    if variant == "slow_only":
        return tuple(f"slow:{family}" for family in data.family_names)
    return tuple(f"{state}:{family}" for state in ("fast", "slow") for family in data.family_names)


def selected_on_boundary(candidate: Candidate) -> bool:
    """Whether any nonlinear parameter lies on its frozen grid boundary."""
    return (
        candidate.acquisition_rate in (min(ACQUISITION_GRID), max(ACQUISITION_GRID))
        or candidate.forgetting_rate in (min(FORGETTING_GRID), max(FORGETTING_GRID))
        or candidate.consolidation_rate in (min(CONSOLIDATION_GRID), max(CONSOLIDATION_GRID))
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = one_state.load_pair_data()
    cached_features = feature_cache(data)
    selected, sweep = select_candidate(data, cached_features)
    sweep.to_csv(args.output_dir / "candidate_sweep.csv", index=False)

    selected_features = cached_features[
        (selected.acquisition_rate, selected.forgetting_rate, selected.consolidation_rate)
    ]
    family_count = len(data.family_names)
    variants = {
        "fast_slow": selected_features,
        "fast_only": selected_features[:, :, :family_count],
        "slow_only": selected_features[:, :, family_count:],
    }
    splits = one_state.pair_splits(len(data.keys), one_state.SPLIT_SEED)
    evaluations = {
        name: evaluate_state_variant(name, data, features, selected.ridge, splits) for name, features in variants.items()
    }

    metrics = pd.DataFrame(row for evaluation in evaluations.values() for row in evaluation.metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    all_pairs = np.arange(len(data.keys))
    stability_rows = []
    for fold, (_train, excluded) in enumerate(splits):
        retained = np.setdiff1d(all_pairs, excluded)
        fold_selected, _fold_sweep = select_candidate(
            data,
            cached_features,
            retained,
            seed=one_state.SPLIT_SEED + 100 * (fold + 1),
        )
        stability_rows.append(
            {
                "excluded_fold": fold,
                **asdict(fold_selected),
                "nonlinear_parameter_on_boundary": selected_on_boundary(fold_selected),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(args.output_dir / "transition_stability.csv", index=False)

    parameter_rows = []
    for variant, evaluation in evaluations.items():
        labels = feature_labels(data, variant)
        for label, value in zip(labels, evaluation.full_coefficients, strict=True):
            parameter_rows.append(
                {
                    "variant": variant,
                    "fit": "uncheatable_pre_final_full",
                    "fold": -1,
                    "feature": label,
                    "response_bpb_per_mean_state": value,
                }
            )
        for fold, coefficients in enumerate(evaluation.fold_coefficients):
            for label, value in zip(labels, coefficients, strict=True):
                parameter_rows.append(
                    {
                        "variant": variant,
                        "fit": "uncheatable_pre_final_oof",
                        "fold": fold,
                        "feature": label,
                        "response_bpb_per_mean_state": value,
                    }
                )
        for fold, coefficients in enumerate(evaluation.table9_coefficients):
            for label, value in zip(labels, coefficients, strict=True):
                parameter_rows.append(
                    {
                        "variant": variant,
                        "fit": "table9_endpoint_oof",
                        "fold": fold,
                        "feature": label,
                        "response_bpb_per_mean_state": value,
                    }
                )
    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(args.output_dir / "response_parameters.csv", index=False)

    full = evaluations["fast_slow"]
    endpoint_frame = pd.DataFrame(
        {
            "phase_correspondence_key": data.keys,
            "uncheatable_observed_delta": data.endpoint_delta,
            "uncheatable_predicted_delta_strict_holdout": full.endpoint_prediction,
            "table9_observed_delta": data.table9_delta,
            "table9_predicted_delta_frozen_state_oof": full.table9_prediction,
        }
    )
    endpoint_frame.to_csv(args.output_dir / "endpoint_predictions.csv", index=False)
    one_state.write_plots(
        data,
        full.predicted_trajectory,
        full.endpoint_prediction,
        args.output_dir,
    )

    fast_total = float(full.full_coefficients[:family_count].sum())
    slow_total = float(full.full_coefficients[family_count:].sum())
    hpr_u = one_state.HPR_PAIR_RMSE["uncheatable"]
    hpr_t9 = one_state.HPR_PAIR_RMSE["table9"]
    full_metric = metrics.loc[metrics["variant"].eq("fast_slow")].set_index("evaluation")
    heldout_row = full_metric.loc["uncheatable_21000_to_22000_holdout"]
    endpoint_u_row = full_metric.loc["uncheatable_final_endpoint_strict_holdout"]
    endpoint_t9_row = full_metric.loc["table9_final_endpoint_frozen_state_oof"]
    passes = {
        "heldout_interval_beats_zero_null": bool(heldout_row["rmse"] < heldout_row["zero_delta_null_rmse"]),
        "uncheatable_endpoint_within_five_percent_hpr": bool(endpoint_u_row["rmse"] <= 1.05 * hpr_u),
        "table9_endpoint_within_five_percent_hpr": bool(endpoint_t9_row["rmse"] <= 1.05 * hpr_t9),
        "fast_block_active": bool(fast_total > 1e-8),
        "slow_block_active": bool(slow_total > 1e-8),
        "nonlinear_parameters_interior": not selected_on_boundary(selected),
    }
    decision = "promote_to_wsd80_shape_audit" if all(passes.values()) else "reject_before_wsd80"
    selected_record = {
        **asdict(selected),
        "family_names": data.family_names,
        "phase_boundary_step": one_state.PHASE_BOUNDARY_STEP,
        "transition_training_end_step": one_state.TRANSITION_TRAIN_END_STEP,
        "transition_holdout_end_step": one_state.TRANSITION_HOLDOUT_END_STEP,
        "final_endpoint_step": one_state.FINAL_STEP,
        "fast_response_total": fast_total,
        "slow_response_total": slow_total,
        "passes": passes,
        "decision": decision,
    }
    (args.output_dir / "selected_model.json").write_text(
        json.dumps(selected_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = f"""# Trajectory-identified fast/slow consolidation state

## Frozen model

The bounded fast state obeys `dz_fast/du = alpha*q*(1-z_fast) - lambda*z_fast`.
The bounded slow state obeys
`dz_slow/du = kappa*z_fast*(1-z_slow)`. The slow state consolidates fast
mastery and does not forget in this first falsification.

- Selected acquisition rate: `{selected.acquisition_rate}`
- Selected forgetting rate: `{selected.forgetting_rate}`
- Selected consolidation rate: `{selected.consolidation_rate}`
- Selected response ridge: `{selected.ridge}`
- Transition selection used only increments ending by step
  {one_state.TRANSITION_TRAIN_END_STEP}.
- The {one_state.TRANSITION_TRAIN_END_STEP}-to-{one_state.TRANSITION_HOLDOUT_END_STEP}
  increment and final step {one_state.FINAL_STEP} were held out.
- Table-9 used the frozen transition and ridge; only six response amplitudes
  were refit inside correspondence-grouped folds.

## Decision

`{decision}`

{pd.DataFrame([passes]).to_markdown(index=False)}

## Falsification metrics

{metrics.to_markdown(index=False)}

## Transition stability

{stability.to_markdown(index=False)}

HPR's persisted exact-pair endpoint RMSE is {hpr_u:.6f} on Uncheatable and
{hpr_t9:.6f} on Table-9. Fixed-state `fast_only` and `slow_only` rows are nested
ablations; they do not reselect nonlinear parameters.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
