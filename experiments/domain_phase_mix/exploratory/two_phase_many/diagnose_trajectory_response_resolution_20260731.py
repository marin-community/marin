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
"""Locate whether trajectory-state failure comes from response resolution.

This diagnostic freezes the transition parameters already selected by
WSD80-SUR-060 and WSD80-SUR-061. It changes only the response basis:
predeclared family averages versus bucket-resolved nonnegative amplitudes.
Only ridge is selected from pre-final paired increments. No result from this
script is sufficient to promote a high-dimensional bucket head.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_trajectory_identified_acquisition_forgetting_20260731 as one_state,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_trajectory_identified_fast_slow_20260731 as fast_slow,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "trajectory_response_resolution_20260731"

FAST_ACQUISITION = 10.0
FAST_FORGETTING = 10.0
CONSOLIDATION = 10.0
RIDGE_GRID = (0.0, 0.1, 1.0, 10.0, 100.0, 1_000.0)


@dataclass(frozen=True)
class ResponseFit:
    """Selected response basis and its frozen evaluation."""

    variant: str
    ridge: float
    evaluation: fast_slow.Evaluation
    feature_labels: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def raw_fast_delta(data: one_state.PairData) -> np.ndarray:
    """Return bucket-resolved fast-state difference."""
    asymmetric = one_state.states_at_progress(
        data.asymmetric_weights,
        data.c0,
        data.c1,
        data.progress,
        FAST_ACQUISITION,
        FAST_FORGETTING,
    )
    tied = one_state.states_at_progress(
        data.tied_weights,
        data.c0,
        data.c1,
        data.progress,
        FAST_ACQUISITION,
        FAST_FORGETTING,
    )
    return asymmetric - tied


def raw_fast_slow_delta(data: one_state.PairData) -> tuple[np.ndarray, np.ndarray]:
    """Return bucket-resolved fast and slow state differences."""
    asymmetric_fast, asymmetric_slow = fast_slow.states_at_progress(
        data.asymmetric_weights,
        data.c0,
        data.c1,
        data.progress,
        FAST_ACQUISITION,
        FAST_FORGETTING,
        CONSOLIDATION,
    )
    tied_fast, tied_slow = fast_slow.states_at_progress(
        data.tied_weights,
        data.c0,
        data.c1,
        data.progress,
        FAST_ACQUISITION,
        FAST_FORGETTING,
        CONSOLIDATION,
    )
    return asymmetric_fast - tied_fast, asymmetric_slow - tied_slow


def family_pool(data: one_state.PairData, bucket_features: np.ndarray) -> np.ndarray:
    """Average bucket states inside each predeclared family."""
    return np.stack(
        [bucket_features[:, :, members].mean(axis=2) for members in data.family_members],
        axis=2,
    )


def feature_sets(data: one_state.PairData) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    """Return the four frozen response-resolution arms."""
    fast_bucket = raw_fast_delta(data)
    fast_bucket_two_state, slow_bucket = raw_fast_slow_delta(data)
    if not np.allclose(fast_bucket, fast_bucket_two_state):
        raise ValueError("Fast state differs between one-state and two-state transitions")

    family_fast = family_pool(data, fast_bucket)
    family_slow = family_pool(data, slow_bucket)
    domain_names = tuple(str(name) for name in one_state.benchmark.load_300m("uncheatable").domain_names)
    return {
        "family_fast": (
            family_fast,
            tuple(f"fast:{family}" for family in data.family_names),
        ),
        "bucket_fast": (
            fast_bucket,
            tuple(f"fast:{domain}" for domain in domain_names),
        ),
        "family_fast_slow": (
            np.concatenate([family_fast, family_slow], axis=2),
            tuple(f"{state}:{family}" for state in ("fast", "slow") for family in data.family_names),
        ),
        "bucket_fast_slow": (
            np.concatenate([fast_bucket, slow_bucket], axis=2),
            tuple(f"{state}:{domain}" for state in ("fast", "slow") for domain in domain_names),
        ),
    }


def select_ridge(
    data: one_state.PairData,
    features: np.ndarray,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, pd.DataFrame]:
    """Select ridge from phase-1 and all pre-final increment OOF error."""
    rows = []
    best: tuple[float, float, float] | None = None
    for ridge in RIDGE_GRID:
        prediction, target, phase1, _coefficients = one_state.candidate_oof(
            data,
            features,
            ridge,
            splits,
        )
        phase1_rmse = one_state.rmse(target[:, phase1], prediction[:, phase1])
        all_rmse = one_state.rmse(target, prediction)
        rows.append(
            {
                "ridge": ridge,
                "phase1_interval_oof_rmse": phase1_rmse,
                "all_interval_oof_rmse": all_rmse,
            }
        )
        key = (phase1_rmse, all_rmse, ridge)
        if best is None or key < best:
            best = key
    if best is None:
        raise RuntimeError("No response ridge was scored")
    return best[2], pd.DataFrame(rows)


def trajectory_correlations(data: one_state.PairData) -> pd.DataFrame:
    """Quantify when intermediate pair deltas become predictive of final deltas."""
    rows = []
    for index, step in enumerate(data.steps):
        intermediate = data.observed_delta[:, index]
        finite = np.isfinite(intermediate) & np.isfinite(data.endpoint_delta)
        if int(finite.sum()) < 10:
            continue
        x = intermediate[finite]
        y = data.endpoint_delta[finite]
        rows.append(
            {
                "step": int(step),
                "n_pairs": int(finite.sum()),
                "pearson_with_final": float(np.corrcoef(x, y)[0, 1]),
                "spearman_with_final": float(spearmanr(x, y).statistic),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = one_state.load_pair_data()
    splits = one_state.pair_splits(len(data.keys), one_state.SPLIT_SEED)

    fits = []
    ridge_rows = []
    for variant, (features, labels) in feature_sets(data).items():
        ridge, sweep = select_ridge(data, features, splits)
        sweep.insert(0, "variant", variant)
        ridge_rows.append(sweep)
        evaluation = fast_slow.evaluate_state_variant(
            variant,
            data,
            features,
            ridge,
            splits,
        )
        fits.append(ResponseFit(variant, ridge, evaluation, labels))

    pd.concat(ridge_rows, ignore_index=True).to_csv(args.output_dir / "ridge_sweep.csv", index=False)
    metrics = pd.DataFrame(row for fit in fits for row in fit.evaluation.metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    selected_rows = []
    parameter_rows = []
    endpoint = pd.DataFrame(
        {
            "phase_correspondence_key": data.keys,
            "uncheatable_observed_delta": data.endpoint_delta,
            "table9_observed_delta": data.table9_delta,
        }
    )
    for fit in fits:
        selected_rows.append(
            {
                "variant": fit.variant,
                "ridge": fit.ridge,
                "feature_count": len(fit.feature_labels),
                "active_uncheatable_full_coefficients": int(np.count_nonzero(fit.evaluation.full_coefficients > 1e-10)),
                "active_table9_fold_coefficients_mean": float(
                    np.mean(np.count_nonzero(fit.evaluation.table9_coefficients > 1e-10, axis=1))
                ),
            }
        )
        for label, value in zip(
            fit.feature_labels,
            fit.evaluation.full_coefficients,
            strict=True,
        ):
            parameter_rows.append(
                {
                    "variant": fit.variant,
                    "feature": label,
                    "uncheatable_pre_final_full_coefficient": value,
                }
            )
        endpoint[f"{fit.variant}_uncheatable_prediction"] = fit.evaluation.endpoint_prediction
        endpoint[f"{fit.variant}_table9_prediction"] = fit.evaluation.table9_prediction
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.output_dir / "selected_response_heads.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "response_parameters.csv", index=False)
    endpoint.to_csv(args.output_dir / "endpoint_predictions.csv", index=False)

    correlations = trajectory_correlations(data)
    correlations.to_csv(args.output_dir / "trajectory_correlations.csv", index=False)

    plotted = metrics.loc[
        metrics["evaluation"].isin(
            [
                "uncheatable_pre_final_interval_oof_phase1",
                "uncheatable_21000_to_22000_holdout",
                "uncheatable_final_endpoint_strict_holdout",
                "table9_final_endpoint_frozen_state_oof",
            ]
        )
    ].copy()
    figure = px.bar(
        plotted,
        x="evaluation",
        y="rmse",
        color="variant",
        barmode="group",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Frozen-state response-resolution diagnostic",
    )
    figure.update_layout(template="plotly_white", height=700, width=1300)
    figure.write_html(args.output_dir / "response_resolution_metrics.html", include_plotlyjs="cdn")

    endpoint_metrics = metrics.loc[
        metrics["evaluation"].isin(
            [
                "uncheatable_21000_to_22000_holdout",
                "uncheatable_final_endpoint_strict_holdout",
                "table9_final_endpoint_frozen_state_oof",
            ]
        ),
        ["variant", "evaluation", "rmse", "zero_delta_null_rmse", "hpr_reference_rmse", "spearman"],
    ]
    phase_boundary_steps = [17_000, 18_000, 19_000, 20_000, 21_000, 22_000, 22_887]
    phase_boundary_correlations = correlations.loc[correlations["step"].isin(phase_boundary_steps)].to_markdown(
        index=False
    )
    report = f"""# Trajectory response-resolution diagnostic

## Frozen protocol

The one-state transition is fixed at acquisition `{FAST_ACQUISITION}` and
forgetting `{FAST_FORGETTING}`. The two-state transition additionally fixes
consolidation `{CONSOLIDATION}`. These are the already exposed selections from
SUR-060 and SUR-061. Only response ridge is selected from paired increments
ending by step {one_state.TRANSITION_TRAIN_END_STEP}.

Bucket-resolved heads are diagnostic only. They cannot promote a model because
they replace three or six predeclared family amplitudes with 39 or 78 target-
specific coefficients.

## Selected heads

{selected.to_markdown(index=False)}

## Held-out decision metrics

{endpoint_metrics.to_markdown(index=False)}

## Phase-boundary sign reversal

{phase_boundary_correlations}

Before the phase boundary, intermediate pair effects are negatively correlated
with final effects. The correlation changes sign immediately after the switch
and approaches one near the endpoint. This is evidence for a fast phase-local
response, not for a generic scalar rescaling of BPB over training progress.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
