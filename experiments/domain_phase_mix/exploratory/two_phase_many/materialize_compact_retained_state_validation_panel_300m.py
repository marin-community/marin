# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Materialize the compact retained-state surrogate's 2x2 validation panel.

The panel crosses policy class (one phase or two phases) with a structural
prior (bucket-only responses or bucket responses plus three family-coverage
channels). Each cell selects ridge regularization independently by repeated
five-fold CV. The one-phase fit uses total exposure and removes all retention
parameters. The paired two-phase candidate keeps its one-phase aggregate
exactly fixed and optimizes only phase order under a small information budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_decoupled_phase_information_constraints_300m as phase_information,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_decoupled_phase_information_model_family_panel_300m as decoupled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_original_style_matched_sepheads_ablation_300m as matched,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as objective_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_retained_state_validation_panel_20260713"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_compact_retained_state_validation_20260713/mixtures"
)
DEFAULT_L2_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0)
CV_SEEDS = (0, 1, 2)
AGGREGATE_KL = 0.05
PHASE_INFORMATION_BUDGET = 0.005
FIT_MAXITER = 24
FIT_TOP_K = 2
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Arm:
    grouping: str
    policy: str
    config: retained.ModelConfig

    @property
    def candidate(self) -> str:
        return f"retstate_unch_{self.grouping}_{self.policy}"


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def model_config(policy: str, family_coverage: bool) -> retained.ModelConfig:
    grouping = "grouped" if family_coverage else "nogroup"
    if policy == "1p":
        return retained.ModelConfig(
            name=f"total_exposure_weibull_shared_replay_{grouping}",
            signal=retained.SignalKind.TOTAL_EXPOSURE,
            response=retained.ResponseKind.WEIBULL,
            retention=retained.RetentionKind.CONSTANT,
            replay_penalty=retained.ReplayPenaltyKind.SHARED,
            family_coverage=family_coverage,
        )
    if policy == "2p":
        return retained.ModelConfig(
            name=f"retained_state_weibull_shared_replay_{grouping}",
            signal=retained.SignalKind.RETAINED_STATE,
            response=retained.ResponseKind.WEIBULL,
            retention=retained.RetentionKind.REVISIT_GATED,
            replay_penalty=retained.ReplayPenaltyKind.SHARED,
            family_coverage=family_coverage,
        )
    raise ValueError(f"Unknown policy {policy!r}")


def panel_arms() -> tuple[Arm, ...]:
    return tuple(
        Arm(grouping, policy, model_config(policy, grouping == "grouped"))
        for grouping in ("nogroup", "grouped")
        for policy in ("1p", "2p")
    )


def cv_metrics(
    arm: Arm,
    dataset: pooled.Dataset,
    l2_values: tuple[float, ...],
    cv_seeds: tuple[int, ...],
    output_dir: Path,
    *,
    force: bool,
) -> list[dict[str, object]]:
    rows = []
    for l2 in l2_values:
        for seed in cv_seeds:
            prediction, _parameters = retained.oof_prediction(
                dataset,
                arm.config,
                l2,
                seed,
                output_dir,
                maxiter=FIT_MAXITER,
                top_k=FIT_TOP_K,
                force=force,
            )
            row = asdict(retained.metric_row(dataset, arm.config, l2, prediction, seed))
            row.update({"grouping": arm.grouping, "policy": arm.policy})
            rows.append(row)
    return rows


def summarize_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["grouping", "policy", "model", "l2", "nominal_parameter_count"],
            as_index=False,
        )
        .agg(
            cv_seeds=("seed", "nunique"),
            oof_rmse=("oof_rmse", "mean"),
            oof_rmse_sd=("oof_rmse", "std"),
            oof_spearman=("oof_spearman", "mean"),
            fold_mean_regret_at_1=("fold_mean_regret_at_1", "mean"),
            lower_tail_optimism=("lower_tail_optimism", "mean"),
            low_tail_rmse=("low_tail_rmse", "mean"),
        )
        .sort_values(["grouping", "policy", "oof_rmse", "lower_tail_optimism", "l2"])
        .reset_index(drop=True)
    )


def selected_l2(summary: pd.DataFrame, arm: Arm) -> float:
    rows = summary.loc[summary["grouping"].eq(arm.grouping) & summary["policy"].eq(arm.policy)]
    if rows.empty:
        raise ValueError(f"No CV rows for {arm.candidate}")
    return float(rows.iloc[0]["l2"])


def scalar_predictor(model: retained.FittedModel):
    def predict(weights: np.ndarray) -> float:
        return float(model.predict(np.asarray(weights, dtype=float)[None, :, :])[0])

    return predict


def phase_order_kl(weights: np.ndarray, aggregate: np.ndarray) -> float:
    alpha0, alpha1 = matched.PHASE_FRACTIONS
    return float(phase_information.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1))


def model_record(model: retained.FittedModel, l2: float) -> dict[str, object]:
    return {
        "config": asdict(model.config),
        "l2": l2,
        "shape": asdict(model.shape),
        "intercept": model.intercept,
        "signal_coef": model.signal_coef.tolist(),
        "replay_coef": model.replay_coef.tolist(),
        "family_members": [members.tolist() for members in model.family_members],
    }


def candidate_row(
    arm: Arm,
    model: retained.FittedModel,
    dataset: pooled.Dataset,
    l2: float,
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    aggregate: np.ndarray,
    optimizer_successful_starts: int,
) -> dict[str, object]:
    prediction = scalar_predictor(model)(weights)
    tied_prediction = scalar_predictor(model)(np.stack([aggregate, aggregate]))
    epochs = matched.olmix.simulated_epochs(weights, token_counts, target_budget=target_budget)
    actual_aggregate = matched.PHASE_FRACTIONS[0] * weights[0] + matched.PHASE_FRACTIONS[1] * weights[1]
    return {
        "candidate": arm.candidate,
        "grouping": arm.grouping,
        "policy": arm.policy,
        "selected_l2": l2,
        "nominal_parameter_count": retained.nominal_parameter_count(dataset, arm.config),
        "aggregate_kl_coefficient": AGGREGATE_KL,
        "phase_information_budget": 0.0 if arm.policy == "1p" else PHASE_INFORMATION_BUDGET,
        "predicted_bpb": prediction,
        "predicted_tied_bpb": tied_prediction,
        "predicted_phase_gain": prediction - tied_prediction,
        "aggregate_kl_to_proportional": matched.weighted_kl(np.stack([aggregate, aggregate]), natural),
        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
        "phase_information_kl": phase_order_kl(weights, aggregate),
        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "max_weight": float(weights.max()),
        "max_simulated_epoch": float(epochs.max()),
        "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
        "aggregate_match_max_abs_error": float(np.max(np.abs(actual_aggregate - aggregate))),
        "optimizer_successful_starts": optimizer_successful_starts,
    }


def write_candidate(
    output_dir: Path,
    gcs_output_dir: str,
    candidate: str,
    frame: pd.DataFrame,
    *,
    upload: bool,
) -> None:
    mixture_dir = output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(mixture_dir / f"{candidate}.csv", index=False)
    if upload:
        with fsspec.open(f"{gcs_output_dir.rstrip('/')}/{candidate}.csv", "wt") as handle:
            frame.to_csv(handle, index=False)


def plot_candidates(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure = make_subplots(
        rows=len(frames),
        cols=1,
        subplot_titles=list(frames),
        shared_xaxes=True,
        vertical_spacing=0.025,
    )
    colors = ("#238443", "#d73027")
    for row, (_candidate, frame) in enumerate(frames.items(), start=1):
        ordered = frame.sort_values("aggregate_weight")
        for phase, color in enumerate(colors):
            figure.add_bar(
                x=ordered[f"phase_{phase}_weight"],
                y=ordered["domain"],
                orientation="h",
                name=f"Phase {phase}",
                legendgroup=f"phase-{phase}",
                showlegend=row == 1,
                marker_color=color,
                customdata=np.column_stack([ordered["simulated_epochs"], ordered["aggregate_weight"]]),
                hovertemplate=(
                    "%{y}<br>weight=%{x:.5f}<br>total simulated epochs=%{customdata[0]:.3f}"
                    "<br>aggregate weight=%{customdata[1]:.5f}<extra></extra>"
                ),
                row=row,
                col=1,
            )
    figure.update_layout(
        title="Compact retained-state 2x2 validation candidates",
        template="plotly_white",
        barmode="group",
        height=2300,
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(cv_summary: pd.DataFrame, manifest: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Compact retained-state 3e18 ablation panel",
        "",
        "The one-phase model removes the retention state entirely:",
        "",
        "$$q_i=e_i^{(0)}+e_i^{(1)},\\qquad " "\\hat L=b-\\sum_i a_i(1-e^{-(\\rho q_i)^p})+c\\sum_i[q_i-1]_+^2.$$",
        "The two-phase model uses retained phase-0 state:",
        "",
        "$$z_i=e^{-\\lambda(1-w_i^{(1)})}e_i^{(0)}+\\eta e_i^{(1)},\\qquad "
        "\\hat L=b-\\sum_i a_i(1-e^{-(\\rho z_i)^p})+c\\sum_i[q_i-1]_+^2.$$",
        "The grouped variant adds three nonnegative family-coverage response channels. The no-group variant "
        "retains one amplitude per bucket but receives no prior bucket-to-family assignment. No data bucket is removed.",
        "",
        f"Each cell selects ridge from repeated five-fold CV. The tied aggregate uses KL coefficient "
        f"{AGGREGATE_KL:g}; the 2p candidate holds that aggregate exactly fixed and has phase-information "
        f"budget {PHASE_INFORMATION_BUDGET:g}.",
        "",
        "## CV summary",
        "",
        cv_summary.to_markdown(index=False),
        "",
        "## Candidates",
        "",
        manifest.to_markdown(index=False),
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "materialize_compact_retained_state_validation_panel_300m.py --upload",
        "```",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--l2-values", default=",".join(str(value) for value in DEFAULT_L2_VALUES))
    parser.add_argument("--cv-seeds", default=",".join(str(value) for value in CV_SEEDS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    two_phase = pooled.load_300m_dataset("uncheatable")
    one_phase = observatory.load_300m_single_phase_dataset("uncheatable", two_phase)
    if two_phase.n != 280 or one_phase.n != 280:
        raise ValueError(f"Expected 280 rows per policy class, got 1p={one_phase.n}, 2p={two_phase.n}")
    datasets = {"1p": one_phase, "2p": two_phase}
    arms = panel_arms()
    l2_values = parse_float_tuple(args.l2_values)
    cv_seeds = parse_int_tuple(args.cv_seeds)

    metric_rows = []
    for arm in arms:
        metric_rows.extend(
            cv_metrics(
                arm,
                datasets[arm.policy],
                l2_values,
                cv_seeds,
                args.output_dir,
                force=args.force,
            )
        )
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_cv(metrics)
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)

    _packet, domains, natural, token_counts, target_budget, _folds = objective_data.load_objective("uncheatable")
    if domains != two_phase.domain_names or domains != one_phase.domain_names:
        raise ValueError("Optimization and fit-panel domain orders differ")

    fitted_models: dict[tuple[str, str], retained.FittedModel] = {}
    selected_l2s: dict[tuple[str, str], float] = {}
    for arm in arms:
        l2 = selected_l2(summary, arm)
        selected_l2s[(arm.grouping, arm.policy)] = l2
        fitted_models[(arm.grouping, arm.policy)] = retained.fit_model(
            datasets[arm.policy],
            np.arange(datasets[arm.policy].n),
            arm.config,
            l2,
            maxiter=FIT_MAXITER,
            top_k=FIT_TOP_K,
        )

    manifest_rows = []
    mixture_frames: dict[str, pd.DataFrame] = {}
    model_records = {}
    for grouping in ("nogroup", "grouped"):
        one_arm = next(arm for arm in arms if arm.grouping == grouping and arm.policy == "1p")
        two_arm = next(arm for arm in arms if arm.grouping == grouping and arm.policy == "2p")
        one_model = fitted_models[(grouping, "1p")]
        two_model = fitted_models[(grouping, "2p")]
        one_result = matched.optimize(
            scalar_predictor(one_model),
            one_phase,
            natural,
            AGGREGATE_KL,
            "1p",
        )
        aggregate = np.asarray(one_result.weights[0], dtype=float)
        two_result = decoupled.generic_optimize_fixed_aggregate(
            scalar_predictor(two_model),
            aggregate,
            PHASE_INFORMATION_BUDGET,
        )
        candidates = (
            (one_arm, one_model, one_result.weights, one_result.successful_starts),
            (two_arm, two_model, two_result.weights, two_result.successful_starts),
        )
        for arm, model, weights, successful_starts in candidates:
            if np.min(weights) < -1e-8 or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8):
                raise ValueError(f"Invalid simplex weights for {arm.candidate}")
            row = candidate_row(
                arm,
                model,
                datasets[arm.policy],
                selected_l2s[(grouping, arm.policy)],
                weights,
                natural,
                token_counts,
                target_budget,
                aggregate,
                successful_starts,
            )
            mixture = per_component.mixture_frame(
                domains=domains,
                natural=natural,
                weights=weights,
                token_counts=token_counts,
                target_budget=target_budget,
            )
            write_candidate(
                args.output_dir,
                args.gcs_output_dir,
                arm.candidate,
                mixture,
                upload=args.upload,
            )
            manifest_rows.append(row)
            mixture_frames[arm.candidate] = mixture
            model_records[arm.candidate] = model_record(model, selected_l2s[(grouping, arm.policy)])
            print(
                f"{arm.candidate}: pred={row['predicted_bpb']:.6f}, "
                f"max_epoch={row['max_simulated_epoch']:.3f}, phase_KL={row['phase_information_kl']:.6f}",
                flush=True,
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["grouping", "policy"]).reset_index(drop=True)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    (args.output_dir / "fitted_models.json").write_text(json.dumps(model_records, indent=2, allow_nan=False) + "\n")
    plot_candidates(mixture_frames, args.output_dir / "candidate_mixtures.html")
    write_report(summary, manifest, args.output_dir)
    print(f"Wrote {len(manifest)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
