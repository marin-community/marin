# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Audit unregularized optima of the exact-budget marginal-acquisition model.

This script deliberately separates surrogate correctness from deployment
regularization. It refits the frozen 280-checkpoint protocols, then optimizes
four nested surfaces over either a tied policy or two independent phase
simplexes:

* the phase-invariant physical pooled-acquisition aggregate spine;
* aggregate plus the global odd order potential;
* aggregate plus the even family switching cost;
* aggregate plus both phase mechanisms.

No KL penalty, phase-information budget, epoch cap, or trust region is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.special import softmax

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_aggregate_comparators_20260724 as comparators,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_marginal_acquisition_joint_20260724 as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_pooled_acquisition_protocol_20260724 as strict_protocol,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_frontier_control_aggregate_identification_20260724 as aggregate_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_marginal_acquisition_phase_potential_20260724 as phase_potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_marginal_acquisition_raw_optima_20260724"
DEFAULT_SEEDS = (20260724, 20260725, 20260726)
DEFAULT_STARTS = 32
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}
PHASE_ARMS = tuple(arm for arm in strict_protocol.ARMS if arm.name in {"phase_probe_32", "phase_probe_112"})
VARIANTS = (
    "aggregate_only",
    "global_order",
    "family_switching",
    "global_order_plus_family_switching",
)
POLICIES = ("single_phase", "two_phase")


class Predictor(Protocol):
    def predict(self, weights: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class JointPredictor:
    """One nested phase-correction variant over a fixed aggregate fit."""

    aggregate: orthogonal.AggregateModel
    phase: joint.FittedPhaseCorrection
    variant: str

    def predict(self, weights: np.ndarray) -> np.ndarray:
        aggregate_prediction = self.aggregate.predict(weights)
        global_odd, _family_odd, switching = joint.phase_components(weights, self.aggregate, self.phase)
        if self.variant == "aggregate_only":
            return aggregate_prediction
        if self.variant == "global_order":
            return aggregate_prediction + global_odd
        if self.variant == "family_switching":
            return aggregate_prediction + switching
        if self.variant == "global_order_plus_family_switching":
            return aggregate_prediction + global_odd + switching
        raise ValueError(f"Unknown variant {self.variant!r}")


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    predicted_bpb: float
    converged: bool
    converged_starts: int
    best_start: int
    objective_spread_top5: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate and finalize an already-computed raw_optima.csv without rerunning optimization.",
    )
    return parser.parse_args()


def proportional_weights(c_total: np.ndarray) -> np.ndarray:
    weights = 1.0 / np.maximum(np.asarray(c_total, dtype=float), 1e-12)
    return weights / weights.sum()


def logits_to_weights(logits: np.ndarray, bucket_count: int, policy: str) -> np.ndarray:
    if policy == "single_phase":
        tied = softmax(np.asarray(logits, dtype=float))
        return np.stack([tied, tied], axis=0)
    if policy == "two_phase":
        return softmax(np.asarray(logits, dtype=float).reshape(2, bucket_count), axis=1)
    raise ValueError(f"Unknown policy {policy!r}")


def weights_to_logits(weights: np.ndarray, policy: str) -> np.ndarray:
    values = 0.5 * (weights[0] + weights[1]) if policy == "single_phase" else weights
    logits = np.log(np.maximum(values, 1e-12))
    return np.asarray(logits - np.mean(logits, axis=-1, keepdims=True), dtype=float).ravel()


def optimization_starts(
    training: pooled.Dataset,
    policy: str,
    seed: int,
    count: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    tied_proportional = proportional_weights(training.c0 + training.c1)
    starts = [weights_to_logits(np.stack([tied_proportional, tied_proportional]), policy)]
    best_observed = training.weights[int(np.argmin(training.y))]
    starts.append(weights_to_logits(best_observed, policy))
    while len(starts) < count:
        concentration = (0.15, 0.5, 1.0, 4.0, 16.0)[(len(starts) - 2) % 5]
        if policy == "single_phase":
            sample = rng.dirichlet(np.full(training.m, concentration))
            weights = np.stack([sample, sample], axis=0)
        else:
            weights = np.stack(
                [
                    rng.dirichlet(np.full(training.m, concentration)),
                    rng.dirichlet(np.full(training.m, concentration)),
                ],
                axis=0,
            )
        starts.append(weights_to_logits(weights, policy))
    return starts


def optimize(
    predictor: Predictor,
    training: pooled.Dataset,
    policy: str,
    seed: int,
    count: int,
) -> OptimizationResult:
    starts = optimization_starts(training, policy, seed, count)
    results = []

    def objective(logits: np.ndarray) -> float:
        weights = logits_to_weights(logits, training.m, policy)
        prediction = predictor.predict(weights[None, :, :])
        return float(prediction[0])

    for start_index, start in enumerate(starts):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 1200, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50},
        )
        if np.isfinite(result.fun):
            results.append((float(result.fun), start_index, np.asarray(result.x, dtype=float), bool(result.success)))
    if not results:
        raise RuntimeError("No finite optimization result")
    results.sort(key=lambda item: item[0])
    best = results[0]
    top = results[: min(5, len(results))]
    return OptimizationResult(
        weights=logits_to_weights(best[2], training.m, policy),
        predicted_bpb=best[0],
        converged=best[3],
        converged_starts=sum(item[3] for item in results),
        best_start=best[1],
        objective_spread_top5=float(top[-1][0] - top[0][0]),
    )


def support_distance(training: pooled.Dataset, weights: np.ndarray) -> float:
    fit = training.weights.reshape(training.n, -1)
    scale = np.maximum(np.std(fit, axis=0), 1e-3)
    distances = np.linalg.norm((fit - weights.reshape(1, -1)) / scale, axis=1)
    return float(np.min(distances))


def optimum_row(
    target: str,
    arm: strict_protocol.BudgetArm,
    seed: int,
    variant: str,
    policy: str,
    predictor: JointPredictor,
    training: pooled.Dataset,
    result: OptimizationResult,
) -> dict[str, object]:
    weights = result.weights
    aggregate = orthogonal.aggregate_weights(weights[None, :, :], predictor.aggregate.phase_fraction)[0]
    exposure = weights[0] * training.c0 + weights[1] * training.c1
    proportional = proportional_weights(training.c0 + training.c1)
    proportional_policy = np.stack([proportional, proportional], axis=0)
    return {
        "target": target,
        "arm": arm.name,
        "seed": seed,
        "variant": variant,
        "policy": policy,
        **asdict(arm),
        "predicted_bpb": result.predicted_bpb,
        "predicted_gain_vs_proportional": float(
            predictor.predict(proportional_policy[None, :, :])[0] - result.predicted_bpb
        ),
        "optimizer_converged": result.converged,
        "converged_starts": result.converged_starts,
        "best_start": result.best_start,
        "objective_spread_top5": result.objective_spread_top5,
        "max_bucket_weight": float(np.max(weights)),
        "max_simulated_epochs": float(np.max(exposure)),
        "mean_simulated_epochs": float(np.mean(exposure)),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "aggregate_hhi": float(np.sum(np.square(aggregate))),
        "fit_support_distance": support_distance(training, weights),
        "phase_0_weights_json": json.dumps(
            dict(zip(training.domain_names, weights[0].tolist(), strict=True)),
            separators=(",", ":"),
        ),
        "phase_1_weights_json": json.dumps(
            dict(zip(training.domain_names, weights[1].tolist(), strict=True)),
            separators=(",", ":"),
        ),
    }


def pairwise_policy_tv(group: pd.DataFrame) -> tuple[float, float]:
    policies = []
    for row in group.itertuples(index=False):
        phase_0 = np.asarray(list(json.loads(row.phase_0_weights_json).values()), dtype=float)
        phase_1 = np.asarray(list(json.loads(row.phase_1_weights_json).values()), dtype=float)
        policies.append(np.stack([phase_0, phase_1], axis=0))
    distances = [
        float(0.25 * np.abs(policies[left] - policies[right]).sum())
        for left in range(len(policies))
        for right in range(left + 1, len(policies))
    ]
    if not distances:
        return 0.0, 0.0
    return float(np.mean(distances)), float(np.max(distances))


def stability_rows(optima: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "arm", "variant", "policy"]
    for key, group in optima.groupby(keys, sort=True):
        mean_tv, max_tv = pairwise_policy_tv(group)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "replicates": len(group),
                "predicted_bpb_sd": float(group["predicted_bpb"].std(ddof=0)),
                "phase_tv_mean": float(group["phase_total_variation"].mean()),
                "phase_tv_sd": float(group["phase_total_variation"].std(ddof=0)),
                "max_weight_mean": float(group["max_bucket_weight"].mean()),
                "max_epochs_mean": float(group["max_simulated_epochs"].mean()),
                "support_distance_mean": float(group["fit_support_distance"].mean()),
                "pairwise_policy_tv_mean": mean_tv,
                "pairwise_policy_tv_max": max_tv,
            }
        )
    return pd.DataFrame(rows)


def weights_frame(optima: pd.DataFrame, domains: list[str], c0: np.ndarray, c1: np.ndarray) -> pd.DataFrame:
    rows = []
    phase_fraction = float(np.mean(c0 / (c0 + c1)))
    for optimum in optima.itertuples(index=False):
        phase_0 = json.loads(optimum.phase_0_weights_json)
        phase_1 = json.loads(optimum.phase_1_weights_json)
        for index, domain in enumerate(domains):
            p0 = float(phase_0[domain])
            p1 = float(phase_1[domain])
            rows.append(
                {
                    "target": optimum.target,
                    "arm": optimum.arm,
                    "seed": optimum.seed,
                    "variant": optimum.variant,
                    "policy": optimum.policy,
                    "domain": domain,
                    "phase_0_weight": p0,
                    "phase_1_weight": p1,
                    "aggregate_weight": phase_fraction * p0 + (1.0 - phase_fraction) * p1,
                    "aggregate_exposure": p0 * c0[index] + p1 * c1[index],
                }
            )
    return pd.DataFrame(rows)


def plot_optima(optima: pd.DataFrame, weights: pd.DataFrame, output_dir: Path) -> None:
    summary = optima.groupby(["target", "arm", "variant", "policy"], as_index=False).agg(
        predicted_bpb=("predicted_bpb", "mean"),
        phase_total_variation=("phase_total_variation", "mean"),
        max_simulated_epochs=("max_simulated_epochs", "mean"),
        fit_support_distance=("fit_support_distance", "mean"),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Predicted raw optimum",
            "Phase divergence",
            "Maximum exposure",
            "Distance from fit support",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    metrics = (
        ("predicted_bpb", "predicted BPB"),
        ("phase_total_variation", "phase TV"),
        ("max_simulated_epochs", "max simulated epochs"),
        ("fit_support_distance", "standardized nearest-fit distance"),
    )
    colors = {
        "aggregate_only": "#2c7bb6",
        "global_order": "#abd9e9",
        "family_switching": "#fdae61",
        "global_order_plus_family_switching": "#d7191c",
    }
    for panel, (metric, axis_title) in enumerate(metrics):
        row = panel // 2 + 1
        column = panel % 2 + 1
        for variant in VARIANTS:
            local = summary[summary["variant"].eq(variant)].copy()
            local["x"] = (
                local["target"]
                + " / "
                + local["arm"].str.replace("phase_probe_", "phase ")
                + " / "
                + local["policy"].str.replace("_", " ")
            )
            figure.add_trace(
                go.Scatter(
                    x=local["x"],
                    y=local[metric],
                    mode="lines+markers",
                    name=variant.replace("_", " "),
                    legendgroup=variant,
                    showlegend=panel == 0,
                    marker={"size": 9, "color": colors[variant]},
                    line={"color": colors[variant]},
                    customdata=np.column_stack(
                        [
                            local["target"],
                            local["arm"],
                            local["policy"],
                            local["phase_total_variation"],
                            local["max_simulated_epochs"],
                            local["fit_support_distance"],
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]} / %{customdata[1]}</b><br>"
                        "policy=%{customdata[2]}<br>"
                        f"{axis_title}=%{{y:.6f}}<br>"
                        "phase TV=%{customdata[3]:.4f}<br>"
                        "max epochs=%{customdata[4]:.3f}<br>"
                        "support distance=%{customdata[5]:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_yaxes(title_text=axis_title, row=row, col=column)
    figure.update_layout(
        title={
            "text": (
                "Unregularized exact-budget optima"
                "<br><span style='font-size:14px;color:#64748b'>"
                "No KL, phase-information budget, epoch cap, or trust region.</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        width=1650,
        height=1050,
        margin={"l": 110, "r": 70, "t": 130, "b": 180},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.15},
    )
    figure.write_html(
        output_dir / "raw_optimum_diagnostics.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    selected = weights[
        weights["arm"].eq("phase_probe_112") & weights["policy"].eq("two_phase") & weights["seed"].eq(DEFAULT_SEEDS[0])
    ].copy()
    selected["label"] = selected["target"] + " / " + selected["variant"].str.replace("_", " ")
    labels = selected["label"].drop_duplicates().tolist()
    anatomy = make_subplots(
        rows=len(labels),
        cols=3,
        subplot_titles=["Phase 0 weight", "Phase 1 weight", "Aggregate exposure"] + [""] * (3 * (len(labels) - 1)),
        row_titles=labels,
        shared_yaxes="rows",
        horizontal_spacing=0.045,
        vertical_spacing=0.025,
    )
    for row_index, label in enumerate(labels, start=1):
        local = selected[selected["label"].eq(label)].sort_values("aggregate_exposure")
        for column, (value, title, color) in enumerate(
            (
                ("phase_0_weight", "phase 0 weight", "#e76f51"),
                ("phase_1_weight", "phase 1 weight", "#2a9d8f"),
                ("aggregate_exposure", "simulated epochs", "#457b9d"),
            ),
            start=1,
        ):
            anatomy.add_trace(
                go.Bar(
                    x=local[value],
                    y=local["domain"],
                    orientation="h",
                    marker_color=color,
                    showlegend=False,
                    hovertemplate=f"<b>%{{y}}</b><br>{title}=%{{x:.6f}}<extra></extra>",
                ),
                row=row_index,
                col=column,
            )
            anatomy.update_xaxes(title_text=title, row=row_index, col=column)
    anatomy.update_layout(
        title="Phase-112 raw two-phase optima, first subset seed",
        template="plotly_white",
        width=2100,
        height=max(1000, 520 * len(labels)),
        margin={"l": 250, "r": 240, "t": 120, "b": 80},
    )
    anatomy.write_html(
        output_dir / "raw_optimum_mixture_anatomy.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_report(output_dir: Path, optima: pd.DataFrame, stability: pd.DataFrame) -> None:
    means = (
        optima.groupby(["target", "arm", "variant", "policy"], as_index=False)
        .agg(
            predicted_bpb=("predicted_bpb", "mean"),
            predicted_bpb_sd=("predicted_bpb", lambda values: float(np.std(values))),
            max_bucket_weight=("max_bucket_weight", "mean"),
            max_simulated_epochs=("max_simulated_epochs", "mean"),
            phase_total_variation=("phase_total_variation", "mean"),
            fit_support_distance=("fit_support_distance", "mean"),
            objective_spread_top5=("objective_spread_top5", "mean"),
        )
        .sort_values(["target", "arm", "policy", "predicted_bpb"])
    )
    lines = [
        "# Raw optimum audit for the exact-budget marginal-acquisition model",
        "",
        "## Boundary",
        "",
        (
            "Every row is an unregularized continuous optimum. No deployment KL, phase-information "
            "budget, epoch cap, or trust region is present. The same frozen aggregate and phase fits used "
            "by the exact-280 audit are refit independently for each subset seed."
        ),
        "",
        "## Mean diagnostics across subset seeds",
        "",
        means.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Policy stability",
        "",
        stability.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation guardrail",
        "",
        (
            "The even family-switching term can make a raw optimum look conservative. That is evidence for "
            "the model form only if its coefficient is identified by controlled phase data and transfers "
            "across targets; conservatism by itself is not a mechanistic validation."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) < 2:
        raise ValueError("At least two subset seeds are required for a stability audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_single = comparators.target_data(orthogonal.TARGETS[0])[3]
    domains = list(first_single.domain_names)
    c0 = np.asarray(first_single.c0, dtype=float)
    c1 = np.asarray(first_single.c1, dtype=float)
    raw_optima_path = args.output_dir / "raw_optima.csv"
    if args.finalize_existing:
        if not raw_optima_path.exists():
            raise FileNotFoundError(raw_optima_path)
        optima = pd.read_csv(raw_optima_path)
        key_columns = ["target", "arm", "seed", "variant", "policy"]
        expected_keys = {
            (target, arm.name, seed, variant, policy)
            for target in orthogonal.TARGETS
            for arm in PHASE_ARMS
            for seed in seeds
            for variant in VARIANTS
            for policy in POLICIES
        }
        actual_keys = set(optima[key_columns].itertuples(index=False, name=None))
        if actual_keys != expected_keys or len(optima) != len(expected_keys):
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Existing optima do not match the requested audit: "
                f"rows={len(optima)}, expected={len(expected_keys)}, "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    else:
        optimum_rows = []
        for target in orthogonal.TARGETS:
            (
                _reference,
                _heldout_frame,
                _heldout_weights,
                single,
                controls,
                _evaluation_frame,
                _evaluation_weights,
                _observed,
                _clusters,
            ) = comparators.target_data(target)
            if list(single.domain_names) != domains:
                raise ValueError("Target datasets do not share an ordered bucket schema")
            pair_dataset = phase_potential.pair_datasets()[target]
            for arm in PHASE_ARMS:
                for seed in seeds:
                    training = strict_protocol.aggregate_training_dataset(target, single, controls, arm, seed)
                    fold = strict_protocol.grouped_stratified_folds(training, seed)
                    aggregate_fit = aggregate_audit.frozen_pooled_fit(training, fold)
                    phase_fit = joint.fit_phase_correction(
                        pair_dataset,
                        aggregate_fit.model,
                        arm.treatment_count,
                        seed,
                    )
                    if phase_fit is None:
                        raise AssertionError("A phase-probe arm must fit a phase correction")
                    for variant in VARIANTS:
                        predictor = JointPredictor(aggregate_fit.model, phase_fit, variant)
                        for policy in POLICIES:
                            result = optimize(
                                predictor,
                                training,
                                policy,
                                seed + 10_000 * PHASE_ARMS.index(arm) + 100_000 * VARIANTS.index(variant),
                                args.starts,
                            )
                            optimum_rows.append(
                                optimum_row(
                                    target,
                                    arm,
                                    seed,
                                    variant,
                                    policy,
                                    predictor,
                                    training,
                                    result,
                                )
                            )
        optima = pd.DataFrame(optimum_rows)
    optima["optimizer_starts"] = args.starts
    stability = stability_rows(optima)
    weights = weights_frame(optima, domains, c0, c1)
    optima.to_csv(raw_optima_path, index=False)
    stability.to_csv(args.output_dir / "raw_optimum_stability.csv", index=False)
    weights.to_csv(args.output_dir / "raw_optimum_weights.csv", index=False)
    plot_optima(optima, weights, args.output_dir)
    write_report(args.output_dir, optima, stability)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "total_checkpoint_budget": strict_protocol.TOTAL_BUDGET,
        "arms": [asdict(arm) for arm in PHASE_ARMS],
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "policies": list(POLICIES),
        "optimizer_starts": args.starts,
        "deployment_regularization": None,
        "sealed_targeted_pairwise_panel_accessed": False,
        "script_sha256": script_hash,
        "git": {
            "commit": (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
