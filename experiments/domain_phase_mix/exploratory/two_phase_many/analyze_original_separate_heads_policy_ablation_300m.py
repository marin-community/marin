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
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Cross-validate and ablate the original 280-row separate-heads surrogate.

This preserves the exact original training panel and functional form used to
materialize ``seplf_*_sep_*``. It answers two questions without changing the
data weighting:

1. Was the manually pinned ridge penalty ``L2=0.1`` supported by grouped CV?
2. What tied one-phase optimum does the same fitted two-phase surrogate imply
   when deployment is constrained to ``w0 == w1``?

The one-phase arm is therefore a policy-class/optimizer ablation, not a second
surrogate fitted to a separately collected one-phase swarm.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_sep_lf_kl_sweep_panel_300m as original_panel,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_lf_sepheads_kl_sweep_300m as original_fit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_two_phase_canonical_bowl_kl_sweep_300m import (  # noqa: E402
    optimize_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "original_separate_heads_policy_ablation_20260712"
ORIGINAL_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "sep_lf_kl_sweep_panel_20260706"
OBJECTIVES = ("uncheatable", "table9")
FIXED_L2 = 0.1
DEFAULT_L2_VALUES = (
    0.0,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
)
DEFAULT_KL_VALUES = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4)
CV_SEEDS = (0, 1, 2)
N_SPLITS = 5
LOWER_TAIL_FRAC = 0.15


@dataclass(frozen=True)
class SeparateHeadsModel:
    l2: float
    mu0: np.ndarray
    mu1: np.ndarray
    intercept: float
    coefficients: np.ndarray


@dataclass(frozen=True)
class CvMetric:
    objective: str
    l2: float
    seed: int
    oof_rmse: float
    oof_spearman: float
    fold_mean_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def separate_design(packet, weights: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> np.ndarray:
    zero_c0 = np.zeros_like(packet.c0)
    phase0 = bowl.abowl_design(weights, packet.c0, packet.c1, mu0, 0.0)
    phase1 = bowl.abowl_design(weights, zero_c0, packet.c1, mu1, 1.0)
    return np.hstack([phase0, phase1])


def fit_separate_heads(packet, indices: np.ndarray, l2: float) -> SeparateHeadsModel:
    weights = packet.w[indices]
    target = packet.y[indices]
    zero_c0 = np.zeros_like(packet.c0)
    mu0 = original_fit._gridmu(weights, packet.c0, 0.0, packet.c1, target, l2)
    mu1 = original_fit._gridmu(weights, zero_c0, 1.0, packet.c1, target, l2)
    design = separate_design(packet, weights, mu0, mu1)
    intercept, coefficients = bowl.fit_head(design, target, l2)
    return SeparateHeadsModel(
        l2=l2,
        mu0=mu0,
        mu1=mu1,
        intercept=intercept,
        coefficients=coefficients,
    )


def predict_separate_heads(model: SeparateHeadsModel, packet, weights: np.ndarray) -> np.ndarray:
    design = separate_design(packet, np.asarray(weights, dtype=float), model.mu0, model.mu1)
    return model.intercept + design @ model.coefficients


def cv_metrics(
    objective: str,
    target: np.ndarray,
    prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    l2: float,
    seed: int,
) -> CvMetric:
    residual = prediction - target
    fold_regrets = []
    for _train_indices, test_indices in folds:
        selected = test_indices[int(np.argmin(prediction[test_indices]))]
        fold_regrets.append(float(target[selected] - np.min(target[test_indices])))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(target))))
    tail = np.argsort(prediction)[:tail_count]
    tail_residual = residual[tail]
    return CvMetric(
        objective=objective,
        l2=l2,
        seed=seed,
        oof_rmse=float(np.sqrt(np.mean(residual**2))),
        oof_spearman=float(spearmanr(target, prediction).statistic),
        fold_mean_regret_at_1=float(np.mean(fold_regrets)),
        lower_tail_optimism=float(np.mean(np.maximum(-tail_residual, 0.0))),
        low_tail_rmse=float(np.sqrt(np.mean(tail_residual**2))),
    )


def cross_validate(objective: str, packet, l2_values: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for l2 in l2_values:
        for seed in CV_SEEDS:
            folds = component_dsp.panel_stratified_folds(packet.frame, n_splits=N_SPLITS, seed=seed)
            prediction = np.zeros_like(packet.y, dtype=float)
            for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
                print(
                    f"{objective}: L2={l2:g}, seed={seed}, fold={fold_index}/{N_SPLITS}",
                    flush=True,
                )
                model = fit_separate_heads(packet, train_indices, l2)
                prediction[test_indices] = predict_separate_heads(model, packet, packet.w[test_indices])
            rows.append(asdict(cv_metrics(objective, packet.y, prediction, folds, l2, seed)))
    return pd.DataFrame(rows)


def summarize_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["objective", "l2"], as_index=False)
        .agg(
            oof_rmse_mean=("oof_rmse", "mean"),
            oof_rmse_sd=("oof_rmse", "std"),
            oof_spearman_mean=("oof_spearman", "mean"),
            fold_mean_regret_at_1_mean=("fold_mean_regret_at_1", "mean"),
            lower_tail_optimism_mean=("lower_tail_optimism", "mean"),
            low_tail_rmse_mean=("low_tail_rmse", "mean"),
        )
        .sort_values(["objective", "oof_rmse_mean", "fold_mean_regret_at_1_mean", "l2"])
        .reset_index(drop=True)
    )


def selected_l2(summary: pd.DataFrame, objective: str) -> float:
    row = summary.loc[summary["objective"].eq(objective)].iloc[0]
    return float(row["l2"])


def predictor(model: SeparateHeadsModel, packet):
    def predict(weights: np.ndarray) -> float:
        return float(predict_separate_heads(model, packet, np.asarray(weights)[None, :, :])[0])

    return predict


def original_candidate_path(objective: str) -> Path:
    abbr = "unch" if objective == "uncheatable" else "t9"
    return ORIGINAL_PANEL_DIR / f"seplf_{abbr}_sep_kl0p1" / "proposed_mixture_weights.csv"


def weights_from_frame(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    indexed = frame.set_index("domain").loc[domains]
    return np.stack(
        [
            indexed["phase_0_weight"].to_numpy(dtype=float),
            indexed["phase_1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )


def optimize_paths(
    objective: str,
    packet,
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    cv_l2: float,
    kl_values: tuple[float, ...],
    output_dir: Path,
    maxiter: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | int | str | bool]] = []
    all_indices = np.arange(len(packet.y), dtype=int)
    starts = bowl.opt_starts(packet, packet.m, k=4)
    fits = [("fixed_0p1", FIXED_L2), ("cv_selected", cv_l2)]
    fit_metadata: dict[str, float] = {"cv_selected_l2": cv_l2}
    for fit_label, l2 in fits:
        model = fit_separate_heads(packet, all_indices, l2)
        predict_fn = predictor(model, packet)
        for kl in kl_values:
            for policy in ("2p", "1p"):
                if policy == "2p":
                    weights = optimize_fast(predict_fn, packet.m, natural, kl, starts, maxiter)
                else:
                    weights = original_panel.one_phase_argmin(
                        predict_fn,
                        packet.m,
                        natural,
                        kl,
                        starts,
                        maxiter,
                    )
                predicted = predict_fn(weights)
                regularized = predicted
                if kl > 0:
                    regularized += kl * float(base.weighted_multiclass_kl(weights, natural, base.PHASE_FRACTIONS))
                frame = per_component.mixture_frame(
                    domains=domains,
                    natural=natural,
                    weights=weights,
                    token_counts=token_counts,
                    target_budget=target_budget,
                )
                candidate = f"original_sep_{objective}_{fit_label}_{policy}_kl{kl:g}".replace(".", "p")
                frame.to_csv(output_dir / f"{candidate}.csv", index=False)
                aggregate = base.aggregate_phase_weights(weights)
                epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
                rows.append(
                    {
                        "objective": objective,
                        "fit": fit_label,
                        "l2": l2,
                        "policy": policy,
                        "kl_reg": kl,
                        "candidate": candidate,
                        "predicted_bpb": predicted,
                        "regularized_objective": regularized,
                        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
                        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
                        "max_weight": float(np.max(weights)),
                        "max_simulated_epoch": float(np.max(epochs)),
                    }
                )
                print(
                    f"{objective}: {fit_label}/{policy}/KL={kl:g}: "
                    f"pred={predicted:.6f}, max_epoch={float(np.max(epochs)):.3f}",
                    flush=True,
                )
        if fit_label == "fixed_0p1":
            reproduced = next(
                row for row in rows if row["fit"] == fit_label and row["policy"] == "2p" and row["kl_reg"] == 0.1
            )
            generated = pd.read_csv(output_dir / str(reproduced["candidate"] + ".csv"))
            original = pd.read_csv(original_candidate_path(objective))
            generated_weights = weights_from_frame(generated, domains)
            original_weights = weights_from_frame(original, domains)
            fit_metadata["fixed_0p1_reproduction_tv"] = float(
                0.5 * np.abs(generated_weights - original_weights).sum(axis=1).mean()
            )
    return pd.DataFrame(rows), fit_metadata


def plot_cv(summary: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9"])
    for column, objective in enumerate(OBJECTIVES, start=1):
        subset = summary.loc[summary["objective"].eq(objective)].sort_values("l2")
        plotted = subset.loc[subset["l2"].gt(0)]
        figure.add_trace(
            go.Scatter(
                x=plotted["l2"],
                y=plotted["oof_rmse_mean"],
                error_y={"type": "data", "array": plotted["oof_rmse_sd"], "visible": True},
                mode="lines+markers",
                name=objective,
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_vline(x=FIXED_L2, line_dash="dash", line_color="#D95F02", row=1, col=column)
        figure.update_xaxes(type="log", title_text="ridge L2", row=1, col=column)
        figure.update_yaxes(title_text="grouped OOF RMSE", row=1, col=column)
    figure.update_layout(
        title="Original 280-row two-phase separate-heads: ridge CV",
        template="plotly_white",
        width=1200,
        height=480,
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    figure.write_html(
        output,
        include_plotlyjs=True,
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def plot_kl_paths(paths: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Uncheatable predicted BPB",
            "Table-9 predicted BPB",
            "Uncheatable maximum epochs",
            "Table-9 maximum epochs",
        ],
    )
    colors = {
        ("fixed_0p1", "2p"): "#1B9E77",
        ("fixed_0p1", "1p"): "#66A61E",
        ("cv_selected", "2p"): "#D95F02",
        ("cv_selected", "1p"): "#E6AB02",
    }
    for column, objective in enumerate(OBJECTIVES, start=1):
        subset = paths.loc[paths["objective"].eq(objective)]
        for (fit_label, policy), group in subset.groupby(["fit", "policy"], sort=True):
            group = group.sort_values("kl_reg")
            name = f"{fit_label}, {policy}"
            style = {"color": colors[(fit_label, policy)], "dash": "solid" if policy == "2p" else "dash"}
            figure.add_trace(
                go.Scatter(
                    x=group["kl_reg"],
                    y=group["predicted_bpb"],
                    mode="lines+markers",
                    name=name,
                    line=style,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=group["kl_reg"],
                    y=group["max_simulated_epoch"],
                    mode="lines+markers",
                    name=name,
                    line=style,
                    showlegend=False,
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(title_text="deployment KL", row=2, col=1)
    figure.update_xaxes(title_text="deployment KL", row=2, col=2)
    figure.update_yaxes(title_text="predicted BPB", row=1, col=1)
    figure.update_yaxes(title_text="predicted BPB", row=1, col=2)
    figure.update_yaxes(title_text="max simulated epochs", row=2, col=1)
    figure.update_yaxes(title_text="max simulated epochs", row=2, col=2)
    figure.update_layout(
        title="Original separate-heads: fixed versus CV ridge and tied-policy ablation",
        template="plotly_white",
        width=1300,
        height=850,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.09},
        margin={"l": 70, "r": 30, "t": 90, "b": 110},
    )
    figure.write_html(
        output,
        include_plotlyjs=True,
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def write_report(
    cv_summary: pd.DataFrame,
    paths: pd.DataFrame,
    metadata: dict[str, dict[str, float]],
    output: Path,
) -> None:
    lines = [
        "# Original separate-heads ridge CV and policy ablation",
        "",
        (
            "The fit panel is unchanged from the original separate-heads experiment: "
            "241 qsplit rows plus 39 domain deletions, with the 11 proportional "
            "observations collapsed to one mean target."
        ),
        "",
        "## CV-selected ridge",
        "",
        cv_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selected fits",
        "",
    ]
    for objective in OBJECTIVES:
        selected = metadata[objective]["cv_selected_l2"]
        reproduction_tv = metadata[objective]["fixed_0p1_reproduction_tv"]
        lines.append(
            f"- **{objective}:** CV-selected L2={selected:g}; fixed-L2 reproduction "
            f"TV to the original KL=0.1 candidate={reproduction_tv:.3e}."
        )
    lines.extend(["", "## KL paths", "", paths.to_markdown(index=False, floatfmt=".6f"), ""])
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l2-values", default=",".join(str(value) for value in DEFAULT_L2_VALUES))
    parser.add_argument("--kl-values", default=",".join(str(value) for value in DEFAULT_KL_VALUES))
    parser.add_argument("--maxiter", type=int, default=250)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    l2_values = parse_float_tuple(args.l2_values)
    kl_values = parse_float_tuple(args.kl_values)
    all_cv = []
    loaded: dict[str, tuple] = {}
    for objective in OBJECTIVES:
        packet, domains, natural, token_counts, target_budget, _folds = bowl.load_objective(objective)
        if len(packet.y) != 280:
            raise ValueError(f"Expected original 280-row panel for {objective}, found {len(packet.y)}")
        loaded[objective] = (packet, domains, natural, token_counts, target_budget)
        all_cv.append(cross_validate(objective, packet, l2_values))
    cv_frame = pd.concat(all_cv, ignore_index=True)
    cv_summary = summarize_cv(cv_frame)
    cv_frame.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    cv_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)

    all_paths = []
    metadata: dict[str, dict[str, float]] = {}
    for objective in OBJECTIVES:
        packet, domains, natural, token_counts, target_budget = loaded[objective]
        best_l2 = selected_l2(cv_summary, objective)
        path, objective_metadata = optimize_paths(
            objective,
            packet,
            domains,
            natural,
            token_counts,
            target_budget,
            best_l2,
            kl_values,
            args.output_dir,
            args.maxiter,
        )
        all_paths.append(path)
        metadata[objective] = objective_metadata
    paths = pd.concat(all_paths, ignore_index=True)
    paths.to_csv(args.output_dir / "kl_policy_paths.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "fit_panel_rows": 280,
                "fixed_l2": FIXED_L2,
                "selected": metadata,
                "cv_seeds": CV_SEEDS,
                "n_splits": N_SPLITS,
            },
            indent=2,
        )
        + "\n"
    )
    plot_cv(cv_summary, args.output_dir / "cv_l2_sweep.html")
    plot_kl_paths(paths, args.output_dir / "kl_policy_paths.html")
    write_report(cv_summary, paths, metadata, args.output_dir / "report.md")

    print("\n=== CV summary ===")
    print(cv_summary.to_string(index=False))
    print("\n=== KL/policy paths ===")
    print(paths.to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
