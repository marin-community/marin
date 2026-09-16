# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate", "matplotlib"]
# ///
"""Export uncertainty contrasts, noise diagnostics, and shortlist curves offline."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import delphi_selection_models_20260906 as alternatives
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_selection_20260906 as scoring


def panel_contrasts(metrics: pd.DataFrame, output: Path) -> pd.DataFrame:
    folds = metrics[metrics.population.eq("panel_oof") & metrics.fold.ge(0)]
    contrasts = []
    comparisons = {(spec.name, spec.parent) for spec in alternatives.SPECS}
    comparisons |= {(method, benchmark.BASELINES[0]) for method in scoring.METHODS if method != benchmark.BASELINES[0]}
    for target in benchmark.TARGETS:
        for candidate, reference in sorted(comparisons):
            merged = folds[folds.target.eq(target) & folds.method.eq(candidate)].merge(
                folds[folds.target.eq(target) & folds.method.eq(reference)],
                on=["repeat", "fold"],
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            for metric in ("rmse", "regret_at_1", "best_of_5_regret", "best_of_10_regret", "optimism", "spearman"):
                delta = (merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]).to_numpy(float)
                ratio = np.mean(
                    [
                        len(benchmark.partition(output, int(f), int(r)).test)
                        / len(benchmark.partition(output, int(f), int(r)).train)
                        for r, f in zip(merged.repeat, merged.fold, strict=True)
                    ]
                )
                width = float(stats.t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) * np.sqrt(1 / len(delta) + ratio))
                contrasts.append(
                    {
                        "target": target,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "folds": len(delta),
                        "mean_delta": float(delta.mean()),
                        "ci_low": float(delta.mean() - width),
                        "ci_high": float(delta.mean() + width),
                        "correction": "Nadeau-Bengio using realized mean test/train ratio; screening only",
                    }
                )
    return pd.DataFrame(contrasts)


def noise_diagnostics(output: Path) -> dict:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    repeat_path = benchmark.REFERENCE / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
    noise = pd.read_csv(repeat_path, index_col=0)
    covariance = noise.cov().to_numpy()
    result: dict[str, object] = {
        "repeat_source": str(repeat_path.relative_to(benchmark.REPO_ROOT)),
        "repeat_sha256": benchmark.sha256(repeat_path),
        "repeats": len(noise),
        "macro_repeat_sd": float(noise.mean(axis=1).std()),
        "independent_component_macro_sd": float(np.sqrt(np.trace(covariance)) / len(covariance)),
        "off_diagonal_fraction_of_macro_variance": float(1 - np.trace(covariance) / covariance.sum()),
    }
    predictions = pd.read_csv(output / "predictions.csv")
    for target in benchmark.TARGETS:
        macro = predictions[predictions.method.eq("ridge_log_epoch_macro") & predictions.target.eq(target)]
        shared = predictions[predictions.method.eq("ridge_log_epoch_shared") & predictions.target.eq(target)]
        joined = macro.merge(
            shared, on=["target", "population", "repeat", "fold", "row_id"], suffixes=("_macro", "_shared")
        )
        result[f"{target}_linear_pooling_max_difference"] = float(
            np.abs(joined.prediction_macro - joined.prediction_shared).max()
        )
    components = data["table9_components"]
    error = np.zeros((280, len(components)))
    for component in range(len(components)):
        for fold in range(5):
            shard = benchmark.read_npz(
                output / "baseline_shards" / benchmark.BASELINES[0] / "table9" / f"r0_f{fold}_c{component}.npz"
            )
            error[shard["test"], component] = shard["prediction"] - data["table9_outcomes"][shard["test"], component]
    eigenvalues = np.linalg.eigvalsh(np.cov(error.T))
    result["table9_wspu_error_first_pc_variance_fraction"] = float(eigenvalues[-1] / eigenvalues.sum())
    result["table9_wspu_error_effective_covariance_rank"] = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    noise_sd = noise.std()
    lookup = {name.split("/")[-2]: value for name, value in noise_sd.items()}
    repeat_sd = np.array([lookup[name.split("/")[-2] if "/" in name else name] for name in components])
    result["table9_component_rmse_vs_repeat_sd_spearman"] = float(
        stats.spearmanr(np.sqrt(np.square(error).mean(axis=0)), repeat_sd).statistic
    )
    pilot = benchmark.REFERENCE / "delphi_apriori_swarm_280_20260904"
    evidence = [
        pilot / "pilot_gate" / "gate_manifest.json",
        pilot / "pilot_gate" / "gate_anchor_summary.csv",
        pilot / "predictive_value_20260906" / "archive_selection_metrics.csv",
        pilot / "predictive_value_20260906" / "paired_bootstrap.csv",
    ]
    result["pilot_development_evidence"] = {
        str(path.relative_to(benchmark.REPO_ROOT)): benchmark.sha256(path) for path in evidence
    }
    return result


def shortlist_figure(output: Path) -> None:
    plt.switch_backend("Agg")
    mpl.rcParams["svg.hashsalt"] = "delphi-selection-20260906"
    prediction = pd.read_csv(output / "predictions.csv")
    curves = []
    specifications = [
        (benchmark.BASELINES[0], "point", "WSPU", "#0072B2", "-"),
        (benchmark.BASELINES[1], "point", "Canonical DSP", "#D55E00", "-"),
        (benchmark.BASELINES[2], "point", "Taskwise OLMix", "#009E73", "-"),
        (benchmark.BASELINES[1], "diverse_top20", "DSP: diverse top 20", "#D55E00", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, target, title in zip(axes, benchmark.TARGETS, ("Uncheatable", "Table 9"), strict=True):
        labels = pd.read_csv(output / "inputs" / f"{target}_bank_labels.csv")
        feature = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        mask = np.array([not bool(set(text.split(";")) & benchmark.INTERVENTIONS) for text in labels.sources])
        labels = labels.loc[mask]
        measured = labels.measured_mean_bpb.to_numpy(float)
        for method, policy, label, color, linestyle in specifications:
            sub = (
                prediction[
                    prediction.method.eq(method)
                    & prediction.target.eq(target)
                    & prediction.population.eq("external_development")
                ]
                .set_index("row_id")
                .loc[labels.coordinate_id]
            )
            values = sub.prediction.to_numpy(float)
            order = (
                scoring.diverse_order(values, feature["weights"][mask])
                if policy == "diverse_top20"
                else np.argsort(values, kind="stable")
            )
            regret = np.minimum.accumulate(measured[order[:10]]) - measured.min()
            axis.plot(
                np.arange(1, 11),
                regret,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                linewidth=1.7,
                label=label,
            )
            curves.extend(
                {"target": target, "method": method, "policy": policy, "k": i + 1, "regret": float(value)}
                for i, value in enumerate(regret)
            )
        axis.set_title(f"{title} · {len(labels)} archived optima", fontsize=12)
        axis.set_xlabel("Shortlist size k")
        axis.set_ylabel("Best measured regret in shortlist (BPB)")
        axis.set_xticks(range(1, 11))
        axis.set_ylim(bottom=-0.0004)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Retrospective shortlist coverage does not establish a new optimum", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output / "shortlist_regret.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "shortlist_regret.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    pd.DataFrame(curves).to_csv(output / "shortlist_curves.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    benchmark.verify_inputs(output)
    metrics = pd.read_csv(output / "metrics.csv")
    panel_contrasts(metrics, output).to_csv(output / "panel_fold_contrasts.csv", index=False)
    benchmark.write_json(output / "diagnosis.json", noise_diagnostics(output))
    loso = pd.read_csv(output / "source_disjoint_selection.csv")
    loso.groupby("target")[
        ["regret_at_1", "best_of_5_regret", "best_of_10_regret", "selected_rank", "optimism", "rmse", "spearman"]
    ].mean().to_csv(output / "source_disjoint_summary.csv")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "matplotlib")
        },
        "uv_lock_sha256": benchmark.sha256(benchmark.REPO_ROOT / "uv.lock"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=benchmark.REPO_ROOT).strip(),
    }
    benchmark.write_json(output / "environment.json", environment)
    shortlist_figure(output)


if __name__ == "__main__":
    main()
