# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How well each frozen surrogate orders the best-measured band of the Delphi bank.

Selection regret is decided by the ordering among the few coordinates that are already near the optimum, not by
the ordering of the whole bank. For each target the optima stratum is sorted by measured value and the best k
coordinates form the band; inside the band every method is scored by pairwise sign accuracy on pairs whose
measured difference exceeds one run SD, Spearman correlation, and the regret of its argmin inside the band. The
noise ceiling is the sign accuracy a perfect predictor of the true means would reach against the same single-run
measurements (mean of Phi(|dy| / sd_diff) over the pairs, with sd_diff from the run counts). Uncertainty: a
source-block bootstrap over the band's coordinates. Predictions come from the frozen evaluation packages; nothing
is fitted or launched.

usage: uv run python analyze_delphi_top_band_ordering_20260906.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
BENCHMARK = REFERENCE / "delphi_offline_selection_20260906"
DEFAULT_OUTPUT = REFERENCE / "delphi_top_band_ordering_20260906"
INTERVENTIONS = ("conditional_epoch_dose_response", "archive::delphi_baseline_mixtures_issue6607_20260623")
RUN_SD = {"table9": 0.0038, "uncheatable": 0.0009}
BAND_SIZES = (10, 20, 30, 50, 100, 0)  # 0 = whole optima stratum
BOOTSTRAP_DRAWS = 2000
SEED = 20260906
PACKAGES = {
    "delphi_offline_selection_20260906": {
        "weibull_softplus_unscaled": "WSPU",
        "dsp_total_exposure": "DSP",
        "olmix_loglinear_taskwise": "OLMix",
        "wspu_direct_macro": "WSPU, direct macro",
        "ridge_log_epoch_taskwise": "ridge log-epoch, taskwise",
        "matern_share_macro": "Matern kernel, share",
        "ridge_log_epoch_wls": "ridge log-epoch, WLS",
    },
    "delphi_link_variants_selection_20260906": {
        "weibull_softplus_unscaled@log_deficit_bounded_link": "WSPU, bounded link",
        "weibull_softplus_unscaled@log_deficit_bounded_link_floor_cv": "WSPU, link, floor by CV",
        "weibull_softplus_unscaled@log_deficit_bounded_link_total_hub": "WSPU, link + hub",
    },
    "delphi_link_pairs_selection_20260906": {
        "weibull_softplus_unscaled@named_pairs": "WSPU, named pairs",
        "weibull_softplus_unscaled@log_deficit_bounded_link_named_pairs": "WSPU, link + named pairs",
    },
    "delphi_link_selection_20260906": {
        "weibull_softplus_unscaled@link_by_cv": "WSPU, link by CV",
        "link_bounded_coupling_kappa_1": "bounded link + coupling",
    },
    "delphi_coupling_followup_20260906/incumbent_coupling": {"wspu_coupling_kappa_1": "WSPU + coupling"},
}
PLOTTED = (
    "WSPU",
    "WSPU, bounded link",
    "WSPU, link + hub",
    "DSP",
    "OLMix",
    "bank kernel LOSO, TV 0.05",
    "bank kernel LOSO, TV 0.1",
)


def load_predictions() -> pd.DataFrame:
    frames = []
    for package, methods in PACKAGES.items():
        table = pd.read_csv(REFERENCE / package / "predictions.csv")
        table = table[table.population.eq("external_development") & table.method.isin(methods)]
        frames.append(table.assign(label=table.method.map(methods))[["target", "label", "row_id", "prediction"]])
    return pd.concat(frames, ignore_index=True)


def band_scores(
    measured: np.ndarray, predicted: np.ndarray, threshold: float, sd_diff: np.ndarray, blocks: np.ndarray
) -> dict:
    pairs = np.array(list(itertools.combinations(range(len(measured)), 2)))
    dy = measured[pairs[:, 0]] - measured[pairs[:, 1]]
    dp = predicted[pairs[:, 0]] - predicted[pairs[:, 1]]
    keep = np.abs(dy) > threshold
    same = blocks[pairs[:, 0]] == blocks[pairs[:, 1]]
    correct = np.sign(dp) == np.sign(dy)
    accuracy = float(np.mean(correct[keep])) if keep.any() else np.nan
    ceiling = float(np.mean(stats.norm.cdf(np.abs(dy[keep]) / sd_diff[keep]))) if keep.any() else np.nan
    return {
        "pairs": int(keep.sum()),
        "sign_accuracy": accuracy,
        "within_block_pairs": int((keep & same).sum()),
        "within_block_sign_accuracy": float(np.mean(correct[keep & same])) if (keep & same).any() else np.nan,
        "cross_block_sign_accuracy": float(np.mean(correct[keep & ~same])) if (keep & ~same).any() else np.nan,
        "noise_ceiling": ceiling,
        "spearman": float(stats.spearmanr(predicted, measured).statistic) if len(measured) > 2 else np.nan,
        "argmin_regret": float(measured[int(np.argmin(predicted))] - measured.min()),
        "argmin_rank": int(stats.rankdata(measured, method="min")[int(np.argmin(predicted))]),
    }


def bootstrap_accuracy(
    measured: np.ndarray, predicted: np.ndarray, blocks: np.ndarray, threshold: float, sd_diff_matrix: np.ndarray
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(blocks)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
        if len(rows) < 3:
            continue
        pairs = np.array(list(itertools.combinations(range(len(rows)), 2)))
        dy = measured[rows[pairs[:, 0]]] - measured[rows[pairs[:, 1]]]
        dp = predicted[rows[pairs[:, 0]]] - predicted[rows[pairs[:, 1]]]
        keep = np.abs(dy) > threshold
        if keep.any():
            draws.append(np.mean(np.sign(dp[keep]) == np.sign(dy[keep])))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


KERNEL_BANDWIDTHS = (0.05, 0.1, 0.2)


def kernel_loso(weights: np.ndarray, measured: np.ndarray, blocks: np.ndarray, bandwidth: float) -> np.ndarray:
    """Nadaraya-Watson smoother on TV distance, each coordinate predicted from the other source blocks only."""
    distance = np.abs(weights[:, None, :] - weights[None, :, :]).sum(axis=2) / 2
    kernel = np.exp(-0.5 * (distance / bandwidth) ** 2)
    kernel[blocks[:, None] == blocks[None, :]] = 0.0
    mass = kernel.sum(axis=1)
    prediction = np.where(mass > 0, (kernel @ measured) / np.maximum(mass, 1e-300), measured.mean())
    return prediction


def bank_kernel_predictions(
    target: str, labels: pd.DataFrame, blocks: np.ndarray, surrogate: pd.DataFrame
) -> pd.DataFrame:
    """Held-out-source kernel predictions on the whole bank (all strata): the bank alone, and WSPU plus a
    kernel-smoothed residual of WSPU (both leave the held-out coordinate's source block out)."""
    features = np.load(BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
    order = pd.Index(features["coordinate_id"]).get_indexer(labels.coordinate_id)
    weights = features["weights"][order]
    measured = labels.measured_mean_bpb.to_numpy(float)
    wspu = surrogate[surrogate.target.eq(target) & surrogate.label.eq("WSPU")].set_index("row_id")
    wspu = wspu.loc[labels.coordinate_id].prediction.to_numpy(float)
    rows = []
    for bandwidth in KERNEL_BANDWIDTHS:
        for label, prediction in (
            (f"bank kernel LOSO, TV {bandwidth}", kernel_loso(weights, measured, blocks, bandwidth)),
            (
                f"WSPU + LOSO kernel residual, TV {bandwidth}",
                wspu + kernel_loso(weights, measured - wspu, blocks, bandwidth),
            ),
        ):
            rows.extend(
                {"target": target, "label": label, "row_id": row, "prediction": value}
                for row, value in zip(labels.coordinate_id, prediction, strict=True)
            )
    return pd.DataFrame(rows)


ENSEMBLES = {
    "ensemble mean rank (WSPU, DSP, OLMix)": ("WSPU", "DSP", "OLMix"),
    "ensemble mean rank (WSPU, link, hub, DSP, OLMix)": (
        "WSPU",
        "WSPU, bounded link",
        "WSPU, link + hub",
        "DSP",
        "OLMix",
    ),
}


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean rank over the bank of several methods' predictions, as a rank-valued prediction."""
    rows = []
    for target, frame in predictions.groupby("target"):
        wide = frame.pivot(index="row_id", columns="label", values="prediction")
        for label, members in ENSEMBLES.items():
            ranks = wide[list(members)].rank().mean(axis=1)
            rows.extend(
                {"target": target, "label": label, "row_id": row, "prediction": float(value)}
                for row, value in ranks.items()
            )
    return pd.DataFrame(rows)


def analyse(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores every method on each band; extends `predictions` in place with the bank-kernel methods."""
    predictions = pd.concat([predictions, ensemble_predictions(predictions)], ignore_index=True)
    blocks_table = pd.read_csv(BENCHMARK / "source_blocks.csv")
    rows = []
    extra = []
    for target in ("table9", "uncheatable"):
        labels = pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
        all_blocks = blocks_table[blocks_table.target.eq(target)].set_index("coordinate_id")
        kernel = bank_kernel_predictions(
            target, labels, all_blocks.loc[labels.coordinate_id].source_block.to_numpy(), predictions
        )
        extra.append(kernel)
        predictions = pd.concat([predictions, kernel], ignore_index=True)
        labels = labels[~labels.sources.str.contains("|".join(INTERVENTIONS))].sort_values("measured_mean_bpb")
        blocks = all_blocks.loc[labels.coordinate_id]
        block_ids = blocks.source_block.to_numpy()
        measured = labels.measured_mean_bpb.to_numpy(float)
        sd_run = RUN_SD[target] / np.sqrt(labels.run_count.to_numpy(float))
        sd_diff_matrix = np.sqrt(sd_run[:, None] ** 2 + sd_run[None, :] ** 2)
        threshold = RUN_SD[target]
        for label, frame in predictions[predictions.target.eq(target)].groupby("label"):
            predicted = frame.set_index("row_id").loc[labels.coordinate_id].prediction.to_numpy(float)
            for size in BAND_SIZES:
                k = len(measured) if size == 0 else size
                band = np.arange(k)
                pairs = np.array(list(itertools.combinations(band, 2)))
                scores = band_scores(
                    measured[band], predicted[band], threshold, sd_diff_matrix[pairs[:, 0], pairs[:, 1]], block_ids[band]
                )
                low, high = bootstrap_accuracy(
                    measured[band], predicted[band], block_ids[band], threshold, sd_diff_matrix
                )
                rows.append(
                    {
                        "target": target,
                        "method": label,
                        "band": k,
                        "band_width_bpb": float(measured[band].max() - measured[band].min()),
                        "source_blocks": len(np.unique(block_ids[band])),
                        **scores,
                        "sign_accuracy_ci_low": low,
                        "sign_accuracy_ci_high": high,
                    }
                )
    return pd.DataFrame(rows), pd.concat(extra, ignore_index=True)


def half_accuracy(measured: np.ndarray, prediction: np.ndarray, rows: np.ndarray, threshold: float) -> float:
    """Pairwise sign accuracy over the selected rows, pairs below the threshold excluded."""
    index = np.flatnonzero(rows)
    pairs = np.array(list(itertools.combinations(index, 2)))
    if len(pairs) == 0:
        return np.nan
    dy = measured[pairs[:, 0]] - measured[pairs[:, 1]]
    dp = prediction[pairs[:, 0]] - prediction[pairs[:, 1]]
    keep = np.abs(dy) > threshold
    return float(np.mean(np.sign(dp[keep]) == np.sign(dy[keep]))) if keep.any() else np.nan


def split_half_bandwidth(predictions: pd.DataFrame, band: int = 30, splits: int = 200) -> pd.DataFrame:
    """Choose the kernel bandwidth on half the source blocks of the band and score it on the other half."""
    blocks_table = pd.read_csv(BENCHMARK / "source_blocks.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for target in ("table9", "uncheatable"):
        labels = pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
        labels = labels[~labels.sources.str.contains("|".join(INTERVENTIONS))].sort_values("measured_mean_bpb")
        labels = labels.head(band)
        blocks = blocks_table[blocks_table.target.eq(target)].set_index("coordinate_id").loc[labels.coordinate_id]
        block_ids = blocks.source_block.to_numpy()
        measured = labels.measured_mean_bpb.to_numpy(float)
        threshold = RUN_SD[target]
        candidates = {}
        for bandwidth in KERNEL_BANDWIDTHS:
            for prefix in ("bank kernel LOSO", "WSPU + LOSO kernel residual"):
                label = f"{prefix}, TV {bandwidth}"
                frame = predictions[predictions.target.eq(target) & predictions.label.eq(label)].set_index("row_id")
                candidates[label] = frame.loc[labels.coordinate_id].prediction.to_numpy(float)
        reference = predictions[predictions.target.eq(target) & predictions.label.eq("WSPU")].set_index("row_id")
        reference = reference.loc[labels.coordinate_id].prediction.to_numpy(float)
        unique = np.unique(block_ids)
        chosen_scores, reference_scores, chosen_labels = [], [], []
        for _ in range(splits):
            permuted = rng.permutation(unique)
            first = np.isin(block_ids, permuted[: len(unique) // 2])
            second = ~first
            scores = {label: half_accuracy(measured, values, first, threshold) for label, values in candidates.items()}
            if all(np.isnan(list(scores.values()))):
                continue
            best = max(scores, key=lambda key: (np.nan_to_num(scores[key], nan=-1.0), key))
            chosen = half_accuracy(measured, candidates[best], second, threshold)
            base = half_accuracy(measured, reference, second, threshold)
            if np.isnan(chosen) or np.isnan(base):
                continue
            chosen_scores.append(chosen)
            reference_scores.append(base)
            chosen_labels.append(best)
        delta = np.array(chosen_scores) - np.array(reference_scores)
        counts = pd.Series(chosen_labels).value_counts()
        rows.append(
            {
                "target": target,
                "band": band,
                "splits": len(delta),
                "chosen_mean_accuracy": float(np.mean(chosen_scores)),
                "wspu_mean_accuracy": float(np.mean(reference_scores)),
                "mean_delta": float(delta.mean()),
                "delta_q025": float(np.quantile(delta, 0.025)),
                "delta_q975": float(np.quantile(delta, 0.975)),
                "fraction_better": float(np.mean(delta > 0)),
                "most_chosen": f"{counts.index[0]} ({counts.iloc[0]}/{len(delta)})",
            }
        )
    return pd.DataFrame(rows)


TWO_STAGE_SHORTLISTS = (5, 10, 20, 30)
TWO_STAGE_MASS_FLOOR = 0.5


def two_stage_policy(predictions: pd.DataFrame) -> pd.DataFrame:
    """Surrogate picks the basin (its predicted top-k of the optima stratum); the held-out-source bank kernel orders
    the floor (the shortlist member with the lowest kernel forecast among those with kernel mass >= 0.5)."""
    blocks_table = pd.read_csv(BENCHMARK / "source_blocks.csv")
    rows = []
    for target in ("table9", "uncheatable"):
        labels = pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
        features = np.load(BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
        order = pd.Index(features["coordinate_id"]).get_indexer(labels.coordinate_id)
        weights = features["weights"][order]
        measured = labels.measured_mean_bpb.to_numpy(float)
        blocks = blocks_table[blocks_table.target.eq(target)].set_index("coordinate_id")
        blocks = blocks.loc[labels.coordinate_id].source_block.to_numpy()
        optima = np.flatnonzero(~labels.sources.str.contains("|".join(INTERVENTIONS)))
        best = measured[optima].min()
        distance = np.abs(weights[:, None, :] - weights[None, :, :]).sum(axis=2) / 2
        for label in ("WSPU", "DSP", "WSPU, bounded link", "WSPU, link + hub"):
            frame = predictions[predictions.target.eq(target) & predictions.label.eq(label)].set_index("row_id")
            predicted = frame.loc[labels.coordinate_id].prediction.to_numpy(float)
            for bandwidth in KERNEL_BANDWIDTHS:
                kernel = np.exp(-0.5 * (distance / bandwidth) ** 2)
                kernel[blocks[:, None] == blocks[None, :]] = 0.0
                mass = kernel.sum(axis=1)
                forecast = np.where(mass > 0, (kernel @ measured) / np.maximum(mass, 1e-300), np.inf)
                for size in TWO_STAGE_SHORTLISTS:
                    shortlist = optima[np.argsort(predicted[optima], kind="stable")[:size]]
                    supported = shortlist[mass[shortlist] >= TWO_STAGE_MASS_FLOOR]
                    pick = supported[np.argmin(forecast[supported])] if len(supported) else shortlist[0]
                    rows.append(
                        {
                            "target": target,
                            "method": label,
                            "bandwidth": bandwidth,
                            "shortlist": size,
                            "supported": len(supported),
                            "regret_at_1": float(measured[pick] - best),
                            "selected_rank": int((measured[optima] <= measured[pick]).sum()),
                            "plain_regret_at_1": float(measured[shortlist[0]] - best),
                            "plain_rank": int((measured[optima] <= measured[shortlist[0]]).sum()),
                        }
                    )
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for axis, target in zip(axes, ("table9", "uncheatable"), strict=True):
        sub = table[table.target.eq(target)]
        for label in PLOTTED:
            line = sub[sub.method.eq(label)].sort_values("band")
            axis.plot(line.band, line.sign_accuracy, marker="o", ms=3, label=label)
        ceiling = sub[sub.method.eq("WSPU")].sort_values("band")
        axis.plot(ceiling.band, ceiling.noise_ceiling, color="black", ls="--", label="noise ceiling")
        axis.axhline(0.5, color="grey", lw=0.6)
        axis.set_xscale("log")
        axis.set_xlabel("band: k best-measured coordinates")
        axis.set_ylabel("pairwise sign accuracy")
        axis.set_title("OlmoBaseEval Easy" if target == "table9" else "Uncheatable")
        axis.set_ylim(0.0, 1.02)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "top_band_sign_accuracy.png", dpi=200)
    fig.savefig(output / "top_band_sign_accuracy.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    table, kernel = analyse(predictions)
    table.to_csv(args.output_dir / "top_band_ordering.csv", index=False)
    two_stage = two_stage_policy(predictions)
    two_stage.to_csv(args.output_dir / "two_stage_policy.csv", index=False)
    print("two-stage policy (surrogate shortlist, then held-out bank kernel), Table-9 regret@1:")
    print(
        two_stage[two_stage.target.eq("table9")]
        .pivot_table(index=["method", "bandwidth"], columns="shortlist", values="regret_at_1")
        .round(4)
        .to_string()
    )
    halves = split_half_bandwidth(pd.concat([predictions, kernel], ignore_index=True))
    print("split-half bandwidth check (band 30):")
    print(halves.round(3).to_string(index=False))
    halves.to_csv(args.output_dir / "split_half_bandwidth.csv", index=False)
    plot(table, args.output_dir)
    pd.set_option("display.width", 250)
    for target in ("table9", "uncheatable"):
        for band in (30, 0):
            sizes = table[table.target.eq(target)].band
            sub = table[table.target.eq(target) & table.band.eq(band if band else sizes.max())]
            print(f"== {target}, band {sub.band.iloc[0]} (width {sub.band_width_bpb.iloc[0]:.4f} BPB)")
            print(
                sub.sort_values("sign_accuracy", ascending=False)[
                    [
                        "method",
                        "pairs",
                        "sign_accuracy",
                        "sign_accuracy_ci_low",
                        "sign_accuracy_ci_high",
                        "within_block_pairs",
                        "within_block_sign_accuracy",
                        "cross_block_sign_accuracy",
                        "noise_ceiling",
                        "spearman",
                        "argmin_regret",
                        "argmin_rank",
                    ]
                ]
                .round(3)
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
