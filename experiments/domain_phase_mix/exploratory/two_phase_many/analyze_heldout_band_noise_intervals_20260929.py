# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Measurement-noise intervals for the best-k ordering of the held-out optima.

For each objective (its mean loss) and for arXiv physics and SciQ alone, the interval around a predictor's pairwise
sign accuracy among the k best-measured held-out optima is the 2.5th to 97.5th percentile over draws in which every
optimum's measured loss is redrawn from a normal with its run standard deviation over the square root of its run count,
the k best are re-selected under the redrawn losses, and the accuracy is rescored over pairs whose redrawn difference
exceeds the run standard deviation. Predictors: MARINER and Olmix from the frozen prediction packages, tuned RegMix from
`regmix_heldout_task_predictions.csv`, and the leave-source-out kernel over the other optima (recomputed per draw from
the redrawn losses). The point estimate is the unperturbed accuracy, as in `top_band_ordering.csv`.

Writes `noise_band_ordering.csv` to `reference_outputs/delphi_top_band_ordering_20260928/`.

usage: uv run --offline --no-sync python analyze_heldout_band_noise_intervals_20260929.py [--draws 2000]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import analyze_heldout_band_ordering_pbt_20260929 as pbt  # noqa: E402
import analyze_heldout_component_ordering_20260929 as ordering  # noqa: E402
import analyze_heldout_component_predictability_20260929 as predictability  # noqa: E402
import analyze_top_band_ordering_refresh_20260928 as refresh  # noqa: E402
import preview_band_ordering_snr_subsets_20260929 as preview  # noqa: E402

OUTPUT = pbt.OUTPUT
REGMIX_TASKS = OUTPUT / "regmix_heldout_task_predictions.csv"
COMPONENTS = {"uncheatable": "arxiv_physics", "table9": "sciq"}
SEED = 20260929


def noise_accuracies(
    measured: np.ndarray,
    predictions: dict[str, np.ndarray],
    blocks: np.ndarray,
    run_sd: float,
    counts: np.ndarray,
    weights: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sign accuracy per band and draw, shape (draws, bands), for each predictor.

    The kernel is redrawn with the losses.
    """
    sd = run_sd / np.sqrt(counts)
    out = {name: np.full((draws, len(pbt.BAND_SIZES)), np.nan) for name in (*predictions, "kernel")}
    for d in range(draws):
        y = measured + rng.normal(0.0, sd)
        kernel = top_band.kernel_loso(weights, y, blocks, predictability.KERNEL_BANDWIDTH)
        order = np.argsort(y, kind="stable")
        for j, size in enumerate(pbt.BAND_SIZES):
            k = len(y) if size == 0 else size
            band = order[:k]
            pairs = np.array(list(itertools.combinations(range(k), 2)))
            dy = y[band][pairs[:, 0]] - y[band][pairs[:, 1]]
            keep = np.abs(dy) > run_sd
            if not keep.any():
                continue
            for name, p in (*predictions.items(), ("kernel", kernel)):
                dp = p[band][pairs[:, 0]] - p[band][pairs[:, 1]]
                out[name][d, j] = np.mean(np.sign(dp[keep]) == np.sign(dy[keep]))
    return out


def summarize(target: str, component: str | None, measured, predictions, blocks, run_sd, counts, weights, draws, rng):
    accuracies = noise_accuracies(measured, predictions, blocks, run_sd, counts, weights, draws, rng)
    kernel = top_band.kernel_loso(weights, measured, blocks, predictability.KERNEL_BANDWIDTH)
    rows = []
    for name, p in (*predictions.items(), ("kernel", kernel)):
        for j, size in enumerate(pbt.BAND_SIZES):
            k = len(measured) if size == 0 else size
            point = predictability.band_statistics(measured, p, blocks, run_sd, counts, k)
            rows.append(
                {
                    "target": target,
                    "component": component,
                    "method": name,
                    "band": k,
                    "sign_accuracy": point["sign_accuracy"],
                    "noise_ceiling": point["noise_ceiling"],
                    "noise_low": float(np.nanquantile(accuracies[name][:, j], 0.025)),
                    "noise_high": float(np.nanquantile(accuracies[name][:, j], 0.975)),
                    "noise_mean": float(np.nanmean(accuracies[name][:, j])),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    top_band.PACKAGES = refresh.PACKAGES
    frozen = top_band.load_predictions()
    regmix_tasks = pd.read_csv(REGMIX_TASKS)
    panel, swarm, _ = ordering.swarm_inputs()
    blocks_table = pd.read_csv(preview.BENCHMARK / "source_blocks.csv")
    rows = []
    for target, component in COMPONENTS.items():
        labels = ordering.held_out_labels(target)
        blocks = (
            blocks_table[blocks_table.target.eq(target)]
            .set_index("coordinate_id")
            .loc[labels.coordinate_id]
            .source_block.to_numpy()
        )
        counts = labels.run_count.to_numpy(float)
        measured = labels.measured_mean_bpb.to_numpy(float)
        features = np.load(preview.BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
        weights = (
            pd.DataFrame(
                features["weights"],
                index=[str(c) for c in features["coordinate_id"]],
                columns=[str(b) for b in panel.buckets],
            )
            .loc[labels.coordinate_id]
            .to_numpy(float)
        )
        group = panel.group(target)
        names = [preview.short_name(str(c)) for c in group.components]
        tasks = regmix_tasks[regmix_tasks.target.eq(target)].pivot(
            index="coordinate_id", columns="component", values="prediction"
        )
        regmix = tasks[names].to_numpy(float) @ np.asarray(group.aggregation_weights, float)
        regmix = pd.Series(regmix, index=tasks.index).reindex(labels.coordinate_id).to_numpy()
        predictions = {
            label: frozen[frozen.target.eq(target) & frozen.label.eq(label)]
            .set_index("row_id")
            .prediction.reindex(labels.coordinate_id)
            .to_numpy(float)
            for label in ("MARINER", "OLMix")
        }
        predictions["RegMix"] = regmix
        rows.extend(
            summarize(
                target, None, measured, predictions, blocks, top_band.RUN_SD[target], counts, weights, args.draws, rng
            )
        )
        print(f"{target} objective: done", flush=True)
        frame = ordering.component_frame(target, component)
        component_regmix = tasks[component].reindex(frame.coordinate_id).to_numpy()
        component_weights = frame.exposures.to_numpy() / np.asarray(swarm.inventory, float)[None, :]
        rows.extend(
            summarize(
                target,
                component,
                frame.measured,
                {"MARINER": frame.mariner, "OLMix": frame.olmix, "RegMix": component_regmix},
                frame.blocks,
                frame.run_sd,
                frame.counts,
                component_weights,
                args.draws,
                rng,
            )
        )
        print(f"{target} {component}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "noise_band_ordering.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
