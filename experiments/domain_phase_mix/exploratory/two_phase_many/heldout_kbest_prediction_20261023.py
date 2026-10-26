# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prediction error and rank correlation on the k best held-out mixtures (Qwen3 360M/1.6B).

Each model's full-swarm fit (k = 279 swarm runs; MARINER, Olmix and the natural cubic spline from their stored
learning-curve heads, RegMix's tuned trees refit with the hyperparameters of its bank pass, all as in
``learning_curve_expanded_bank_20260913``) predicts every coordinate of a held-out bank: the frozen held-out set the
paper describes (default) or the expanded set that adds the validation launches. For each objective, the coordinates
are sorted by measured loss and, for each k, the k best are scored: mean absolute error (BPB), mean signed error, and
Spearman correlation between predicted and measured loss. Intervals are 95% bootstrap intervals over the k
coordinates (2,000 resamples).

The noise references assume every coordinate is as noisy as the proportional repeats (std sigma). They score exact
predictions of each coordinate's mean, with the measured losses standing in for the means: each of 2,000 draws adds
N(0, sigma^2) noise to every coordinate, takes the k best noisy losses, and scores the means against them. Choosing
the k best by noisy loss favours negative noise, so at small k the reference exceeds sigma * sqrt(2 / pi).

Writes ``predictions.csv``, ``kbest_metrics.csv`` and ``summary.md`` under
``reference_outputs/heldout_kbest_prediction_20261023/<bank>``.

usage: LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 uv run --offline --no-sync --with lightgbm python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.heldout_kbest_prediction_20261023 [--bank expanded]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    learning_curve_expanded_bank_20260913 as bank_study,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    learning_curve_mariner_fits_20260908 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    learning_curve_metrics_20260905 as metrics,
)

OUT = Path(__file__).resolve().parent / "reference_outputs" / "heldout_kbest_prediction_20261023"
FULL_SWARM = 279
DRAW = 0
MODELS = ("mariner", "olmix", "regmix", "spline")
TARGETS = ("uncheatable", "table9")
K_VALUES = (10, 15, 20, 30, 50, 75, 100, 150, 200, 300)
BOOTSTRAP = 2000
SEED = 20261023


def predictions_for(target: str, model: str, panel, query) -> np.ndarray:
    if model == "regmix":
        component, _ = bank_study.regmix_predictions(target, FULL_SWARM, DRAW, panel, query)
    else:
        component, _ = bank_study.stored_predictions(target, model, FULL_SWARM, DRAW, panel, query)
    return component @ panel.group(target).aggregation_weights


def scored(measured: np.ndarray, predicted: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    error = predicted - measured
    out = {
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "spearman": float(spearmanr(predicted, measured).statistic),
    }
    n = len(measured)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    mae = np.abs(error[draws]).mean(axis=1)
    rho = np.array([spearmanr(predicted[d], measured[d]).statistic for d in draws])
    out |= {
        "mae_low": float(np.quantile(mae, 0.025)),
        "mae_high": float(np.quantile(mae, 0.975)),
        "spearman_low": float(np.nanquantile(rho, 0.025)),
        "spearman_high": float(np.nanquantile(rho, 0.975)),
    }
    return out


def noise_reference(measured: np.ndarray, k: int, sigma: float, rng: np.random.Generator) -> dict[str, float]:
    """Error and rank correlation of exact mean predictions on the k best of noisy losses (see the module doc)."""
    noisy = measured[None, :] + rng.normal(0.0, sigma, size=(BOOTSTRAP, len(measured)))
    best = np.argsort(noisy, axis=1)[:, :k]
    means, observed = measured[best], np.take_along_axis(noisy, best, axis=1)
    rho = np.array([spearmanr(m, o).statistic for m, o in zip(means, observed, strict=True)])
    return {"noise_mae": float(np.abs(means - observed).mean()), "noise_spearman": float(np.nanmean(rho))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction error on the k best held-out mixtures.")
    parser.add_argument("--bank", choices=("frozen", "expanded"), default="frozen")
    args = parser.parse_args()
    out = OUT / args.bank
    out.mkdir(parents=True, exist_ok=True)
    bank_study.use_bank(args.bank)
    metrics.select_study("mariner")
    panel = benchmark.load_panel(fits.PANEL)
    rng = np.random.default_rng(SEED)
    prediction_frames, rows = [], []
    for target in TARGETS:
        bank, query = benchmark.heldout_features(panel, target)
        _count_column, mean_column = benchmark.HELDOUT_TARGET_COLUMNS[target]
        frame = pd.DataFrame(
            {
                "target": target,
                "coordinate_id": bank["coordinate_id"].to_numpy(str),
                "sources": bank["sources"].to_numpy(str),
                "measured": bank[mean_column].to_numpy(float),
            }
        )
        for model in MODELS:
            frame[model] = predictions_for(target, model, panel, query)
            print(f"{target} {model}: {len(frame)} coordinates predicted", flush=True)
        frame = frame.sort_values("measured").reset_index(drop=True)
        prediction_frames.append(frame)
        sigma = float(panel.repeat_sd[target])
        measured = frame["measured"].to_numpy()
        for k in [k for k in K_VALUES if k < len(frame)] + [len(frame)]:
            band = frame.iloc[:k]
            reference = noise_reference(measured, k, sigma, rng)
            for model in MODELS:
                rows.append(
                    {
                        "target": target,
                        "k": k,
                        "model": model,
                        "band_width_bpb": float(band["measured"].iloc[-1] - band["measured"].iloc[0]),
                        "sigma": sigma,
                        **reference,
                        **scored(band["measured"].to_numpy(), band[model].to_numpy(), rng),
                    }
                )
    pd.concat(prediction_frames).to_csv(out / "predictions.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "kbest_metrics.csv", index=False)
    lines = [f"# k best held-out mixtures ({args.bank} held-out set, full-swarm fits)", ""]
    for target in TARGETS:
        part = table[table.target.eq(target)]
        lines += [
            f"## {target} (n = {int(part.k.max())}, proportional repeat std {part.sigma.iloc[0]:.4f})",
            "",
            "| k | " + " | ".join(f"{m} MAE / rho" for m in MODELS) + " | noise MAE / rho |",
            "|---" * (len(MODELS) + 2) + "|",
        ]
        for k, group in part.groupby("k"):
            by = group.set_index("model")
            lines.append(
                f"| {k} | "
                + " | ".join(f"{by.loc[m, 'mae']:.4f} / {by.loc[m, 'spearman']:.2f}" for m in MODELS)
                + f" | {by.iloc[0]['noise_mae']:.4f} / {by.iloc[0]['noise_spearman']:.2f} |"
            )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
