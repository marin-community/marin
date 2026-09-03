# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Selection-value scoring of frozen heldout predictions on the Delphi 3e18 bank, with paired bootstrap.

Reads the heldout prediction shards written by the benchmark harness, joins them to the measured coordinate
means of a heldout registry directory (the canonical one or a corrected copy), and reports, per target and
stratum, the regret of the predicted argmin, the best-of-top-k regret, the rank the true frontier receives,
bias by distance from the fit panel, and paired bootstrap intervals against a reference model.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)

PANEL = "delphi_3e18_39bucket"
TARGETS = ("uncheatable", "table9")
DOSE_SOURCE = "conditional_epoch_dose_response"
DISTANCE_EDGES = (0.0, 0.25, 0.5, 0.75, float("inf"))
BOOTSTRAP_SEED = 20_260_903
TOP_KS = (5, 10)


@dataclasses.dataclass(frozen=True)
class Bank:
    target: str
    coordinate_id: np.ndarray
    measured: np.ndarray
    sources: np.ndarray
    run_count: np.ndarray
    distance: np.ndarray
    tolerance: float


def load_bank(panel: harness.BenchPanel, target: str) -> Bank:
    frame, features = harness.heldout_features(panel, target)
    _count, mean_column = harness.HELDOUT_TARGET_COLUMNS[target]
    distance = np.abs(features.weights[:, None, :] - panel.features.weights[None, :, :]).sum(axis=-1).min(axis=1)
    return Bank(
        target=target,
        coordinate_id=frame["coordinate_id"].to_numpy(str),
        measured=frame[mean_column].to_numpy(float),
        sources=frame["sources"].to_numpy(str),
        run_count=frame["run_count"].to_numpy(int),
        distance=distance,
        tolerance=harness.BASIN_TOLERANCE_SD * panel.repeat_sd.get(target, float("nan")),
    )


def model_predictions(output_dir: Path, model_id: str, panel: harness.BenchPanel, bank: Bank) -> np.ndarray | None:
    group = panel.group(bank.target)
    matrix = np.full((len(bank.measured), len(group.components)), np.nan)
    for index, component in enumerate(group.components):
        payload = harness.load_shard(
            harness.heldout_shard_path(output_dir, model_id, PANEL, bank.target, index, component)
        )
        if payload is None or str(payload["status"].item()) != "ok":
            return None
        if not np.array_equal(payload["coordinate_id"].astype(str), bank.coordinate_id):
            raise ValueError(f"{model_id}/{bank.target}: shard coordinates do not match the registry order")
        matrix[:, index] = payload["prediction"]
    return matrix @ group.aggregation_weights


def selection_row(loss: np.ndarray, guess: np.ndarray, tolerance: float) -> dict[str, float]:
    order = np.argsort(guess, kind="stable")
    selected = int(order[0])
    ranks = stats.rankdata(loss, method="average")
    guess_ranks = stats.rankdata(guess, method="average")
    frontier = int(np.argmin(loss))
    best_quartile = loss <= np.quantile(loss, 0.25)
    row = {
        "bank_size": len(loss),
        "best_measured_bpb": float(loss.min()),
        "selected_measured_bpb": float(loss[selected]),
        "regret_at_1": float(loss[selected] - loss.min()),
        "selected_rank": float(ranks[selected]),
        "selected_percentile": float((ranks[selected] - 1) / max(len(loss) - 1, 1)),
        "frontier_predicted_rank": float(guess_ranks[frontier]),
        "selection_optimism": float(loss[selected] - guess[selected]),
        "basin_hit": float(loss[selected] - loss.min() <= tolerance) if np.isfinite(tolerance) else float("nan"),
        "rmse": float(np.sqrt(np.mean((guess - loss) ** 2))),
        "bias": float(np.mean(guess - loss)),
        "spearman": harness._safe_spearman(loss, guess),
        "spearman_best_quartile": (
            harness._safe_spearman(loss[best_quartile], guess[best_quartile])
            if best_quartile.sum() >= 5
            else float("nan")
        ),
    }
    for k in TOP_KS:
        row[f"top{k}_regret"] = float(loss[order[:k]].min() - loss.min())
    row.update(harness.random_ranking_expectations(loss, harness.TOP_K))
    return row


def strata_for(bank: Bank) -> list[tuple[str, np.ndarray]]:
    dose = np.array([DOSE_SOURCE in source for source in bank.sources])
    strata = [("pooled", np.ones(len(bank.measured), dtype=bool)), ("archive", ~dose), ("dose_response", dose)]
    for source in sorted(set(bank.sources)):
        mask = bank.sources == source
        if mask.sum() >= 5 and source != DOSE_SOURCE:
            strata.append((f"source:{source.removeprefix('archive::')}", mask))
    return [(name, mask) for name, mask in strata if mask.sum() >= 5]


def bootstrap_rows(
    bank: Bank, predictions: dict[str, np.ndarray], reference: str, draws: int, stratum: str, mask: np.ndarray
) -> list[dict[str, object]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    loss = bank.measured[mask]
    count = len(loss)
    samples = rng.integers(0, count, size=(draws, count))
    stat_names = ("regret_at_1", "top5_regret", "frontier_predicted_rank")
    per_model: dict[str, np.ndarray] = {}
    for model_id, guess in predictions.items():
        guess = guess[mask]
        values = np.empty((draws, len(stat_names)))
        for draw, sample in enumerate(samples):
            row_loss, row_guess = loss[sample], guess[sample]
            order = np.argsort(row_guess, kind="stable")
            frontier = int(np.argmin(row_loss))
            values[draw] = (
                row_loss[order[0]] - row_loss.min(),
                row_loss[order[:5]].min() - row_loss.min(),
                stats.rankdata(row_guess, method="average")[frontier],
            )
        per_model[model_id] = values
    rows = []
    for model_id, values in per_model.items():
        for column, name in enumerate(stat_names):
            row = {
                "model": model_id,
                "target": bank.target,
                "stratum": stratum,
                "statistic": name,
                "mean": float(values[:, column].mean()),
                "ci_low": float(np.quantile(values[:, column], 0.025)),
                "ci_high": float(np.quantile(values[:, column], 0.975)),
            }
            if reference in per_model and model_id != reference:
                difference = values[:, column] - per_model[reference][:, column]
                row.update(
                    {
                        "difference_vs_reference": float(difference.mean()),
                        "difference_ci_low": float(np.quantile(difference, 0.025)),
                        "difference_ci_high": float(np.quantile(difference, 0.975)),
                        "share_better_than_reference": float(np.mean(difference < 0)),
                    }
                )
            rows.append(row)
    return rows


def measurement_bootstrap_rows(
    bank: Bank,
    predictions: dict[str, np.ndarray],
    reference: str,
    draws: int,
    stratum: str,
    mask: np.ndarray,
    noise_sd: float,
) -> list[dict[str, object]]:
    """Fixed-bank bootstrap: perturb the measured means by run noise, keep every coordinate, rescore selection."""
    if not np.isfinite(noise_sd):
        return []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    loss = bank.measured[mask]
    scale = noise_sd / np.sqrt(np.maximum(bank.run_count[mask], 1))
    noise = rng.normal(0.0, 1.0, size=(draws, len(loss))) * scale[None, :]
    stat_names = ("regret_at_1", "top5_regret", "frontier_predicted_rank")
    per_model: dict[str, np.ndarray] = {}
    for model_id, guess in predictions.items():
        guess = guess[mask]
        order = np.argsort(guess, kind="stable")
        guess_ranks = stats.rankdata(guess, method="average")
        values = np.empty((draws, len(stat_names)))
        for draw in range(draws):
            perturbed = loss + noise[draw]
            frontier = int(np.argmin(perturbed))
            values[draw] = (
                perturbed[order[0]] - perturbed.min(),
                perturbed[order[:5]].min() - perturbed.min(),
                guess_ranks[frontier],
            )
        per_model[model_id] = values
    rows = []
    for model_id, values in per_model.items():
        for column, name in enumerate(stat_names):
            row = {
                "model": model_id,
                "target": bank.target,
                "stratum": stratum,
                "statistic": name,
                "noise_sd": noise_sd,
                "mean": float(values[:, column].mean()),
                "ci_low": float(np.quantile(values[:, column], 0.025)),
                "ci_high": float(np.quantile(values[:, column], 0.975)),
            }
            if reference in per_model and model_id != reference:
                difference = values[:, column] - per_model[reference][:, column]
                row.update(
                    {
                        "difference_vs_reference": float(difference.mean()),
                        "difference_ci_low": float(np.quantile(difference, 0.025)),
                        "difference_ci_high": float(np.quantile(difference, 0.975)),
                        "share_better_than_reference": float(np.mean(difference < 0)),
                    }
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir", type=Path, required=True, help="heldout registry directory (use the corrected view)"
    )
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", required=True, help="comma-separated model ids with heldout shards")
    parser.add_argument("--reference", default="weibull_softplus_unscaled")
    parser.add_argument("--report-subdir", default="heldout_round3")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    model_ids = [token.strip() for token in args.models.split(",") if token.strip()]
    metric_rows: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for target in TARGETS:
        bank = load_bank(panel, target)
        runs = harness.heldout_runs_for(PANEL, target)
        coverage_rows.append(
            {
                "target": target,
                "coordinates": len(bank.measured),
                "runs": len(runs),
                "dose_response_coordinates": int(sum(DOSE_SOURCE in source for source in bank.sources)),
                "archive_coordinates": int(sum(DOSE_SOURCE not in source for source in bank.sources)),
                "frontier_bpb": float(bank.measured.min()),
                "frontier_source": str(bank.sources[int(np.argmin(bank.measured))]),
                "frontier_distance_l1": float(bank.distance[int(np.argmin(bank.measured))]),
            }
        )
        predictions: dict[str, np.ndarray] = {}
        for model_id in model_ids:
            guess = model_predictions(args.output_dir, model_id, panel, bank)
            if guess is None:
                metric_rows.append({"model": model_id, "target": target, "stratum": "pooled", "status": "incomplete"})
                continue
            predictions[model_id] = guess
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "model": model_id,
                        "target": target,
                        "coordinate_id": bank.coordinate_id,
                        "sources": bank.sources,
                        "run_count": bank.run_count,
                        "distance_l1": bank.distance,
                        "prediction": guess,
                        "measured_mean_bpb": bank.measured,
                    }
                )
            )
            for name, mask in strata_for(bank):
                row = {"model": model_id, "target": target, "stratum": name, "status": "ok"}
                row.update(selection_row(bank.measured[mask], guess[mask], bank.tolerance))
                row["selected_distance_l1"] = float(bank.distance[mask][int(np.argmin(guess[mask]))])
                metric_rows.append(row)
            for low, high in itertools.pairwise(DISTANCE_EDGES):
                mask = (bank.distance >= low) & (bank.distance < high)
                if mask.sum() < 5:
                    continue
                error = guess[mask] - bank.measured[mask]
                metric_rows.append(
                    {
                        "model": model_id,
                        "target": target,
                        "stratum": f"distance:{low:.2f}-{high:.2f}",
                        "status": "ok",
                        "bank_size": int(mask.sum()),
                        "bias": float(error.mean()),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "spearman": harness._safe_spearman(bank.measured[mask], guess[mask]),
                    }
                )
        for name, mask in strata_for(bank)[:3]:
            boot_rows.extend(bootstrap_rows(bank, predictions, args.reference, args.bootstrap, name, mask))
            measurement_rows.extend(
                measurement_bootstrap_rows(
                    bank,
                    predictions,
                    args.reference,
                    args.bootstrap,
                    name,
                    mask,
                    panel.repeat_sd.get(target, float("nan")),
                )
            )
    report_dir = args.output_dir / args.report_subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(report_dir / "selection_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(report_dir / "selection_bootstrap.csv", index=False)
    pd.DataFrame(measurement_rows).to_csv(report_dir / "selection_measurement_bootstrap.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(report_dir / "predictions.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(report_dir / "coverage.csv", index=False)
    manifest_hash = hashlib.sha256((args.registry_dir.resolve() / "manifest.json").read_bytes()).hexdigest()
    (report_dir / "registry_dir.txt").write_text(f"{args.registry_dir.resolve()}\nmanifest_sha256 {manifest_hash}\n")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(coverage_rows).round(4).to_string(index=False))
    columns = [
        "model",
        "stratum",
        "bank_size",
        "regret_at_1",
        "top5_regret",
        "top10_regret",
        "selected_rank",
        "frontier_predicted_rank",
        "selection_optimism",
        "bias",
        "rmse",
        "spearman",
        "spearman_best_quartile",
        "selected_distance_l1",
        "random_regret_at_1",
        "random_best_of_5_regret",
    ]
    for target in TARGETS:
        for stratum in ("pooled", "archive", "dose_response"):
            subset = metrics[metrics["target"].eq(target) & metrics["stratum"].eq(stratum) & metrics["status"].eq("ok")]
            if subset.empty:
                continue
            print(f"\n=== {target} / {stratum}")
            print(
                subset.loc[:, [column for column in columns if column in subset.columns]].round(4).to_string(index=False)
            )
    boot = pd.DataFrame(boot_rows)
    if not boot.empty:
        print("\n=== paired bootstrap against", args.reference, "(pooled; difference < 0 favours the model)")
        pooled = boot[boot["stratum"].eq("pooled") & boot["statistic"].isin(["regret_at_1", "top5_regret"])]
        print(pooled.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
