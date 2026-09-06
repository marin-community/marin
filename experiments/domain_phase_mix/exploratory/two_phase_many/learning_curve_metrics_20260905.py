# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the learning-curve records of ``learning_curve_fits_20260905`` and summarise them over draws.

Every complete (target, model, k, draw) record set is scored the way the Observatory benchmark scores a
fit: out-of-fold aggregate metrics pooled over the subset and per outer fold, the same metrics on the
complement rows the subset never saw, the frozen held-out bank (pooled, per source with at least five
coordinates, and at run level), and per-component out-of-fold metrics. Summaries report the mean over
draws with a t interval and a draw-level bootstrap interval, and the paired WSPU minus OLMix difference
on the same subsets and folds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_fits_20260905 as fits,
)

LOGGER = logging.getLogger("learning_curve_metrics")

METRICS_LONG = "metrics_long.csv"
SUMMARY = "summary.csv"
PAIRED_SUMMARY = "paired_summary.csv"
COMPLETENESS = "completeness.csv"
HYPERPARAMETERS = "hyperparameters.csv"
REPORT = "report.md"
EFFICIENCY = "efficiency.csv"
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20_260_905
MIN_STRATUM = 5
DOSE_RESPONSE_SOURCE = "conditional_epoch_dose_response"
KEY_COLUMNS = ["target", "model", "k", "draw"]
SUMMARY_METRICS = (
    "rmse",
    "mae",
    "spearman",
    "pearson",
    "kendall",
    "regret_at_1",
    "regret_at_top_k",
    "selection_optimism",
    "calibration_slope",
    "basin_rmse",
    "basin_spearman",
    "top5_regret",
    "top10_regret",
    "selected_percentile",
    "rmse_over_repeat_sd",
    "random_regret_at_1",
    "regret_ratio",
    "calibration_slope_error",
    "selection_optimism_abs",
)
# Metrics where a smaller value is better; the remaining directional metrics are better when larger.
LOWER_IS_BETTER = frozenset(
    {
        "rmse",
        "mae",
        "regret_at_1",
        "regret_at_top_k",
        "top5_regret",
        "top10_regret",
        "selected_percentile",
        "rmse_over_repeat_sd",
        "regret_ratio",
        "calibration_slope_error",
        "selection_optimism_abs",
        "basin_rmse",
    }
)
# Signed diagnostics with no better direction; the paired win fraction is not reported for them.
UNDIRECTED = frozenset({"calibration_slope", "selection_optimism", "random_regret_at_1"})


class RecordSet:
    """All component chunks of one (target, model, k, draw), assembled in the target's component order."""

    def __init__(self, target: str, model: str, k: int, draw: int, payloads: list[dict[str, Any]], count: int):
        self.target, self.model, self.k, self.draw = target, model, k, draw
        first = payloads[0]
        self.rows = first["rows"]
        self.outer_labels = first["outer_labels"]
        self.complement_rows = first["complement_rows"]
        self.elapsed = float(sum(float(item["elapsed"]) for item in payloads))
        order = np.concatenate([item["component_indices"] for item in payloads])
        if sorted(order.tolist()) != list(range(count)):
            raise ValueError(f"{target}/{model}/k{k}/draw{draw}: chunks do not cover the components")
        inverse = np.argsort(order)
        self.oof = np.concatenate([item["oof_prediction"] for item in payloads], axis=1)[:, inverse]
        self.fit_prediction = np.concatenate([item["fit_prediction"] for item in payloads], axis=1)[:, inverse]
        self.complement_prediction = np.concatenate([item["complement_prediction"] for item in payloads], axis=1)[
            :, inverse
        ]
        self.heldout_prediction = np.concatenate([item["heldout_prediction"] for item in payloads], axis=1)[:, inverse]
        self.ridge = np.concatenate([item["ridge"] for item in payloads], axis=1)[:, inverse]
        self.shape_json = np.concatenate([item["shape_json"] for item in payloads], axis=1)[:, inverse]
        self.diagnostics_json = np.concatenate([item["diagnostics_json"] for item in payloads], axis=1)[:, inverse]
        for item in payloads:
            if not (np.array_equal(item["rows"], self.rows) and np.array_equal(item["outer_labels"], self.outer_labels)):
                raise ValueError(f"{target}/{model}/k{k}/draw{draw}: chunks disagree on the subset")


def load_record_set(
    target: str, model: str, k: int, draw: int, count: int, record_dir: Path, strict: bool = True
) -> tuple[RecordSet | None, str]:
    """The record set and its status: complete | missing | failed | stale.

    With ``strict=False`` a record written under an earlier protocol hash is still returned (status
    ``stale``); its out-of-fold and complement predictions remain valid, its held-out predictions may not.
    """
    chunk_count = len(fits.component_chunks(tuple(range(count))))
    payloads = []
    stale = False
    for chunk in range(chunk_count):
        path = fits.job_path(fits.Job(target, model, k, draw, chunk, (), ()), record_dir)
        payload = benchmark.load_shard(path)
        if payload is None:
            return None, "missing"
        if str(payload["protocol_hash"].item()) != fits.protocol_hash():
            if strict:
                return None, "stale"
            stale = True
        if str(payload["status"].item()) != "ok":
            return None, "failed"
        identity = (
            str(payload["target"].item()),
            str(payload["model"].item()),
            int(payload["k"]),
            int(payload["draw"]),
            int(payload["chunk"]),
        )
        if identity != (target, model, k, draw, chunk):
            raise ValueError(f"{path}: record identity {identity} does not match its path")
        payloads.append(payload)
    return RecordSet(target, model, k, draw, payloads, count), ("stale" if stale else "complete")


def correlations(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if len(observed) < 3 or np.ptp(predicted) <= 1e-12 or np.ptp(observed) <= 1e-12:
        return {"pearson": float("nan"), "kendall": float("nan")}
    return {
        "pearson": float(stats.pearsonr(predicted, observed).statistic),
        "kendall": float(stats.kendalltau(predicted, observed).statistic),
    }


def with_derived(row: dict[str, float]) -> dict[str, float]:
    """Regret relative to a random pick (comparable across evaluation-set sizes) and unsigned diagnostics."""
    random_regret = row.get("random_regret_at_1", float("nan"))
    row["regret_ratio"] = row["regret_at_1"] / random_regret if random_regret > 0 else float("nan")
    row["calibration_slope_error"] = abs(row["calibration_slope"] - 1.0) if "calibration_slope" in row else float("nan")
    row["selection_optimism_abs"] = abs(row["selection_optimism"])
    return row


def scored(observed: np.ndarray, predicted: np.ndarray, basin: np.ndarray, repeat_sd: float) -> dict[str, float]:
    row = benchmark.metric_row(observed, predicted, basin) | correlations(observed, predicted)
    row["rmse_over_repeat_sd"] = row["rmse"] / repeat_sd if np.isfinite(repeat_sd) and repeat_sd > 0 else float("nan")
    # Exact expectation of a uniformly random pick among the evaluation rows.
    row["random_regret_at_1"] = float(observed.mean() - observed.min())
    return with_derived(row)


def heldout_rows(
    panel: benchmark.BenchPanel, target: str, predicted: np.ndarray, bank: pd.DataFrame, common: dict[str, Any]
) -> list[dict[str, Any]]:
    """The benchmark's held-out selection rows (pooled, per source, run level) for one prediction vector."""
    _count_column, mean_column = benchmark.HELDOUT_TARGET_COLUMNS[target]
    measured = bank[mean_column].to_numpy(float)
    tolerance = benchmark.BASIN_TOLERANCE_SD * panel.repeat_sd.get(target, float("nan"))
    sources = bank["sources"].to_numpy(str)
    strata = [
        ("pooled", np.ones(len(bank), dtype=bool)),
        # The benchmark's frozen bank before the dose-response coordinates were registered.
        ("pooled_without_dose_response", sources != DOSE_RESPONSE_SOURCE),
    ] + [(source, sources == source) for source in bank["sources"].unique() if (sources == source).sum() >= MIN_STRATUM]
    rows = []
    for stratum, mask in strata:
        loss = measured[mask]
        guess = predicted[mask]
        order = np.argsort(guess, kind="stable")
        selected = int(order[0])
        ranks = stats.rankdata(loss, method="average")
        row = (
            common
            | {
                "evaluation": "heldout",
                "stratum": stratum,
                "fold": -1,
                "n": int(mask.sum()),
                "selected_coordinate_id": str(bank["coordinate_id"].to_numpy(str)[mask][selected]),
                "selected_measured_bpb": float(loss[selected]),
                "selected_percentile": float((ranks[selected] - 1) / max(mask.sum() - 1, 1)),
                "best_measured_bpb": float(loss.min()),
                "regret_at_1": float(loss[selected] - loss.min()),
                "top5_regret": float(loss[order[:5]].min() - loss.min()),
                "top10_regret": float(loss[order[:10]].min() - loss.min()),
                "rmse": float(np.sqrt(np.mean((guess - loss) ** 2))),
                "spearman": benchmark._safe_spearman(loss, guess),
                "selection_optimism": float(loss[selected] - guess[selected]),
                "basin_hit": bool(loss[selected] - loss.min() <= tolerance) if np.isfinite(tolerance) else float("nan"),
            }
            | correlations(loss, guess)
        )
        row.update(benchmark.random_ranking_expectations(loss, benchmark.TOP_K))
        rows.append(with_derived(row))
    runs = benchmark.heldout_runs_for(panel.name, target)
    joined = runs.merge(
        pd.DataFrame({"coordinate_id": bank["coordinate_id"], "prediction": predicted, "coordinate_mean": measured}),
        on="coordinate_id",
        how="inner",
    )
    if len(joined) >= MIN_STRATUM:
        run_values = joined[benchmark.HELDOUT_RUN_COLUMNS[target]].to_numpy(float)
        guess = joined["prediction"].to_numpy(float)
        rows.append(
            common
            | {
                "evaluation": "heldout",
                "stratum": "run_level",
                "fold": -1,
                "n": len(joined),
                "rmse": float(np.sqrt(np.mean((guess - run_values) ** 2))),
                "spearman": benchmark._safe_spearman(run_values, guess),
                "noise_floor_rmse": float(
                    np.sqrt(np.mean((joined["coordinate_mean"].to_numpy(float) - run_values) ** 2))
                ),
            }
            | correlations(run_values, guess)
        )
    return rows


def evaluate(panel: benchmark.BenchPanel, bank: pd.DataFrame, record: RecordSet) -> list[dict[str, Any]]:
    """Every metric row of one record set."""
    group = panel.group(record.target)
    weights = group.aggregation_weights
    repeat_sd = panel.repeat_sd.get(record.target, float("nan"))
    basin = np.zeros(panel.rows, dtype=bool)
    basin[panel.basin_rows(record.target)] = True
    common = {"target": record.target, "model": record.model, "k": record.k, "draw": record.draw}
    observed = group.aggregate[record.rows]
    predicted = record.oof @ weights
    rows: list[dict[str, Any]] = [
        common
        | {"evaluation": "oof", "stratum": "pooled", "fold": -1}
        | scored(observed, predicted, basin[record.rows], repeat_sd)
    ]
    fold_rows = []
    for fold in range(fits.OUTER_FOLDS):
        mask = record.outer_labels == fold
        fold_rows.append(
            common
            | {"evaluation": "oof", "stratum": "fold", "fold": fold}
            | scored(observed[mask], predicted[mask], basin[record.rows][mask], repeat_sd)
        )
    rows.extend(fold_rows)
    fold_frame = pd.DataFrame(fold_rows)
    rows.append(
        common
        | {"evaluation": "oof", "stratum": "fold_mean", "fold": -1, "n": int(fold_frame["n"].sum())}
        | {metric: float(fold_frame[metric].mean()) for metric in SUMMARY_METRICS if metric in fold_frame}
    )
    fitted = record.fit_prediction @ weights
    rows.append(
        common
        | {"evaluation": "in_sample", "stratum": "pooled", "fold": -1}
        | scored(observed, fitted, basin[record.rows], repeat_sd)
    )
    if len(record.complement_rows):
        complement_observed = group.aggregate[record.complement_rows]
        complement_predicted = record.complement_prediction @ weights
        rows.append(
            common
            | {"evaluation": "complement", "stratum": "pooled", "fold": -1}
            | scored(complement_observed, complement_predicted, basin[record.complement_rows], repeat_sd)
        )
    rows.extend(heldout_rows(panel, record.target, record.heldout_prediction @ weights, bank, common))
    component_rows = []
    for index, component in enumerate(group.components):
        component_sd = panel.component_repeat_sd.get(component, float("nan"))
        component_rows.append(
            common
            | {"evaluation": "oof_component", "stratum": component, "fold": -1}
            | scored(group.outcomes[record.rows, index], record.oof[:, index], basin[record.rows], component_sd)
        )
    rows.extend(component_rows)
    component_frame = pd.DataFrame(component_rows)
    rows.append(
        common
        | {"evaluation": "oof_component", "stratum": "component_mean", "fold": -1, "n": record.k}
        | {metric: float(component_frame[metric].mean()) for metric in SUMMARY_METRICS if metric in component_frame}
    )
    for row in rows:
        row["elapsed"] = record.elapsed
    return rows


def hyperparameter_rows(record: RecordSet, components: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for fit in range(record.ridge.shape[0]):
        for index, component in enumerate(components):
            diagnostics = json.loads(str(record.diagnostics_json[fit, index]))
            rows.append(
                {
                    "target": record.target,
                    "model": record.model,
                    "k": record.k,
                    "draw": record.draw,
                    "fit": fit,
                    "component": component,
                    "ridge": float(record.ridge[fit, index]),
                    "shape_json": str(record.shape_json[fit, index]),
                    "inner_cv_rmse": diagnostics.get("inner_cv_rmse", float("nan")),
                    "boundary_hits": diagnostics.get("boundary_hits", float("nan")),
                    "fitted_dof": diagnostics.get("fitted_dof", float("nan")),
                    "huber_loss": diagnostics.get("huber_loss", float("nan")),
                }
            )
    return rows


def collect(record_dir: Path, draws: int, sizes: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Metric rows, hyperparameter rows and completeness for every requested (target, model, k, draw)."""
    panel = benchmark.load_panel(fits.PANEL)
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    for target in fits.TARGETS:
        components = panel.group(target).components
        bank, _features = benchmark.heldout_features(panel, target)
        for model in fits.MODEL_KEYS:
            for k in sizes:
                for draw in range(draws):
                    record, status = load_record_set(target, model, k, draw, len(components), record_dir)
                    completeness.append({"target": target, "model": model, "k": k, "draw": draw, "status": status})
                    if record is None:
                        continue
                    metric_rows.extend(evaluate(panel, bank, record))
                    parameter_rows.extend(hyperparameter_rows(record, components))
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows), pd.DataFrame(completeness)


def interval_rows(values: np.ndarray, generator: np.random.Generator) -> dict[str, float]:
    """Mean, spread and two 95% intervals (t and draw-level percentile bootstrap) of one metric over draws."""
    missing = int((~np.isfinite(values)).sum())
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return {"n_draws": 0, "n_missing": missing}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if count > 1 else float("nan")
    half = float(stats.t.ppf(0.975, count - 1) * sd / np.sqrt(count)) if count > 1 else float("nan")
    resampled = values[generator.integers(0, count, size=(BOOTSTRAP_DRAWS, count))].mean(axis=1) if count > 1 else None
    return {
        "n_draws": count,
        "n_missing": missing,
        "mean": mean,
        "sd": sd,
        "se": sd / np.sqrt(count) if count > 1 else float("nan"),
        "t_low": mean - half,
        "t_high": mean + half,
        "boot_low": float(np.percentile(resampled, 2.5)) if resampled is not None else float("nan"),
        "boot_high": float(np.percentile(resampled, 97.5)) if resampled is not None else float("nan"),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(target, model, evaluation, stratum, k) summaries and paired WSPU minus OLMix differences."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    keys = ["target", "evaluation", "stratum", "fold", "k"]
    summary_rows = []
    paired_rows = []
    present = [metric for metric in SUMMARY_METRICS if metric in metrics.columns]
    for identity, group in metrics.groupby(keys, sort=True):
        base = dict(zip(keys, identity, strict=True))
        by_model = {model: frame.set_index("draw") for model, frame in group.groupby("model")}
        for model, frame in by_model.items():
            for metric in present:
                summary_rows.append(
                    base | {"model": model, "metric": metric} | interval_rows(frame[metric].to_numpy(float), generator)
                )
        if set(by_model) >= {"wspu", "olmix"}:
            shared = by_model["wspu"].index.intersection(by_model["olmix"].index)
            for metric in present:
                wspu = by_model["wspu"].loc[shared, metric].to_numpy(float)
                olmix = by_model["olmix"].loc[shared, metric].to_numpy(float)
                difference = wspu - olmix
                finite = np.isfinite(difference)
                better = (difference[finite] < 0) if metric in LOWER_IS_BETTER else (difference[finite] > 0)
                fraction = float(better.mean()) if finite.any() and metric not in UNDIRECTED else float("nan")
                paired_rows.append(
                    base | {"metric": metric, "wspu_better_fraction": fraction} | interval_rows(difference, generator)
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


EFFICIENCY_CELLS = (
    ("oof", "pooled", "spearman"),
    ("oof", "pooled", "rmse"),
    ("complement", "pooled", "spearman"),
    ("heldout", "pooled", "spearman"),
    ("heldout", "pooled", "regret_at_1"),
    ("heldout", "pooled", "top5_regret"),
)


def data_efficiency(summary: pd.DataFrame) -> pd.DataFrame:
    """Smallest k at which one model's mean curve reaches the other's value at the largest k.

    ``first_k`` is the first crossing; ``sustained_k`` is the smallest k from which the curve stays at or
    beyond the reference for every larger k. Both are in fitted runs; NaN means the curve never gets there.
    """
    rows = []
    for target in fits.TARGETS:
        for evaluation, stratum, metric in EFFICIENCY_CELLS:
            curves = {}
            for model in fits.MODEL_KEYS:
                frame = summary[
                    summary["target"].eq(target)
                    & summary["model"].eq(model)
                    & summary["evaluation"].eq(evaluation)
                    & summary["stratum"].eq(stratum)
                    & summary["fold"].eq(-1)
                    & summary["metric"].eq(metric)
                ].sort_values("k")
                curves[model] = frame.set_index("k")["mean"].dropna()
            if any(curve.empty for curve in curves.values()):
                continue
            lower = metric in LOWER_IS_BETTER
            for model, other in (("wspu", "olmix"), ("olmix", "wspu")):
                reference_k = int(curves[other].index.max())
                reference = float(curves[other].loc[reference_k])
                curve = curves[model]
                reached = (curve <= reference) if lower else (curve >= reference)
                first = int(reached[reached].index.min()) if reached.any() else float("nan")
                sustained = float("nan")
                for k in curve.index:
                    if bool(reached.loc[k:].all()):
                        sustained = int(k)
                        break
                rows.append(
                    {
                        "target": target,
                        "evaluation": evaluation,
                        "stratum": stratum,
                        "metric": metric,
                        "model": model,
                        "reference_model": other,
                        "reference_k": reference_k,
                        "reference_value": reference,
                        "first_k": first,
                        "sustained_k": sustained,
                        "value_at_reference_k": (
                            float(curve.loc[reference_k]) if reference_k in curve.index else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_report(path: Path, summary: pd.DataFrame, paired: pd.DataFrame, completeness: pd.DataFrame) -> None:
    lines = [
        "# Learning curves: WSPU vs OLMix on the Delphi 3e18 panel",
        "",
        "Each draw resamples the fitted runs without replacement and re-blocks the folds with the benchmark's "
        "repeat seeds; intervals are over draws (t and draw-level bootstrap). At k = 280 every draw uses the "
        "same 280 runs, so the out-of-fold spread there is fold-seed spread only (the benchmark's repeats) "
        "and the held-out and complement fits are identical across draws (the benchmark's held-out fit). "
        "Out-of-fold and complement regret are measured against the minimum of the evaluation rows, which "
        "improves with k; `regret_ratio` divides by the random-pick expectation on the same rows. The held-out "
        "bank is the only evaluation set that is fixed across k.",
        "",
    ]
    counts = completeness.groupby(["target", "model", "status"]).size().unstack(fill_value=0)
    lines += ["## Completeness", "", counts.to_markdown(), ""]
    for target in fits.TARGETS:
        for evaluation, stratum, metric in (
            ("oof", "pooled", "spearman"),
            ("oof", "pooled", "rmse"),
            ("oof", "pooled", "regret_at_1"),
            ("oof", "pooled", "regret_ratio"),
            ("complement", "pooled", "spearman"),
            ("complement", "pooled", "regret_ratio"),
            ("heldout", "pooled", "spearman"),
            ("heldout", "pooled", "regret_at_1"),
            ("heldout", "pooled", "top5_regret"),
        ):
            table = summary[
                summary["target"].eq(target)
                & summary["evaluation"].eq(evaluation)
                & summary["stratum"].eq(stratum)
                & summary["metric"].eq(metric)
            ]
            if table.empty:
                continue
            pivot = table.pivot_table(index="k", columns="model", values=["mean", "t_low", "t_high", "n_draws"])
            diff = paired[
                paired["target"].eq(target)
                & paired["evaluation"].eq(evaluation)
                & paired["stratum"].eq(stratum)
                & paired["metric"].eq(metric)
            ].set_index("k")
            frame = pd.DataFrame(
                {
                    "draws": pivot[("n_draws", "wspu")] if ("n_draws", "wspu") in pivot else np.nan,
                    "wspu": pivot[("mean", "wspu")] if ("mean", "wspu") in pivot else np.nan,
                    "wspu_95": (
                        pivot[("t_low", "wspu")].map("{:.3f}".format)
                        + ".."
                        + pivot[("t_high", "wspu")].map("{:.3f}".format)
                        if ("t_low", "wspu") in pivot
                        else ""
                    ),
                    "olmix": pivot[("mean", "olmix")] if ("mean", "olmix") in pivot else np.nan,
                    "olmix_95": (
                        pivot[("t_low", "olmix")].map("{:.3f}".format)
                        + ".."
                        + pivot[("t_high", "olmix")].map("{:.3f}".format)
                        if ("t_low", "olmix") in pivot
                        else ""
                    ),
                    "wspu_minus_olmix": diff["mean"] if not diff.empty else np.nan,
                    "paired_95": (
                        diff["t_low"].map("{:.3f}".format) + ".." + diff["t_high"].map("{:.3f}".format)
                        if not diff.empty
                        else ""
                    ),
                    "wspu_better": diff["wspu_better_fraction"] if not diff.empty else np.nan,
                }
            )
            lines += [f"## {target}: {evaluation}/{stratum} {metric}", "", frame.to_markdown(floatfmt=".4f"), ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-dir", type=Path, default=fits.RECORD_DIR)
    parser.add_argument("--output-dir", type=Path, default=fits.OUTPUT_DIR)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(fits.SUBSET_SIZES))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics, parameters, completeness = collect(args.record_dir, args.draws, tuple(args.sizes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completeness.to_csv(args.output_dir / COMPLETENESS, index=False)
    LOGGER.info("completeness:\n%s", completeness.groupby(["model", "status"]).size().to_string())
    if metrics.empty:
        LOGGER.warning("no complete record sets")
        return
    metrics.to_csv(args.output_dir / METRICS_LONG, index=False)
    parameters.to_csv(args.output_dir / HYPERPARAMETERS, index=False)
    summary, paired = summarize(metrics)
    summary.to_csv(args.output_dir / SUMMARY, index=False)
    paired.to_csv(args.output_dir / PAIRED_SUMMARY, index=False)
    data_efficiency(summary).to_csv(args.output_dir / EFFICIENCY, index=False)
    write_report(args.output_dir / REPORT, summary, paired, completeness)
    LOGGER.info("wrote %s", args.output_dir)


if __name__ == "__main__":
    main()
