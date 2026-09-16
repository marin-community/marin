# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bank regret of the saved learning-curve fits on the expanded retrospective bank.

The learning-curve records of ``learning_curve_mariner_fits_20260908`` store every full-subset head: shape, floor,
kappa, link and NNLS coefficients for MARINER, the quadratic and the spline, and log_c with its coefficients for
Olmix. This script rebuilds those heads, predicts every coordinate of a bank registry (the frozen one or
``expand_heldout_bank_20260913``'s), checks that the coordinates shared with the frozen registry reproduce the
stored predictions, and scores the bank with the study's own held-out metrics. RegMix has no stored boosters; it
is refit with the trees and leaves its bank pass selected (``learning_curve_regmix_bank_20260912``), at seed 42 on
one thread, and checked the same way. ``regmix_official`` instead fits the released notebook cell on each subset
(``regmix_official_oof_20260913``: raw weights, up to 1,000 rounds, early stopping after three rounds on a held-out
third of the subset's training runs, anchor always in training), which is the RegMix drawn in the paper. Besides the
study's strata, the rows carry ``frozen_bank`` (coordinates of the frozen registry) and ``validation_cohort``
(coordinates added by the validation launches).

Writes ``metrics_long.csv``, ``summary.csv``, ``reproduction.csv`` and the regret figure under
``learning_curve_mariner_delphi_3e18_20260908/<bank>_bank_20260913``.

Progress (units done, elapsed, projected total) streams to stdout and every scored unit is appended to
``units.jsonl`` immediately, so a killed run resumes where it stopped.

usage: LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 uv run --offline --no-sync --with lightgbm python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_expanded_bank_20260913 \\
    --bank expanded [--workers 8] > run.log 2>&1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_metrics_20260905 as metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_regmix_bank_20260912 as regmix_bank,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_learning_curve_20260905 as plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    regmix_official_oof_20260913 as official,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

BANK_DIRS = {
    "frozen": benchmark.HELDOUT_DIR,
    "expanded": SCRIPT_DIR / "reference_outputs" / "single_phase_heldout_benchmark_expanded_20260913",
}
FROZEN_DIR = benchmark.HELDOUT_DIR
DRAWS = 10
REGMIX_MODEL = "regmix"  # the harness's inner-fold-tuned trees of the bank pass
REGMIX_OFFICIAL = "regmix_official"  # the released notebook cell, the RegMix of the paper's figures
STORED_MODELS = ("mariner", "olmix", "quadratic", "spline")
REFIT_MODELS = (REGMIX_MODEL, REGMIX_OFFICIAL)
DEFAULT_MODELS = ("mariner", "olmix", REGMIX_OFFICIAL)
ANCHOR_MIXTURES = 1  # the study counts the pinned proportional anchor as a mixture
REPRODUCTION_TOLERANCE = 1e-8
TUNED_TREES_COLOR = "#E69F00"


def use_bank(name: str) -> Path:
    """Point the benchmark's registry readers at one bank directory."""
    benchmark.HELDOUT_DIR = BANK_DIRS[name]
    benchmark.heldout_registry.cache_clear()
    benchmark.heldout_runs_for.cache_clear()
    return BANK_DIRS[name]


def frozen_positions(bank: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``bank`` that are frozen coordinates, and their positions in the frozen bank of ``target``."""
    coordinates = pd.read_csv(FROZEN_DIR / "heldout_coordinates.csv")
    count_column, _ = benchmark.HELDOUT_TARGET_COLUMNS[target]
    frozen = coordinates[coordinates["panel"].eq(fits.PANEL) & coordinates[count_column].fillna(0).gt(0)].reset_index(
        drop=True
    )
    index = {coordinate: position for position, coordinate in enumerate(frozen["coordinate_id"])}
    rows = np.asarray([i for i, coordinate in enumerate(bank["coordinate_id"]) if coordinate in index], dtype=int)
    return rows, np.asarray([index[bank["coordinate_id"].iloc[i]] for i in rows], dtype=int)


def stored_heads(target: str, model: str, k: int, draw: int, count: int) -> dict[str, Any]:
    """Full-subset heads of one record set in component order, plus its stored bank predictions."""
    chunks = len(fits.component_chunks(tuple(range(count))))
    order, shape, diagnostics, ridge, intercept, coefficients, heldout = [], [], [], [], [], [], []
    for chunk in range(chunks):
        payload = benchmark.load_shard(fits.job_path(fits.Job(target, model, k, draw, chunk, (), ()), fits.RECORD_DIR))
        if payload is None or str(payload["status"].item()) != "ok":
            raise FileNotFoundError(f"{target}/{model}/k{k}/draw{draw}/chunk{chunk}")
        order.extend(int(i) for i in payload["component_indices"])
        for column in range(len(payload["component_indices"])):
            shape.append(json.loads(str(payload["shape_json"][fits.FULL_FIT, column])))
            diagnostics.append(json.loads(str(payload["diagnostics_json"][fits.FULL_FIT, column])))
            ridge.append(float(payload["ridge"][fits.FULL_FIT, column]))
            intercept.append(float(payload["intercept"][fits.FULL_FIT, column]))
            vector = payload["coefficients"][fits.FULL_FIT, column]
            coefficients.append(vector[np.isfinite(vector)])
        heldout.append(payload["heldout_prediction"])
    inverse = np.argsort(order)
    return {
        "shape": [shape[i] for i in inverse],
        "diagnostics": [diagnostics[i] for i in inverse],
        "ridge": [ridge[i] for i in inverse],
        "intercept": [intercept[i] for i in inverse],
        "coefficients": [coefficients[i] for i in inverse],
        "heldout": np.concatenate(heldout, axis=1)[:, inverse],
    }


def stored_predictions(
    target: str, model: str, k: int, draw: int, panel: benchmark.BenchPanel, query: models.Features
) -> tuple[np.ndarray, np.ndarray]:
    """Bank predictions of one record set's stored full-subset heads, and the predictions it recorded."""
    group = panel.group(target)
    heads = stored_heads(target, model, k, draw, len(group.components))
    entry = registry.ENTRY_BY_ID[fits.MODEL_IDS[model]]
    base = registry.apply_transform(panel.features, entry)
    query_base = registry.apply_transform(query, entry)
    rows = np.arange(len(query.weights))
    predictions = np.empty((len(rows), len(group.components)))
    if model in fits.FLOORED_MODELS:
        for column, name in enumerate(group.components):
            fitting = dataclasses.replace(base, component=name)
            querying = dataclasses.replace(query_base, component=name)
            head_model = entry.build(fitting)  # the spline binds its knots to the fitting features here
            diagnostics = heads["diagnostics"][column]
            coefficients = np.asarray(heads["coefficients"][column], dtype=float)
            head = models.FittedHead(
                intercept=heads["intercept"][column],
                coefficients=coefficients,
                floor=float(diagnostics["floor"]),
                active=int((coefficients != 0).sum()),
                kappa=float(diagnostics["kappa"]),
                converged=bool(diagnostics["converged"]),
            )
            fitted = models.Fitted(heads["shape"][column], heads["ridge"][column], head, diagnostics)
            predictions[:, column] = head_model.predict(fitted, querying, rows)
        return predictions, heads["heldout"]
    fitting = dataclasses.replace(base, component=group.components[0])
    querying = dataclasses.replace(query_base, component=group.components[0])
    head_model = entry.build(fitting)
    for column in range(len(group.components)):
        diagnostics = heads["diagnostics"][column]
        head = olmix_loglinear.OlmixLoglinearFit(
            log_c=heads["intercept"][column],
            coefficients=tuple(float(v) for v in heads["coefficients"][column]),
            huber_loss=float(diagnostics["huber_loss"]),
        )
        fitted = models.Fitted(heads["shape"][column], heads["ridge"][column], head, diagnostics)
        predictions[:, column] = head_model.predict(fitted, querying, rows)
    return predictions, heads["heldout"]


def regmix_predictions(
    target: str, k: int, draw: int, panel: benchmark.BenchPanel, query: models.Features
) -> tuple[np.ndarray, np.ndarray]:
    """Refit RegMix's trees with the hyperparameters its bank pass selected and predict the bank."""
    group = panel.group(target)
    count = len(group.components)
    chunks = len(fits.component_chunks(tuple(range(count))))
    order, shapes, heldout = [], [], []
    for chunk in range(chunks):
        payload = benchmark.load_shard(regmix_bank.record_path(fits.Job(target, REGMIX_MODEL, k, draw, chunk, (), ())))
        if payload is None:
            raise FileNotFoundError(f"{target}/regmix/k{k}/draw{draw}/chunk{chunk}")
        order.extend(int(i) for i in payload["component_indices"])
        shapes.extend(json.loads(str(item)) for item in payload["shape_json"])
        heldout.append(payload["heldout_prediction"])
    inverse = np.argsort(order)
    shapes = [shapes[i] for i in inverse]
    design = fits.cached_subset_design(k, draw)
    train, _inner, _test = design.fit_rows(fits.FULL_FIT)
    model = models.LightGBMModel()
    matrix = panel.features.weights
    predictions = np.empty((len(query.weights), count))
    for column in range(count):
        make = model._make(int(shapes[column]["n_estimators"]), int(shapes[column]["num_leaves"]))
        head = models._fit_estimator_head(make, matrix[train], group.outcomes[train, column])
        predictions[:, column] = models._predict_estimator_head(head, query.weights)
    return predictions, np.concatenate(heldout, axis=1)[:, inverse]


def regmix_official_predictions(
    target: str, k: int, draw: int, panel: benchmark.BenchPanel, query: models.Features
) -> np.ndarray:
    """Fit the released RegMix cell on the subset's full-fit rows and predict the bank."""
    group = panel.group(target)
    design = fits.cached_subset_design(k, draw)
    train, _inner, _test = design.fit_rows(fits.FULL_FIT)
    training, validation = official.early_stopping_split(train, design.calibration)
    x = panel.features.weights
    predictions = np.empty((len(query.weights), len(group.components)))
    for column in range(len(group.components)):
        y = group.outcomes[:, column]
        regressor = official.lightgbm.LGBMRegressor(**official.HYPER_PARAMS, num_threads=1)
        regressor.fit(
            x[training],
            y[training],
            eval_set=[(x[validation], y[validation])],
            eval_metric="l2",
            callbacks=[official.lightgbm.early_stopping(stopping_rounds=official.EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        predictions[:, column] = regressor.predict(query.weights)
    return predictions


def score_unit(bank_name: str, target: str, model: str, k: int, draw: int) -> tuple[list[dict[str, Any]], dict]:
    """Held-out metric rows of one (target, model, k, draw) on the bank, and its reproduction check."""
    use_bank(bank_name)
    metrics.select_study("mariner")
    panel = benchmark.load_panel(fits.PANEL)
    bank, query = benchmark.heldout_features(panel, target)
    started = time.monotonic()
    rows, positions = frozen_positions(bank, target)
    if model == REGMIX_OFFICIAL:
        predictions = regmix_official_predictions(target, k, draw, panel, query)
        difference = float("nan")  # nothing stored to reproduce
    else:
        if model == REGMIX_MODEL:
            predictions, stored = regmix_predictions(target, k, draw, panel, query)
        else:
            predictions, stored = stored_predictions(target, model, k, draw, panel, query)
        difference = float(np.abs(predictions[rows] - stored[positions]).max())
    weights = panel.group(target).aggregation_weights
    objective = predictions @ weights
    common = {"target": target, "model": model, "k": k, "draw": draw}
    metric_rows = metrics.heldout_rows(panel, target, objective, bank, common)
    frozen_mask = np.zeros(len(bank), dtype=bool)
    frozen_mask[rows] = True
    for stratum, mask in (("frozen_bank", frozen_mask), ("validation_cohort", ~frozen_mask)):
        if mask.sum() < metrics.MIN_STRATUM:
            continue
        subset = bank[mask].reset_index(drop=True)
        pooled = [
            row
            for row in metrics.heldout_rows(panel, target, objective[mask], subset, common)
            if row["stratum"] == "pooled"
        ]
        for row in pooled:
            row["stratum"] = stratum
        metric_rows.extend(pooled)
    check = common | {
        "bank": bank_name,
        "coordinates": len(bank),
        "frozen_coordinates": int(frozen_mask.sum()),
        "max_abs_difference": difference,
        "seconds": time.monotonic() - started,
    }
    return metric_rows, check


def regret_figure(
    summary: pd.DataFrame, metrics_long: pd.DataFrame, shown: tuple[str, ...], output_dir: Path, stem: str
) -> plot.plt.Figure:
    """The regret figure for ``shown`` models; sizes where a model's predictions tied on every draw carry the
    random pick's regret and are drawn as hollow diamonds on a dotted line, joined to its first ranking size."""
    saved_models = plot.REGRET_MODELS
    plot.REGRET_MODELS = shown
    try:
        tied = (
            metrics_long[metrics_long["stratum"].eq("pooled") & metrics_long["model"].isin(shown)]
            .groupby(["model", "target", "k"])["degenerate"]
            .all()
        )
        tied_keys = {key for key, flag in tied.items() if flag}
        regret_rows = summary["metric"].eq("regret_at_1") & summary["evaluation"].eq("heldout")
        tied_mask = [
            regret_rows.loc[index] and (row["model"], row["target"], row["k"]) in tied_keys
            for index, row in summary.iterrows()
        ]
        tied_summary = summary[tied_mask]
        ranked_summary = summary.drop(tied_summary.index)
        figure, points = plot.build_regret_figure(ranked_summary, metrics_long, 5.5, 1.8, True, "t")
        for axis, target in zip(figure.axes[: len(fits.TARGETS)], fits.TARGETS, strict=True):
            for model in shown:
                part = tied_summary[
                    tied_summary["target"].eq(target)
                    & tied_summary["model"].eq(model)
                    & tied_summary["stratum"].eq("pooled")
                ].sort_values("k")
                if not len(part):
                    continue
                first_ranked = points[
                    points["target"].eq(target) & points["model"].eq(model) & points["k"].gt(part["k"].max())
                ].sort_values("k")
                xs = [*part["mixtures"], *first_ranked["mixtures"].iloc[:1]]
                ys = [*part["mean"], *first_ranked["center"].iloc[:1]]
                color = plot.MODEL_COLORS[model]
                axis.plot(xs, ys, linestyle=":", color=color, linewidth=1.0, zorder=4)
                axis.plot(
                    part["mixtures"],
                    part["mean"],
                    linestyle="none",
                    marker="D",
                    markersize=3.0,
                    markerfacecolor=plot.PAPER,
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    zorder=5,
                )
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"{stem}.{extension}", dpi=plot.DPI)
        points.to_csv(output_dir / f"{stem}_points.csv", index=False)
        return figure
    finally:
        plot.REGRET_MODELS = saved_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", choices=tuple(BANK_DIRS), default="expanded")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=(*STORED_MODELS, *REFIT_MODELS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()
    if cpu_count(only_physical_cores=True) != 1:
        raise ValueError("Set LOKY_MAX_CPU_COUNT=1 so LightGBM's sklearn wrapper fits on one thread per worker.")
    output_dir = fits.OUTPUT_DIR / f"{args.bank}_bank_20260913"
    output_dir.mkdir(parents=True, exist_ok=True)
    units = [
        (args.bank, target, model, k, draw)
        for target in fits.TARGETS
        for model in args.models
        for k in fits.SUBSET_SIZES
        for draw in range(args.draws)
    ]
    # Slow RegMix suite refits first so the projection settles early; finished units are reused.
    units.sort(key=lambda unit: (unit[2] not in REFIT_MODELS, unit[1] != "table9"))
    partial_path = output_dir / "units.jsonl"
    done: dict[tuple, dict] = {}
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            item = json.loads(line)
            done[(item["check"]["target"], item["check"]["model"], item["check"]["k"], item["check"]["draw"])] = item
    pending = [unit for unit in units if unit[1:] not in done]
    print(f"{len(done)} units cached, {len(pending)} pending on {args.workers} workers", flush=True)
    started = time.monotonic()
    with partial_path.open("a") as handle:
        stream = Parallel(n_jobs=args.workers, backend="loky", return_as="generator")(
            delayed(score_unit)(*unit) for unit in pending
        )
        for index, (rows, check) in enumerate(stream, start=1):
            item = {"rows": rows, "check": check}
            handle.write(json.dumps(item, default=float) + "\n")
            handle.flush()
            done[(check["target"], check["model"], check["k"], check["draw"])] = item
            if index % 10 == 0 or index == len(pending):
                elapsed = (time.monotonic() - started) / 60
                print(
                    f"{index}/{len(pending)} units done in {elapsed:.1f} min; "
                    f"projected total {elapsed * len(pending) / index:.1f} min",
                    flush=True,
                )
    results = [(item["rows"], item["check"]) for unit in units for item in [done[unit[1:]]]]
    metric_rows = [row for rows, _check in results for row in rows]
    checks = pd.DataFrame([check for _rows, check in results])
    metrics_long = pd.DataFrame(metric_rows)
    # A fit whose bank predictions are all equal ranks nothing: the released recipe cannot split its two-thirds
    # training portion below about 60 runs (LightGBM's default 20-run leaf minimum). Every coordinate then ties, so
    # the selection is a random pick; such rows carry the random pick's expected regret and are drawn hollow.
    degenerate = metrics_long["spearman"].isna() & metrics_long["n"].ge(metrics.MIN_STRATUM)
    metrics_long["degenerate"] = degenerate
    metrics_long.loc[degenerate, "regret_at_1"] = metrics_long.loc[degenerate, "random_regret_at_1"]
    metrics_long.loc[degenerate, "top5_regret"] = metrics_long.loc[degenerate, "random_best_of_5_regret"]
    for column in ("top10_regret", "selected_percentile", "selection_optimism"):
        metrics_long.loc[degenerate, column] = np.nan
    print(
        "degenerate units (constant predictions):",
        metrics_long[degenerate].groupby(["model", "k"]).size().to_dict() or "none",
    )
    metrics.select_study("mariner")
    summary, paired = metrics.summarize(metrics_long)
    summary = summary[summary["n_draws"].gt(0)].reset_index(drop=True)
    summary["mixtures"] = summary["k"] + ANCHOR_MIXTURES
    metrics_long.to_csv(output_dir / metrics.METRICS_LONG, index=False)
    summary.to_csv(output_dir / metrics.SUMMARY, index=False)
    paired.to_csv(output_dir / "paired_summary.csv", index=False)
    checks.to_csv(output_dir / "reproduction.csv", index=False)
    worst = checks.groupby("model")["max_abs_difference"].max()
    print("reproduction of the stored frozen-bank predictions, max |difference| by model:")
    print(worst.to_string())
    if (worst[[m for m in worst.index if m not in REFIT_MODELS]] > REPRODUCTION_TOLERANCE).any():
        raise ValueError("stored heads do not reproduce the recorded bank predictions")
    plot.plt.rcParams.update(plot.PLOT_STYLE)
    # The plot names the third curve "RegMix": the released recipe in the main figure, the tuned trees in the variant.
    for variant, regmix_key in (("", REGMIX_OFFICIAL), ("_tuned", REGMIX_MODEL)):
        if regmix_key not in set(summary["model"]):
            continue
        shown_summary = summary[summary["model"].isin(["mariner", "olmix", regmix_key])].copy()
        shown_metrics = metrics_long[metrics_long["model"].isin(["mariner", "olmix", regmix_key])].copy()
        shown_summary["model"] = shown_summary["model"].replace({regmix_key: REGMIX_MODEL})
        shown_metrics["model"] = shown_metrics["model"].replace({regmix_key: REGMIX_MODEL})
        regret_figure(
            shown_summary,
            shown_metrics,
            ("mariner", "olmix", REGMIX_MODEL),
            output_dir,
            f"learning_curve_regret{variant}_logx",
        )
    if {REGMIX_OFFICIAL, REGMIX_MODEL} <= set(summary["model"]):
        # Both RegMix variants beside MARINER, for the appendix comparison.
        labels = {REGMIX_OFFICIAL: "RegMix, released recipe", REGMIX_MODEL: "RegMix, tuned trees"}
        colors = {REGMIX_OFFICIAL: plot.MODEL_COLORS[REGMIX_MODEL], REGMIX_MODEL: TUNED_TREES_COLOR}
        saved = {key: (plot.MODEL_LABELS.get(key), plot.MODEL_COLORS.get(key)) for key in labels}
        plot.MODEL_LABELS.update(labels)
        plot.MODEL_COLORS.update(colors)
        try:
            regret_figure(
                summary,
                metrics_long,
                ("mariner", REGMIX_OFFICIAL, REGMIX_MODEL),
                output_dir,
                "learning_curve_regret_regmix_variants_logx",
            )
        finally:
            for key, (label, color) in saved.items():
                if label is None:
                    plot.MODEL_LABELS.pop(key, None)
                    plot.MODEL_COLORS.pop(key, None)
                else:
                    plot.MODEL_LABELS[key], plot.MODEL_COLORS[key] = label, color
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
