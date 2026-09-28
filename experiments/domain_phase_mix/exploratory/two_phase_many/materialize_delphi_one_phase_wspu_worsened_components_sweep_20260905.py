# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "joblib>=1.4",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.5",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Materialize WSPU epoch-cap optima for the three Uncheatable components the full optimum worsened.

The Uncheatable-optimized WSPU mixture (epoch cap 6) improves GitHub and arXiv but worsens
bbc_news, ao3_english and wikipedia_english. This script re-targets the same reconstructed
component fits at a byte-weighted aggregate of only those three components, sweeps the epoch
cap from 2 upward and stops reporting movement once the runtime-grid optimum stops changing.
It reuses the fits, the exact dynamic-programming optimizer and the outer-fold sensitivity
machinery of the full sweep. It does not launch training.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902 as sweep,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

TARGET = "uncheatable"
WORSENED = (
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
CAPS = tuple(range(2, 21))
LAUNCH_CAPS = (4, 6, 7)
LAUNCH_TARGET = "uncheatable_worsened"
LAUNCH_TARGET_LABEL = "Uncheatable worsened trio"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_wspu_worsened_components_sweep_20260905"
FULL_SWEEP_WEIGHTS = sweep.OUTPUT_DIR / "candidate_weights.csv"
HELDOUT_COMPONENTS = (
    SCRIPT_DIR / "reference_outputs" / "single_phase_heldout_benchmark_20260902" / "heldout_components.csv"
)
FULL_OPTIMUM_ID = "wspu_uncheatable_cap06"
FULL_OPTIMUM_ROW = "delphi_recent::weibull_softplus_unscaled_epoch_cap::wspu_uncheatable_cap06"
PROPORTIONAL_RUN = "singleavg_fit_000_baseline_proportional"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser.parse_args()


def short_name(component: str) -> str:
    return component.removeprefix("eval/uncheatable_eval/").removesuffix("/bpb")


def reconstruct_fits(panel: benchmark.BenchPanel, workers: int) -> list[sweep.ComponentFit]:
    entry = registry.ENTRY_BY_ID[sweep.MODEL_ID]
    transformed = registry.apply_transform(panel.features, entry)
    if transformed.cache_key != panel.features.cache_key:
        raise ValueError("The successor should use the unmodified true-inventory features")
    model = entry.build(transformed)
    if not isinstance(model, models.GridModel):
        raise TypeError(f"Expected GridModel, got {type(model).__name__}")
    group = panel.group(TARGET)
    metadata = [
        sweep.metadata_for_component(panel, TARGET, position, source)
        for position in range(len(group.components))
        for source in sweep.SOURCE_LABELS
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: sweep.reconstruct_component(item, panel, model), metadata))


def predictor_for(
    fits: list[sweep.ComponentFit],
    *,
    source: str,
    weights: dict[str, float],
    buckets: int,
    label: str,
) -> sweep.AggregatePredictor:
    selected = sorted(
        (fit for fit in fits if fit.metadata.source == source and fit.metadata.component in weights),
        key=lambda fit: fit.metadata.component_position,
    )
    if len(selected) != len(weights):
        raise ValueError(f"Missing component fits for {label}/{source}")
    reweighted = tuple(
        replace(fit, metadata=replace(fit.metadata, aggregation_weight=weights[fit.metadata.component]))
        for fit in selected
    )
    return sweep.AggregatePredictor(label, source, reweighted, buckets)


def component_prediction(fit: sweep.ComponentFit, exposures: np.ndarray, buckets: int) -> np.ndarray:
    values = np.atleast_2d(np.asarray(exposures, dtype=float))
    benefit = models.weibull_response(values, fit.rate, fit.power)
    harm = models.softplus_harm(values, fit.threshold)
    coefficients = fit.head.coefficients
    return fit.head.intercept - benefit @ coefficients[:buckets] + harm @ coefficients[buckets:]


def full_optimum_weights(panel: benchmark.BenchPanel) -> np.ndarray:
    frame = pd.read_csv(FULL_SWEEP_WEIGHTS)
    rows = frame.loc[frame["candidate_id"].eq(FULL_OPTIMUM_ID)].set_index("domain")
    missing = set(panel.buckets) - set(rows.index)
    if missing:
        raise ValueError(f"Full-sweep optimum is missing buckets: {sorted(missing)}")
    return rows.loc[list(panel.buckets), "weight"].to_numpy(float)


def observed_full_optimum_components() -> dict[str, float]:
    frame = pd.read_csv(HELDOUT_COMPONENTS)
    rows = frame.loc[frame["row_id"].eq(FULL_OPTIMUM_ROW) & frame["target"].eq(TARGET)]
    if rows.empty:
        raise ValueError("The full Uncheatable optimum has no component rows in the heldout benchmark")
    return dict(zip(rows["component"], rows["bpb"].astype(float), strict=True))


def tv(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    references: pd.DataFrame,
    sensitivity: pd.DataFrame,
    plateau_cap: int | None,
    worsened_weights: dict[str, float],
) -> None:
    final_cap = int(summary["epoch_cap"].max())
    plateau = summary.loc[summary["epoch_cap"].eq(plateau_cap)].iloc[0] if plateau_cap else summary.iloc[-1]
    predicted = references.pivot(index="component", columns="mixture", values="predicted_bpb")
    observed = references.pivot(index="component", columns="mixture", values="observed_bpb")
    objective = ", ".join(f"{short_name(name)} ({weight:.3f})" for name, weight in worsened_weights.items())
    at_proportional = predicted.loc["worsened_aggregate", "proportional"]
    at_full_optimum = predicted.loc["worsened_aggregate", "full_uncheatable_optimum_cap06"]
    full_at_proportional = predicted.loc["full_aggregate", "proportional"]
    summary_columns = [
        "epoch_cap",
        "predicted_worsened_aggregate_bpb",
        "outer_fold_prediction_sd",
        "predicted_bbc_news_bpb",
        "predicted_ao3_english_bpb",
        "predicted_wikipedia_english_bpb",
        "predicted_full_uncheatable_bpb",
        "tv_to_previous_cap",
        "tv_to_proportional",
        "fold_optimum_tv_median",
        "support_buckets",
        "largest_weight",
    ]
    weight_columns = [
        "domain",
        "weight",
        "proportional_weight",
        "full_uncheatable_optimum_weight",
        "materialized_epochs",
    ]
    plateau_value = float(plateau["predicted_worsened_aggregate_bpb"])
    first_value = float(summary.iloc[0]["predicted_worsened_aggregate_bpb"])
    full_value = float(plateau["predicted_full_uncheatable_bpb"])
    plateau_weights = weights.loc[weights["epoch_cap"].eq(plateau_cap)].sort_values("weight", ascending=False)
    lines = [
        "# WSPU epoch-cap sweep for the three worsened Uncheatable components",
        "",
        f"Objective: byte-weighted aggregate of {objective}. Same reconstructed component fits and exact",
        "runtime-grid optimizer as the full sweep. No training launched.",
        "",
        f"- The runtime-grid optimum stops moving at cap {plateau_cap}; caps {plateau_cap} to {final_cap}",
        "  share one mixture.",
        f"- Predicted three-component aggregate: {first_value:.4f} at cap 2, {plateau_value:.4f} at the plateau,",
        f"  {at_proportional:.4f} predicted at proportional, {at_full_optimum:.4f} predicted at the full",
        "  Uncheatable optimum.",
        f"- Predicted full Uncheatable aggregate at the plateau mixture: {full_value:.4f}",
        f"  (proportional predicted {full_at_proportional:.4f}).",
        f"- Outer-fold prediction SD at the plateau: {plateau['outer_fold_prediction_sd']:.4f}; fold optima sit TV",
        f"  {plateau['fold_optimum_tv_median']:.3f} (median) from the full-selection optimum.",
        f"- Plateau mixture: {int(plateau['support_buckets'])} supported buckets, largest {plateau['largest_bucket']}",
        f"  at {plateau['largest_weight']:.3f}, max materialized epochs {plateau['max_materialized_epoch']:.2f},",
        f"  TV {plateau['tv_to_proportional']:.3f} to proportional, {plateau['tv_to_full_uncheatable_optimum']:.3f}",
        "  to the full Uncheatable optimum.",
        "",
        "## Cap sweep",
        "",
        summary[summary_columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Predicted versus observed at the reference mixtures",
        "",
        pd.concat({"predicted": predicted, "observed": observed}, axis=1).to_markdown(floatfmt=".4f"),
        "",
        f"## Plateau mixture (cap {plateau_cap})",
        "",
        plateau_weights[weight_columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Outer-fold optima at the plateau cap",
        "",
        sensitivity.loc[sensitivity["epoch_cap"].eq(plateau_cap)].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = benchmark.load_panel(sweep.PANEL_ID)
    group = panel.group(TARGET)
    components = tuple(group.components)
    buckets = len(panel.buckets)
    inventory = panel.features.inventory
    byte_weights = dict(zip(components, group.aggregation_weights.astype(float), strict=True))
    worsened_total = sum(byte_weights[name] for name in WORSENED)
    worsened_weights = {name: byte_weights[name] / worsened_total for name in WORSENED}

    fits = reconstruct_fits(panel, args.workers)
    worsened = {
        source: predictor_for(
            fits, source=source, weights=worsened_weights, buckets=buckets, label="uncheatable_worsened"
        )
        for source in sweep.SOURCE_LABELS
    }
    full_aggregate = predictor_for(fits, source="full", weights=byte_weights, buckets=buckets, label=TARGET)
    full_fits = {fit.metadata.component: fit for fit in fits if fit.metadata.source == "full"}

    optima = [sweep.exact_runtime_optimum(worsened["full"], inventory, cap) for cap in CAPS]
    plateau_cap = None
    for previous, current in itertools.pairwise(optima):
        if np.array_equal(previous.counts, current.counts):
            plateau_cap = plateau_cap or previous.cap
        else:
            plateau_cap = None
    sensitivity = []
    if not args.skip_sensitivity:
        folds = [source for source in sweep.SOURCE_LABELS if source != "full"]
        tasks = [(worsened[source], inventory, cap) for source in folds for cap in CAPS]
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            sensitivity = list(pool.map(lambda item: sweep.exact_runtime_optimum(*item), tasks))
    sensitivity_lookup: dict[tuple[str, int], sweep.RuntimeOptimum] = {
        (item.source, item.cap): item for item in sensitivity
    }

    proportional = sweep.proportional_weights(panel)
    full_optimum = full_optimum_weights(panel)
    proportional_row = list(panel.runs).index(PROPORTIONAL_RUN)
    observed_proportional = dict(zip(components, group.outcomes[proportional_row].astype(float), strict=True))
    observed_full_optimum = observed_full_optimum_components()

    def predictions_at(weights: np.ndarray) -> dict[str, float]:
        exposures = (inventory * weights)[None, :]
        values = {
            short_name(name): float(component_prediction(full_fits[name], exposures, buckets)[0]) for name in components
        }
        values["worsened_aggregate"] = float(worsened["full"].predict(exposures)[0])
        values["full_aggregate"] = float(full_aggregate.predict(exposures)[0])
        return values

    summary_rows = []
    weight_rows = []
    sensitivity_rows = []
    previous_weights = None
    for optimum in optima:
        cap = optimum.cap
        weights = optimum.weights
        exposures = inventory * weights
        maximum = np.floor(np.minimum(1.0, cap / inventory) * sweep.MIXTURE_BLOCK_SIZE + 1e-12).astype(int)
        predicted = predictions_at(weights)
        fold_predictions = np.asarray(
            [worsened[source].predict(exposures[None, :])[0] for source in sweep.SOURCE_LABELS if source != "full"]
        )
        fold_tvs = [
            tv(sensitivity_lookup[source, cap].weights, weights)
            for source in sweep.SOURCE_LABELS
            if source != "full" and (source, cap) in sensitivity_lookup
        ]
        summary_rows.append(
            {
                "candidate_id": f"wspu_worsened_cap{cap:02d}",
                "epoch_cap": cap,
                "predicted_worsened_aggregate_bpb": optimum.prediction,
                "outer_fold_prediction_mean": float(fold_predictions.mean()),
                "outer_fold_prediction_sd": float(fold_predictions.std(ddof=1)),
                **{f"predicted_{short_name(name)}_bpb": predicted[short_name(name)] for name in components},
                "predicted_full_uncheatable_bpb": predicted["full_aggregate"],
                "tv_to_previous_cap": None if previous_weights is None else tv(weights, previous_weights),
                "tv_to_proportional": tv(weights, proportional),
                "tv_to_full_uncheatable_optimum": tv(weights, full_optimum),
                "fold_optimum_tv_median": float(np.median(fold_tvs)) if fold_tvs else None,
                "fold_optimum_tv_max": float(np.max(fold_tvs)) if fold_tvs else None,
                "max_materialized_epoch": float(exposures.max()),
                "cap_active_buckets": int((optimum.counts == maximum).sum()),
                "support_buckets": int(np.count_nonzero(optimum.counts)),
                "effective_buckets": sweep.effective_buckets(weights),
                "largest_bucket": str(panel.buckets[int(np.argmax(weights))]),
                "largest_weight": float(weights.max()),
                "one_exchange_improvement": sweep.one_exchange_improvement(worsened["full"], inventory, optimum),
                "at_plateau": bool(plateau_cap is not None and cap >= plateau_cap),
            }
        )
        for position, bucket in enumerate(panel.buckets):
            weight_rows.append(
                {
                    "candidate_id": f"wspu_worsened_cap{cap:02d}",
                    "target": LAUNCH_TARGET,
                    "target_label": LAUNCH_TARGET_LABEL,
                    "epoch_cap": cap,
                    "bucket_position": position,
                    "domain": bucket,
                    "runtime_count": int(optimum.counts[position]),
                    "weight": float(weights[position]),
                    "proportional_weight": float(proportional[position]),
                    "full_uncheatable_optimum_weight": float(full_optimum[position]),
                    "materialized_epochs": float(exposures[position]),
                    "cap_active": bool(optimum.counts[position] == maximum[position]),
                }
            )
        for source in sweep.SOURCE_LABELS:
            if (source, cap) in sensitivity_lookup:
                item = sensitivity_lookup[source, cap]
                sensitivity_rows.append(
                    {
                        "epoch_cap": cap,
                        "source": source,
                        "predicted_worsened_aggregate_bpb": item.prediction,
                        "tv_to_full_selection_optimum": tv(item.weights, weights),
                        "support_buckets": int(np.count_nonzero(item.counts)),
                    }
                )
        previous_weights = weights

    reference_rows = []
    for label, weights, observed in (
        ("proportional", proportional, observed_proportional),
        ("full_uncheatable_optimum_cap06", full_optimum, observed_full_optimum),
    ):
        predicted = predictions_at(weights)
        for name in components:
            reference_rows.append(
                {
                    "mixture": label,
                    "component": short_name(name),
                    "predicted_bpb": predicted[short_name(name)],
                    "observed_bpb": observed[name],
                    "byte_weight": byte_weights[name],
                    "in_worsened_aggregate": name in WORSENED,
                }
            )
        reference_rows.append(
            {
                "mixture": label,
                "component": "full_aggregate",
                "predicted_bpb": predicted["full_aggregate"],
                "observed_bpb": sum(byte_weights[name] * observed[name] for name in components),
                "byte_weight": 1.0,
                "in_worsened_aggregate": False,
            }
        )
        reference_rows.append(
            {
                "mixture": label,
                "component": "worsened_aggregate",
                "predicted_bpb": predicted["worsened_aggregate"],
                "observed_bpb": sum(worsened_weights[name] * observed[name] for name in WORSENED),
                "byte_weight": worsened_total,
                "in_worsened_aggregate": True,
            }
        )

    summary = pd.DataFrame(summary_rows)
    weights_frame = pd.DataFrame(weight_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    references = pd.DataFrame(reference_rows)
    launch_weights = weights_frame.loc[weights_frame["epoch_cap"].isin(LAUNCH_CAPS)].reset_index(drop=True)
    for name, frame in (
        ("candidate_summary.csv", summary),
        ("candidate_weights.csv", weights_frame),
        ("launch_candidate_weights.csv", launch_weights),
        ("sensitivity_optima.csv", sensitivity_frame),
        ("reference_predictions.csv", references),
    ):
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "model": sweep.MODEL_ID,
        "panel": sweep.PANEL_ID,
        "objective": "byte-weighted aggregate of the three Uncheatable components worsened by the full optimum",
        "components": {short_name(name): worsened_weights[name] for name in WORSENED},
        "caps": list(CAPS),
        "launch_caps": list(LAUNCH_CAPS),
        "plateau_cap": plateau_cap,
        "optimizer": "exact separable integer dynamic programming on the 1/2048 runtime grid",
        "inputs": {
            "full_sweep_candidate_weights": hashlib.sha256(FULL_SWEEP_WEIGHTS.read_bytes()).hexdigest(),
            "heldout_components": hashlib.sha256(HELDOUT_COMPONENTS.read_bytes()).hexdigest(),
            "benchmark_dir": str(sweep.BENCHMARK_DIR),
        },
        "note": "No training launched.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(args.output_dir, summary, weights_frame, references, sensitivity_frame, plateau_cap, worsened_weights)
    print(json.dumps({"plateau_cap": plateau_cap, "output": str(args.output_dir)}))
    columns = [
        "epoch_cap",
        "predicted_worsened_aggregate_bpb",
        "outer_fold_prediction_sd",
        "predicted_bbc_news_bpb",
        "predicted_ao3_english_bpb",
        "predicted_wikipedia_english_bpb",
        "predicted_full_uncheatable_bpb",
        "tv_to_previous_cap",
        "tv_to_proportional",
        "tv_to_full_uncheatable_optimum",
        "fold_optimum_tv_median",
        "support_buckets",
        "largest_bucket",
        "largest_weight",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
