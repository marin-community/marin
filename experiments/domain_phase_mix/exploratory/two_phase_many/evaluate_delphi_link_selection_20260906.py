# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the log-deficit links on the frozen Delphi selection benchmark, alone and under Codex's coupling.

Fits WSPU, WSPU with the bounded log-deficit link, and WSPU with the inner-CV-selected link on the frozen
280-row panel of `delphi_offline_selection_20260906` with its outer/inner partitions, predicts the frozen
held-out bank, and adds a composite method that applies the fixed cross-bucket product of
`audit_delphi_wspu_coupling_20260906` (kappa = 1) to the bounded-link heads. Everything is then scored with
`score_delphi_selection_20260906.score_predictions`, and the optima-stratum rows are tabulated next to the
reference rows for WSPU, DSP, OLMix and Codex's WSPU + coupling. No training or evaluation job is launched.

usage: uv run python evaluate_delphi_link_selection_20260906.py [--output-dir DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_delphi_wspu_coupling_20260906 as coupling,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    score_delphi_selection_20260906 as scorer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

REFERENCE = benchmark.DEFAULT_OUTPUT
FOLLOWUP = SCRIPT_DIR / "reference_outputs" / "delphi_coupling_followup_20260906"
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_link_selection_20260906"
WSPU = "weibull_softplus_unscaled"
LINK = "weibull_softplus_unscaled@log_deficit_bounded_link"
LINK_CV = "weibull_softplus_unscaled@link_by_cv"
FITTED_METHODS = (WSPU, LINK, LINK_CV)
COMPOSITE = "link_bounded_coupling_kappa_1"
COMPOSITE_KAPPA = 1.0
FOLDS = (-1, 0, 1, 2, 3, 4)
BUCKETS = 39
PARITY_TOLERANCE = 1e-8
DISPLAY = {
    WSPU: "WSPU (refit here)",
    LINK: "WSPU, bounded log-deficit link",
    LINK_CV: "WSPU, link chosen by inner CV",
    COMPOSITE: "Bounded link + Codex coupling (kappa 1)",
}
REFERENCE_DISPLAY = {
    WSPU: "WSPU (reference)",
    "dsp_total_exposure": "DSP (reference)",
    "olmix_loglinear_taskwise": "OLMix (reference)",
}
CODEX_DISPLAY = {"wspu_coupling_kappa_1": "WSPU + Codex coupling (kappa 1)"}


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "inputs").exists():
        shutil.copytree(REFERENCE / "inputs", output / "inputs")
        shutil.copy(REFERENCE / "input_hashes.json", output / "input_hashes.json")
    benchmark.verify_inputs(output)


def fit_all(output: Path, workers: int) -> None:
    fingerprint = benchmark.source_fingerprint(output)
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    tasks = [
        (method, target, component, fold, 0)
        for method in FITTED_METHODS
        for target in benchmark.TARGETS
        for component in range(data[f"{target}_outcomes"].shape[1])
        for fold in FOLDS
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        counts = Parallel(n_jobs=workers, verbose=5)(
            delayed(benchmark.fit_baseline)(output, *task, fingerprint) for task in tasks
        )
    print(pd.Series(counts).value_counts().to_dict(), flush=True)


def bucket_contributions(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Per-bucket linear-predictor contributions of the 78-column WSPU design, shape (rows, buckets)."""
    return matrix[:, :BUCKETS] * coefficients[None, :BUCKETS] + matrix[:, BUCKETS:] * coefficients[None, BUCKETS:]


def composite_fold(output: Path, target: str, fold: int) -> dict:
    """Bounded-link heads for one fold, then Codex's product over one-bucket-at-a-time link responses."""
    start = time.monotonic()
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
    feature = benchmark.feature_set(data, benchmark.PANEL, data["weights"], data["exposures"])
    query = benchmark.feature_set(data, f"{benchmark.PANEL}|frozen-bank", bank["weights"], bank["exposures"])
    entry = registry.ENTRY_BY_ID[LINK]
    feature = registry.apply_transform(feature, entry)
    query = registry.apply_transform(query, entry)
    components = data[f"{target}_components"]
    shards = [
        benchmark.read_npz(output / "baseline_shards" / LINK / target / f"r0_f{fold}_c{index}.npz")
        for index in range(len(components))
    ]
    train, test = shards[0]["train"], shards[0]["test"]
    anchor_weights = data["weights"][train].mean(axis=0, keepdims=True)
    anchor = dataclasses.replace(
        feature,
        weights=anchor_weights,
        exposures=anchor_weights * feature.inventory[None, :],
        label=f"{feature.label}|anchor",
    )
    outcomes = data[f"{target}_outcomes"]
    anchors, test_link, bank_link, test_deltas, bank_deltas = [], [], [], [], []
    for index, shard in enumerate(shards):
        if not np.array_equal(shard["train"], train) or not np.array_equal(shard["test"], test):
            raise ValueError("Bounded-link components have inconsistent split rows")
        model = entry.build(dataclasses.replace(feature, component=str(components[index])))
        if not isinstance(model, models.GridModel):
            raise ValueError("The bounded-link model is not the expected grid model")
        shape = json.loads(str(shard["shape_json"]))
        design = model.design(feature, shape)
        expected = tuple(f"bucket_signal:{i}" for i in range(BUCKETS)) + tuple(
            f"bucket_overexposure:{i}" for i in range(BUCKETS)
        )
        if design.names != expected:
            raise ValueError("The bounded-link model is not the expected 78-column design")
        spec = model.head_for(shape)
        head = models.fit_head(
            models.Design(design.values[train], design.ridge, design.names),
            outcomes[train, index],
            float(shard["ridge"]),
            spec,
        )
        anchor_parts = bucket_contributions(model.design(anchor, shape).values, head.coefficients)[0]
        anchor_value = float(models.link_inverse(head.intercept + anchor_parts.sum(), head.floor, spec, head.cap))
        anchors.append(anchor_value)
        for rows_features, rows, link_store, delta_store, parity in (
            (feature, test, test_link, test_deltas, shard["prediction"]),
            (query, np.arange(len(bank["weights"])), bank_link, bank_deltas, shard["bank_prediction"]),
        ):
            parts = bucket_contributions(model.design(rows_features, shape).values[rows], head.coefficients)
            full = models.link_inverse(head.intercept + parts.sum(axis=1), head.floor, spec, head.cap)
            if len(rows) and np.max(np.abs(full - parity)) > PARITY_TOLERANCE:
                raise ValueError(f"Bounded-link reconstruction parity failed: {target}/{fold}/{index}")
            # One bucket at a time off the anchor: linear predictor of the anchor with bucket b replaced.
            one_at_a_time = anchor_parts.sum() - anchor_parts[None, :] + parts
            single = models.link_inverse(head.intercept + one_at_a_time, head.floor, spec, head.cap)
            link_store.append(full)
            delta_store.append(single - anchor_value)
    anchors = np.array(anchors)
    weights = data[f"{target}_aggregation_weights"]
    result = {"test": test, "train": train, "elapsed": time.monotonic() - start}
    for name, link_store, delta_store in (
        ("prediction", test_link, test_deltas),
        ("bank_prediction", bank_link, bank_deltas),
    ):
        link_matrix = np.stack(link_store, axis=1) if len(link_store[0]) else np.empty((0, len(anchors)))
        deltas = np.stack(delta_store, axis=1) if len(delta_store[0]) else np.empty((0, len(anchors), BUCKETS))
        if len(deltas):
            coupled, _counts = coupling.coupled_values(anchors, deltas, COMPOSITE_KAPPA)
            additive, _zero = coupling.coupled_values(anchors, deltas, 0.0)
            result[f"{name}_link_macro"] = link_matrix @ weights
            result[f"{name}_additive_macro"] = additive @ weights
            result[name] = coupled @ weights
        else:
            result[f"{name}_link_macro"] = np.empty(0)
            result[f"{name}_additive_macro"] = np.empty(0)
            result[name] = np.empty(0)
    return result


def write_composite(output: Path, workers: int) -> pd.DataFrame:
    jobs = [(target, fold) for target in benchmark.TARGETS for fold in FOLDS]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=workers, verbose=5)(delayed(composite_fold)(output, *job) for job in jobs)
    rows = []
    for (target, fold), result in zip(jobs, results, strict=True):
        path = output / "alternative_shards" / COMPOSITE / target / f"r0_f{fold}.npz"
        benchmark.harness.atomic_save(
            path,
            {
                "prediction": result["prediction"],
                "bank_prediction": result["bank_prediction"],
                "test": result["test"],
                "train": result["train"],
                "elapsed": result["elapsed"],
                "dof": np.nan,
                "effective_rows": np.nan,
                "selected_json": json.dumps({"kappa": COMPOSITE_KAPPA, "base": LINK}),
            },
        )
        rows.append(
            {
                "target": target,
                "fold": fold,
                "bank_rows": len(result["bank_prediction"]),
                "max_abs_change_bank": float(
                    np.max(np.abs(result["bank_prediction"] - result["bank_prediction_link_macro"]))
                ),
                "mean_change_bank": float(np.mean(result["bank_prediction"] - result["bank_prediction_link_macro"])),
                "max_abs_change_test": float(
                    np.max(np.abs(result["prediction"] - result["prediction_link_macro"]), initial=0.0)
                ),
                "additive_sum_vs_link_max_abs": float(
                    np.max(np.abs(result["bank_prediction_additive_macro"] - result["bank_prediction_link_macro"]))
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "composite_change.csv", index=False)
    return table


def collect(output: Path, with_composite: bool = True) -> pd.DataFrame:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    records = []
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        weights = data[f"{target}_aggregation_weights"]
        for method in (*FITTED_METHODS, *((COMPOSITE,) if with_composite else ())):
            replicates = []
            final = None
            for fold in FOLDS:
                if method == COMPOSITE:
                    shard = benchmark.read_npz(output / "alternative_shards" / method / target / f"r0_f{fold}.npz")
                    prediction, bank_prediction, test = shard["prediction"], shard["bank_prediction"], shard["test"]
                else:
                    shards = [
                        benchmark.read_npz(output / "baseline_shards" / method / target / f"r0_f{fold}_c{c}.npz")
                        for c in range(len(weights))
                    ]
                    prediction = np.stack([s["prediction"] for s in shards], axis=1) @ weights
                    bank_prediction = np.stack([s["bank_prediction"] for s in shards], axis=1) @ weights
                    test = shards[0]["test"]
                if fold == -1:
                    final = bank_prediction
                    continue
                replicates.append(bank_prediction)
                records.extend(
                    {
                        "method": method,
                        "target": target,
                        "population": "panel_oof",
                        "repeat": 0,
                        "fold": fold,
                        "row_id": str(int(row)),
                        "prediction": float(value),
                        "uncertainty": np.nan,
                    }
                    for row, value in zip(test, prediction, strict=True)
                )
            spread = np.std(np.stack(replicates), axis=0, ddof=1)
            records.extend(
                {
                    "method": method,
                    "target": target,
                    "population": "external_development",
                    "repeat": 0,
                    "fold": -1,
                    "row_id": str(row),
                    "prediction": float(value),
                    "uncertainty": float(error),
                }
                for row, value, error in zip(bank["coordinate_id"], final, spread, strict=True)
            )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "predictions.csv", index=False)
    return frame


def wspu_parity(output: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE / "predictions.csv")
    rows = []
    for target in benchmark.TARGETS:
        for population in ("panel_oof", "external_development"):
            keys = ["target", "population", "fold", "row_id"]
            mine = predictions[
                predictions.method.eq(WSPU) & predictions.target.eq(target) & predictions.population.eq(population)
            ]
            theirs = reference[
                reference.method.eq(WSPU) & reference.target.eq(target) & reference.population.eq(population)
            ]
            mine = mine.assign(row_id=mine.row_id.astype(str))
            theirs = theirs.assign(row_id=theirs.row_id.astype(str))
            merged = mine.merge(theirs, on=keys, suffixes=("_mine", "_reference"), validate="one_to_one")
            rows.append(
                {
                    "target": target,
                    "population": population,
                    "rows": len(merged),
                    "max_abs_difference": float(np.max(np.abs(merged.prediction_mine - merged.prediction_reference))),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "wspu_parity.csv", index=False)
    return table


def comparison_table(output: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "method",
        "regret_at_1",
        "best_of_5_regret",
        "best_of_10_regret",
        "selected_rank",
        "rows",
        "optimism",
        "rmse",
        "spearman",
    ]
    mine = metrics[
        metrics.stratum.eq("optima") & metrics.policy.eq("point") & metrics.population.eq("external_development")
    ][columns]
    mine = mine.assign(model=mine.method.map(lambda m: DISPLAY.get(m, m)), source="this run")
    reference = pd.read_csv(REFERENCE / "metrics.csv")
    reference = reference[
        reference.stratum.eq("optima")
        & reference.policy.eq("point")
        & reference.population.eq("external_development")
        & reference.method.isin(REFERENCE_DISPLAY)
    ][columns]
    reference = reference.assign(
        model=reference.method.map(REFERENCE_DISPLAY), source="delphi_offline_selection_20260906"
    )
    codex = pd.read_csv(FOLLOWUP / "incumbent_coupling" / "metrics.csv")
    codex = codex[
        codex.stratum.eq("optima")
        & codex.policy.eq("point")
        & codex.population.eq("external_development")
        & codex.method.isin(CODEX_DISPLAY)
    ][columns]
    codex = codex.assign(model=codex.method.map(CODEX_DISPLAY), source="delphi_coupling_followup_20260906")
    table = pd.concat([reference, codex, mine], ignore_index=True)
    table["rank"] = table.selected_rank.astype(int).astype(str) + "/" + table.rows.astype(int).astype(str)
    order = {"uncheatable": 0, "table9": 1}
    table = table.sort_values(["target", "source", "method"], key=lambda s: s.map(order) if s.name == "target" else s)
    table.to_csv(output / "comparison.csv", index=False)
    return table


def main() -> None:
    global FITTED_METHODS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--methods",
        default=",".join(FITTED_METHODS),
        help="comma-separated registry ids to fit; WSPU must be first for the paired contrasts",
    )
    parser.add_argument("--no-composite", action="store_true", help="skip the bounded link + Codex coupling method")
    args = parser.parse_args()
    FITTED_METHODS = tuple(args.methods.split(","))
    if FITTED_METHODS[0] != WSPU:
        raise ValueError("The first method must be WSPU, the reference of the paired contrasts")
    for method in FITTED_METHODS:
        if method not in registry.ENTRY_BY_ID:
            raise ValueError(f"Unknown registry entry: {method}")
    prepare(args.output_dir)
    fit_all(args.output_dir, args.workers)
    with_composite = not args.no_composite and LINK in FITTED_METHODS
    if with_composite:
        change = write_composite(args.output_dir, args.workers)
        print("composite change against the bounded link (macro BPB):")
        print(change.round(5).to_string(index=False))
    predictions = collect(args.output_dir, with_composite)
    print("WSPU parity against the reference package:")
    print(wspu_parity(args.output_dir, predictions).to_string(index=False))
    scorer.METHODS = tuple(predictions.method.unique())
    metrics, assignments, loso = scorer.score_predictions(args.output_dir, predictions)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    assignments.to_csv(args.output_dir / "source_blocks.csv", index=False)
    loso.to_csv(args.output_dir / "source_disjoint_selection.csv", index=False)
    scorer.paired_sources(metrics, loso).to_csv(args.output_dir / "paired_source_contrasts.csv", index=False)
    table = comparison_table(args.output_dir, metrics)
    pd.set_option("display.width", 250)
    print(
        table[
            [
                "target",
                "model",
                "regret_at_1",
                "best_of_5_regret",
                "best_of_10_regret",
                "rank",
                "optimism",
                "rmse",
                "spearman",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
