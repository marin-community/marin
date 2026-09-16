# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Test whether the a-priori pilot improves held-out Table-9 prediction.

The analysis keeps the frozen 280-row Delphi panel as the baseline and evaluates
the WSPU surrogate in two leakage-free settings:

* cross-block: train on one pilot seed block and predict the other block's 16
  support interventions;
* external bank: train on all 37 pilot runs and predict the complete pre-pilot
  registry, excluding every coordinate sourced from the a-priori swarm.

The external comparison is intentionally over budget (317 versus 280 rows). It
measures the incremental value of the completed data, not a matched-budget swarm
design. Every fit uses the existing model registry and its training-only inner
fold selection.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_union_loso_20260903 as union_harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round6_training_sets_20260904 as training_sets,
)

PANEL = "delphi_3e18_39bucket"
TARGET = "table9"
MODEL = "weibull_softplus_unscaled"
PILOT_SOURCE = "prospectively_frozen_apriori_swarm"
ELIGIBLE_EXTERNAL_SOURCES = training_sets.ELIGIBLE_SOURCES
BOOTSTRAP_SEED = 20_260_906
BOOTSTRAP_DRAWS = 2_000

REFERENCE_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_apriori_swarm_280_20260904"
DEFAULT_REGISTRY_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_heldout_benchmark_20260902"
DEFAULT_OUTPUT_DIR = REFERENCE_DIR / "predictive_value_20260906"
DESIGN_PATH = REFERENCE_DIR / "swarm_mixtures.csv"
RESULTS_PATH = REFERENCE_DIR / "pilot_materialization" / "heldout_results.csv"
COMPONENTS_PATH = REFERENCE_DIR / "pilot_materialization" / "table9_components.csv"


@dataclasses.dataclass(frozen=True)
class FitRequest:
    name: str
    train: np.ndarray
    test: np.ndarray


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def table9_component_matrix(
    table: pd.DataFrame, row_key: str, rows: pd.Series, components: tuple[str, ...]
) -> np.ndarray:
    pivot = table.pivot(index=row_key, columns="component", values="bpb").reindex(index=rows, columns=list(components))
    if pivot.isna().any().any():
        missing = pivot.isna().sum(axis=1)
        raise ValueError(f"Incomplete Table-9 components: {missing[missing.gt(0)].to_dict()}")
    return pivot.to_numpy(float)


def registry_component_matrix(registry_dir: Path, coordinate_ids: pd.Series, components: tuple[str, ...]) -> np.ndarray:
    table = pd.read_csv(registry_dir / "heldout_coordinate_components.csv")
    table = table[table["panel"].eq(PANEL) & table["target"].eq(TARGET)].copy()
    full_name = {name.split("/")[-2] if "/" in name else name: name for name in components}
    table["component"] = table["component"].map(lambda name: name if name in components else full_name.get(name))
    if table["component"].isna().any():
        raise ValueError("Registry contains an unknown Table-9 component name")
    pivot = table.pivot_table(index="coordinate_id", columns="component", values="bpb_mean", aggfunc="first")
    pivot = pivot.reindex(index=coordinate_ids, columns=list(components))
    if pivot.isna().any().any():
        missing = pivot.isna().sum(axis=1)
        raise ValueError(f"External bank has incomplete Table-9 components: {missing[missing.gt(0)].to_dict()}")
    return pivot.to_numpy(float)


def build_union(registry_dir: Path) -> tuple[union_harness.Union, pd.DataFrame, pd.DataFrame]:
    harness.HELDOUT_DIR = registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    group = panel.group(TARGET)
    design = pd.read_csv(DESIGN_PATH).reset_index(names="run_order")
    results = pd.read_csv(RESULTS_PATH)
    pilot = design.merge(results, on=["run_order", "run_name"], suffixes=("_design", "_observed"), validate="one_to_one")
    if len(pilot) != 37 or not pilot["source"].eq("new").all():
        raise ValueError(f"Expected the 37 frozen new pilot rows, found {len(pilot)}")

    component_table = pd.read_csv(COMPONENTS_PATH)
    pilot_outcomes = table9_component_matrix(component_table, "run_name", pilot["run_name"], tuple(group.components))
    pilot_aggregate = pilot["table9_macro_bpb"].to_numpy(float)
    if not np.allclose(pilot_outcomes @ group.aggregation_weights, pilot_aggregate, atol=2e-4):
        raise ValueError("Pilot components do not reconstruct the materialized Table-9 macro")

    buckets = tuple(panel.buckets)
    pilot_weights = pilot.loc[:, [f"phase_0_{bucket}" for bucket in buckets]].to_numpy(float)
    pilot_pool_fractions = pilot.loc[:, [f"pool_fraction_{bucket}" for bucket in buckets]].to_numpy(float)
    pilot_exposures = pilot_weights * panel.features.inventory[None, :] / pilot_pool_fractions
    expected_exposures = pilot.loc[:, [f"materialized_epochs_{bucket}" for bucket in buckets]].to_numpy(float)
    if not np.allclose(pilot_exposures, expected_exposures, atol=1e-12, rtol=0.0):
        raise ValueError("Pilot exposures do not match the frozen materialized-epoch columns")

    bank, bank_features = harness.heldout_features(panel, TARGET)
    external_mask = ~bank["sources"].str.contains(PILOT_SOURCE, regex=False, na=False)
    external = bank.loc[external_mask].reset_index(drop=True)
    external_weights = bank_features.weights[external_mask.to_numpy()]
    external_exposures = bank_features.exposures[external_mask.to_numpy()]
    external_outcomes = registry_component_matrix(registry_dir, external["coordinate_id"], tuple(group.components))
    external_aggregate = external["table9_macro_mean_bpb"].to_numpy(float)
    if not np.allclose(external_outcomes @ group.aggregation_weights, external_aggregate, atol=2e-4):
        raise ValueError("External components do not reconstruct the registry Table-9 macro")

    weights = np.vstack([panel.features.weights, pilot_weights, external_weights])
    exposures = np.vstack([panel.features.exposures, pilot_exposures, external_exposures])
    features = dataclasses.replace(
        panel.features,
        weights=weights,
        exposures=exposures,
        label=f"{PANEL}|apriori_pilot_predictive_value",
    )
    outcomes = np.vstack([group.outcomes, pilot_outcomes, external_outcomes])
    aggregate = np.concatenate([group.aggregate, pilot_aggregate, external_aggregate])
    panel_rows = panel.rows
    pilot_rows = len(pilot)
    distance = np.abs(weights[:, None, :] - panel.features.weights[None, :, :]).sum(axis=-1).min(axis=1)
    memberships = (
        tuple([frozenset({"panel"})] * panel_rows)
        + tuple([frozenset({PILOT_SOURCE})] * pilot_rows)
        + tuple(frozenset(union_harness.parse_sources(text)) for text in external["sources"])
    )
    primary = np.array(
        ["panel"] * panel_rows
        + [PILOT_SOURCE] * pilot_rows
        + [union_harness.parse_sources(text)[0] for text in external["sources"]]
    )
    coordinate_id = np.concatenate(
        [
            np.array([f"panel:{index}" for index in range(panel_rows)]),
            ("pilot_run::" + pilot["run_name"]).to_numpy(str),
            external["coordinate_id"].to_numpy(str),
        ]
    )
    union = union_harness.Union(
        target=TARGET,
        features=features,
        outcomes=outcomes,
        aggregate=aggregate,
        memberships=memberships,
        primary=primary,
        coordinate_id=coordinate_id,
        distance=distance,
        trainable=np.ones(len(aggregate), dtype=bool),
    )
    pilot = pilot.copy()
    pilot["union_row"] = np.arange(panel_rows, panel_rows + pilot_rows)
    external = external.copy()
    external["union_row"] = np.arange(panel_rows + pilot_rows, len(aggregate))
    return union, pilot, external


def fit_request(union: union_harness.Union, request: FitRequest, workers: int) -> np.ndarray:
    panel = harness.load_panel(PANEL)
    group = panel.group(TARGET)
    with harness.parallel_config(backend="loky", inner_max_num_threads=1):
        predictions = Parallel(n_jobs=workers, verbose=5)(
            delayed(training_sets.fit_rows)(MODEL, union, index, request.train, request.test)
            for index in range(len(group.components))
        )
    matrix = np.stack([prediction for prediction, _curve in predictions], axis=1)
    return matrix @ group.aggregation_weights


def metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - measured
    return {
        "rows": len(measured),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "spearman": harness._safe_spearman(measured, predicted),
    }


def paired_bootstrap(
    measured: np.ndarray,
    baseline: np.ndarray,
    augmented: np.ndarray,
    clusters: np.ndarray,
    draws: int,
) -> dict[str, float]:
    unique = np.unique(clusters)
    members = [np.where(clusters == cluster)[0] for cluster in unique]
    if len({len(rows) for rows in members}) != 1:
        raise ValueError("Bootstrap clusters must have equal size")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(unique), size=(draws, len(unique)))
    base_error = baseline - measured
    augmented_error = augmented - measured
    rmse_delta = np.empty(draws)
    mae_delta = np.empty(draws)
    for draw, sample in enumerate(samples):
        rows = np.concatenate([members[index] for index in sample])
        rmse_delta[draw] = np.sqrt(np.mean(np.square(augmented_error[rows]))) - np.sqrt(
            np.mean(np.square(base_error[rows]))
        )
        mae_delta[draw] = np.mean(np.abs(augmented_error[rows])) - np.mean(np.abs(base_error[rows]))
    return {
        "clusters": len(unique),
        "rmse_delta": float(np.sqrt(np.mean(np.square(augmented_error))) - np.sqrt(np.mean(np.square(base_error)))),
        "rmse_delta_ci_low": float(np.quantile(rmse_delta, 0.025)),
        "rmse_delta_ci_high": float(np.quantile(rmse_delta, 0.975)),
        "rmse_share_better": float(np.mean(rmse_delta < 0)),
        "mae_delta": float(np.mean(np.abs(augmented_error)) - np.mean(np.abs(base_error))),
        "mae_delta_ci_low": float(np.quantile(mae_delta, 0.025)),
        "mae_delta_ci_high": float(np.quantile(mae_delta, 0.975)),
        "mae_share_better": float(np.mean(mae_delta < 0)),
    }


def source_is_eligible(text: str) -> bool:
    membership = union_harness.parse_sources(text)
    return any(source in membership for source in ELIGIBLE_EXTERNAL_SOURCES)


def report_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, divider]
    for row in frame.itertuples(index=False):
        values = []
        for column in columns:
            value = getattr(row, column)
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    union, pilot, external = build_union(args.registry_dir)
    panel = harness.load_panel(PANEL)
    panel_train = np.arange(panel.rows)
    pilot_subsampled = pilot[pilot["target_bucket"].notna()]
    external_test = external["union_row"].to_numpy(int)
    pilot_test = pilot_subsampled["union_row"].to_numpy(int)
    baseline_test = np.concatenate([external_test, pilot_test])
    baseline_prediction = fit_request(union, FitRequest("panel_280", panel_train, baseline_test), args.workers)
    baseline_lookup = dict(zip(union.coordinate_id[baseline_test], baseline_prediction, strict=True))

    all_pilot_train = np.concatenate([panel_train, pilot["union_row"].to_numpy(int)])
    external_augmented = fit_request(
        union, FitRequest("panel_plus_all_37_pilot", all_pilot_train, external_test), args.workers
    )

    cross_predictions: dict[int, np.ndarray] = {}
    for heldout_block in (0, 1):
        other_block = 1 - heldout_block
        train_rows = pilot[pilot["seed_block"].eq(other_block)]["union_row"].to_numpy(int)
        test_rows = pilot_subsampled[pilot_subsampled["seed_block"].eq(heldout_block)]["union_row"].to_numpy(int)
        cross_predictions[heldout_block] = fit_request(
            union,
            FitRequest(
                f"panel_plus_pilot_block{other_block}",
                np.concatenate([panel_train, train_rows]),
                test_rows,
            ),
            args.workers,
        )

    prediction_rows: list[dict[str, object]] = []
    for position, (_index, row) in enumerate(external.reset_index(drop=True).iterrows()):
        coordinate = str(row["coordinate_id"])
        common = {
            "evaluation": "external_pre_pilot_bank",
            "item_id": coordinate,
            "stratum": "eligible_intervention" if source_is_eligible(str(row["sources"])) else "model_optimum_archive",
            "sources": str(row["sources"]),
            "measured_bpb": float(row["table9_macro_mean_bpb"]),
        }
        prediction_rows.append({**common, "design": "panel_280", "prediction_bpb": baseline_lookup[coordinate]})
        prediction_rows.append(
            {**common, "design": "panel_plus_all_37_pilot", "prediction_bpb": float(external_augmented[position])}
        )

    for heldout_block in (0, 1):
        rows = pilot_subsampled[pilot_subsampled["seed_block"].eq(heldout_block)].reset_index(drop=True)
        for position, (_index, row) in enumerate(rows.iterrows()):
            item = f"pilot_run::{row['run_name']}"
            condition = (
                f"{row['anchor']}|{row['target_bucket']}|{row['run_name'].split('_pool', 1)[1].split('_block', 1)[0]}"
            )
            common = {
                "evaluation": "cross_block_pilot",
                "item_id": item,
                "condition": condition,
                "heldout_block": heldout_block,
                "stratum": str(row["target_bucket"]),
                "sources": PILOT_SOURCE,
                "measured_bpb": float(row["table9_macro_bpb"]),
            }
            prediction_rows.append({**common, "design": "panel_280", "prediction_bpb": baseline_lookup[item]})
            prediction_rows.append(
                {
                    **common,
                    "design": "panel_plus_opposite_pilot_block",
                    "prediction_bpb": float(cross_predictions[heldout_block][position]),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    stratum_specs = [
        ("cross_block_pilot", "pooled", lambda frame: np.ones(len(frame), dtype=bool)),
        ("external_pre_pilot_bank", "pooled", lambda frame: np.ones(len(frame), dtype=bool)),
        (
            "external_pre_pilot_bank",
            "eligible_intervention",
            lambda frame: frame["stratum"].eq("eligible_intervention").to_numpy(),
        ),
        (
            "external_pre_pilot_bank",
            "model_optimum_archive",
            lambda frame: frame["stratum"].eq("model_optimum_archive").to_numpy(),
        ),
    ]
    for evaluation, stratum, mask_fn in stratum_specs:
        block = predictions[predictions["evaluation"].eq(evaluation)]
        wide = block.pivot(index="item_id", columns="design", values=["measured_bpb", "prediction_bpb"])
        metadata = block.drop_duplicates("item_id").set_index("item_id")
        keep = mask_fn(metadata.loc[wide.index])
        wide = wide.loc[keep]
        metadata = metadata.loc[wide.index]
        designs = [
            "panel_280",
            "panel_plus_opposite_pilot_block" if evaluation == "cross_block_pilot" else "panel_plus_all_37_pilot",
        ]
        measured = wide[("measured_bpb", designs[0])].to_numpy(float)
        design_predictions: dict[str, np.ndarray] = {}
        for design in designs:
            guess = wide[("prediction_bpb", design)].to_numpy(float)
            design_predictions[design] = guess
            metric_rows.append(
                {"evaluation": evaluation, "stratum": stratum, "design": design, **metrics(measured, guess)}
            )
        clusters = (
            metadata["condition"].to_numpy(str) if evaluation == "cross_block_pilot" else metadata.index.to_numpy(str)
        )
        comparison_rows.append(
            {
                "evaluation": evaluation,
                "stratum": stratum,
                "baseline": designs[0],
                "augmented": designs[1],
                **paired_bootstrap(
                    measured,
                    design_predictions[designs[0]],
                    design_predictions[designs[1]],
                    clusters,
                    args.bootstrap,
                ),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)

    archive = predictions[
        predictions["evaluation"].eq("external_pre_pilot_bank") & predictions["stratum"].eq("model_optimum_archive")
    ]
    selection_rows = []
    for design, block in archive.groupby("design"):
        row = {"design": design}
        row.update(
            selection.selection_row(
                block["measured_bpb"].to_numpy(float),
                block["prediction_bpb"].to_numpy(float),
                harness.BASIN_TOLERANCE_SD * panel.repeat_sd[TARGET],
            )
        )
        selection_rows.append(row)
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(args.output_dir / "archive_selection_metrics.csv", index=False)

    manifest = {
        "model": MODEL,
        "panel": PANEL,
        "target": TARGET,
        "baseline_rows": len(panel_train),
        "external_augmented_rows": len(all_pilot_train),
        "cross_block_augmented_rows": int(len(panel_train) + pilot["seed_block"].eq(0).sum()),
        "pilot_runs": len(pilot),
        "pilot_subsampled_runs": len(pilot_subsampled),
        "external_coordinates": len(external),
        "bootstrap_draws": args.bootstrap,
        "input_sha256": {
            str(DESIGN_PATH.relative_to(REPO_ROOT)): file_sha256(DESIGN_PATH),
            str(RESULTS_PATH.relative_to(REPO_ROOT)): file_sha256(RESULTS_PATH),
            str(COMPONENTS_PATH.relative_to(REPO_ROOT)): file_sha256(COMPONENTS_PATH),
            str((args.registry_dir / "manifest.json").resolve().relative_to(REPO_ROOT)): file_sha256(
                args.registry_dir / "manifest.json"
            ),
        },
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    comparison_view = comparisons[
        [
            "evaluation",
            "stratum",
            "clusters",
            "rmse_delta",
            "rmse_delta_ci_low",
            "rmse_delta_ci_high",
            "rmse_share_better",
            "mae_delta",
            "mae_delta_ci_low",
            "mae_delta_ci_high",
        ]
    ]
    report = [
        "# Delphi a-priori pilot predictive value",
        "",
        (
            "Negative deltas favor the pilot-augmented WSPU fit. Cross-block intervals resample the 16 support "
            "conditions; external intervals resample coordinates."
        ),
        "",
        *report_table(comparison_view, list(comparison_view.columns)),
        "",
        "## External model-optimum selection",
        "",
        *report_table(
            selection_frame[
                [
                    "design",
                    "bank_size",
                    "regret_at_1",
                    "top5_regret",
                    "selected_rank",
                    "frontier_predicted_rank",
                    "rmse",
                    "spearman",
                ]
            ],
            [
                "design",
                "bank_size",
                "regret_at_1",
                "top5_regret",
                "selected_rank",
                "frontier_predicted_rank",
                "rmse",
                "spearman",
            ],
        ),
        "",
        (
            "The 317-row external fit measures incremental data value and is not a matched-budget replacement "
            "for the 280-row panel."
        ),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(comparison_view.round(6).to_string(index=False))
    print("\nExternal model-optimum selection:")
    print(selection_frame.round(6).to_string(index=False))
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
