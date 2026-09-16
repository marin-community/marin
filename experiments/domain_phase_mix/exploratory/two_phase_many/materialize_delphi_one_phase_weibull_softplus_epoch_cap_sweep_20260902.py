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

"""Materialize Delphi one-phase epoch-cap optima for the successor surrogate.

The successor is fitted componentwise exactly as in the single-phase
Observatory benchmark. Its identity-link prediction is separable by bucket, so
the constrained optimum on Marin's exact 1/2048 mixture grid is found by
dynamic programming rather than a local continuous optimizer. This script does
not launch training.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

MODEL_ID = "weibull_softplus_unscaled"
PANEL_ID = "delphi_3e18_39bucket"
TARGETS = ("uncheatable", "table9")
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "Table-9 macro"}
CAPS = tuple(range(2, 9))
SENSITIVITY_CAPS = CAPS
MIXTURE_BLOCK_SIZE = 2_048
BENCHMARK_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_observatory_benchmark_20260902"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"
PRIOR_SWEEPS = {
    "shared_shape_dsp": SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_dsp_epoch_cap_sweep_20260828",
    "full_canonical_dsp": (
        SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901"
    ),
}
SOURCE_LABELS = {
    "full": "Full-panel selection",
    **{f"outer_fold_{fold}": f"Outer fold {fold + 1}" for fold in range(benchmark.OUTER_FOLDS)},
}
COLORS = {
    "uncheatable": "#178A72",
    "table9": "#D95F32",
    "successor": "#1B8A5A",
    "shared_shape_dsp": "#E8A11B",
    "full_canonical_dsp": "#CF3E2E",
    "proportional": "#78909C",
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
VERIFICATION_TOLERANCE = 2e-9


@dataclass(frozen=True)
class FitMetadata:
    """Persisted shape selection plus the rows used for its linear head."""

    target: str
    component_position: int
    component: str
    aggregation_weight: float
    source: str
    train: np.ndarray
    test: np.ndarray
    shape: dict[str, float]
    ridge: float
    protocol_hash: str
    expected_train: np.ndarray
    expected_test: np.ndarray


@dataclass(frozen=True)
class ComponentFit:
    """One reconstructed atomic component fit."""

    metadata: FitMetadata
    head: models.FittedHead
    verification_max_abs: float

    @property
    def rate(self) -> float:
        return float(self.metadata.shape["rate"])

    @property
    def power(self) -> float:
        return float(self.metadata.shape["power"])

    @property
    def threshold(self) -> float:
        return float(self.metadata.shape["threshold"])


@dataclass(frozen=True)
class AggregatePredictor:
    """Fixed weighted aggregate of componentwise successor fits."""

    target: str
    source: str
    components: tuple[ComponentFit, ...]
    buckets: int

    @property
    def intercept(self) -> float:
        return float(
            sum(component.metadata.aggregation_weight * component.head.intercept for component in self.components)
        )

    def predict(self, exposures: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(exposures, dtype=float))
        if values.shape[1] != self.buckets:
            raise ValueError(f"Expected {self.buckets} exposure columns, got {values.shape[1]}")
        prediction = np.full(values.shape[0], self.intercept)
        for component in self.components:
            benefit = models.weibull_response(values, component.rate, component.power)
            harm = models.softplus_harm(values, component.threshold)
            coefficients = component.head.coefficients
            contribution = -benefit @ coefficients[: self.buckets] + harm @ coefficients[self.buckets :]
            prediction += component.metadata.aggregation_weight * contribution
        return prediction

    def bucket_cost(self, bucket: int, exposures: np.ndarray) -> np.ndarray:
        values = np.asarray(exposures, dtype=float)
        result = np.zeros_like(values)
        for component in self.components:
            coefficients = component.head.coefficients
            benefit = models.weibull_response(values, component.rate, component.power)
            harm = models.softplus_harm(values, component.threshold)
            result += component.metadata.aggregation_weight * (
                -coefficients[bucket] * benefit + coefficients[self.buckets + bucket] * harm
            )
        return result


@dataclass(frozen=True)
class RuntimeOptimum:
    """Exact fitted optimum on one capped 1/2048 mixture grid."""

    target: str
    source: str
    cap: int
    counts: np.ndarray
    prediction: float

    @property
    def weights(self) -> np.ndarray:
        return self.counts / MIXTURE_BLOCK_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--skip-sensitivity-optima", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    if str(payload["status"].item()) != "ok":
        raise ValueError(f"Benchmark shard failed: {path}: {payload['error'].item()}")
    return payload


def metadata_for_component(
    panel: benchmark.BenchPanel,
    target: str,
    component_position: int,
    source: str,
) -> FitMetadata:
    group = panel.group(target)
    component = group.components[component_position]
    if source == "full":
        path = benchmark.heldout_shard_path(BENCHMARK_DIR, MODEL_ID, panel.name, target, component_position, component)
        payload = load_npz(path)
        train = np.arange(panel.rows)
        test = np.asarray([], dtype=int)
        expected_train = np.asarray(payload["fit_prediction"], dtype=float)
        expected_test = np.asarray([], dtype=float)
    else:
        fold = int(source.removeprefix("outer_fold_"))
        split = next(item for item in benchmark.panel_splits(panel, 1) if item.repeat == 0 and item.fold == fold)
        task = benchmark.FitTask(MODEL_ID, panel.name, target, component_position, component, 0, fold)
        path = benchmark.shard_path(BENCHMARK_DIR, task)
        payload = load_npz(path)
        train = split.train
        test = split.test
        expected_train = np.asarray(payload["train_prediction"], dtype=float)
        expected_test = np.asarray(payload["prediction"], dtype=float)
    if str(payload["model_id"].item()) != MODEL_ID or str(payload["component"].item()) != component:
        raise ValueError(f"Shard identity mismatch: {path}")
    return FitMetadata(
        target=target,
        component_position=component_position,
        component=component,
        aggregation_weight=float(group.aggregation_weights[component_position]),
        source=source,
        train=train,
        test=test,
        shape={key: float(value) for key, value in json.loads(str(payload["shape_json"].item())).items()},
        ridge=float(payload["ridge"].item()),
        protocol_hash=str(payload["protocol_hash"].item()),
        expected_train=expected_train,
        expected_test=expected_test,
    )


def reconstruct_component(
    metadata: FitMetadata,
    panel: benchmark.BenchPanel,
    model: models.GridModel,
) -> ComponentFit:
    group = panel.group(metadata.target)
    response = group.outcomes[:, metadata.component_position]
    design = model.design(panel.features, metadata.shape)
    spec = model.head_for(metadata.shape)
    train_design = models.Design(design.values[metadata.train], design.ridge, design.names)
    head = models.fit_head(train_design, response[metadata.train], metadata.ridge, spec)
    train_prediction = models.predict_head(head, design.values[metadata.train], spec)
    errors = [float(np.max(np.abs(train_prediction - metadata.expected_train), initial=0.0))]
    if len(metadata.test):
        test_prediction = models.predict_head(head, design.values[metadata.test], spec)
        errors.append(float(np.max(np.abs(test_prediction - metadata.expected_test), initial=0.0)))
    maximum = max(errors)
    if maximum > VERIFICATION_TOLERANCE:
        raise ValueError(
            f"Could not reproduce {metadata.target}/{metadata.component}/{metadata.source}: max abs {maximum:.3g}"
        )
    return ComponentFit(metadata=metadata, head=head, verification_max_abs=maximum)


def reconstruct_predictors(
    panel: benchmark.BenchPanel,
    workers: int,
) -> tuple[dict[tuple[str, str], AggregatePredictor], pd.DataFrame]:
    entry = registry.ENTRY_BY_ID[MODEL_ID]
    transformed = registry.apply_transform(panel.features, entry)
    if transformed.cache_key != panel.features.cache_key:
        raise ValueError("The successor should use the unmodified true-inventory features")
    model = entry.build(transformed)
    if not isinstance(model, models.GridModel):
        raise TypeError(f"Expected GridModel, got {type(model).__name__}")
    sources = tuple(SOURCE_LABELS)
    metadata = [
        metadata_for_component(panel, target, position, source)
        for target in TARGETS
        for position in range(len(panel.group(target).components))
        for source in sources
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fits = list(pool.map(lambda item: reconstruct_component(item, panel, model), metadata))

    predictors = {}
    for target in TARGETS:
        for source in sources:
            selected = tuple(
                sorted(
                    (fit for fit in fits if fit.metadata.target == target and fit.metadata.source == source),
                    key=lambda fit: fit.metadata.component_position,
                )
            )
            if len(selected) != len(panel.group(target).components):
                raise ValueError(f"Incomplete reconstructed predictor: {target}/{source}")
            predictors[target, source] = AggregatePredictor(target, source, selected, len(panel.buckets))

    rows = []
    for fit in fits:
        metadata = fit.metadata
        rows.append(
            {
                "model": MODEL_ID,
                "target": metadata.target,
                "component_position": metadata.component_position,
                "component": metadata.component,
                "aggregation_weight": metadata.aggregation_weight,
                "source": metadata.source,
                "training_rows": len(metadata.train),
                "rate": fit.rate,
                "power": fit.power,
                "threshold": fit.threshold,
                "ridge": metadata.ridge,
                "intercept": fit.head.intercept,
                "active_coefficients": fit.head.active,
                "benefit_amplitudes_json": json.dumps(fit.head.coefficients[: len(panel.buckets)].tolist()),
                "harm_amplitudes_json": json.dumps(fit.head.coefficients[len(panel.buckets) :].tolist()),
                "protocol_hash": metadata.protocol_hash,
                "reproduction_max_abs": fit.verification_max_abs,
            }
        )
    return predictors, pd.DataFrame(rows)


def exact_runtime_optimum(
    predictor: AggregatePredictor,
    inventory: np.ndarray,
    cap: int,
) -> RuntimeOptimum:
    """Solve the capped integer mixture allocation exactly by min-plus DP."""
    maximum = np.floor(np.minimum(1.0, cap / inventory) * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
    if int(maximum.sum()) < MIXTURE_BLOCK_SIZE:
        raise ValueError(f"Epoch cap {cap} is infeasible on the runtime grid")
    dp = np.full(MIXTURE_BLOCK_SIZE + 1, np.inf)
    dp[0] = 0.0
    choices = np.full((len(inventory), MIXTURE_BLOCK_SIZE + 1), -1, dtype=np.int16)
    for bucket, limit in enumerate(maximum):
        counts = np.arange(int(limit) + 1)
        exposures = inventory[bucket] * counts / MIXTURE_BLOCK_SIZE
        costs = predictor.bucket_cost(bucket, exposures)
        updated = np.full_like(dp, np.inf)
        selected = np.full(MIXTURE_BLOCK_SIZE + 1, -1, dtype=np.int16)
        for count, cost in enumerate(costs):
            candidate = dp[: MIXTURE_BLOCK_SIZE + 1 - count] + cost
            destination = updated[count:]
            better = candidate < destination
            destination[better] = candidate[better]
            selected_view = selected[count:]
            selected_view[better] = count
        dp = updated
        choices[bucket] = selected
    if not np.isfinite(dp[MIXTURE_BLOCK_SIZE]):
        raise RuntimeError(f"No runtime-grid solution for cap {cap}")
    counts = np.zeros(len(inventory), dtype=np.int64)
    remaining = MIXTURE_BLOCK_SIZE
    for bucket in range(len(inventory) - 1, -1, -1):
        count = int(choices[bucket, remaining])
        if count < 0:
            raise RuntimeError(f"Broken DP backpointer for cap {cap}, bucket {bucket}")
        counts[bucket] = count
        remaining -= count
    if remaining != 0 or int(counts.sum()) != MIXTURE_BLOCK_SIZE or np.any(counts > maximum):
        raise RuntimeError(f"Invalid DP reconstruction for cap {cap}")
    weights = counts / MIXTURE_BLOCK_SIZE
    direct = float(predictor.predict((inventory * weights)[None, :])[0])
    dynamic = float(predictor.intercept + dp[MIXTURE_BLOCK_SIZE])
    if not math.isclose(direct, dynamic, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError(f"DP and direct predictions disagree: {direct} != {dynamic}")
    return RuntimeOptimum(predictor.target, predictor.source, cap, counts, direct)


def one_exchange_improvement(
    predictor: AggregatePredictor,
    inventory: np.ndarray,
    optimum: RuntimeOptimum,
) -> float:
    counts = optimum.counts
    maximum = np.floor(np.minimum(1.0, optimum.cap / inventory) * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
    proposals = []
    for donor in np.flatnonzero(counts > 0):
        for receiver in np.flatnonzero(counts < maximum):
            if donor == receiver:
                continue
            candidate = counts.copy()
            candidate[donor] -= 1
            candidate[receiver] += 1
            proposals.append(candidate)
    if not proposals:
        return 0.0
    weights = np.asarray(proposals) / MIXTURE_BLOCK_SIZE
    values = predictor.predict(weights * inventory[None, :])
    return max(0.0, optimum.prediction - float(values.min()))


def materialize_primary(
    predictors: dict[tuple[str, str], AggregatePredictor],
    inventory: np.ndarray,
    workers: int,
) -> list[RuntimeOptimum]:
    tasks = [(predictors[target, "full"], inventory, cap) for target in TARGETS for cap in CAPS]
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        return list(pool.map(lambda item: exact_runtime_optimum(*item), tasks))


def materialize_sensitivity(
    predictors: dict[tuple[str, str], AggregatePredictor],
    inventory: np.ndarray,
    workers: int,
) -> list[RuntimeOptimum]:
    tasks = [
        (predictors[target, source], inventory, cap)
        for target in TARGETS
        for source in SOURCE_LABELS
        if source != "full"
        for cap in SENSITIVITY_CAPS
    ]
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        return list(pool.map(lambda item: exact_runtime_optimum(*item), tasks))


def hellinger(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.square(np.sqrt(first) - np.sqrt(second)).sum()))


def effective_buckets(weights: np.ndarray) -> float:
    positive = weights > 0.0
    return float(np.exp(-np.sum(weights[positive] * np.log(weights[positive]))))


def proportional_weights(panel: benchmark.BenchPanel) -> np.ndarray:
    """Return the inventory-proportional policy, which equalizes bucket epochs."""
    weights = 1.0 / panel.features.inventory
    return weights / weights.sum()


def prior_weights(panel: benchmark.BenchPanel) -> dict[tuple[str, str, int], np.ndarray]:
    target_map = {"uncheatable_bpb": "uncheatable", "table9_macro_bpb": "table9"}
    result = {}
    for name, directory in PRIOR_SWEEPS.items():
        frame = pd.read_csv(directory / "candidate_weights.csv")
        for (target, cap), rows in frame.groupby(["target", "epoch_cap"], sort=False):
            mapped = target_map[str(target)]
            indexed = rows.set_index("domain")
            missing = set(panel.buckets) - set(indexed.index)
            if missing:
                raise ValueError(f"{name} is missing buckets: {sorted(missing)}")
            result[name, mapped, int(cap)] = indexed.loc[list(panel.buckets), "weight"].to_numpy(float)
    return result


def heldout_bank(panel: benchmark.BenchPanel, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bank, features = benchmark.heldout_features(panel, target)
    _count_column, mean_column = benchmark.HELDOUT_TARGET_COLUMNS[target]
    return (
        features.weights,
        bank[mean_column].to_numpy(float),
        bank["coordinate_id"].to_numpy(str),
    )


def build_candidate_tables(
    panel: benchmark.BenchPanel,
    predictors: dict[tuple[str, str], AggregatePredictor],
    primary: list[RuntimeOptimum],
    sensitivity: list[RuntimeOptimum],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = panel.features.inventory
    prior = prior_weights(panel)
    proportional = proportional_weights(panel)
    primary_lookup = {(item.target, item.cap): item for item in primary}
    sensitivity_lookup = {(item.target, item.source, item.cap): item for item in sensitivity}
    summary_rows = []
    weight_rows = []
    comparison_rows = []
    sensitivity_rows = []
    for target in TARGETS:
        group = panel.group(target)
        full = predictors[target, "full"]
        bank_weights, bank_outcomes, bank_ids = heldout_bank(panel, target)
        bank_exposures = bank_weights * inventory[None, :]
        bank_predictions = full.predict(bank_exposures)
        for cap in CAPS:
            optimum = primary_lookup[target, cap]
            weights = optimum.weights
            exposures = inventory * weights
            maximum = np.floor(np.minimum(1.0, cap / inventory) * MIXTURE_BLOCK_SIZE + 1e-12).astype(int)
            active = optimum.counts == maximum
            fit_distances = 0.5 * np.abs(panel.features.weights - weights).sum(axis=1)
            nearest_fit = int(np.argmin(fit_distances))
            heldout_distances = 0.5 * np.abs(bank_weights - weights).sum(axis=1)
            nearest_heldout = int(np.argmin(heldout_distances))
            feasible_heldout = np.max(bank_exposures, axis=1) <= cap + 1e-12
            if not np.any(feasible_heldout):
                raise ValueError(f"No heldout policy is feasible at cap {cap}")
            feasible_heldout_rows = np.flatnonzero(feasible_heldout)
            best_observed_heldout = int(feasible_heldout_rows[np.argmin(bank_outcomes[feasible_heldout])])
            best_predicted_heldout = int(feasible_heldout_rows[np.argmin(bank_predictions[feasible_heldout])])
            feasible = np.max(panel.features.exposures, axis=1) <= cap + 1e-12
            if not np.any(feasible):
                raise ValueError(f"No fit-panel policy is feasible at cap {cap}")
            feasible_rows = np.flatnonzero(feasible)
            best_feasible = int(feasible_rows[np.argmin(group.aggregate[feasible])])
            observed_weights = panel.features.weights[best_feasible]
            fold_predictions = np.asarray(
                [
                    predictors[target, source].predict(exposures[None, :])[0]
                    for source in SOURCE_LABELS
                    if source != "full"
                ]
            )
            candidate_id = f"wspu_{target}_cap{cap:02d}"
            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model": MODEL_ID,
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "epoch_cap": cap,
                    "runtime_predicted_bpb": optimum.prediction,
                    "outer_fold_prediction_mean": float(fold_predictions.mean()),
                    "outer_fold_prediction_sd": float(fold_predictions.std(ddof=1)),
                    "outer_fold_prediction_min": float(fold_predictions.min()),
                    "outer_fold_prediction_max": float(fold_predictions.max()),
                    "max_materialized_epoch": float(exposures.max()),
                    "cap_active_buckets": int(active.sum()),
                    "support_buckets": int(np.count_nonzero(optimum.counts)),
                    "effective_buckets": effective_buckets(weights),
                    "tv_to_proportional": float(0.5 * np.abs(weights - proportional).sum()),
                    "hellinger_to_proportional": hellinger(weights, proportional),
                    "nearest_fit_panel_row": str(panel.runs[nearest_fit]),
                    "nearest_fit_panel_tv": float(fit_distances[nearest_fit]),
                    "nearest_fit_panel_observed_bpb": float(group.aggregate[nearest_fit]),
                    "nearest_heldout_coordinate": str(bank_ids[nearest_heldout]),
                    "nearest_heldout_tv": float(heldout_distances[nearest_heldout]),
                    "nearest_heldout_observed_bpb": float(bank_outcomes[nearest_heldout]),
                    "best_feasible_heldout_observed_coordinate": str(bank_ids[best_observed_heldout]),
                    "best_feasible_heldout_observed_bpb": float(bank_outcomes[best_observed_heldout]),
                    "prediction_at_best_feasible_heldout_observed": float(bank_predictions[best_observed_heldout]),
                    "best_feasible_heldout_predicted_coordinate": str(bank_ids[best_predicted_heldout]),
                    "best_feasible_heldout_predicted_bpb": float(bank_predictions[best_predicted_heldout]),
                    "observed_at_best_feasible_heldout_prediction": float(bank_outcomes[best_predicted_heldout]),
                    "predicted_gain_beyond_best_feasible_heldout": float(
                        bank_predictions[best_predicted_heldout] - optimum.prediction
                    ),
                    "best_feasible_fit_panel_row": str(panel.runs[best_feasible]),
                    "best_feasible_fit_panel_observed_bpb": float(group.aggregate[best_feasible]),
                    "prediction_at_best_feasible_fit_panel": float(
                        full.predict(panel.features.exposures[[best_feasible]])[0]
                    ),
                    "predicted_gain_vs_best_feasible_fit_panel": float(
                        full.predict(panel.features.exposures[[best_feasible]])[0] - optimum.prediction
                    ),
                    "tv_to_best_feasible_fit_panel": float(0.5 * np.abs(weights - observed_weights).sum()),
                    "largest_bucket": panel.buckets[int(np.argmax(weights))],
                    "largest_weight": float(weights.max()),
                    "one_exchange_improvement": one_exchange_improvement(full, inventory, optimum),
                }
            )
            for position, bucket in enumerate(panel.buckets):
                weight_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "epoch_cap": cap,
                        "bucket_position": position,
                        "domain": bucket,
                        "runtime_count": int(optimum.counts[position]),
                        "weight": float(weights[position]),
                        "proportional_weight": float(proportional[position]),
                        "weight_ratio_to_proportional": float(weights[position] / proportional[position]),
                        "materialized_epochs": float(exposures[position]),
                        "cap_fraction": float(exposures[position] / cap),
                        "cap_active": bool(active[position]),
                    }
                )
            alternatives = {"proportional": proportional}
            alternatives.update(
                {name: prior[name, target, cap] for name in PRIOR_SWEEPS if (name, target, cap) in prior}
            )
            for name, alternative in alternatives.items():
                alternative_prediction = float(full.predict((inventory * alternative)[None, :])[0])
                comparison_rows.append(
                    {
                        "target": target,
                        "epoch_cap": cap,
                        "candidate_family": name,
                        "successor_predicted_bpb": alternative_prediction,
                        "successor_candidate_gain_bpb": alternative_prediction - optimum.prediction,
                        "tv_to_successor_candidate": float(0.5 * np.abs(alternative - weights).sum()),
                        "alternative_max_materialized_epoch": float(np.max(inventory * alternative)),
                    }
                )
            if cap in SENSITIVITY_CAPS and sensitivity:
                for source in SOURCE_LABELS:
                    if source == "full":
                        continue
                    fold_optimum = sensitivity_lookup[target, source, cap]
                    fold_predictor = predictors[target, source]
                    sensitivity_rows.append(
                        {
                            "target": target,
                            "epoch_cap": cap,
                            "source": source,
                            "source_label": SOURCE_LABELS[source],
                            "fold_own_optimum_bpb": fold_optimum.prediction,
                            "fold_prediction_at_full_optimum": float(fold_predictor.predict(exposures[None, :])[0]),
                            "full_prediction_at_fold_optimum": float(
                                full.predict((inventory * fold_optimum.weights)[None, :])[0]
                            ),
                            "tv_fold_optimum_to_full_optimum": float(0.5 * np.abs(fold_optimum.weights - weights).sum()),
                        }
                    )
    target_order = {target: position for position, target in enumerate(TARGETS)}
    summary = pd.DataFrame(summary_rows).assign(_target_order=lambda frame: frame.target.map(target_order))
    summary = summary.sort_values(["_target_order", "epoch_cap"]).drop(columns="_target_order").reset_index(drop=True)
    weights = pd.DataFrame(weight_rows).assign(_target_order=lambda frame: frame.target.map(target_order))
    weights = weights.sort_values(["_target_order", "epoch_cap", "bucket_position"]).drop(columns="_target_order")
    comparison = pd.DataFrame(comparison_rows).assign(_target_order=lambda frame: frame.target.map(target_order))
    comparison = comparison.sort_values(["_target_order", "epoch_cap", "candidate_family"]).drop(columns="_target_order")
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    if not sensitivity_frame.empty:
        sensitivity_frame = sensitivity_frame.assign(_target_order=lambda frame: frame.target.map(target_order))
        sensitivity_frame = sensitivity_frame.sort_values(["_target_order", "epoch_cap", "source"]).drop(
            columns="_target_order"
        )
    return summary, weights, comparison, sensitivity_frame


def fit_diagnostics(
    panel: benchmark.BenchPanel,
    predictors: dict[tuple[str, str], AggregatePredictor],
) -> pd.DataFrame:
    benchmark_metrics = pd.read_csv(BENCHMARK_DIR / "finalist" / "aggregate_metrics.csv")
    heldout_metrics = pd.read_csv(BENCHMARK_DIR / "external_heldout_selection_metrics.csv")
    rows = []
    for target in TARGETS:
        group = panel.group(target)
        full = predictors[target, "full"]
        prediction = full.predict(panel.features.exposures)
        finalist = benchmark_metrics[
            benchmark_metrics["model"].eq(MODEL_ID)
            & benchmark_metrics["panel"].eq(panel.name)
            & benchmark_metrics["target"].eq(target)
        ].iloc[0]
        heldout = heldout_metrics[
            heldout_metrics["model"].eq(MODEL_ID)
            & heldout_metrics["panel"].eq(panel.name)
            & heldout_metrics["target"].eq(target)
            & heldout_metrics["stratum"].eq("pooled")
        ].iloc[0]
        rows.append(
            {
                "target": target,
                "components": len(group.components),
                "in_sample_rmse": float(np.sqrt(np.mean((prediction - group.aggregate) ** 2))),
                "in_sample_spearman": float(spearmanr(prediction, group.aggregate).statistic),
                "outer_cv_rmse": float(finalist.rmse),
                "outer_cv_spearman": float(finalist.spearman),
                "outer_cv_mean_fold_regret_at_1": float(finalist.mean_fold_regret_at_1),
                "heldout_bank_size": int(heldout.bank_size),
                "heldout_rmse": float(heldout.rmse),
                "heldout_spearman": float(heldout.spearman),
                "heldout_regret_at_1": float(heldout.regret_at_1),
                "heldout_top5_regret": float(heldout.top5_regret),
                "noise_floor_rmse": float(heldout.basin_tolerance),
            }
        )
    return pd.DataFrame(rows)


def prior_validation() -> pd.DataFrame:
    rows = []
    target_columns = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
    for family, directory in PRIOR_SWEEPS.items():
        path = directory / "measured_results.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for target, metric in target_columns.items():
            selected = frame[frame["target"].eq(f"{metric}") & frame[metric].notna()]
            for row in selected.itertuples(index=False):
                rows.append(
                    {
                        "candidate_family": family,
                        "target": target,
                        "epoch_cap": int(row.epoch_cap),
                        "measured_bpb": float(getattr(row, metric)),
                    }
                )
    return pd.DataFrame(rows)


def short_bucket_name(bucket: str) -> str:
    value = bucket.replace("dolma3_cc/", "CC ").replace("dolma3_", "").replace("dolmino_", "")
    return value.replace("_high", " H").replace("_low", " L").replace("_", " ")


def base_layout(figure: go.Figure, *, height: int, title: str) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        height=height,
        margin={"l": 75, "r": 35, "t": 100, "b": 80},
        paper_bgcolor="#FBF7EF",
        plot_bgcolor="#FBF7EF",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#17324A", "size": 15},
        hoverlabel={"font": {"family": "Avenir Next, Avenir, sans-serif"}},
    )


def build_figures(
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    buckets: tuple[str, ...],
) -> list[go.Figure]:
    overview = make_subplots(
        rows=1,
        cols=3,
        specs=[[{}, {}, {"secondary_y": True}]],
        subplot_titles=("Predicted optimum", "Distance to measured policies", "Mixture geometry"),
        horizontal_spacing=0.09,
    )
    for target in TARGETS:
        frame = summary[summary.target.eq(target)]
        color = COLORS[target]
        overview.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.runtime_predicted_bpb,
                mode="lines+markers",
                name=TARGET_LABELS[target],
                line={"color": color, "width": 3},
                marker={"size": 9},
                error_y={"type": "data", "array": frame.outer_fold_prediction_sd, "visible": True},
                customdata=np.column_stack(
                    [frame.outer_fold_prediction_min, frame.outer_fold_prediction_max, frame.one_exchange_improvement]
                ),
                hovertemplate=(
                    "Cap %{x}<br>Full fit %{y:.6f}<br>Fold range %{customdata[0]:.6f} to "
                    "%{customdata[1]:.6f}<br>Best one-count improvement %{customdata[2]:.2g}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        overview.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.best_feasible_heldout_predicted_bpb,
                mode="lines+markers",
                name=f"{TARGET_LABELS[target]}: best scored measured policy",
                legendgroup=target,
                line={"color": color, "width": 2, "dash": "dot"},
                marker={"size": 8, "symbol": "circle-open"},
                customdata=np.column_stack(
                    [
                        frame.observed_at_best_feasible_heldout_prediction,
                        frame.predicted_gain_beyond_best_feasible_heldout,
                    ]
                ),
                hovertemplate=(
                    "Cap %{x}<br>Heldout prediction %{y:.6f}<br>Heldout observation "
                    "%{customdata[0]:.6f}<br>Predicted candidate gain %{customdata[1]:.6f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        overview.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.nearest_fit_panel_tv,
                mode="lines+markers",
                name=f"{TARGET_LABELS[target]}: fit panel",
                legendgroup=target,
                showlegend=False,
                line={"color": color, "width": 3},
            ),
            row=1,
            col=2,
        )
        overview.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.nearest_heldout_tv,
                mode="lines+markers",
                name=f"{TARGET_LABELS[target]}: heldout",
                legendgroup=target,
                showlegend=False,
                line={"color": color, "width": 2, "dash": "dot"},
            ),
            row=1,
            col=2,
        )
        overview.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.tv_to_proportional,
                mode="lines+markers",
                name=f"{TARGET_LABELS[target]} TV",
                legendgroup=target,
                showlegend=False,
                line={"color": color, "width": 3},
            ),
            row=1,
            col=3,
            secondary_y=False,
        )
        overview.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.support_buckets,
                mode="lines+markers",
                name=f"{TARGET_LABELS[target]} support",
                legendgroup=target,
                showlegend=False,
                line={"color": color, "width": 2, "dash": "dash"},
            ),
            row=1,
            col=3,
            secondary_y=True,
        )
    overview.update_xaxes(title_text="Whole-run epoch cap", tickvals=CAPS)
    overview.update_yaxes(title_text="Predicted BPB", row=1, col=1)
    overview.update_yaxes(title_text="Nearest-policy TV", row=1, col=2)
    overview.update_yaxes(title_text="TV from proportional", row=1, col=3, secondary_y=False)
    overview.update_yaxes(
        title_text="Supported buckets",
        range=[0, 40],
        tickvals=[0, 10, 20, 30, 39],
        row=1,
        col=3,
        secondary_y=True,
    )
    base_layout(overview, height=560, title="Exact runtime-grid optima, measured-policy benchmark, and geometry")

    comparison_plot = make_subplots(rows=1, cols=2, subplot_titles=[TARGET_LABELS[target] for target in TARGETS])
    family_labels = {
        "successor": "Successor optimum",
        "shared_shape_dsp": "Shared-shape DSP mixture",
        "full_canonical_dsp": "Full canonical DSP mixture",
        "proportional": "Proportional",
    }
    for column, target in enumerate(TARGETS, start=1):
        own = summary[summary.target.eq(target)][["epoch_cap", "runtime_predicted_bpb"]].copy()
        own["candidate_family"] = "successor"
        own = own.rename(columns={"runtime_predicted_bpb": "successor_predicted_bpb"})
        frame = pd.concat([own, comparison[comparison.target.eq(target)]], ignore_index=True, sort=False)
        for family in family_labels:
            selected = frame[frame.candidate_family.eq(family)].sort_values("epoch_cap")
            if selected.empty:
                continue
            comparison_plot.add_trace(
                go.Scatter(
                    x=selected.epoch_cap,
                    y=selected.successor_predicted_bpb,
                    mode="lines+markers",
                    name=family_labels[family],
                    legendgroup=family,
                    showlegend=column == 1,
                    line={"color": COLORS[family], "width": 3 if family == "successor" else 2},
                    marker={"size": 8},
                    hovertemplate="Cap %{x}<br>Successor prediction %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
    comparison_plot.update_xaxes(title_text="Whole-run epoch cap", tickvals=CAPS)
    comparison_plot.update_yaxes(title_text="Successor-predicted BPB")
    base_layout(
        comparison_plot,
        height=560,
        title="Does the successor prefer its own candidate to earlier DSP candidates?",
    )

    labels = [short_bucket_name(bucket) for bucket in buckets]
    heatmap = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[f"{TARGET_LABELS[target]}: weight relative to proportional" for target in TARGETS],
    )
    for row, target in enumerate(TARGETS, start=1):
        frame = weights[weights.target.eq(target)]
        ratio = frame.pivot(index="epoch_cap", columns="domain", values="weight_ratio_to_proportional").reindex(
            index=CAPS, columns=buckets
        )
        epochs = frame.pivot(index="epoch_cap", columns="domain", values="materialized_epochs").reindex(
            index=CAPS, columns=buckets
        )
        heatmap.add_trace(
            go.Heatmap(
                z=np.log2(np.clip(ratio.to_numpy(), 1 / 32, 32)),
                x=labels,
                y=CAPS,
                zmin=-5,
                zmax=5,
                zmid=0,
                colorscale="RdYlGn_r",
                colorbar={"title": "log2 ratio"} if row == 1 else None,
                showscale=row == 1,
                customdata=np.dstack([ratio.to_numpy(), epochs.to_numpy()]),
                hovertemplate=(
                    "%{x}<br>Cap %{y}<br>%{customdata[0]:.3f}x proportional"
                    "<br>%{customdata[1]:.3f} materialized epochs<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
    heatmap.update_yaxes(title_text="Epoch cap", tickvals=CAPS)
    heatmap.update_xaxes(tickangle=-55, row=2, col=1)
    base_layout(heatmap, height=940, title="Successor mixture path")

    sensitivity_plot = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Outer-fold optimum displacement", "Full-fit penalty at fold optima"),
    )
    if not sensitivity.empty:
        for target in TARGETS:
            frame = sensitivity[sensitivity.target.eq(target)]
            grouped = frame.groupby("epoch_cap", as_index=False).agg(
                tv_mean=("tv_fold_optimum_to_full_optimum", "mean"),
                tv_min=("tv_fold_optimum_to_full_optimum", "min"),
                tv_max=("tv_fold_optimum_to_full_optimum", "max"),
                penalty_mean=("full_prediction_at_fold_optimum", "mean"),
                penalty_min=("full_prediction_at_fold_optimum", "min"),
                penalty_max=("full_prediction_at_fold_optimum", "max"),
            )
            own = summary[summary.target.eq(target)].set_index("epoch_cap").runtime_predicted_bpb
            grouped["full_optimum"] = grouped.epoch_cap.map(own)
            grouped["penalty_mean"] -= grouped.full_optimum
            grouped["penalty_min"] -= grouped.full_optimum
            grouped["penalty_max"] -= grouped.full_optimum
            color = COLORS[target]
            sensitivity_plot.add_trace(
                go.Scatter(
                    x=grouped.epoch_cap,
                    y=grouped.tv_mean,
                    mode="lines+markers",
                    name=TARGET_LABELS[target],
                    line={"color": color, "width": 3},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": grouped.tv_max - grouped.tv_mean,
                        "arrayminus": grouped.tv_mean - grouped.tv_min,
                    },
                ),
                row=1,
                col=1,
            )
            sensitivity_plot.add_trace(
                go.Scatter(
                    x=grouped.epoch_cap,
                    y=grouped.penalty_mean,
                    mode="lines+markers",
                    name=TARGET_LABELS[target],
                    legendgroup=target,
                    showlegend=False,
                    line={"color": color, "width": 3},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": grouped.penalty_max - grouped.penalty_mean,
                        "arrayminus": grouped.penalty_mean - grouped.penalty_min,
                    },
                ),
                row=1,
                col=2,
            )
    sensitivity_plot.update_xaxes(title_text="Whole-run epoch cap", tickvals=SENSITIVITY_CAPS)
    sensitivity_plot.update_yaxes(title_text="TV from full-fit optimum", row=1, col=1)
    sensitivity_plot.update_yaxes(title_text="Full-fit predicted BPB penalty", row=1, col=2)
    base_layout(sensitivity_plot, height=540, title="Hyperparameter-selection sensitivity")
    return [overview, comparison_plot, heatmap, sensitivity_plot]


def assessment(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fit_metrics: pd.DataFrame,
) -> tuple[str, list[str]]:
    findings = []
    gate = "plausible validation candidates; no surrogate-resolved frontier gain"
    for target in TARGETS:
        frame = summary[summary.target.eq(target)].sort_values("epoch_cap")
        stabilized = frame.loc[frame.runtime_predicted_bpb <= frame.runtime_predicted_bpb.min() + 1e-6, "epoch_cap"]
        first_stable = int(stabilized.min()) if len(stabilized) else int(frame.iloc[-1].epoch_cap)
        plateau = frame[frame.epoch_cap.eq(first_stable)].iloc[0]
        metric = fit_metrics[fit_metrics.target.eq(target)].iloc[0]
        gain_to_resolution = plateau.predicted_gain_beyond_best_feasible_heldout / metric.outer_cv_rmse
        findings.append(
            f"{TARGET_LABELS[target]} reaches its fitted plateau by cap {first_stable}; larger caps produce the "
            f"same runtime-grid mixture. The path gains "
            f"{frame.iloc[0].runtime_predicted_bpb - plateau.runtime_predicted_bpb:.4f} predicted BPB from cap 2."
        )
        findings.append(
            f"At that plateau, it improves only {plateau.predicted_gain_beyond_best_feasible_heldout:.4f} BPB over "
            f"the best model-ranked, cap-feasible measured heldout policy. This is {gain_to_resolution:.2f} times "
            f"the {metric.outer_cv_rmse:.4f} outer-CV RMSE, so the incremental gain is not resolved by this surrogate."
        )
        findings.append(
            f"The model-ranked measured policy was predicted at {plateau.best_feasible_heldout_predicted_bpb:.4f} "
            f"but observed at {plateau.observed_at_best_feasible_heldout_prediction:.4f}; absolute BPB is optimistic "
            f"by "
            f"{plateau.observed_at_best_feasible_heldout_prediction - plateau.best_feasible_heldout_predicted_bpb:.4f}."
        )
        findings.append(
            f"The plateau mixture is nondegenerate ({int(plateau.support_buckets)} supported buckets, largest weight "
            f"{plateau.largest_weight:.1%}) and lies TV {plateau.nearest_heldout_tv:.3f} from its nearest measured "
            f"heldout policy and TV {plateau.tv_to_proportional:.3f} from proportional."
        )
        if not sensitivity.empty:
            selected = sensitivity[sensitivity.target.eq(target) & sensitivity.epoch_cap.eq(first_stable)]
            findings.append(
                f"Across five benchmark outer-fold fits at cap {first_stable}, optima move by median TV "
                f"{selected.tv_fold_optimum_to_full_optimum.median():.3f} and maximum TV "
                f"{selected.tv_fold_optimum_to_full_optimum.max():.3f}."
            )
        prior = comparison[
            comparison.target.eq(target)
            & comparison.epoch_cap.eq(first_stable)
            & comparison.candidate_family.isin(PRIOR_SWEEPS)
        ]
        if not prior.empty:
            findings.append(
                f"When the successor rescored matched earlier DSP mixtures at cap {first_stable}, it preferred its "
                f"own candidate by {prior.successor_candidate_gain_bpb.min():.4f}-"
                f"{prior.successor_candidate_gain_bpb.max():.4f} BPB; this is model-internal, not validation."
            )
    if summary.nearest_fit_panel_tv.max() > 0.65:
        gate = "too extrapolative to launch without regularization"
    if not sensitivity.empty and sensitivity.tv_fold_optimum_to_full_optimum.median() > 0.25:
        gate = "fit is predictive, but the raw optimum is not coordinate-stable"
    if float(summary.one_exchange_improvement.max()) > 1e-11:
        raise ValueError("Exact runtime-grid solution admits an improving one-count exchange")
    return gate, findings


def dataframe_html(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column, spec in formats.items():
        selected[column] = selected[column].map(spec.format)
    return selected.to_html(index=False, border=0, classes="dataframe", escape=True)


def render_report(
    output_dir: Path,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    prior_results: pd.DataFrame,
    component_fits: pd.DataFrame,
    figures: list[go.Figure],
    gate: str,
    findings: list[str],
) -> None:
    fragments = [
        pio.to_html(
            figure,
            include_plotlyjs=index == 0,
            full_html=False,
            config=PLOT_CONFIG,
            div_id=f"figure-{index}",
        )
        for index, figure in enumerate(figures)
    ]
    cards = "".join(
        f"""
        <article class="metric">
          <span>{html.escape(TARGET_LABELS[row.target])}</span>
          <strong>{row.outer_cv_spearman:.3f}</strong>
          <small>outer-CV Spearman</small>
          <p>RMSE {row.outer_cv_rmse:.4f}; heldout top-5 regret {row.heldout_top5_regret:.4f}.</p>
        </article>
        """
        for row in fit_metrics.itertuples(index=False)
    )
    finding_items = "".join(f"<li>{html.escape(item)}</li>" for item in findings)
    summary_table = dataframe_html(
        summary,
        [
            "target_label",
            "epoch_cap",
            "runtime_predicted_bpb",
            "outer_fold_prediction_sd",
            "nearest_fit_panel_tv",
            "nearest_heldout_tv",
            "best_feasible_heldout_predicted_bpb",
            "predicted_gain_beyond_best_feasible_heldout",
            "tv_to_proportional",
            "support_buckets",
        ],
        {
            "runtime_predicted_bpb": "{:.6f}",
            "outer_fold_prediction_sd": "{:.5f}",
            "nearest_fit_panel_tv": "{:.3f}",
            "nearest_heldout_tv": "{:.3f}",
            "best_feasible_heldout_predicted_bpb": "{:.6f}",
            "predicted_gain_beyond_best_feasible_heldout": "{:.6f}",
            "tv_to_proportional": "{:.3f}",
        },
    )
    shape_summary = (
        component_fits[component_fits.source.eq("full")]
        .groupby(["target", "rate", "power", "threshold", "ridge"], as_index=False)
        .size()
        .sort_values(["target", "size"], ascending=[True, False])
    )
    shape_table = dataframe_html(
        shape_summary,
        ["target", "rate", "power", "threshold", "ridge", "size"],
        {"rate": "{:.2g}", "power": "{:.2g}", "threshold": "{:.1f}", "ridge": "{:.3g}"},
    )
    prior_table = (
        "<p>No prior measured sweep table was available.</p>"
        if prior_results.empty
        else dataframe_html(
            prior_results.sort_values(["candidate_family", "target", "epoch_cap"]),
            ["candidate_family", "target", "epoch_cap", "measured_bpb"],
            {"measured_bpb": "{:.6f}"},
        )
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delphi successor epoch-cap materialization</title>
<style>
:root {{ --ink:#17324a; --muted:#5d7082; --paper:#fbf7ef; --card:#fffdf8; --accent:#d9542d; --line:#d8cdbb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"Avenir Next",Avenir,sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:48px 32px 80px; }}
h1,h2 {{ font-family:Georgia,serif; margin:0; }}
h1 {{ font-size:clamp(2.3rem,5vw,4.5rem); line-height:1.02; max-width:1100px; }}
h2 {{ font-size:2rem; margin-top:48px; }}
p,li {{ font-size:1.08rem; line-height:1.6; }}
.lede {{ max-width:1080px; color:var(--muted); font-size:1.25rem; }}
.verdict {{ margin:30px 0; padding:24px 28px; border-left:8px solid var(--accent); background:var(--card); }}
.verdict strong {{ display:block; font-size:1.55rem; margin-top:4px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:18px; margin:28px 0; }}
.metric {{ background:var(--card); border:1px solid var(--line); padding:20px; }}
.metric span,.metric small {{ display:block; color:var(--muted); }}
.metric strong {{ display:block; font:2.5rem Georgia,serif; margin:8px 0 2px; }}
.metric p {{ font-size:.95rem; margin:10px 0 0; }}
.plot {{ background:var(--card); border:1px solid var(--line); margin:22px 0; overflow:hidden; }}
.note {{ color:var(--muted); max-width:1150px; }}
.table-wrap {{ overflow:auto; background:var(--card); border:1px solid var(--line); padding:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th,td {{ padding:9px 11px; border-bottom:1px solid #e8dfd1; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
code {{ background:#eee6d8; padding:.1rem .3rem; }}
@media (max-width:700px) {{ main {{ padding:28px 14px 60px; }} h1 {{ font-size:2.4rem; }} }}
</style>
</head>
<body><main>
<h1>Successor model: epoch-capped optima</h1>
<p class="lede">This report asks a narrower question than the benchmark: when
<code>weibull_softplus_unscaled</code> is optimized continuously over mixture space, do its proposed Delphi 3e18
one-phase mixtures remain sensible? We fit all seven Uncheatable and 51 Table-9 components separately, combine
them with their frozen aggregation weights, and optimize the exact 1/2048 runtime grid under every integer
whole-run per-bucket epoch cap from 2 through 8.</p>
<section class="verdict"><span>Current assessment</span><strong>{html.escape(gate)}</strong>
<p>No new training result is used here. This is a candidate-geometry and fit-sensitivity audit, not validation of a
new frontier point.</p></section>
<div class="metrics">{cards}</div>
<h2>What the model is</h2>
<p class="note">For bucket <i>b</i>, materialized exposure is
<i>E</i><sub>b</sub>=<i>s</i><sub>b</sub><i>w</i><sub>b</sub>,
where <i>s</i><sub>b</sub> is the true full-share inventory epoch count. Each atomic objective is an intercept minus a
nonnegative, saturating Weibull benefit plus a nonnegative softplus-squared overexposure harm. Rate, Weibull power,
and harm threshold are shared across buckets within an atomic fit; benefit and harm amplitudes are bucket-specific.
There are no semantic families. The nonlinear shape and ridge are selected by blocked inner CV.</p>
<div class="table-wrap">{shape_table}</div>
<h2>Bottom line</h2><ul>{finding_items}</ul>
<div class="plot">{fragments[0]}</div>
<p class="note">Error bars are the standard deviation of predictions from five genuine benchmark outer-fold fits,
evaluated at the full-panel candidate. The dotted lines in the first panel are the lowest successor prediction among
already measured, cap-feasible heldout policies; they expose how much optimization extrapolates beyond measured
support. Dotted distance curves use the measured heldout bank; solid curves use the 280-row fit panel.</p>
<div class="plot">{fragments[1]}</div>
<p class="note">All lines in this panel are scored by the successor, so it diagnoses what its fitted surface prefers;
it does not establish that the preference is correct. Earlier DSP mixtures are included only where the cap matches.</p>
<div class="plot">{fragments[2]}</div>
<div class="plot">{fragments[3]}</div>
<p class="note">The sensitivity panel re-optimizes four anchor caps under each of the five repeat-0 outer-fold fits.
Large coordinate movement with a small full-fit objective penalty indicates a broad, weakly identified optimum basin;
large movement with a large penalty indicates genuine model-selection instability.</p>
<h2>Candidate table</h2><div class="table-wrap">{summary_table}</div>
<h2>Prior measured cap sweeps</h2>
<p class="note">These rows are context for candidate quality. They are observed outcomes of earlier DSP mixtures,
not observations of the successor mixtures.</p><div class="table-wrap">{prior_table}</div>
<h2>Reproducibility</h2>
<p class="note">The full-panel shape/ridge selections and five outer-fold selections are read from the frozen
Observatory benchmark shards. Their linear heads are reconstructed and required to reproduce every stored prediction
within {VERIFICATION_TOLERANCE:g} BPB. Dynamic programming proves the exact global optimum on the runtime grid;
all candidates are independently checked for an improving one-count exchange. Tables beside this report contain
the fitted component parameters, exact counts, comparisons, and split-sensitivity optima.</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(document)


def write_manifest(
    output_dir: Path,
    generated: list[Path],
    component_fits: pd.DataFrame,
) -> None:
    benchmark_inputs = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in (
            Path(__file__).resolve(),
            Path(benchmark.__file__).resolve(),
            Path(models.__file__).resolve(),
            Path(registry.__file__).resolve(),
            BENCHMARK_DIR / "protocol.json",
            BENCHMARK_DIR / "finalist" / "aggregate_metrics.csv",
            BENCHMARK_DIR / "external_heldout_selection_metrics.csv",
        )
    }
    manifest = {
        "model": MODEL_ID,
        "panel": PANEL_ID,
        "targets": list(TARGETS),
        "caps": list(CAPS),
        "runtime_grid": MIXTURE_BLOCK_SIZE,
        "fit_protocol": (
            "atomic component fits; frozen aggregation; full-panel heldout shape/ridge selections; "
            "five repeat-0 outer folds for sensitivity"
        ),
        "optimizer": "exact separable integer dynamic programming",
        "component_fit_rows": len(component_fits),
        "max_reproduction_error": float(component_fits.reproduction_max_abs.max()),
        "inputs": benchmark_inputs,
        "outputs": {path.name: file_sha256(path) for path in generated},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_markdown_report(path: Path, gate: str, findings: list[str], summary: pd.DataFrame) -> None:
    lines = [
        "# Delphi successor epoch-cap materialization",
        "",
        f"**Assessment: {gate}.**",
        "",
        *[f"- {finding}" for finding in findings],
        "",
        "The candidates are exact optima of the fitted successor on the 1/2048 runtime grid. They are not fresh",
        "training validations. See `index.html` for mixture geometry and split sensitivity.",
        "",
        summary[
            [
                "target_label",
                "epoch_cap",
                "runtime_predicted_bpb",
                "outer_fold_prediction_sd",
                "nearest_fit_panel_tv",
                "nearest_heldout_tv",
                "best_feasible_heldout_predicted_bpb",
                "predicted_gain_beyond_best_feasible_heldout",
                "tv_to_proportional",
                "support_buckets",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = benchmark.load_panel(PANEL_ID)
    predictors, component_fits = reconstruct_predictors(panel, args.workers)
    primary = materialize_primary(predictors, panel.features.inventory, args.workers)
    sensitivity = (
        []
        if args.skip_sensitivity_optima
        else materialize_sensitivity(predictors, panel.features.inventory, args.workers)
    )
    summary, weights, comparison, sensitivity_frame = build_candidate_tables(panel, predictors, primary, sensitivity)
    fit_metrics = fit_diagnostics(panel, predictors)
    prior_results = prior_validation()
    gate, findings = assessment(summary, comparison, sensitivity_frame, fit_metrics)
    figures = build_figures(summary, weights, comparison, sensitivity_frame, panel.buckets)

    tables = {
        "candidate_summary.csv": summary,
        "candidate_weights.csv": weights,
        "candidate_comparisons.csv": comparison,
        "sensitivity_optima.csv": sensitivity_frame,
        "fit_metrics.csv": fit_metrics,
        "prior_measured_results.csv": prior_results,
        "component_fits.csv": component_fits,
    }
    generated = []
    for name, frame in tables.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        generated.append(path)
    report_path = args.output_dir / "report.md"
    write_markdown_report(report_path, gate, findings, summary)
    generated.append(report_path)
    render_report(
        args.output_dir,
        summary,
        weights,
        comparison,
        sensitivity_frame,
        fit_metrics,
        prior_results,
        component_fits,
        figures,
        gate,
        findings,
    )
    generated.append(args.output_dir / "index.html")
    write_manifest(args.output_dir, generated, component_fits)
    print(json.dumps({"assessment": gate, "findings": findings, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
