# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark canonical single-phase DSP by fitting every atomic objective.

Each atomic metric gets an independent full canonical DSP fit. Mixture-blocked
outer folds measure generalization, and nonlinear shape selection is repeated
inside each outer training split. Published aggregates are reconstructed only
with their fixed evaluator aggregation rule; no aggregate labels are fitted.

Work is saved atomically at ``(panel, target, component, repeat, fold)``
granularity, so the complete benchmark can be resumed without repeating fits.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_dsp_single_phase_ladder_20260824 as dsp_ladder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_olmix_swarm_single_phase_dsp_20260901 as olmix_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_surrogates_20260824 as single_phase,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm39,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_componentwise_canonical_dsp_20260902"
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OLMIX_INPUT_DIR = olmix_benchmark.DEFAULT_INPUT_DIR
TABLE9_METADATA = REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_parity_panel_300m_20260628" / "component_metadata.json"
SIXTY_M_PANEL = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724" / "fit_single_phase.csv"
SIXTY_M_TABLE9 = REFERENCE_OUTPUTS / "60m_table9_gap_completion_20260725" / "table9_eval_results.csv"
THREE_HUNDRED_M_TABLE9 = (
    REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_parity_panel_300m_20260628" / "one_phase_augmented_fit_panel.csv"
)
DELPHI_COMPONENTS = REFERENCE_OUTPUTS / "delphi_3e18_observed_components_20260724" / "observed_component_panel.csv"
OLMIX_REFERENCE_METRICS = REFERENCE_OUTPUTS / "olmix_swarm_single_phase_dsp_20260901" / "aggregate_metrics.csv"

PANEL_NAMES = (
    "60m_39bucket",
    "300m_39bucket",
    "delphi_3e18_39bucket",
    "dclm_10k",
    "high_quality_10k",
)
OUTER_FOLDS = 5
INNER_FOLDS = 3
FOLD_SEED = 20_260_902
UNCHEATABLE_AGGREGATE = "eval/uncheatable_eval/bpb"
UNCHEATABLE_COMPONENTS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
# Fixed byte-mixture weights recovered from each evaluator payload generation.
# The older 60M/300M jobs share one payload and Delphi used a later payload.
# Both reconstruct their published micro BPB below 2e-7; neither is estimated
# inside a benchmark fold.
LEGACY_UNCHEATABLE_WEIGHTS = np.asarray(
    [
        0.1725475920,
        0.1456785645,
        0.1673678370,
        0.1207508824,
        0.1540951309,
        0.1479443173,
        0.0916157061,
    ],
    dtype=float,
)
DELPHI_UNCHEATABLE_WEIGHTS = np.asarray(
    [
        0.1725032419832083,
        0.1459150285318545,
        0.1673147061106415,
        0.1206224088062254,
        0.1543450059149921,
        0.1478598606213357,
        0.0914397685598754,
    ],
    dtype=float,
)
RECONSTRUCTION_TOLERANCE = 3e-6


@dataclasses.dataclass(frozen=True)
class TargetGroup:
    name: str
    components: tuple[str, ...]
    outcomes: np.ndarray
    aggregate: np.ndarray
    aggregation_weights: np.ndarray
    aggregation: str


@dataclasses.dataclass(frozen=True)
class Panel:
    name: str
    runs: tuple[str, ...]
    buckets: tuple[str, ...]
    weights: np.ndarray
    exposures: np.ndarray
    groups: tuple[TargetGroup, ...]
    input_hashes: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


@dataclasses.dataclass(frozen=True)
class FitTask:
    panel: Panel
    group: TargetGroup
    component_index: int
    split: Fold
    path: Path
    protocol_hash: str
    compatible_protocol_hashes: tuple[str, ...]
    maxiter: int
    restarts: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_matrix(name: str, matrix: np.ndarray, rows: int) -> None:
    if matrix.shape[0] != rows or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: invalid matrix shape or values: {matrix.shape}")


def _target_group(
    *,
    name: str,
    components: tuple[str, ...],
    outcomes: np.ndarray,
    aggregate: np.ndarray,
    aggregation_weights: np.ndarray,
    aggregation: str,
) -> TargetGroup:
    if outcomes.shape != (len(aggregate), len(components)):
        raise ValueError(f"{name}: component matrix has shape {outcomes.shape}")
    if len(aggregation_weights) != len(components) or not np.isclose(aggregation_weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"{name}: invalid aggregation weights")
    reconstructed = outcomes @ aggregation_weights
    error = float(np.max(np.abs(reconstructed - aggregate)))
    if error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(f"{name}: component reconstruction differs from aggregate by {error:.3g}")
    return TargetGroup(
        name=name,
        components=components,
        outcomes=outcomes,
        aggregate=aggregate,
        aggregation_weights=aggregation_weights,
        aggregation=aggregation,
    )


def table9_components() -> tuple[str, ...]:
    payload = json.loads(TABLE9_METADATA.read_text())
    components = tuple(str(component) for component in payload["components"])
    if len(components) != 51 or len(set(components)) != 51:
        raise ValueError("Expected the fixed 51-component Table-9 inventory")
    return components


def _sixty_m_table9_column(component: str) -> str:
    prefix = "olmo_base_eval/easy_bpb/"
    if component.startswith(prefix):
        return f"table9/{component.removeprefix(prefix)}"
    return f"table9/{component}/bpb"


def _singleavg_row_id(name: str) -> str:
    return name if name.startswith("singleavg_") else f"singleavg_{name}"


def _reindex_unique(frame: pd.DataFrame, identity: str, runs: tuple[str, ...], label: str) -> pd.DataFrame:
    selected = frame[frame[identity].astype(str).isin(runs)].copy()
    if selected[identity].duplicated().any():
        duplicates = selected.loc[selected[identity].duplicated(False), identity].astype(str).tolist()
        raise ValueError(f"{label}: duplicate identities {duplicates}")
    selected = selected.set_index(identity).reindex(runs)
    if selected.index.hasnans or selected.isna().all(axis=1).any():
        missing = selected.index[selected.isna().all(axis=1)].astype(str).tolist()
        raise ValueError(f"{label}: missing rows {missing}")
    return selected


def _load_uncheatable(
    path: Path,
    runs: tuple[str, ...],
    expected: np.ndarray,
    aggregation_weights: np.ndarray,
    label: str,
) -> TargetGroup:
    frame = _reindex_unique(pd.read_csv(path), "row_id", runs, label)
    aggregate = frame[UNCHEATABLE_AGGREGATE].to_numpy(float)
    if float(np.max(np.abs(aggregate - expected))) > RECONSTRUCTION_TOLERANCE:
        raise ValueError(f"{label}: frozen W&B aggregate does not match canonical panel")
    return _target_group(
        name="uncheatable",
        components=UNCHEATABLE_COMPONENTS,
        outcomes=frame.loc[:, UNCHEATABLE_COMPONENTS].to_numpy(float),
        aggregate=aggregate,
        aggregation_weights=aggregation_weights,
        aggregation="fixed byte-weighted micro BPB",
    )


def _load_60m(output_dir: Path) -> Panel:
    base = pd.read_csv(SIXTY_M_PANEL)
    runs = tuple(base["run_name"].astype(str))
    buckets, c0, c1, _family_index, _family_names = swarm39._exposure("delphi_3e18_two_phase_fit")
    weights = base.loc[:, [f"phase_0_{bucket}" for bucket in buckets]].to_numpy(float)
    exposures = weights * (c0 + c1)[None, :]

    components = table9_components()
    table9 = _reindex_unique(pd.read_csv(SIXTY_M_TABLE9), "run_name", runs, "60M Table-9")
    table9_outcomes = table9.loc[:, [_sixty_m_table9_column(component) for component in components]].to_numpy(float)
    table9_group = _target_group(
        name="table9",
        components=components,
        outcomes=table9_outcomes,
        aggregate=base["table9_macro_bpb"].to_numpy(float),
        aggregation_weights=np.full(len(components), 1.0 / len(components)),
        aggregation="unweighted 51-component mean",
    )
    uncheatable_path = output_dir / "input" / "60m_uncheatable_components.csv"
    uncheatable = _load_uncheatable(
        uncheatable_path,
        runs,
        base["uncheatable_bpb"].to_numpy(float),
        LEGACY_UNCHEATABLE_WEIGHTS,
        "60M Uncheatable",
    )
    paths = (SIXTY_M_PANEL, SIXTY_M_TABLE9, TABLE9_METADATA, uncheatable_path, swarm39.CATALOG)
    return Panel(
        name="60m_39bucket",
        runs=runs,
        buckets=buckets,
        weights=weights,
        exposures=exposures,
        groups=(uncheatable, table9_group),
        input_hashes={str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
    )


def _table9_group_from_frame(
    frame: pd.DataFrame,
    runs: tuple[str, ...],
    aggregate: np.ndarray,
    *,
    identity: str,
    label: str,
) -> TargetGroup:
    indexed = _reindex_unique(frame, identity, runs, label)
    components = table9_components()
    return _target_group(
        name="table9",
        components=components,
        outcomes=indexed.loc[:, components].to_numpy(float),
        aggregate=aggregate,
        aggregation_weights=np.full(len(components), 1.0 / len(components)),
        aggregation="unweighted 51-component mean",
    )


def _load_300m(output_dir: Path) -> Panel:
    panel = single_phase.one_phase_panel("300m")
    runs = tuple(str(value) for value in panel.row_id)
    source = pd.read_csv(THREE_HUNDRED_M_TABLE9)
    benchmark_row_id = source["run_name"].astype(str).map(_singleavg_row_id)
    source = pd.concat([source, benchmark_row_id.rename("benchmark_row_id")], axis=1)
    table9 = _table9_group_from_frame(
        source,
        runs,
        panel.targets[swarm39.TABLE9],
        identity="benchmark_row_id",
        label="300M Table-9",
    )
    uncheatable_path = output_dir / "input" / "300m_uncheatable_components.csv"
    uncheatable = _load_uncheatable(
        uncheatable_path,
        runs,
        panel.targets[swarm39.UNCHEATABLE],
        LEGACY_UNCHEATABLE_WEIGHTS,
        "300M Uncheatable",
    )
    paths = (
        swarm39.CANONICAL / "300m_one_phase_fit.csv",
        THREE_HUNDRED_M_TABLE9,
        TABLE9_METADATA,
        uncheatable_path,
        swarm39.CATALOG,
    )
    return Panel(
        name="300m_39bucket",
        runs=runs,
        buckets=panel.buckets,
        weights=panel.phase0,
        exposures=panel.epochs,
        groups=(uncheatable, table9),
        input_hashes={str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
    )


def _load_delphi() -> Panel:
    panel = single_phase.one_phase_panel("delphi_3e18")
    runs = tuple(str(value) for value in panel.row_id)
    components = pd.read_csv(DELPHI_COMPONENTS)
    components = components[components["panel"].eq("one_phase_fit")].copy()
    table9 = _table9_group_from_frame(
        components,
        runs,
        panel.targets[swarm39.TABLE9],
        identity="row_name",
        label="Delphi Table-9",
    )
    uncheatable_frame = _reindex_unique(components, "row_name", runs, "Delphi Uncheatable")
    uncheatable = _target_group(
        name="uncheatable",
        components=UNCHEATABLE_COMPONENTS,
        outcomes=uncheatable_frame.loc[:, UNCHEATABLE_COMPONENTS].to_numpy(float),
        aggregate=panel.targets[swarm39.UNCHEATABLE],
        aggregation_weights=DELPHI_UNCHEATABLE_WEIGHTS,
        aggregation="fixed byte-weighted micro BPB",
    )
    paths = (
        swarm39.CANONICAL / "delphi_3e18_one_phase_fit.csv",
        DELPHI_COMPONENTS,
        TABLE9_METADATA,
        swarm39.CATALOG,
    )
    return Panel(
        name="delphi_3e18_39bucket",
        runs=runs,
        buckets=panel.buckets,
        weights=panel.phase0,
        exposures=panel.epochs,
        groups=(uncheatable, table9),
        input_hashes={str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
    )


def _load_olmix(name: str) -> Panel:
    pool = olmix_benchmark.load_pool(OLMIX_INPUT_DIR, name)
    weights = np.full(len(pool.tasks), 1.0 / len(pool.tasks))
    group = _target_group(
        name="native_42_task_mean",
        components=pool.tasks,
        outcomes=pool.outcomes,
        aggregate=pool.outcomes.mean(axis=1),
        aggregation_weights=weights,
        aggregation="unweighted 42-task mean",
    )
    return Panel(
        name=name,
        runs=pool.runs,
        buckets=pool.buckets,
        weights=pool.weights,
        exposures=pool.exposures,
        groups=(group,),
        input_hashes=pool.input_hashes,
    )


def load_panel(name: str, output_dir: Path) -> Panel:
    if name == "60m_39bucket":
        return _load_60m(output_dir)
    if name == "300m_39bucket":
        return _load_300m(output_dir)
    if name == "delphi_3e18_39bucket":
        return _load_delphi()
    if name in olmix_benchmark.POOLS:
        return _load_olmix(name)
    raise ValueError(f"Unknown panel {name!r}")


def panel_folds(panel: Panel, repeats: int) -> tuple[Fold, ...]:
    rows = np.arange(len(panel.runs))
    result: list[Fold] = []
    for repeat in range(repeats):
        labels = olmix_benchmark.block_labels(panel.weights, OUTER_FOLDS, FOLD_SEED + 100 * repeat)
        for fold in range(OUTER_FOLDS):
            result.append(Fold(repeat, fold, rows[labels != fold], rows[labels == fold]))
    return tuple(result)


def _component_token(component: str) -> str:
    label = component.removeprefix("eval/").removeprefix("olmo_base_eval/easy_bpb/").removesuffix("/bpb")
    return label.replace("/", "__")


def shard_path(output_dir: Path, panel: str, group: str, component_index: int, component: str, split: Fold) -> Path:
    filename = f"component_{component_index:03d}_{_component_token(component)}.npz"
    return output_dir / "shards" / panel / group / f"repeat_{split.repeat:02d}" / f"fold_{split.fold:02d}" / filename


def atomic_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def valid_shard(task: FitTask) -> bool:
    if not task.path.is_file() or task.path.stat().st_size == 0:
        return False
    try:
        with np.load(task.path) as payload:
            return (
                str(payload["protocol_hash"].item()) in task.compatible_protocol_hashes
                and str(payload["component"].item()) == task.group.components[task.component_index]
                and np.array_equal(payload["test"], task.split.test)
                and np.asarray(payload["prediction"]).shape == (len(task.split.test),)
                and np.isfinite(payload["prediction"]).all()
            )
    except (KeyError, OSError, ValueError):
        return False


def _seed(task: FitTask) -> int:
    identity = f"{task.panel.name}|{task.group.name}|{task.component_index}|" f"{task.split.repeat}|{task.split.fold}"
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def fit_task(task: FitTask) -> str:
    if valid_shard(task):
        return "cached"
    response = task.group.outcomes[:, task.component_index]
    inner_labels = olmix_benchmark.block_labels(
        task.panel.weights[task.split.train],
        INNER_FOLDS,
        FOLD_SEED + 10_000 * task.split.repeat + 100 * task.split.fold + task.component_index,
    )
    inner_folds = tuple(
        (np.flatnonzero(inner_labels != fold), np.flatnonzero(inner_labels == fold)) for fold in range(INNER_FOLDS)
    )
    canonical = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    started = time.monotonic()
    vector, intercept, coefficients = dsp_ladder.fit_rung(
        task.panel.exposures[task.split.train],
        response[task.split.train],
        canonical,
        inner_folds,
        (),
        seed=_seed(task),
        maxiter=task.maxiter,
        restarts=task.restarts,
    )
    prediction = (
        intercept
        + dsp_ladder.rung_design(task.panel.exposures[task.split.test], vector, canonical, len(task.panel.buckets))
        @ coefficients
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"{task.panel.name}/{task.group.name}: nonfinite canonical DSP prediction")
    atomic_save(
        task.path,
        protocol_hash=np.asarray(task.protocol_hash),
        component=np.asarray(task.group.components[task.component_index]),
        component_index=np.asarray(task.component_index),
        test=task.split.test,
        observed=response[task.split.test],
        prediction=prediction,
        constant_prediction=np.full(len(task.split.test), response[task.split.train].mean()),
        vector=vector,
        intercept=np.asarray(intercept),
        coefficients=coefficients,
        elapsed=np.asarray(time.monotonic() - started),
    )
    return "fitted"


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    order = np.argsort(predicted)
    constant_prediction = bool(np.ptp(predicted) <= 1e-12)
    spearman = 0.0 if constant_prediction else float(stats.spearmanr(predicted, actual).statistic)
    calibration = 0.0 if constant_prediction else float(stats.linregress(predicted, actual).slope)
    return {
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "spearman": spearman,
        "calibration_slope": calibration,
        "selection_regret": float(actual[order[0]] - actual.min()),
        "regret_at_5": float(actual[order[:5]].min() - actual.min()),
    }


def collect_predictions(tasks: list[FitTask]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in tasks:
        if not valid_shard(task):
            continue
        with np.load(task.path) as payload:
            observed = np.asarray(payload["observed"], dtype=float)
            for model, key in (("canonical_dsp", "prediction"), ("fold_mean", "constant_prediction")):
                prediction = np.asarray(payload[key], dtype=float)
                for local, row_index in enumerate(task.split.test):
                    rows.append(
                        {
                            "panel": task.panel.name,
                            "target": task.group.name,
                            "component_index": task.component_index,
                            "component": task.group.components[task.component_index],
                            "repeat": task.split.repeat,
                            "fold": task.split.fold,
                            "row_index": int(row_index),
                            "run": task.panel.runs[row_index],
                            "model": model,
                            "observed": observed[local],
                            "prediction": prediction[local],
                        }
                    )
    return pd.DataFrame(rows)


def component_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["panel", "target", "component_index", "component", "repeat", "model"]
    for identity, group in predictions.groupby(keys, sort=False):
        rows.append(
            dict(zip(keys, identity, strict=True))
            | metric_row(group["observed"].to_numpy(), group["prediction"].to_numpy())
        )
    return pd.DataFrame(rows)


def aggregate_predictions(predictions: pd.DataFrame, panels: tuple[Panel, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel in panels:
        for target in panel.groups:
            selected = predictions[predictions["panel"].eq(panel.name) & predictions["target"].eq(target.name)]
            for (repeat, model), group in selected.groupby(["repeat", "model"], sort=False):
                pivot = group.pivot(index="row_index", columns="component_index", values="prediction")
                if pivot.shape != (len(panel.runs), len(target.components)):
                    continue
                pivot = pivot.reindex(index=np.arange(len(panel.runs)), columns=np.arange(len(target.components)))
                values = pivot.to_numpy(float) @ target.aggregation_weights
                fold_by_row = group[["row_index", "fold"]].drop_duplicates()
                if fold_by_row["row_index"].duplicated().any() or len(fold_by_row) != len(panel.runs):
                    raise ValueError(f"{panel.name}/{target.name}: inconsistent outer-fold identity")
                fold_lookup = fold_by_row.set_index("row_index")["fold"]
                for row_index, prediction in enumerate(values):
                    rows.append(
                        {
                            "panel": panel.name,
                            "target": target.name,
                            "repeat": int(repeat),
                            "fold": int(fold_lookup.loc[row_index]),
                            "row_index": row_index,
                            "run": panel.runs[row_index],
                            "model": model,
                            "observed": target.aggregate[row_index],
                            "prediction": prediction,
                        }
                    )
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for identity, group in predictions.groupby(keys, sort=False):
        rows.append(
            dict(zip(keys, identity, strict=True))
            | metric_row(group["observed"].to_numpy(), group["prediction"].to_numpy())
        )
    return pd.DataFrame(rows)


def aggregate_metrics(repeat_metrics: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["panel", "target", "model"]
    for identity, group in repeat_metrics.groupby(keys, sort=False):
        row: dict[str, object] = dict(zip(keys, identity, strict=True))
        for metric in ("rmse", "mae", "spearman", "calibration_slope", "selection_regret", "regret_at_5"):
            row[metric] = float(group[metric].mean())
            row[f"{metric}_repeat_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
        selected_folds = fold_metrics
        for key, value in zip(keys, identity, strict=True):
            selected_folds = selected_folds[selected_folds[key].eq(value)]
        row["mean_fold_selection_regret"] = float(selected_folds["selection_regret"].mean())
        row["mean_fold_regret_at_5"] = float(selected_folds["regret_at_5"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def olmix_model_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "panel",
        "model",
        "rmse",
        "rmse_repeat_sd",
        "spearman",
        "spearman_repeat_sd",
        "mean_fold_selection_regret",
    ]
    canonical = aggregate[
        aggregate["panel"].isin(olmix_benchmark.POOLS)
        & aggregate["target"].eq("native_42_task_mean")
        & aggregate["model"].eq("canonical_dsp")
    ].loc[:, [column for column in columns if column != "model"] + ["model"]]
    canonical = canonical.loc[:, columns]
    if not OLMIX_REFERENCE_METRICS.is_file():
        return canonical

    reference = pd.read_csv(OLMIX_REFERENCE_METRICS).rename(columns={"pool": "panel", "variant": "model"})
    selected_models = ("olmix_exact_macro", "linear_epoch_log_link", "dsp_benefit_log_link")
    reference = reference[reference["panel"].isin(olmix_benchmark.POOLS) & reference["model"].isin(selected_models)].loc[
        :, columns
    ]
    order = {"canonical_dsp": 0, "olmix_exact_macro": 1, "linear_epoch_log_link": 2, "dsp_benefit_log_link": 3}
    comparison = pd.concat([canonical, reference], ignore_index=True)
    comparison["model_order"] = comparison["model"].map(order)
    return comparison.sort_values(["panel", "model_order"]).drop(columns="model_order").reset_index(drop=True)


def write_report(
    output_dir: Path,
    panels: tuple[Panel, ...],
    component: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Componentwise canonical DSP benchmark",
        "",
        "Canonical DSP is fitted independently to every atomic metric under mixture-blocked outer cross-validation. "
        "Aggregate predictions are reconstructed after fitting; aggregate labels never enter the component fits.",
        "",
        "## Panels",
        "",
        "| panel | rows | buckets | target groups | atomic fits per repeat |",
        "|---|---:|---:|---|---:|",
    ]
    for panel in panels:
        target_names = ", ".join(group.name for group in panel.groups)
        atomic_fits = sum(len(group.components) for group in panel.groups)
        panel_summary = f"| {panel.name} | {len(panel.runs)} | {len(panel.buckets)} | {target_names}"
        lines.append(f"{panel_summary} | {atomic_fits} |")
    if not aggregate.empty:
        display_columns = [
            "panel",
            "target",
            "rmse",
            "rmse_repeat_sd",
            "spearman",
            "spearman_repeat_sd",
            "calibration_slope",
            "mean_fold_selection_regret",
        ]
        canonical = aggregate[aggregate["model"].eq("canonical_dsp")].loc[:, display_columns]
        lines.extend(["", "## Reconstructed aggregates", "", canonical.to_markdown(index=False, floatfmt=".6f")])
    if not comparison.empty:
        lines.extend(
            [
                "",
                "## Existing Michael-panel baselines",
                "",
                "These rows use the same complete panels and five-repeat mixture-blocked protocol family. "
                "The older baselines use an independently seeded fold partition, so the table supports aggregate "
                "comparison but not paired-fold inference.",
                "",
                comparison.to_markdown(index=False, floatfmt=".6f"),
            ]
        )
    if not component.empty:
        summary = (
            component[component["model"].eq("canonical_dsp")]
            .groupby(["panel", "target"], as_index=False)
            .agg(
                median_component_rmse=("rmse", "median"),
                median_component_spearman=("spearman", "median"),
                median_component_selection_regret=("selection_regret", "median"),
            )
        )
        lines.extend(["", "## Atomic metrics", "", summary.to_markdown(index=False, floatfmt=".6f")])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Canonical DSP is a credible baseline on the 39-bucket panels: aggregate rank correlation is high and "
            "repeat dispersion is small. Full per-bucket canonical DSP is not competitive on the 118/120-bucket "
            "Michael panels. Each atomic fit there has 4B+1 fitted quantities (two nonlinear shapes and two "
            "nonnegative amplitudes per bucket, plus an intercept), exceeding the roughly 290 training rows in an "
            "outer fold. Its unstable magnitude estimates are therefore an identification failure of this model/data "
            "regime, not evidence against epoch exposure itself.",
            "",
            "Selection regret is measured against the best observed row in the same complete panel. Uncheatable uses "
            "its frozen byte-weighted micro-BPB aggregation; Table-9 and the two OLMix proxy swarms use unweighted "
            "task means. The full fold-mean calibration-floor results remain available in aggregate_metrics.csv.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_selection(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value == "all":
        return allowed
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(selected) - set(allowed)
    if not selected or unknown:
        raise ValueError(f"Unknown selection: {sorted(unknown)}")
    return selected


def protocol_payload(args: argparse.Namespace, panels: tuple[Panel, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": "canonical single-phase DSP, full per-bucket shape",
        "fit_granularity": "independent atomic component",
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": args.outer_repeats,
        "inner_folds": INNER_FOLDS,
        "fold_seed": FOLD_SEED,
        "canonical_maxiter": args.canonical_maxiter,
        "canonical_restarts": args.canonical_restarts,
        "panels": [panel.name for panel in panels],
        "input_hashes": {panel.name: panel.input_hashes for panel in panels},
        "uncheatable_aggregation_weights": {
            "legacy_60m_300m": dict(zip(UNCHEATABLE_COMPONENTS, LEGACY_UNCHEATABLE_WEIGHTS, strict=True)),
            "delphi_3e18": dict(zip(UNCHEATABLE_COMPONENTS, DELPHI_UNCHEATABLE_WEIGHTS, strict=True)),
        },
        "optimizer_sha256": file_sha256(Path(dsp_ladder.__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panels", default="all")
    parser.add_argument("--targets", default="all", help="Comma-separated target groups after panels load")
    parser.add_argument("--outer-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 1) - 2)))
    parser.add_argument("--canonical-maxiter", type=int, default=36)
    parser.add_argument("--canonical-restarts", type=int, default=2)
    parser.add_argument("--component-limit", type=int)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.outer_repeats < 1 or args.workers < 1 or args.canonical_restarts < 1 or args.canonical_maxiter < 1:
        raise ValueError("repeat, worker, restart, and iteration counts must be positive")

    selected_names = parse_selection(args.panels, PANEL_NAMES)
    panels = tuple(load_panel(name, args.output_dir) for name in selected_names)
    available_targets = tuple(dict.fromkeys(group.name for panel in panels for group in panel.groups))
    selected_targets = parse_selection(args.targets, available_targets)
    protocol = protocol_payload(args, panels)
    fit_protocol = dict(protocol)
    fit_protocol.pop("outer_repeats")
    protocol_hash = hashlib.sha256(json.dumps(fit_protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    legacy_one_repeat_protocol = dict(protocol)
    legacy_one_repeat_protocol["outer_repeats"] = 1
    legacy_one_repeat_hash = hashlib.sha256(
        json.dumps(legacy_one_repeat_protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    compatible_protocol_hashes = (protocol_hash, legacy_one_repeat_hash)
    protocol["fit_protocol_hash"] = protocol_hash
    protocol["compatible_shard_protocol_hashes"] = list(compatible_protocol_hashes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    registry = {
        panel.name: {
            group.name: {
                "components": list(group.components),
                "aggregation": group.aggregation,
                "aggregation_weights": group.aggregation_weights.tolist(),
            }
            for group in panel.groups
            if group.name in selected_targets
        }
        for panel in panels
    }
    (args.output_dir / "target_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")

    tasks: list[FitTask] = []
    for panel in panels:
        splits = panel_folds(panel, args.outer_repeats)
        if args.fold_limit is not None:
            splits = splits[: args.fold_limit]
        for group in panel.groups:
            if group.name not in selected_targets:
                continue
            component_count = len(group.components)
            if args.component_limit is not None:
                component_count = min(component_count, args.component_limit)
            for component_index in range(component_count):
                for split in splits:
                    tasks.append(
                        FitTask(
                            panel=panel,
                            group=group,
                            component_index=component_index,
                            split=split,
                            path=shard_path(
                                args.output_dir,
                                panel.name,
                                group.name,
                                component_index,
                                group.components[component_index],
                                split,
                            ),
                            protocol_hash=protocol_hash,
                            compatible_protocol_hashes=compatible_protocol_hashes,
                            maxiter=args.canonical_maxiter,
                            restarts=args.canonical_restarts,
                        )
                    )
    cached = sum(valid_shard(task) for task in tasks)
    print(f"benchmark tasks: {len(tasks)} total, {cached} cached, {len(tasks) - cached} pending", flush=True)
    if not args.compile_only and cached < len(tasks):
        with parallel_config(backend="loky", inner_max_num_threads=1):
            status = Parallel(n_jobs=args.workers, verbose=10)(delayed(fit_task)(task) for task in tasks)
        print(f"completed: fitted={status.count('fitted')}, cached={status.count('cached')}", flush=True)

    component_predictions_frame = collect_predictions(tasks)
    component_metrics_frame = (
        component_metrics(component_predictions_frame) if not component_predictions_frame.empty else pd.DataFrame()
    )
    aggregate_predictions_frame = (
        aggregate_predictions(component_predictions_frame, panels)
        if not component_predictions_frame.empty
        else pd.DataFrame()
    )
    aggregate_repeat_metrics_frame = (
        grouped_metrics(aggregate_predictions_frame, ["panel", "target", "repeat", "model"])
        if not aggregate_predictions_frame.empty
        else pd.DataFrame()
    )
    aggregate_fold_metrics_frame = (
        grouped_metrics(aggregate_predictions_frame, ["panel", "target", "repeat", "fold", "model"])
        if not aggregate_predictions_frame.empty
        else pd.DataFrame()
    )
    aggregate_metrics_frame = (
        aggregate_metrics(aggregate_repeat_metrics_frame, aggregate_fold_metrics_frame)
        if not aggregate_repeat_metrics_frame.empty
        else pd.DataFrame()
    )
    comparison_frame = (
        olmix_model_comparison(aggregate_metrics_frame) if not aggregate_metrics_frame.empty else pd.DataFrame()
    )
    outputs = {
        "component_predictions.csv": component_predictions_frame,
        "component_metrics.csv": component_metrics_frame,
        "aggregate_predictions.csv": aggregate_predictions_frame,
        "aggregate_repeat_metrics.csv": aggregate_repeat_metrics_frame,
        "aggregate_fold_metrics.csv": aggregate_fold_metrics_frame,
        "aggregate_metrics.csv": aggregate_metrics_frame,
        "olmix_model_comparison.csv": comparison_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    write_report(args.output_dir, panels, component_metrics_frame, aggregate_metrics_frame, comparison_frame)
    print(f"wrote benchmark outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
