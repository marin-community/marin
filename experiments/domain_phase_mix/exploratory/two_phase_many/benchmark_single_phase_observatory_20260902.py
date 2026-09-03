# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///
"""Shared resumable benchmark for every single-phase Observatory model and its ablations.

One harness fits every registered model on the same panels, targets, outer folds, and inner
folds. Work is saved atomically per ``(model, panel, target, component, repeat, fold)`` shard and
resumes without recomputation. Tiers only select which shards a run needs: ``smoke`` validates
an adapter on one fold, ``screen`` compares mechanisms on the atomic anchors, ``certify`` covers
every component of every panel for the leaderboard, and ``finalist`` adds repeated outer
partitions for the named finalists.

Aggregates are reconstructed from atomic predictions with the frozen evaluator rule; no aggregate
label is fitted. The external heldout stage refits every model on the complete canonical fit
panel, freezes and hashes its predictions on the coordinate-disjoint registry, and only then joins
them to measured outcomes.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_olmix_swarm_single_phase_dsp_20260901 as olmix_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_componentwise_canonical_dsp_20260902 as canonical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_surrogates_20260824 as single_phase,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_starcoder_inputs_20260902 as starcoder_curves,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_epoch_accounting as epoch_accounting,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm39,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_observatory_benchmark_20260902"
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
CANONICAL_INPUT_DIR = canonical.DEFAULT_OUTPUT_DIR
HELDOUT_DIR = REFERENCE_OUTPUTS / "single_phase_heldout_benchmark_20260902"
STARCODER_INVENTORY_DIR = starcoder_curves.INVENTORY_DIR
TIED_DIAGONAL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_tied_diagonal_20260730" / "results_20260731"
SIXTY_M_REPEATS = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724" / "repeat_observations.csv"
DELPHI_NOISE_DIR = REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703"
PROTOCOL_SCHEMA_VERSION = 1
OUTER_FOLDS = 5
INNER_FOLDS = 3
FOLD_SEED = canonical.FOLD_SEED
HELDOUT_INNER_SEED = FOLD_SEED + 90_000
OUTCOME_PERMUTATION_SEED = 20_260_902
BASIN_FRACTION = 0.15
BASIN_MIN_ROWS = 5
TOP_K = 5
BASIN_TOLERANCE_SD = 1.0
FIVE_MINUTE_TARGET = 300.0
EIGHT_MINUTE_FAIL = 480.0
TABULAR_PANELS = ("60m_39bucket", "300m_39bucket", "delphi_3e18_39bucket", "dclm_10k", "high_quality_10k")
THIRTY_NINE_BUCKET_PANELS = ("60m_39bucket", "300m_39bucket", "delphi_3e18_39bucket")
MICHAEL_PANELS = ("dclm_10k", "high_quality_10k")
STARCODER_PANEL_PREFIX = "starcoder::"
STARCODER_TARGET = "programming_languages_bpb"
STARCODER_COMPONENT = starcoder_curves.PRIMARY_TARGET
MICHAEL_TASKS = (
    "codex_humaneval/gold_bpb_3shot",
    "mt_mbpp_python/gold_bpb_3shot",
    "mt_mbpp_cpp/gold_bpb_3shot",
    "gsm8k/gold_bpb_5shot",
    "naturalqs_open/bpb_5shot",
    "sciq/rc_5shot",
    "mmlu_stem/rc_5shot",
    "mmlu_humanities/rc_5shot",
)
MICHAEL_TARGET = "frozen_8_task_mean"
UNCHEATABLE_ANCHOR_TOKENS = ("github_python", "github_cpp")
TABLE9_ANCHOR_TOKENS = ("mt_mbpp_python", "mt_mbpp_cpp", "minerva_math_geometry", "hellaswag")
SMOKE_CURVE = "fixed_model_wsd80_1b__endpoint"
FIXED_MODEL_CURVES = {
    "fixed_model_wsd80_1b__endpoint": 1_000_000_000,
    "fixed_model_wsd80_2b__endpoint": 2_000_000_000,
    "fixed_model_wsd80_4b__endpoint": 4_000_000_000,
    "fixed_model_wsd80_8b__endpoint": 8_000_000_000,
}
COMPARATORS = ("dsp_total_exposure", "olmix_loglinear_taskwise")
TIERS = ("smoke", "screen", "certify", "finalist")
METRIC_NAMES = (
    "rmse",
    "mae",
    "spearman",
    "calibration_intercept",
    "calibration_slope",
    "regret_at_1",
    "regret_at_top_k",
    "selection_optimism",
    "basin_rmse",
    "basin_spearman",
)
PROMOTION_AMENDMENT = (
    "Post hoc (after Screen results were seen): raw pooled differences are dominated by Michael-panel units "
    "whose predictions explode by orders of magnitude, so no raw interval excludes zero for any ablation. An "
    "ablation is additionally promoted when its relative RMSE difference (a - b) / b has a corrected 95% "
    "interval excluding zero or when a two-sided sign test over units on the fold-averaged RMSE difference "
    "has p < 0.05. The frozen decision is reported separately as promoted_by_frozen_rule."
)
PROMOTION_RULE = (
    "An ablation is promoted to Certify when (a) its paired Screen contrast against its parent on "
    "anchor RMSE or anchor regret-at-1, pooled over the six 39-bucket anchors, the eight Michael "
    "tasks, and the 45 StarCoder curves, has a Nadeau-Bengio corrected 95% interval excluding zero, "
    "or (b) it is the matched control (permuted inventory, shuffled families, scrambled harm, signed head, or "
    "retention-gate removal) needed to attribute a parent that beats canonical DSP or taskwise OLMix "
    "on a paper-primary Certify metric. The rule was frozen before any Screen result was inspected; the "
    "45 StarCoder curves enter as four family-macro units (a post-review correction of the unit weighting)."
)


# ---------------------------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BenchPanel:
    """One benchmark source: policies, single-phase features, atomic targets, and noise."""

    name: str
    kind: str  # tabular | curve
    family: str
    runs: tuple[str, ...]
    buckets: tuple[str, ...]
    features: models.Features
    groups: tuple[canonical.TargetGroup, ...]
    input_hashes: dict[str, str]
    repeat_sd: dict[str, float]
    component_repeat_sd: dict[str, float]
    metadata: dict[str, Any]

    @property
    def rows(self) -> int:
        return len(self.runs)

    def group(self, target: str) -> canonical.TargetGroup:
        for group in self.groups:
            if group.name == target:
                return group
        raise KeyError(f"{self.name} has no target {target}")

    def basin_rows(self, target: str) -> np.ndarray:
        aggregate = self.group(target).aggregate
        count = min(len(aggregate), max(BASIN_MIN_ROWS, int(np.ceil(BASIN_FRACTION * len(aggregate)))))
        return np.sort(np.argsort(aggregate, kind="stable")[:count])


def file_sha256(path: Path) -> str:
    return canonical.file_sha256(path)


def _sixty_m_repeat_noise() -> dict[str, float]:
    frame = pd.read_csv(SIXTY_M_REPEATS)
    single = frame[frame["policy_class"].eq("single_phase")]
    grouped = single.groupby("policy_hash")
    result = {}
    for target, column in (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb")):
        deviations = grouped[column].agg(lambda values: values.std(ddof=1) if values.count() > 1 else np.nan).dropna()
        result[target] = float(np.sqrt(np.mean(deviations**2))) if len(deviations) else float("nan")
    return result


def _delphi_repeat_noise() -> tuple[dict[str, float], dict[str, float]]:
    summary = json.loads((DELPHI_NOISE_DIR / "noise_floor_summary.json").read_text())
    heldout = pd.read_csv(HELDOUT_DIR / "heldout_runs.csv")
    delphi = heldout[heldout["panel"].eq("delphi_3e18_39bucket")]
    grouped = delphi.groupby("coordinate_id")
    pooled = {}
    for target, column, floor_key in (
        ("uncheatable", "uncheatable_bpb", "uncheatable_bpb_sd"),
        ("table9", "table9_macro_bpb", "table9_macro_bpb_sd"),
    ):
        deviations = grouped[column].agg(lambda values: values.std(ddof=1) if values.count() > 1 else np.nan).dropna()
        variances = np.concatenate([deviations.to_numpy(float) ** 2, [float(summary[floor_key]) ** 2]])
        pooled[target] = float(np.sqrt(np.mean(variances)))
    matrix = pd.read_csv(DELPHI_NOISE_DIR / "noise_component_matrix.csv", index_col=0)
    components = {}
    table9 = set(canonical.table9_components())
    for column in matrix.columns:
        prefix = "olmo_base_easy/table9/"
        name = column.removeprefix(prefix)
        candidates = (f"olmo_base_eval/easy_bpb/{name}", name.removesuffix("/bpb"), column)
        key = next((candidate for candidate in candidates if candidate in table9), column)
        components[key] = float(matrix[column].std(ddof=1))
    return pooled, components


def _tabular_features(panel: canonical.Panel) -> models.Features:
    if panel.name == "60m_39bucket":
        _domains, c0, c1, _family_index, _family_names = swarm39._exposure("delphi_3e18_two_phase_fit")
    elif panel.name == "300m_39bucket":
        one_phase = single_phase.one_phase_panel("300m")
        c0, c1 = one_phase.c0, one_phase.c1
    elif panel.name == "delphi_3e18_39bucket":
        one_phase = single_phase.one_phase_panel("delphi_3e18")
        c0, c1 = one_phase.c0, one_phase.c1
    else:
        payload = json.loads((olmix_benchmark.DEFAULT_INPUT_DIR / panel.name / "swarm_s42_K363.json").read_text())
        inventory = np.asarray(
            [olmix_benchmark.PROXY_TOKENS / float(payload["tokens"][bucket]) for bucket in panel.buckets]
        )
        features = models.features_from_panel(
            panel.weights, inventory, panel.buckets, early_fraction=None, label=panel.name
        )
        if float(np.abs(features.exposures - panel.exposures).max()) > 1e-9:
            raise ValueError(f"{panel.name}: reconstructed exposures differ from the canonical loader")
        return features
    inventory = c0 + c1
    features = models.features_from_panel(
        panel.weights, inventory, panel.buckets, early_fraction=c0 / inventory, label=panel.name
    )
    if float(np.abs(features.exposures - panel.exposures).max()) > 1e-9:
        raise ValueError(f"{panel.name}: reconstructed exposures differ from the canonical loader")
    return features


@functools.cache
def _anchor_components() -> dict[str, tuple[str, ...]]:
    table9 = canonical.table9_components()
    anchors: dict[str, tuple[str, ...]] = {"uncheatable": tuple(), "table9": tuple()}
    anchors["uncheatable"] = tuple(
        component
        for token in UNCHEATABLE_ANCHOR_TOKENS
        for component in canonical.UNCHEATABLE_COMPONENTS
        if f"/{token}/" in component
    )
    selected = []
    for token in TABLE9_ANCHOR_TOKENS:
        matches = [component for component in table9 if token in component]
        if len(matches) != 1:
            raise ValueError(f"Table-9 anchor {token!r} matched {matches}")
        selected.append(matches[0])
    anchors["table9"] = tuple(selected)
    if len(anchors["uncheatable"]) != 2:
        raise ValueError("expected exactly two Uncheatable anchors")
    return anchors


def load_tabular_panel(name: str) -> BenchPanel:
    base = canonical.load_panel(name, CANONICAL_INPUT_DIR)
    features = _tabular_features(base)
    groups = base.groups
    repeat_sd: dict[str, float] = {}
    component_sd: dict[str, float] = {}
    metadata: dict[str, Any] = {"noise_sources": []}
    if name in MICHAEL_PANELS:
        source = base.groups[0]
        positions = [source.components.index(task) for task in MICHAEL_TASKS]
        outcomes = source.outcomes[:, positions]
        groups = (
            canonical.TargetGroup(
                name=MICHAEL_TARGET,
                components=MICHAEL_TASKS,
                outcomes=outcomes,
                aggregate=outcomes.mean(axis=1),
                aggregation_weights=np.full(len(MICHAEL_TASKS), 1.0 / len(MICHAEL_TASKS)),
                aggregation="unweighted frozen 8-task mean",
            ),
        )
    elif name == "60m_39bucket":
        repeat_sd = _sixty_m_repeat_noise()
        metadata["noise_sources"].append(str(SIXTY_M_REPEATS.relative_to(REPO_ROOT)))
    elif name == "delphi_3e18_39bucket":
        repeat_sd, component_sd = _delphi_repeat_noise()
        metadata["noise_sources"].extend(
            [
                str((DELPHI_NOISE_DIR / "noise_floor_summary.json").relative_to(REPO_ROOT)),
                str((HELDOUT_DIR / "heldout_runs.csv").relative_to(REPO_ROOT)),
            ]
        )
    else:
        metadata["noise_sources"].append("none identified for Uncheatable or Table 9 at 300M")
    return BenchPanel(
        name=name,
        kind="tabular",
        family=name,
        runs=base.runs,
        buckets=base.buckets,
        features=features,
        groups=groups,
        input_hashes=dict(base.input_hashes),
        repeat_sd=repeat_sd,
        component_repeat_sd=component_sd,
        metadata=metadata,
    )


def starcoder_inventory(metadata: pd.Series) -> np.ndarray:
    """Materialized epochs at full share for Nemotron and StarCoder on one physical curve."""
    starcoder = starcoder_curves.starcoder_epoch_scale(metadata)
    support = str(metadata["support_id"])
    nemotron = (
        float(metadata["planned_materialized_tokens"]) / epoch_accounting.NEMOTRON_SOURCE_TOKENS
        if support == "full"
        else epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.NEMOTRON_SOURCE_TOKENS
    )
    return np.asarray([nemotron, starcoder], dtype=float)


@functools.cache
def _starcoder_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    curves, points = starcoder_curves.load_inputs(STARCODER_INVENTORY_DIR)
    hashes = {
        str((STARCODER_INVENTORY_DIR / name).relative_to(REPO_ROOT)): file_sha256(STARCODER_INVENTORY_DIR / name)
        for name in starcoder_curves.INPUT_FILES
    }
    return curves, points, hashes


def starcoder_curve_ids() -> tuple[str, ...]:
    curves, _points, _hashes = _starcoder_inputs()
    return tuple(str(value) for value in curves["curve_id"])


def load_curve_panel(curve_id: str) -> BenchPanel:
    curves, points, hashes = _starcoder_inputs()
    metadata = curves.loc[curves["curve_id"].eq(curve_id)].iloc[0]
    block = points.loc[points["curve_id"].eq(curve_id)].sort_values("starcoder_weight")
    share = block["starcoder_weight"].to_numpy(float)
    response = block["bpb"].to_numpy(float)
    weights = np.column_stack([1.0 - share, share])
    inventory = starcoder_inventory(metadata)
    features = models.features_from_panel(
        weights,
        inventory,
        ("nemotron_full", "starcoder"),
        early_fraction=None,
        label=f"{STARCODER_PANEL_PREFIX}{curve_id}",
    )
    group = canonical.TargetGroup(
        name=STARCODER_TARGET,
        components=(STARCODER_COMPONENT,),
        outcomes=response[:, None],
        aggregate=response,
        aggregation_weights=np.asarray([1.0]),
        aggregation="single atomic target",
    )
    repeat_sd: dict[str, float] = {}
    extra: dict[str, Any] = {
        "family": str(metadata["family"]),
        "support_id": str(metadata["support_id"]),
        "planned_materialized_tokens": float(metadata["planned_materialized_tokens"]),
        "inventory": inventory.tolist(),
        "measured_argmin_weight": float(share[int(np.argmin(response))]),
    }
    if curve_id in FIXED_MODEL_CURVES:
        noise = pd.read_csv(TIED_DIAGONAL_DIR / "repeat_noise.csv").set_index("token_budget_requested")
        optima = pd.read_csv(TIED_DIAGONAL_DIR / "tied_optima.csv").set_index("token_budget_requested")
        budget = FIXED_MODEL_CURVES[curve_id]
        repeat_sd = {STARCODER_TARGET: float(noise.loc[budget, "repeat_sd_bpb"])}
        extra.update(
            {
                "token_budget_requested": budget,
                "sampled_min_weight": float(optima.loc[budget, "sampled_min_weight"]),
                "sampled_min_bpb": float(optima.loc[budget, "sampled_min_bpb"]),
                "one_sd_basin_low": float(optima.loc[budget, "one_sd_basin_low"]),
                "one_sd_basin_high": float(optima.loc[budget, "one_sd_basin_high"]),
            }
        )
        hashes = {
            **hashes,
            **{
                str(path.relative_to(REPO_ROOT)): file_sha256(path)
                for path in (TIED_DIAGONAL_DIR / "repeat_noise.csv", TIED_DIAGONAL_DIR / "tied_optima.csv")
            },
        }
    return BenchPanel(
        name=f"{STARCODER_PANEL_PREFIX}{curve_id}",
        kind="curve",
        family=str(metadata["family"]),
        runs=tuple(str(value) for value in block["training_run_id"]),
        buckets=("nemotron_full", "starcoder"),
        features=features,
        groups=(group,),
        input_hashes=dict(hashes),
        repeat_sd=repeat_sd,
        component_repeat_sd={},
        metadata=extra,
    )


@functools.cache
def load_panel(name: str) -> BenchPanel:
    if name.startswith(STARCODER_PANEL_PREFIX):
        return load_curve_panel(name.removeprefix(STARCODER_PANEL_PREFIX))
    if name in TABULAR_PANELS:
        return load_tabular_panel(name)
    raise ValueError(f"unknown panel {name!r}")


# ---------------------------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    inner: tuple[tuple[np.ndarray, np.ndarray], ...]


def _interleaved(count: int, folds: int) -> np.ndarray:
    return np.arange(count) % folds


def panel_splits(panel: BenchPanel, repeats: int) -> tuple[Split, ...]:
    """Outer mixture-blocked folds and the shared inner partition of each training fold."""
    return _panel_splits(panel.name, repeats)


@functools.cache
def _panel_splits(panel_name: str, repeats: int) -> tuple[Split, ...]:
    panel = load_panel(panel_name)
    rows = np.arange(panel.rows)
    result: list[Split] = []
    for repeat in range(repeats):
        if panel.kind == "tabular":
            labels = olmix_benchmark.block_labels(panel.features.weights, OUTER_FOLDS, FOLD_SEED + 100 * repeat)
        else:
            if repeat > 0:
                raise ValueError("StarCoder curves have one deterministic interleaved partition")
            labels = _interleaved(panel.rows, OUTER_FOLDS)
        for fold in range(OUTER_FOLDS):
            train = rows[labels != fold]
            test = rows[labels == fold]
            if panel.kind == "tabular":
                inner_labels = olmix_benchmark.block_labels(
                    panel.features.weights[train], INNER_FOLDS, FOLD_SEED + 10_000 * repeat + 100 * fold
                )
            else:
                inner_labels = _interleaved(len(train), INNER_FOLDS)
            inner = tuple((train[inner_labels != index], train[inner_labels == index]) for index in range(INNER_FOLDS))
            result.append(Split(repeat, fold, train, test, inner))
    return tuple(result)


def heldout_inner_folds(panel: BenchPanel) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    rows = np.arange(panel.rows)
    labels = olmix_benchmark.block_labels(panel.features.weights, INNER_FOLDS, HELDOUT_INNER_SEED)
    return tuple((rows[labels != index], rows[labels == index]) for index in range(INNER_FOLDS))


def split_manifest(panels: tuple[BenchPanel, ...], repeats: int) -> pd.DataFrame:
    rows = []
    for panel in panels:
        for split in panel_splits(panel, repeats if panel.kind == "tabular" else 1):
            inner_lookup = np.full(panel.rows, -1)
            for index, (_train, validation) in enumerate(split.inner):
                inner_lookup[validation] = index
            for row_index in range(panel.rows):
                rows.append(
                    {
                        "panel": panel.name,
                        "repeat": split.repeat,
                        "fold": split.fold,
                        "row_index": row_index,
                        "run": panel.runs[row_index],
                        "outer_role": "test" if row_index in set(split.test.tolist()) else "train",
                        "inner_fold": int(inner_lookup[row_index]),
                    }
                )
    return pd.DataFrame(rows)


def manifest_hash(manifest: pd.DataFrame) -> str:
    payload = manifest.to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------------------------
# Tiers and tasks
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TierPlan:
    tier: str
    repeats: int
    folds: tuple[int, ...]
    components: dict[str, dict[str, tuple[str, ...]]]  # panel -> target -> component ids
    curves: tuple[str, ...]


def tier_plan(tier: str, *, panels: tuple[str, ...] = TABULAR_PANELS) -> TierPlan:
    anchors = _anchor_components()
    components: dict[str, dict[str, tuple[str, ...]]] = {}
    for name in panels:
        if name in THIRTY_NINE_BUCKET_PANELS:
            if tier in ("smoke", "screen"):
                components[name] = {"uncheatable": anchors["uncheatable"], "table9": anchors["table9"]}
            else:
                components[name] = {
                    "uncheatable": canonical.UNCHEATABLE_COMPONENTS,
                    "table9": canonical.table9_components(),
                }
        elif name in MICHAEL_PANELS:
            components[name] = {MICHAEL_TARGET: MICHAEL_TASKS}
    all_curves = starcoder_curve_ids()
    if tier == "smoke":
        return TierPlan(tier, 1, (0,), components, (SMOKE_CURVE,) if SMOKE_CURVE in all_curves else all_curves[:1])
    if tier in ("screen", "certify"):
        return TierPlan(tier, 1, tuple(range(OUTER_FOLDS)), components, all_curves)
    if tier == "finalist":
        return TierPlan(tier, OUTER_FOLDS, tuple(range(OUTER_FOLDS)), components, all_curves)
    raise ValueError(f"unknown tier {tier!r}")


@dataclasses.dataclass(frozen=True)
class FitTask:
    model_id: str
    panel: str
    target: str
    component_index: int
    component: str
    repeat: int
    fold: int


def shard_path(output_dir: Path, task: FitTask) -> Path:
    panel_token = task.panel.replace("::", "__").replace("/", "__")
    filename = f"component_{task.component_index:03d}_{canonical._component_token(task.component)}.npz"
    return (
        output_dir
        / "shards"
        / task.model_id
        / panel_token
        / task.target
        / f"repeat_{task.repeat:02d}"
        / f"fold_{task.fold:02d}"
        / filename
    )


def plan_tasks(plan: TierPlan, model_ids: tuple[str, ...]) -> list[FitTask]:
    tasks: list[FitTask] = []
    for model_id in model_ids:
        for panel_name, targets in plan.components.items():
            panel = load_panel(panel_name)
            for target, components in targets.items():
                group = panel.group(target)
                for component in components:
                    index = group.components.index(component)
                    for repeat in range(plan.repeats):
                        for fold in plan.folds:
                            tasks.append(FitTask(model_id, panel_name, target, index, component, repeat, fold))
        for curve in plan.curves:
            for fold in plan.folds:
                tasks.append(
                    FitTask(
                        model_id, f"{STARCODER_PANEL_PREFIX}{curve}", STARCODER_TARGET, 0, STARCODER_COMPONENT, 0, fold
                    )
                )
    return tasks


# ---------------------------------------------------------------------------------------------
# Protocol hashing
# ---------------------------------------------------------------------------------------------


@functools.cache
def source_hashes() -> dict[str, str]:
    files = {
        "models": Path(models.__file__),
        "registry": Path(registry.__file__),
        "canonical_loader": Path(canonical.__file__),
        "starcoder_loader": Path(starcoder_curves.__file__),
        "olmix_swarm_loader": Path(olmix_benchmark.__file__),
    }
    return {name: file_sha256(path.resolve()) for name, path in files.items()}


def entry_fingerprint(entry: registry.ModelEntry) -> str:
    """Legacy metadata fingerprint, kept only to recognise shards written before configuration keys."""
    payload = {
        "model_id": entry.model_id,
        "source_model_ids": list(entry.source_model_ids),
        "mechanisms": entry.mechanisms,
        "solver_id": entry.solver_id,
        "hyperparameter_grid": entry.hyperparameter_grid,
        "feature_transform": entry.feature_transform,
        "parent": entry.parent,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def description_hash(entry: registry.ModelEntry, panel: BenchPanel) -> str:
    """Hash of the built model's complete fit configuration on this panel."""
    model = entry.build(registry.apply_transform(panel.features, entry))
    return hashlib.sha256(json.dumps(models.describe_model(model), sort_keys=True, default=str).encode()).hexdigest()


@functools.cache
def fit_path_hash() -> str:
    """Hash of the harness functions on the fitting path, so fitting-logic edits invalidate shards."""
    functions = (
        fit_one,
        fit_heldout_component,
        _seed,
        panel_splits,
        _panel_splits,
        load_tabular_panel,
        load_curve_panel,
        starcoder_inventory,
        _tabular_features,
        heldout_features,
        heldout_inner_folds,
        plan_tasks,
        tier_plan,
        component_cv_table,
        fit_shared_unit,
        shared_unit_key,
    )
    return hashlib.sha256("".join(inspect.getsource(function) for function in functions).encode()).hexdigest()


@functools.cache
def legacy_snapshot(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "legacy_entry_descriptions.json"
    return json.loads(path.read_text()) if path.is_file() else {}


@functools.cache
def cache_generations(output_dir: Path) -> tuple[dict[str, Any], ...]:
    """Recorded models-module, fit-path, and configuration hashes whose keys stay valid for unchanged configurations."""
    return tuple(json.loads(path.read_text()) for path in sorted(output_dir.glob("legacy_entry_descriptions_gen*.json")))


def legacy_accepted(entry: registry.ModelEntry, panel: BenchPanel, output_dir: Path) -> bool:
    """True when this (entry, panel) configuration is unchanged since the recorded legacy snapshot."""
    snapshot = legacy_snapshot(output_dir)
    return snapshot.get("entries", {}).get(f"{entry.model_id}|{panel.name}") == description_hash(entry, panel)


@functools.cache
def fit_protocol(split_hash: str) -> dict[str, Any]:
    """Legacy fit-protocol payload keyed by the whole one-repeat manifest hash (kept for cache compatibility)."""
    return {**fit_protocol_core(), "split_hash": split_hash}


@functools.cache
def fit_protocol_core() -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "fold_seed": FOLD_SEED,
        "heldout_inner_seed": HELDOUT_INNER_SEED,
        "outcome_permutation_seed": OUTCOME_PERMUTATION_SEED,
        "transform_seed": registry.TRANSFORM_SEED,
        "source_hashes": {key: value for key, value in source_hashes().items() if key == "models"},
    }


@functools.cache
def split_fingerprint(panel_name: str, repeat: int, fold: int) -> str:
    """Hash of one outer split and its inner partition, independent of how many repeats a tier plans."""
    split = next(item for item in _panel_splits(panel_name, repeat + 1) if item.repeat == repeat and item.fold == fold)
    digest = hashlib.sha256(f"{panel_name}|{repeat}|{fold}".encode())
    digest.update(np.ascontiguousarray(split.train, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(split.test, dtype=np.int64).tobytes())
    for inner_train, validation in split.inner:
        digest.update(np.ascontiguousarray(inner_train, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(validation, dtype=np.int64).tobytes())
    return digest.hexdigest()


def task_protocol_hashes(
    task: FitTask, entry: registry.ModelEntry, legacy_split_hash: str, panel: BenchPanel, output_dir: Path | None = None
) -> tuple[str, ...]:
    """Accepted protocol hashes for one shard.

    The primary key holds the models-module and fit-path source hashes, the per-split fingerprint,
    the built model's full configuration, the panel inputs and features. Shards written before
    configuration keys existed are accepted only when the recorded legacy snapshot shows that this
    (entry, panel) configuration is unchanged; their keys carried the legacy metadata fingerprint,
    the legacy models-module hash, and either the per-split fingerprint or the one-repeat manifest hash.
    """
    split = split_fingerprint(task.panel, task.repeat, task.fold)
    primary_payload = {
        "fit_protocol": {**fit_protocol_core(), "fit_path_hash": fit_path_hash(), "split_fingerprint": split},
        "entry": {
            "model_id": entry.model_id,
            "feature_transform": entry.feature_transform,
            "parent": entry.parent,
            "description": description_hash(entry, panel),
        },
        "panel_inputs": panel.input_hashes,
        "panel_features": panel.features.cache_key,
    }
    hashes = [hashlib.sha256(json.dumps(primary_payload, sort_keys=True).encode()).hexdigest()]
    snapshot = legacy_snapshot(output_dir) if output_dir is not None else {}
    if snapshot and legacy_accepted(entry, panel, output_dir):
        legacy_core = {**fit_protocol_core(), "source_hashes": {"models": snapshot["legacy_models_hash"]}}
        common = {
            "entry": snapshot["entry_fingerprints"][entry.model_id],
            "panel_inputs": panel.input_hashes,
            "panel_features": panel.features.cache_key,
        }
        legacy_primary = {"fit_protocol": {**legacy_core, "split_fingerprint": split}, **common}
        legacy_manifest = {"fit_protocol": {**legacy_core, "split_hash": legacy_split_hash}, **common}
        hashes.extend(
            hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            for payload in (legacy_primary, legacy_manifest)
        )
    for generation in cache_generations(output_dir) if output_dir is not None else ():
        if generation["entries"].get(f"{entry.model_id}|{panel.name}") != primary_payload["entry"]["description"]:
            continue
        core = {**fit_protocol_core(), "source_hashes": {"models": generation["models_hash"]}}
        payload = {
            **primary_payload,
            "fit_protocol": {**core, "fit_path_hash": generation["fit_path_hash"], "split_fingerprint": split},
        }
        hashes.append(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    return tuple(hashes)


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in payload.items()})
    os.replace(temporary, path)


def load_shard(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    except (OSError, ValueError, KeyError):
        return None


def valid_shard(path: Path, protocol_hashes: tuple[str, ...], component: str, test: np.ndarray) -> bool:
    payload = load_shard(path)
    if payload is None:
        return False
    return (
        str(payload["protocol_hash"].item()) in protocol_hashes
        and str(payload["component"].item()) == component
        and np.array_equal(payload["test"], test)
        and payload["prediction"].shape == (len(test),)
    )


def _seed(task: FitTask) -> int:
    identity = f"{task.panel}|{task.target}|{task.component_index}|{task.repeat}|{task.fold}"
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def fit_one(task: FitTask, output_dir: Path, legacy_split_hash: str) -> str:
    """Fit one shard; returns cached | fitted | failed."""
    panel = load_panel(task.panel)
    entry = registry.ENTRY_BY_ID[task.model_id]
    protocol_hashes = task_protocol_hashes(task, entry, legacy_split_hash, panel, output_dir)
    protocol_hash = protocol_hashes[0]
    path = shard_path(output_dir, task)
    split = next(
        item for item in panel_splits(panel, task.repeat + 1) if item.repeat == task.repeat and item.fold == task.fold
    )
    if valid_shard(path, protocol_hashes, task.component, split.test):
        return "cached"
    group = panel.group(task.target)
    response = group.outcomes[:, task.component_index].copy()
    training_response = response.copy()
    if entry.feature_transform == "outcome_permutation":
        generator = np.random.default_rng(OUTCOME_PERMUTATION_SEED + _seed(task))
        training_response[split.train] = response[split.train][generator.permutation(len(split.train))]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=task.component)
    started = time.monotonic()
    payload: dict[str, Any] = {
        "protocol_hash": protocol_hash,
        "model_id": task.model_id,
        "component": task.component,
        "component_index": task.component_index,
        "test": split.test,
        "train_rows": len(split.train),
        "observed": response[split.test],
        "constant_prediction": np.full(len(split.test), float(training_response[split.train].mean())),
    }
    try:
        model = entry.build(features)
        fitted = model.fit(features, training_response, split.train, split.inner, _seed(task))
        prediction = np.asarray(model.predict(fitted, features, split.test), dtype=float)
        if prediction.shape != (len(split.test),) or not np.isfinite(prediction).all():
            raise ValueError("non-finite or misshaped prediction")
        payload.update(
            {
                "status": "ok",
                "error": "",
                "prediction": prediction,
                "train_prediction": np.asarray(model.predict(fitted, features, split.train), dtype=float),
                "shape_json": json.dumps(fitted.shape, sort_keys=True),
                "ridge": float(fitted.ridge),
                "cv_table": fitted.cv_table if fitted.cv_table is not None else np.zeros((0, 0)),
                "diagnostics_json": json.dumps(
                    {
                        key: value if not isinstance(value, np.generic) else value.item()
                        for key, value in fitted.diagnostics.items()
                    },
                    sort_keys=True,
                    default=float,
                ),
            }
        )
    except Exception as error:
        payload.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()[-2000:]}",
                "prediction": np.full(len(split.test), np.nan),
                "train_prediction": np.full(len(split.train), np.nan),
                "shape_json": "{}",
                "ridge": float("nan"),
                "diagnostics_json": "{}",
            }
        )
    payload["elapsed"] = time.monotonic() - started
    atomic_save(path, payload)
    return str(payload["status"]) if payload["status"] != "ok" else "fitted"


SHARED_CACHE_DIR = "shared_cache"
SCALE_SHARING_PANELS = frozenset({"60m_39bucket", "300m_39bucket", "delphi_3e18_39bucket"})


def shared_unit_key(unit: str, task: FitTask) -> tuple[str, ...]:
    """Grouping key of one task under a shared-shape sharing unit."""
    if unit == "target":
        return (task.panel, task.target, str(task.repeat), str(task.fold))
    if unit == "panel":
        return (task.panel, str(task.repeat), str(task.fold))
    if unit == "scale":
        head = "39bucket" if task.panel in SCALE_SHARING_PANELS else task.panel
        return (head, str(task.repeat), str(task.fold))
    raise ValueError(f"unknown sharing unit {unit}")


def shared_cache_path(output_dir: Path, task: FitTask) -> Path:
    shard = shard_path(output_dir, task)
    return output_dir / SHARED_CACHE_DIR / shard.relative_to(output_dir / "shards")


def component_cv_table(task: FitTask, output_dir: Path, legacy_split_hash: str) -> str:
    """Pass 1 of the shared-shape fit: cache the parent's inner-CV table for one component fit."""
    panel = load_panel(task.panel)
    entry = registry.ENTRY_BY_ID[task.model_id]
    hashes = task_protocol_hashes(task, entry, legacy_split_hash, panel, output_dir)
    path = shared_cache_path(output_dir, task)
    cached = load_shard(path)
    if cached is not None and str(cached["protocol_hash"]) in hashes:
        return "cached"
    split = next(
        item for item in panel_splits(panel, task.repeat + 1) if item.repeat == task.repeat and item.fold == task.fold
    )
    response = panel.group(task.target).outcomes[:, task.component_index].copy()
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=task.component)
    model = entry.build(features)
    fitted = model.fit(features, response, split.train, split.inner, _seed(task))
    if fitted.cv_table is None:
        raise ValueError(f"{task.model_id} does not expose an inner-CV table")
    atomic_save(
        path,
        {
            "protocol_hash": hashes[0],
            "cv_table": fitted.cv_table,
            "train_response_sd": float(np.std(response[split.train])),
        },
    )
    return "fitted"


def fit_shared_unit(
    shared_id: str, parent_id: str, tasks: list[FitTask], output_dir: Path, legacy_split_hash: str
) -> int:
    """Pass 2: one shape per sharing unit (repeat-SD-normalized CV error summed over components), per-component ridge."""
    parent = registry.ENTRY_BY_ID[parent_id]
    shared = registry.ENTRY_BY_ID[shared_id]
    tables: list[np.ndarray] = []
    for task in tasks:
        panel = load_panel(task.panel)
        cached = load_shard(shared_cache_path(output_dir, task))
        if cached is None:
            raise FileNotFoundError(f"missing CV table for {task}")
        scale = float(panel.component_repeat_sd.get(task.component, 0.0)) or float(cached["train_response_sd"])
        tables.append(np.asarray(cached["cv_table"], dtype=float) / max(scale, 1e-9))
    combined = np.sum([np.min(table, axis=1) for table in tables], axis=0)
    shape_index = int(np.argmin(combined))
    # A shared shard is only valid for the exact sharing unit that selected its shape.
    unit_hash = hashlib.sha256(
        json.dumps(sorted(f"{task.panel}|{task.target}|{task.component}" for task in tasks)).encode()
    ).hexdigest()
    written = 0
    for task, table in zip(tasks, tables, strict=True):
        panel = load_panel(task.panel)
        shared_task = dataclasses.replace(task, model_id=shared_id)
        hashes = task_protocol_hashes(shared_task, shared, legacy_split_hash, panel, output_dir)
        path = shard_path(output_dir, shared_task)
        split = next(
            item
            for item in panel_splits(panel, task.repeat + 1)
            if item.repeat == task.repeat and item.fold == task.fold
        )
        existing = load_shard(path) if valid_shard(path, hashes, task.component, split.test) else None
        if existing is not None and str(existing.get("shared_unit_hash", "")) == unit_hash:
            continue
        response = panel.group(task.target).outcomes[:, task.component_index].copy()
        features = dataclasses.replace(registry.apply_transform(panel.features, parent), component=task.component)
        model = parent.build(features)
        candidates = model.candidate_shapes(features)
        shape = dict(candidates[shape_index])
        ridge_index = int(np.argmin(table[shape_index]))
        ridge = float(model.ridge_grid[ridge_index])
        design = model.design(features, shape)
        spec = model.head_for(shape)
        started = time.monotonic()
        head = models.fit_head(
            models.Design(design.values[split.train], design.ridge, design.names), response[split.train], ridge, spec
        )
        prediction = np.asarray(models.predict_head(head, design.values[split.test], spec), dtype=float)
        payload = {
            "protocol_hash": hashes[0],
            "model_id": shared_id,
            "component": task.component,
            "component_index": task.component_index,
            "test": split.test,
            "train_rows": len(split.train),
            "observed": response[split.test],
            "constant_prediction": np.full(len(split.test), float(response[split.train].mean())),
            "status": "ok" if np.isfinite(prediction).all() else "failed",
            "error": "" if np.isfinite(prediction).all() else "non-finite shared-shape prediction",
            "prediction": prediction,
            "train_prediction": np.asarray(models.predict_head(head, design.values[split.train], spec), dtype=float),
            "shape_json": json.dumps(shape, sort_keys=True),
            "ridge": ridge,
            "cv_table": np.zeros((0, 0)),
            "shared_unit_hash": unit_hash,
            "diagnostics_json": json.dumps(
                {
                    "inner_cv_rmse": float(table[shape_index, ridge_index]),
                    "shared_unit_size": len(tasks),
                    "shared_shape_index": shape_index,
                    "candidates": int(table.size),
                    "converged": True,
                    "boundary_hits": 0,
                    "effective_rank": models.effective_rank(design.values[split.train]),
                    "columns": int(design.values.shape[1]),
                    "fitted_dof": head.active + 1 + model.shape_dof,
                    "nonlinear_dof": model.shape_dof,
                },
                sort_keys=True,
            ),
            "elapsed": time.monotonic() - started,
        }
        atomic_save(path, payload)
        written += 1
    return written


def run_shared_stage(
    plan: TierPlan, model_ids: tuple[str, ...], output_dir: Path, legacy_split_hash: str, workers: int
) -> dict[str, int]:
    """Fit every shared-shape entry in ``model_ids``: cache parent CV tables, then refit per sharing unit."""
    counts: dict[str, int] = {}
    for shared_id in model_ids:
        if shared_id not in registry.SHARED_SHAPE_UNITS:
            continue
        parent_id, unit = registry.SHARED_SHAPE_UNITS[shared_id]
        parent_tasks = plan_tasks(plan, (parent_id,))
        print(f"{shared_id}: caching {len(parent_tasks)} parent CV tables", flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=workers, verbose=5, batch_size=1)(
                delayed(component_cv_table)(task, output_dir, legacy_split_hash) for task in parent_tasks
            )
        groups: dict[tuple[str, ...], list[FitTask]] = {}
        for task in parent_tasks:
            groups.setdefault(shared_unit_key(unit, task), []).append(task)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            written = Parallel(n_jobs=workers, verbose=5, batch_size=1)(
                delayed(fit_shared_unit)(shared_id, parent_id, tasks, output_dir, legacy_split_hash)
                for tasks in groups.values()
            )
        counts[shared_id] = int(sum(written))
        print(f"{shared_id}: {len(groups)} units, {counts[shared_id]} shards written", flush=True)
    return counts


def run_tasks(tasks: list[FitTask], output_dir: Path, legacy_split_hash: str, workers: int) -> dict[str, int]:
    pending = []
    for task in tasks:
        panel = load_panel(task.panel)
        entry = registry.ENTRY_BY_ID[task.model_id]
        split = next(
            item
            for item in panel_splits(panel, task.repeat + 1)
            if item.repeat == task.repeat and item.fold == task.fold
        )
        hashes = task_protocol_hashes(task, entry, legacy_split_hash, panel, output_dir)
        if not valid_shard(shard_path(output_dir, task), hashes, task.component, split.test):
            pending.append(task)
    counts = {"total": len(tasks), "cached": len(tasks) - len(pending), "fitted": 0, "failed": 0}
    print(f"tasks: {counts['total']} total, {counts['cached']} cached, {len(pending)} pending", flush=True)
    if not pending:
        return counts
    # Heavier models first so the tail of the run is short.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        statuses = Parallel(n_jobs=workers, verbose=5, batch_size=1)(
            delayed(fit_one)(task, output_dir, legacy_split_hash) for task in pending
        )
    counts["fitted"] = statuses.count("fitted")
    counts["failed"] = statuses.count("failed")
    print(f"completed: fitted={counts['fitted']} failed={counts['failed']}", flush=True)
    return counts


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


def _safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.ptp(predicted) <= 1e-12 or np.ptp(observed) <= 1e-12:
        return float("nan")
    return float(stats.spearmanr(predicted, observed).statistic)


def metric_row(observed: np.ndarray, predicted: np.ndarray, basin: np.ndarray) -> dict[str, float]:
    """Prediction, ranking, selection, and basin metrics for one set of out-of-fold rows."""
    finite = np.isfinite(predicted)
    if not finite.all():
        return {name: float("nan") for name in METRIC_NAMES} | {
            "n": len(observed),
            "n_basin": int(basin.sum()),
            "n_failed": int((~finite).sum()),
        }
    order = np.argsort(predicted, kind="stable")
    selected = int(order[0])
    constant = bool(np.ptp(predicted) <= 1e-12)
    if constant or len(observed) < 3:
        intercept, slope = float(np.mean(observed)), 0.0
    else:
        fit = stats.linregress(predicted, observed)
        intercept, slope = float(fit.intercept), float(fit.slope)
    top = order[: min(TOP_K, len(order))]
    row = {
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "spearman": _safe_spearman(observed, predicted),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "regret_at_1": float(observed[selected] - observed.min()),
        "regret_at_top_k": float(observed[top].min() - observed.min()),
        "selection_optimism": float(observed[selected] - predicted[selected]),
        "basin_rmse": (
            float(np.sqrt(np.mean((predicted[basin] - observed[basin]) ** 2))) if basin.sum() >= 1 else float("nan")
        ),
        "basin_spearman": _safe_spearman(observed[basin], predicted[basin]) if basin.sum() >= 3 else float("nan"),
        "n": len(observed),
        "n_basin": int(basin.sum()),
        "n_failed": 0,
    }
    return row


def collect_shards(tasks: list[FitTask], output_dir: Path, legacy_split_hash: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long prediction table and one fit-diagnostic row per shard (including failures and misses)."""
    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    basin_cache: dict[tuple[str, str], np.ndarray] = {}
    for task in tasks:
        panel = load_panel(task.panel)
        entry = registry.ENTRY_BY_ID[task.model_id]
        payload = load_shard(shard_path(output_dir, task))
        hashes = task_protocol_hashes(task, entry, legacy_split_hash, panel, output_dir)
        if payload is not None and str(payload["protocol_hash"].item()) not in hashes:
            payload = None
            stale = True
        else:
            stale = False
        base = {
            "model": task.model_id,
            "role": entry.role,
            "parent": entry.parent or "",
            "panel": task.panel,
            "panel_kind": panel.kind,
            "curve_family": panel.family if panel.kind == "curve" else "",
            "target": task.target,
            "component_index": task.component_index,
            "component": task.component,
            "repeat": task.repeat,
            "fold": task.fold,
        }
        if payload is None:
            fit_rows.append(
                base
                | {
                    "status": "stale" if stale else "missing",
                    "error": "shard protocol hash not accepted" if stale else "shard not found",
                    "elapsed": float("nan"),
                }
            )
            continue
        status = str(payload["status"].item())
        diagnostics = json.loads(str(payload["diagnostics_json"].item())) if status == "ok" else {}
        fit_rows.append(
            base
            | {
                "status": status,
                "error": str(payload["error"].item()),
                "elapsed": float(payload["elapsed"]),
                "train_rows": int(payload["train_rows"]),
                "test_rows": len(payload["test"]),
                "ridge": float(payload["ridge"]),
                "shape_json": str(payload["shape_json"].item()),
            }
            | {f"diag_{key}": value for key, value in diagnostics.items()}
        )
        if status != "ok":
            continue
        key = (task.panel, task.target)
        if key not in basin_cache:
            basin = np.zeros(panel.rows, dtype=bool)
            basin[panel.basin_rows(task.target)] = True
            basin_cache[key] = basin
        basin = basin_cache[key]
        test = payload["test"]
        observed = payload["observed"]
        prediction = payload["prediction"]
        for local, row_index in enumerate(test):
            prediction_rows.append(
                base
                | {
                    "row_index": int(row_index),
                    "run": panel.runs[row_index],
                    "observed": float(observed[local]),
                    "prediction": float(prediction[local]),
                    "basin": bool(basin[row_index]),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    fits = pd.DataFrame(fit_rows)
    return predictions, fits


COMPONENT_KEYS = [
    "model",
    "role",
    "parent",
    "panel",
    "panel_kind",
    "curve_family",
    "target",
    "component_index",
    "component",
    "repeat",
]


def grouped_metrics(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for identity, group in predictions.groupby(keys, sort=False):
        rows.append(
            dict(zip(keys, identity, strict=True))
            | metric_row(
                group["observed"].to_numpy(float), group["prediction"].to_numpy(float), group["basin"].to_numpy(bool)
            )
        )
    return pd.DataFrame(rows)


def add_noise_ratio(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Divide RMSE by the identified same-mixture repeat SD, when one exists."""
    ratios = []
    for _, row in frame.iterrows():
        panel = load_panel(str(row["panel"]))
        if level == "component":
            sd = panel.component_repeat_sd.get(str(row["component"]), float("nan"))
            if panel.kind == "curve":
                sd = panel.repeat_sd.get(str(row["target"]), float("nan"))
        else:
            sd = panel.repeat_sd.get(str(row["target"]), float("nan"))
        ratios.append(float(row["rmse"]) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan"))
    frame = frame.copy()
    frame["repeat_sd"] = [
        float(r) if np.isfinite(r) and r > 0 else float("nan")
        for r in [
            float(row["rmse"]) / ratio if np.isfinite(ratio) and ratio > 0 else float("nan")
            for ratio, (_, row) in zip(ratios, frame.iterrows(), strict=True)
        ]
    ]
    frame["rmse_over_repeat_sd"] = ratios
    return frame


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct every aggregate whose components are all present for a (model, panel, target, repeat)."""
    rows: list[dict[str, Any]] = []
    tabular = predictions[predictions["panel_kind"].eq("tabular")]
    for (model, panel_name, target, repeat), group in tabular.groupby(
        ["model", "panel", "target", "repeat"], sort=False
    ):
        panel = load_panel(panel_name)
        target_group = panel.group(target)
        pivot = group.pivot_table(index="row_index", columns="component_index", values="prediction", aggfunc="first")
        if pivot.shape != (panel.rows, len(target_group.components)):
            continue
        pivot = pivot.reindex(index=np.arange(panel.rows), columns=np.arange(len(target_group.components)))
        if pivot.isna().any().any():
            continue
        values = pivot.to_numpy(float) @ target_group.aggregation_weights
        fold_lookup = group[["row_index", "fold"]].drop_duplicates().set_index("row_index")["fold"]
        basin = np.zeros(panel.rows, dtype=bool)
        basin[panel.basin_rows(target)] = True
        entry = registry.ENTRY_BY_ID.get(model)
        role = entry.role if entry else ""
        parent = (entry.parent or "") if entry else ""
        for row_index in range(panel.rows):
            rows.append(
                {
                    "model": model,
                    "role": role,
                    "parent": parent,
                    "panel": panel_name,
                    "panel_kind": "tabular",
                    "curve_family": "",
                    "target": target,
                    "repeat": int(repeat),
                    "fold": int(fold_lookup.loc[row_index]),
                    "row_index": row_index,
                    "run": panel.runs[row_index],
                    "observed": float(target_group.aggregate[row_index]),
                    "prediction": float(values[row_index]),
                    "basin": bool(basin[row_index]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_summary(repeat_metrics: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if repeat_metrics.empty:
        return pd.DataFrame()
    rows = []
    keys = ["model", "role", "parent", "panel", "target"]
    for identity, group in repeat_metrics.groupby(keys, sort=False):
        row: dict[str, Any] = dict(zip(keys, identity, strict=True))
        row["repeats"] = len(group)
        for metric in METRIC_NAMES:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_repeat_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
        folds = fold_metrics
        for key, value in zip(keys, identity, strict=True):
            folds = folds[folds[key].eq(value)]
        row["mean_fold_regret_at_1"] = float(folds["regret_at_1"].mean())
        row["mean_fold_regret_at_top_k"] = float(folds["regret_at_top_k"].mean())
        row["mean_fold_rmse"] = float(folds["rmse"].mean())
        row["repeat_sd"] = load_panel(str(identity[3])).repeat_sd.get(str(identity[4]), float("nan"))
        row["rmse_over_repeat_sd"] = (
            row["rmse"] / row["repeat_sd"] if np.isfinite(row["repeat_sd"]) and row["repeat_sd"] > 0 else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_contrast(
    difference: np.ndarray, test_train_ratio: float, folds_per_repeat: int, repeats: int
) -> dict[str, float]:
    """Nadeau-Bengio corrected paired contrast over K x R folds."""
    difference = difference[np.isfinite(difference)]
    count = len(difference)
    if count < 2:
        return {
            "mean": float(difference.mean()) if count else float("nan"),
            "corrected_se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_folds": count,
        }
    factor = 1.0 / count + test_train_ratio
    mean = float(difference.mean())
    se = float(np.sqrt(factor * difference.var(ddof=1)))
    critical = float(stats.t.ppf(0.975, count - 1))
    del folds_per_repeat, repeats
    return {
        "mean": mean,
        "corrected_se": se,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "n_folds": count,
    }


def paired_contrasts(fold_metrics: pd.DataFrame, fits: pd.DataFrame, level: str) -> pd.DataFrame:
    """Fold-paired contrasts of every model against canonical DSP, taskwise OLMix, and its parent."""
    if fold_metrics.empty:
        return pd.DataFrame()
    unit_keys = ["panel", "target"] + (["component"] if level == "component" else [])
    ratio_lookup = (
        fits.groupby(["panel", "repeat", "fold"])[["test_rows", "train_rows"]].first()
        if not fits.empty and "test_rows" in fits
        else None
    )
    rows = []
    for unit, group in fold_metrics.groupby(unit_keys, sort=False):
        unit_dict = dict(zip(unit_keys, unit, strict=True))
        pivot = {
            metric: group.pivot_table(
                index=["repeat", "fold"], columns="model", values=metric, aggfunc="first", dropna=False
            )
            for metric in ("rmse", "regret_at_1", "spearman", "basin_rmse")
        }
        index = pivot["rmse"].index
        if ratio_lookup is not None:
            ratios = [
                float(ratio_lookup.loc[(unit_dict["panel"], repeat, fold), "test_rows"])
                / float(ratio_lookup.loc[(unit_dict["panel"], repeat, fold), "train_rows"])
                for repeat, fold in index
                if (unit_dict["panel"], repeat, fold) in ratio_lookup.index
            ]
            ratio = float(np.mean(ratios)) if ratios else 1.0 / (OUTER_FOLDS - 1)
        else:
            ratio = 1.0 / (OUTER_FOLDS - 1)
        repeats = int(index.get_level_values("repeat").nunique())
        for model in pivot["rmse"].columns:
            entry = registry.ENTRY_BY_ID[model]
            comparators = [item for item in COMPARATORS if item != model and item in pivot["rmse"].columns]
            if entry.parent and entry.parent in pivot["rmse"].columns and entry.parent not in comparators:
                comparators.append(entry.parent)
            for comparator in comparators:
                for metric, table in pivot.items():
                    if model not in table.columns or comparator not in table.columns:
                        continue
                    difference = (table[model] - table[comparator]).to_numpy(float)
                    stats_row = corrected_contrast(difference, ratio, OUTER_FOLDS, repeats)
                    rows.append(
                        unit_dict
                        | {
                            "level": level,
                            "model": model,
                            "comparator": comparator,
                            "comparator_kind": "parent" if comparator == entry.parent else "reference",
                            "metric": metric,
                            "mean_difference": stats_row["mean"],
                            "corrected_se": stats_row["corrected_se"],
                            "ci_low": stats_row["ci_low"],
                            "ci_high": stats_row["ci_high"],
                            "n_folds": stats_row["n_folds"],
                            "test_train_ratio": ratio,
                        }
                    )
    return pd.DataFrame(rows)


def holm_adjusted(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjustment over the finite entries of one family of tests."""
    values = p_values.to_numpy(float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(finite) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=p_values.index)


def with_holm_correction(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Holm-corrected sign-test p-values, one family per (comparator kind, metric)."""
    if frame.empty or "sign_test_p" not in frame.columns:
        return frame
    frame = frame.copy()
    frame["sign_test_p_holm"] = frame.groupby(["comparator_kind", "metric"], dropna=False)["sign_test_p"].transform(
        holm_adjusted
    )
    return frame


def pooled_anchor_contrasts(component_fold_metrics: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Contrasts pooled over every Screen unit (anchors, Michael tasks, curves).

    The frozen rule uses the raw Nadeau-Bengio contrast. Because a few Michael-panel units explode
    by orders of magnitude and dominate a raw pool, two post-hoc robust views are reported beside
    it: the relative difference (a - b) / b per unit-fold, and a unit-level sign test on the
    fold-averaged difference. Both are labelled post hoc in the promotion table.
    """
    del fits
    if component_fold_metrics.empty:
        return pd.DataFrame()
    frame = component_fold_metrics.copy()
    # StarCoder curves are macro-averaged within each physical family first, so the 28
    # horizon-by-replay curves count as one unit rather than 28.
    curves = frame[frame["panel_kind"].eq("curve")]
    tabular = frame[~frame["panel_kind"].eq("curve")].copy()
    tabular["unit"] = tabular["panel"] + "|" + tabular["target"] + "|" + tabular["component"]
    if not curves.empty:
        metrics = [name for name in METRIC_NAMES if name in curves.columns]
        family_rows = curves.groupby(["model", "curve_family", "repeat", "fold"], as_index=False)[metrics].mean()
        family_rows["unit"] = "starcoder_family|" + family_rows["curve_family"]
        frame = pd.concat([tabular, family_rows], ignore_index=True)
    else:
        frame = tabular
    rows = []
    for model in frame["model"].unique():
        entry = registry.ENTRY_BY_ID[model]
        comparators = [item for item in COMPARATORS if item != model]
        if entry.parent and entry.parent not in comparators:
            comparators.append(entry.parent)
        for comparator in comparators:
            if comparator not in set(frame["model"]):
                continue
            for metric in ("rmse", "regret_at_1", "spearman"):
                left = frame[frame["model"].eq(model)].set_index(["unit", "repeat", "fold"])[metric]
                right = frame[frame["model"].eq(comparator)].set_index(["unit", "repeat", "fold"])[metric]
                joined = pd.concat([left.rename("a"), right.rename("b")], axis=1, join="inner").dropna()
                difference = (joined["a"] - joined["b"]).to_numpy(float)
                stats_row = corrected_contrast(difference, 1.0 / (OUTER_FOLDS - 1), OUTER_FOLDS, 1)
                relative = np.divide(
                    difference,
                    np.abs(joined["b"].to_numpy(float)),
                    out=np.full(len(difference), np.nan),
                    where=np.abs(joined["b"].to_numpy(float)) > 1e-12,
                )
                relative_row = corrected_contrast(relative, 1.0 / (OUTER_FOLDS - 1), OUTER_FOLDS, 1)
                per_unit = (joined["a"] - joined["b"]).groupby(level="unit").mean()
                wins = int((per_unit < 0).sum())
                losses = int((per_unit > 0).sum())
                sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses > 0 else float("nan")
                rows.append(
                    {
                        "model": model,
                        "comparator": comparator,
                        "comparator_kind": "parent" if comparator == entry.parent else "reference",
                        "metric": metric,
                        "units": int(joined.index.get_level_values("unit").nunique()),
                        **stats_row,
                        "relative_mean": relative_row["mean"],
                        "relative_ci_low": relative_row["ci_low"],
                        "relative_ci_high": relative_row["ci_high"],
                        "units_better": wins,
                        "units_worse": losses,
                        "sign_test_p": sign_p,
                    }
                )
    return with_holm_correction(pd.DataFrame(rows))


def ablation_promotions(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entry in registry.ABLATIONS + registry.ROW_SCRAMBLED_CONTROLS + registry.SUCCESSOR_ABLATIONS:
        subset = pooled[pooled["model"].eq(entry.model_id) & pooled["comparator"].eq(entry.parent)]
        frozen_hits = []
        posthoc_hits = []
        for metric in ("rmse", "regret_at_1"):
            row = subset[subset["metric"].eq(metric)]
            if row.empty:
                continue
            low, high = float(row["ci_low"].iloc[0]), float(row["ci_high"].iloc[0])
            if np.isfinite(low) and np.isfinite(high) and (low > 0.0 or high < 0.0):
                frozen_hits.append(f"{metric}:{float(row['mean'].iloc[0]):+.5f}")
            if metric == "rmse":
                rel_low, rel_high = float(row["relative_ci_low"].iloc[0]), float(row["relative_ci_high"].iloc[0])
                if np.isfinite(rel_low) and np.isfinite(rel_high) and (rel_low > 0.0 or rel_high < 0.0):
                    posthoc_hits.append(f"relative_rmse:{float(row['relative_mean'].iloc[0]):+.3f}")
                sign_p = float(row["sign_test_p"].iloc[0])
                if np.isfinite(sign_p) and sign_p < 0.05:
                    posthoc_hits.append(
                        f"sign_test:{int(row['units_better'].iloc[0])}/{int(row['units_worse'].iloc[0])} p={sign_p:.3g}"
                    )
        matched_control = entry.role == "control" or (entry.ablated_mechanism or "").startswith(
            ("retention_gate", "head=signed", "families=shuffled", "coordinate=permuted", "harm=scrambled")
        )
        frozen_promoted = bool(frozen_hits) or matched_control
        amended_promoted = frozen_promoted or bool(posthoc_hits)
        reason = (
            "; ".join(frozen_hits + posthoc_hits)
            if (frozen_hits or posthoc_hits)
            else ("matched control retained for attribution" if matched_control else "no interval excludes zero")
        )
        rows.append(
            {
                "ablation": entry.model_id,
                "parent": entry.parent,
                "mechanism": entry.ablated_mechanism,
                "role": entry.role,
                "screened": not subset.empty,
                "promoted_by_frozen_rule": frozen_promoted and not subset.empty,
                "promoted_to_certify": amended_promoted and not subset.empty,
                "posthoc_trigger": "; ".join(posthoc_hits),
                "reason": reason if not subset.empty else "not screened",
                "rule": PROMOTION_RULE,
                "amendment": PROMOTION_AMENDMENT,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# StarCoder one-dimensional suite
# ---------------------------------------------------------------------------------------------


def starcoder_curve_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    curves = predictions[predictions["panel_kind"].eq("curve")]
    if curves.empty:
        return pd.DataFrame()
    rows = []
    for (model, panel_name), group in curves.groupby(["model", "panel"], sort=False):
        panel = load_panel(panel_name)
        share = panel.features.weights[:, 1]
        observed = panel.group(STARCODER_TARGET).aggregate
        prediction = np.full(panel.rows, np.nan)
        prediction[group["row_index"].to_numpy(int)] = group["prediction"].to_numpy(float)
        complete = np.isfinite(prediction).all()
        measured_argmin = int(np.argmin(observed))
        interior_measured = bool(0 < measured_argmin < panel.rows - 1)
        row: dict[str, Any] = {
            "model": model,
            "curve_id": panel_name.removeprefix(STARCODER_PANEL_PREFIX),
            "family": panel.family,
            "support_id": panel.metadata["support_id"],
            "planned_materialized_tokens": panel.metadata["planned_materialized_tokens"],
            "points": panel.rows,
            "complete": complete,
            "measured_argmin_weight": float(share[measured_argmin]),
            "measured_min_bpb": float(observed[measured_argmin]),
            "measured_interior_minimum": interior_measured,
        }
        if complete:
            metrics = metric_row(observed, prediction, np.zeros(panel.rows, dtype=bool))
            selected = int(np.argmin(prediction))
            row.update(
                {
                    "rmse": metrics["rmse"],
                    "spearman": metrics["spearman"],
                    "calibration_intercept": metrics["calibration_intercept"],
                    "calibration_slope": metrics["calibration_slope"],
                    "selected_weight": float(share[selected]),
                    "selected_bpb": float(observed[selected]),
                    "regret_at_1": metrics["regret_at_1"],
                    "regret_at_top_k": metrics["regret_at_top_k"],
                    "selection_optimism": metrics["selection_optimism"],
                    "predicted_interior_minimum": bool(0 < selected < panel.rows - 1),
                    "expresses_interior_minimum": (
                        bool(0 < selected < panel.rows - 1) if interior_measured else float("nan")
                    ),
                }
            )
            sd = panel.repeat_sd.get(STARCODER_TARGET, float("nan"))
            row["repeat_sd"] = sd
            row["rmse_over_repeat_sd"] = metrics["rmse"] / sd if np.isfinite(sd) and sd > 0 else float("nan")
            if "one_sd_basin_low" in panel.metadata:
                row["sampled_min_weight"] = panel.metadata["sampled_min_weight"]
                row["one_sd_basin_low"] = panel.metadata["one_sd_basin_low"]
                row["one_sd_basin_high"] = panel.metadata["one_sd_basin_high"]
                row["selected_in_one_sd_basin"] = bool(
                    panel.metadata["one_sd_basin_low"] - 1e-9
                    <= share[selected]
                    <= panel.metadata["one_sd_basin_high"] + 1e-9
                )
                row["selected_minus_sampled_min_weight"] = float(share[selected] - panel.metadata["sampled_min_weight"])
        rows.append(row)
    return pd.DataFrame(rows)


def starcoder_family_summary(curve_metrics: pd.DataFrame) -> pd.DataFrame:
    if curve_metrics.empty:
        return pd.DataFrame()
    complete = curve_metrics[curve_metrics["complete"].astype(bool)]
    columns = ["rmse", "spearman", "calibration_slope", "regret_at_1", "regret_at_top_k", "selection_optimism"]
    if complete.empty or any(column not in complete.columns for column in columns):
        return pd.DataFrame()
    family = complete.groupby(["model", "family"], as_index=False).agg(
        curves=("curve_id", "size"),
        **{column: (column, "mean") for column in columns},
        interior_expressed=(
            "expresses_interior_minimum",
            lambda values: float(np.nanmean(values.astype(float))) if values.notna().any() else float("nan"),
        ),
    )
    macro = family.groupby("model", as_index=False).agg(
        families=("family", "size"),
        curves=("curves", "sum"),
        **{column: (column, "mean") for column in columns},
        interior_expressed=("interior_expressed", "mean"),
    )
    macro.insert(1, "family", "equal_family_macro")
    return pd.concat([family, macro], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Complexity, runtime, registry tables
# ---------------------------------------------------------------------------------------------


def representative_design_rank(entry: registry.ModelEntry, panel: BenchPanel) -> tuple[int, int, int]:
    """Numerical rank, column count, and nonlinear dof of the model's design on the full panel."""
    features = registry.apply_transform(panel.features, entry)
    model = entry.build(features)
    if isinstance(model, models.GridModel):
        design = model.design(features, model.candidate_shapes(features)[0]).values
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.ProfiledDspModel):
        box = model._bounds(features.buckets)
        design = model.design(features, np.asarray([0.5 * (low + high) for low, high in box]))
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.CompactRetainedModel):
        design = model._design(features, model._starts()[0]).values
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.BowlModel):
        design = models.bowl_design(features, models.base_mu(features.exposures), symmetric=model.symmetric).values
        dof = 1
    elif isinstance(model, models.HierarchicalModel):
        config = {**model.screen.candidate_shapes(features)[0], "residual_shrink": model.residual_grid[0]}
        design = model._design(features, config).values
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.BandModel):
        config = {**model.base.screen.candidate_shapes(features)[0], "residual_shrink": model.base.residual_grid[0]}
        design = model.base._design(features, config).values
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.FamilyOnsetModel):
        shape = model.shared.candidate_shapes(features)[0]
        design = model._design(
            features, shape, np.full(len(features.families.members), float(shape["threshold"]))
        ).values
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.RetainedPowerLawModel):
        design = model._design(features, model.shapes[0]).values
        dof = model.nonlinear_dof(features)
    elif isinstance(model, models.LinearWeightModel | models.OlmixTaskwiseModel):
        design = features.weights
        dof = model.nonlinear_dof(features)
    else:
        return 0, 0, 0
    return models.effective_rank(design), int(design.shape[1]), int(dof)


def grid_summary(entry: registry.ModelEntry, panel: BenchPanel) -> str:
    """Human-readable hyperparameter budget of the model as actually built for this panel."""
    model = entry.build(registry.apply_transform(panel.features, entry))
    if isinstance(model, models.GridModel):
        search = models.describe_model(model)["effective_search"]
        return f"shapes x{len(model.candidate_shapes(panel.features))}, ridge x{len(model.ridge_grid)}, {search} search"
    if isinstance(model, models.HierarchicalModel):
        return (
            f"screen shapes x{len(model.screen.candidate_shapes(panel.features))}, ridge "
            f"x{len(model.screen.ridge_grid)}, "
            f"residual shrink x{len(model.residual_grid)}, top {model.top_shapes}"
        )
    if isinstance(model, models.BandModel):
        base = model.base
        return (
            f"screen shapes x{len(base.screen.candidate_shapes(panel.features))}, ridge x{len(base.screen.ridge_grid)}, "
            f"residual shrink x{len(base.residual_grid)}, band {model.relative_width:.2f} max {model.max_members}"
        )
    if isinstance(model, models.FamilyOnsetModel):
        return (
            f"shapes x{len(model.shared.candidate_shapes(panel.features))}, ridge x{len(model.shared.ridge_grid)}, "
            f"tau shrink x{len(model.tau_shrink_grid)}"
        )
    if isinstance(model, models.RetainedPowerLawModel):
        return (
            f"shapes x{len(model.shapes)}, ridge x{len(model.ridge_grid)}, top "
            f"{model.top_shapes}, {'huber' if model.robust else 'squared'}"
        )
    if isinstance(model, models.ProfiledDspModel):
        return (
            f"continuous {model.nonlinear_dof(panel.features)} dims, maxiter {model.options.maxiter}, "
            f"restarts {model.options.restarts}, ridge {model.options.linear_reg:g}, penalty {model.options.penalty}"
        )
    if isinstance(model, models.CompactRetainedModel):
        return (
            f"continuous {model.nonlinear_dof(panel.features)} dims in-sample, ridge "
            f"x{len(model.ridge_grid)}, benefit {model.benefit}, harm {model.harm}"
        )
    if isinstance(model, models.BowlModel):
        return (
            f"mu shift x{len(models.BOWL_MU_SHIFTS)}, ridge x{len(model.ridge_grid)}, "
            f"{'symmetric' if model.symmetric else 'asymmetric'}"
        )
    if isinstance(model, models.OlmixTaskwiseModel):
        return f"{model.n_starts} starts, {'analytic' if model.analytic_gradient else 'numerical'} gradient"
    return entry.hyperparameter_grid


def registry_table(panels: tuple[BenchPanel, ...]) -> pd.DataFrame:
    rows = []
    reference_panels = [panel for panel in panels if panel.kind == "tabular"]
    for entry in registry.ALL_ENTRIES:
        row: dict[str, Any] = {
            "single_phase_model_id": entry.model_id,
            "role": entry.role,
            "source_model_ids": ";".join(entry.source_model_ids),
            "visible": entry.visible,
            "equivalence_class": entry.equivalence_class,
            "display": entry.display,
            "parent": entry.parent or "",
            "ablated_mechanism": entry.ablated_mechanism or "",
            "feature_transform": entry.feature_transform or "",
            "active_mechanisms": json.dumps(entry.mechanisms, sort_keys=True),
            "removed_phase_terms": entry.removed_phase_terms,
            "allowed_metadata": entry.allowed_metadata,
            "solver_id": entry.solver_id,
            "hyperparameter_grid": (
                grid_summary(entry, reference_panels[0]) if reference_panels else entry.hyperparameter_grid
            ),
            "entry_fingerprint": entry_fingerprint(entry),
            "configuration_hash[300m_39bucket]": (
                description_hash(entry, reference_panels[1]) if len(reference_panels) > 1 else ""
            ),
            "note": entry.note,
        }
        for panel in reference_panels:
            rank, columns, dof = representative_design_rank(entry, panel)
            row[f"identifiable_linear_rank[{panel.name}]"] = rank
            row[f"columns[{panel.name}]"] = columns
            row[f"nonlinear_dof[{panel.name}]"] = dof
        rows.append(row)
    return pd.DataFrame(rows)


def equivalence_markdown() -> str:
    lines = [
        "# Single-phase equivalence classes",
        "",
        "Every Observatory `MODEL_IDS` entry and the exact tied-input model it reduces to. Two "
        "source ids in one class share one design and are benchmarked once.",
        "",
        "| class | source model ids | visibility | removed phase terms | allowed metadata |",
        "|---|---|---|---|---|",
    ]
    for entry in registry.PARENTS:
        lines.append(
            f"| `{entry.model_id}` | {', '.join(f'`{item}`' for item in entry.source_model_ids)} | "
            "{entry.visible} | {entry.removed_phase_terms} | {entry.allowed_metadata} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism flags",
            "",
            "| class | " + " | ".join(registry.PARENTS[0].mechanisms) + " |",
            "|---|" + "---|" * len(registry.PARENTS[0].mechanisms),
        ]
    )
    for entry in registry.PARENTS:
        lines.append(
            f"| `{entry.model_id}` | "
            + " | ".join(entry.mechanisms[key] for key in registry.PARENTS[0].mechanisms)
            + " |"
        )
    lines.extend(["", "## Notes", ""])
    for entry in registry.PARENTS:
        if entry.note:
            lines.append(f"- `{entry.model_id}`: {entry.note}")
    lines.extend(
        [
            "",
            "## Ablations and matched controls",
            "",
            "| ablation | parent | mechanism | role | transform |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in registry.ABLATIONS:
        lines.append(
            f"| `{entry.model_id}` | `{entry.parent}` | {entry.ablated_mechanism} | {entry.role} | "
            "{entry.feature_transform or ''} |"
        )
    return "\n".join(lines) + "\n"


def complexity_runtime(fits: pd.DataFrame, workers: int) -> pd.DataFrame:
    if fits.empty:
        return pd.DataFrame()
    ok = fits[fits["status"].eq("ok")]
    rows = []
    for model, group in fits.groupby("model", sort=False):
        good = ok[ok["model"].eq(model)]
        per_fit = float(good["elapsed"].mean()) if not good.empty else float("nan")
        # Certify holds 870 39-bucket, 80 Michael, and 225 curve fits; project with per-kind means.
        kinds = {
            "39": good[good["panel"].isin(THIRTY_NINE_BUCKET_PANELS)],
            "michael": good[good["panel"].isin(MICHAEL_PANELS)],
            "curve": good[good["panel_kind"].eq("curve")],
        }
        means = {
            name: float(block["elapsed"].mean()) if not block.empty else float("nan") for name, block in kinds.items()
        }
        projected = 870 * means["39"] + 80 * means["michael"] + 225 * means["curve"]
        row: dict[str, Any] = {
            "model": model,
            "shards": len(group),
            "ok": len(good),
            "failed": int((group["status"] == "failed").sum()),
            "missing": int((group["status"] == "missing").sum()),
            "fit_seconds_total": float(good["elapsed"].sum()),
            "fit_seconds_mean": per_fit,
            "fit_seconds_max": float(good["elapsed"].max()) if not good.empty else float("nan"),
            "mean_seconds_39bucket": means["39"],
            "mean_seconds_michael": means["michael"],
            "mean_seconds_curve": means["curve"],
            "projected_certify_wall_clock_seconds": projected / workers if np.isfinite(projected) else float("nan"),
            "converged_fraction": (
                float(good["diag_converged"].astype(float).mean())
                if "diag_converged" in good and not good.empty
                else float("nan")
            ),
            "mean_boundary_hits": (
                float(good["diag_boundary_hits"].astype(float).mean())
                if "diag_boundary_hits" in good and not good.empty
                else float("nan")
            ),
        }
        row["five_minute_target_met"] = (
            bool(row["projected_certify_wall_clock_seconds"] <= FIVE_MINUTE_TARGET)
            if np.isfinite(row["projected_certify_wall_clock_seconds"])
            else False
        )
        row["eight_minute_gate_failed"] = (
            bool(row["projected_certify_wall_clock_seconds"] > EIGHT_MINUTE_FAIL)
            if np.isfinite(row["projected_certify_wall_clock_seconds"])
            else False
        )
        for panel_name, block in good.groupby("panel"):
            if block["panel_kind"].iloc[0] != "tabular":
                continue
            token = panel_name
            row[f"median_effective_rank[{token}]"] = (
                float(block["diag_effective_rank"].astype(float).median())
                if "diag_effective_rank" in block
                else float("nan")
            )
            row[f"median_columns[{token}]"] = (
                float(block["diag_columns"].astype(float).median()) if "diag_columns" in block else float("nan")
            )
            row[f"median_fitted_dof[{token}]"] = (
                float(block["diag_fitted_dof"].astype(float).median()) if "diag_fitted_dof" in block else float("nan")
            )
            row[f"nonlinear_dof[{token}]"] = (
                float(block["diag_nonlinear_dof"].astype(float).median())
                if "diag_nonlinear_dof" in block
                else float("nan")
            )
            row[f"mean_seconds[{token}]"] = float(block["elapsed"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# External heldout optimum-selection test
# ---------------------------------------------------------------------------------------------


HELDOUT_TARGET_COLUMNS = {
    "uncheatable": ("uncheatable_n", "uncheatable_mean_bpb"),
    "table9": ("table9_macro_n", "table9_macro_mean_bpb"),
}
HELDOUT_RUN_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}


@functools.cache
def heldout_runs_for(panel_name: str, target: str) -> pd.DataFrame:
    runs = pd.read_csv(HELDOUT_DIR / "heldout_runs.csv")
    column = HELDOUT_RUN_COLUMNS[target]
    selected = runs[runs["panel"].eq(panel_name) & runs["eligible"].astype(bool) & runs[column].notna()]
    return selected.loc[:, ["coordinate_id", "row_id", column]].reset_index(drop=True)


@functools.cache
def heldout_registry() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    coordinates = pd.read_csv(HELDOUT_DIR / "heldout_coordinates.csv")
    components = pd.read_csv(HELDOUT_DIR / "heldout_coordinate_components.csv")
    hashes = {
        str((HELDOUT_DIR / name).relative_to(REPO_ROOT)): file_sha256(HELDOUT_DIR / name)
        for name in ("heldout_coordinates.csv", "heldout_coordinate_components.csv", "manifest.json")
    }
    return coordinates, components, hashes


def heldout_features(panel: BenchPanel, target: str) -> tuple[pd.DataFrame, models.Features]:
    coordinates, _components, _hashes = heldout_registry()
    count_column, _mean_column = HELDOUT_TARGET_COLUMNS[target]
    bank = coordinates[coordinates["panel"].eq(panel.name) & coordinates[count_column].fillna(0).gt(0)].reset_index(
        drop=True
    )
    weights = bank.loc[:, [f"weight::{bucket}" for bucket in panel.buckets]].to_numpy(float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{panel.name}: heldout weights are not normalized")
    features = dataclasses.replace(
        panel.features,
        exposures=weights * panel.features.inventory[None, :],
        weights=weights,
        label=f"{panel.name}|heldout|{target}",
    )
    return bank, features


def heldout_shard_path(
    output_dir: Path, model_id: str, panel: str, target: str, component_index: int, component: str
) -> Path:
    return (
        output_dir
        / "heldout_shards"
        / model_id
        / panel
        / target
        / f"component_{component_index:03d}_{canonical._component_token(component)}.npz"
    )


def fit_heldout_component(
    model_id: str, panel_name: str, target: str, component_index: int, output_dir: Path, split_hash: str
) -> str:
    panel = load_panel(panel_name)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    component = group.components[component_index]
    bank, bank_features = heldout_features(panel, target)
    _coordinates, _components, registry_hashes = heldout_registry()
    common = {
        "panel_inputs": panel.input_hashes,
        "features": panel.features.cache_key,
        "heldout": registry_hashes,
        "bank": bank_features.cache_key,
    }
    primary = {
        "fit": {**fit_protocol_core(), "fit_path_hash": fit_path_hash(), "inner": "heldout_inner_folds"},
        "entry": {
            "model_id": model_id,
            "feature_transform": entry.feature_transform,
            "parent": entry.parent,
            "description": description_hash(entry, panel),
        },
        **common,
    }
    protocol_hashes = [hashlib.sha256(json.dumps(primary, sort_keys=True).encode()).hexdigest()]
    snapshot = legacy_snapshot(output_dir)
    if snapshot and legacy_accepted(entry, panel, output_dir):
        legacy_core = {**fit_protocol_core(), "source_hashes": {"models": snapshot["legacy_models_hash"]}}
        legacy_common = {"entry": snapshot["entry_fingerprints"][model_id], **common}
        protocol_hashes.extend(
            hashlib.sha256(json.dumps({"fit": fit_payload, **legacy_common}, sort_keys=True).encode()).hexdigest()
            for fit_payload in (
                {**legacy_core, "inner": "heldout_inner_folds"},
                {**legacy_core, "split_hash": split_hash},
            )
        )
    protocol_hash = protocol_hashes[0]
    path = heldout_shard_path(output_dir, model_id, panel_name, target, component_index, component)
    payload = load_shard(path)
    if (
        payload is not None
        and str(payload["protocol_hash"].item()) == protocol_hash
        and payload["prediction"].shape == (len(bank),)
    ):
        return "cached"
    response = group.outcomes[:, component_index]
    features = dataclasses.replace(
        registry.apply_transform(panel.features, entry), component=str(group.components[component_index])
    )
    started = time.monotonic()
    result: dict[str, Any] = {
        "protocol_hash": protocol_hash,
        "model_id": model_id,
        "component": component,
        "coordinate_id": bank["coordinate_id"].to_numpy(str),
    }
    try:
        model = entry.build(features)
        rows = np.arange(panel.rows)
        fitted = model.fit(
            features,
            response,
            rows,
            heldout_inner_folds(panel),
            _seed(FitTask(model_id, panel_name, target, component_index, component, 0, 0)),
        )
        query = registry.apply_transform(bank_features, entry)
        prediction = np.asarray(model.predict(fitted, query, np.arange(len(bank))), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("non-finite heldout prediction")
        result.update(
            {
                "status": "ok",
                "error": "",
                "prediction": prediction,
                "fit_prediction": np.asarray(model.predict(fitted, features, rows), dtype=float),
                "shape_json": json.dumps(fitted.shape, sort_keys=True),
                "ridge": float(fitted.ridge),
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "prediction": np.full(len(bank), np.nan),
                "fit_prediction": np.full(panel.rows, np.nan),
                "shape_json": "{}",
                "ridge": float("nan"),
            }
        )
    result["elapsed"] = time.monotonic() - started
    atomic_save(path, result)
    return "fitted" if result["status"] == "ok" else "failed"


def random_ranking_expectations(loss: np.ndarray, top_k: int) -> dict[str, float]:
    """Exact expectations for a uniformly random ranking of the bank."""
    ordered = np.sort(loss)
    count = len(ordered)
    best = ordered[0]
    result = {"random_regret_at_1": float(ordered.mean() - best)}
    for requested in (top_k, 10):
        k = min(requested, count)
        total = comb(count, k)
        probabilities = np.array([comb(count - 1 - i, k - 1) / total for i in range(count)])
        result[f"random_best_of_{requested}_regret"] = float((probabilities * (ordered - best)).sum())
    return result


def heldout_selection_metrics(
    output_dir: Path, model_ids: tuple[str, ...], split_hash: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join frozen heldout predictions to measured coordinate means and score selection."""
    coordinates, components_table, _hashes = heldout_registry()
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for panel_name in THIRTY_NINE_BUCKET_PANELS:
        panel = load_panel(panel_name)
        for target in ("uncheatable", "table9"):
            group = panel.group(target)
            bank, _features = heldout_features(panel, target)
            _count_column, mean_column = HELDOUT_TARGET_COLUMNS[target]
            measured = bank[mean_column].to_numpy(float)
            for model_id in model_ids:
                matrix = np.full((len(bank), len(group.components)), np.nan)
                status = "ok"
                for component_index, component in enumerate(group.components):
                    payload = load_shard(
                        heldout_shard_path(output_dir, model_id, panel_name, target, component_index, component)
                    )
                    if payload is None or str(payload["status"].item()) != "ok":
                        status = "incomplete"
                        continue
                    matrix[:, component_index] = payload["prediction"]
                if status != "ok" or not np.isfinite(matrix).all():
                    metric_rows.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "stratum": "pooled",
                            "status": "incomplete",
                        }
                    )
                    continue
                predicted = matrix @ group.aggregation_weights
                digest = hashlib.sha256(np.ascontiguousarray(predicted).tobytes()).hexdigest()
                for index in range(len(bank)):
                    prediction_rows.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "coordinate_id": bank["coordinate_id"].iloc[index],
                            "sources": bank["sources"].iloc[index],
                            "run_count": int(bank["run_count"].iloc[index]),
                            "prediction": float(predicted[index]),
                            "measured_mean_bpb": float(measured[index]),
                            "prediction_hash": digest,
                        }
                    )
                strata = [("pooled", np.ones(len(bank), dtype=bool))] + [
                    (source, bank["sources"].eq(source).to_numpy())
                    for source in bank["sources"].unique()
                    if bank["sources"].eq(source).sum() >= 5
                ]
                tolerance = BASIN_TOLERANCE_SD * panel.repeat_sd.get(target, float("nan"))
                for stratum, mask in strata:
                    loss = measured[mask]
                    guess = predicted[mask]
                    order = np.argsort(guess, kind="stable")
                    selected = int(order[0])
                    ranks = stats.rankdata(loss, method="average")
                    row: dict[str, Any] = {
                        "model": model_id,
                        "panel": panel_name,
                        "target": target,
                        "stratum": stratum,
                        "status": "ok",
                        "bank_size": int(mask.sum()),
                        "prediction_hash": digest,
                        "selected_coordinate_id": bank["coordinate_id"].to_numpy(str)[mask][selected],
                        "selected_measured_bpb": float(loss[selected]),
                        "selected_measured_rank": float(ranks[selected]),
                        "selected_percentile": float((ranks[selected] - 1) / max(mask.sum() - 1, 1)),
                        "best_measured_bpb": float(loss.min()),
                        "regret_at_1": float(loss[selected] - loss.min()),
                        "top5_best_measured_bpb": float(loss[order[:5]].min()),
                        "top5_regret": float(loss[order[:5]].min() - loss.min()),
                        "top10_best_measured_bpb": float(loss[order[:10]].min()),
                        "top10_regret": float(loss[order[:10]].min() - loss.min()),
                        "rmse": float(np.sqrt(np.mean((guess - loss) ** 2))),
                        "spearman": _safe_spearman(loss, guess),
                        "selection_optimism": float(loss[selected] - guess[selected]),
                        "basin_tolerance": tolerance,
                        "basin_hit": (
                            bool(loss[selected] - loss.min() <= tolerance) if np.isfinite(tolerance) else float("nan")
                        ),
                    }
                    row.update(random_ranking_expectations(loss, TOP_K))
                    metric_rows.append(row)
                # Run-level seed sensitivity: the same coordinate predictions scored against every
                # eligible run, alongside the noise floor those repeated runs imply.
                runs = heldout_runs_for(panel_name, target)
                joined = runs.merge(
                    pd.DataFrame(
                        {"coordinate_id": bank["coordinate_id"], "prediction": predicted, "coordinate_mean": measured}
                    ),
                    on="coordinate_id",
                    how="inner",
                )
                if len(joined) >= 5:
                    run_values = joined[HELDOUT_RUN_COLUMNS[target]].to_numpy(float)
                    metric_rows.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "stratum": "run_level",
                            "status": "ok",
                            "bank_size": len(joined),
                            "prediction_hash": digest,
                            "rmse": float(np.sqrt(np.mean((joined["prediction"].to_numpy(float) - run_values) ** 2))),
                            "spearman": _safe_spearman(run_values, joined["prediction"].to_numpy(float)),
                            "noise_floor_rmse": float(
                                np.sqrt(np.mean((joined["coordinate_mean"].to_numpy(float) - run_values) ** 2))
                            ),
                            "repeated_coordinates": int((joined.groupby("coordinate_id").size() > 1).sum()),
                        }
                    )
                # Component-level scoring on coordinates whose atomic components are complete.
                complete = components_table[
                    components_table["panel"].eq(panel_name) & components_table["target"].eq(target)
                ]
                pivot = complete.pivot_table(
                    index="coordinate_id", columns="component", values="bpb_mean", aggfunc="first"
                )
                pivot = pivot.reindex(columns=list(group.components)).dropna()
                shared = bank["coordinate_id"].isin(pivot.index).to_numpy()
                if shared.sum() >= 5:
                    observed_components = pivot.loc[bank["coordinate_id"][shared]].to_numpy(float)
                    predicted_components = matrix[shared]
                    component_rmse = np.sqrt(np.mean((predicted_components - observed_components) ** 2, axis=0))
                    metric_rows.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "stratum": "component_complete_subset",
                            "status": "ok",
                            "bank_size": int(shared.sum()),
                            "prediction_hash": digest,
                            "mean_component_rmse": float(component_rmse.mean()),
                            "median_component_rmse": float(np.median(component_rmse)),
                            "mean_component_spearman": float(
                                np.nanmean(
                                    [
                                        _safe_spearman(observed_components[:, j], predicted_components[:, j])
                                        for j in range(observed_components.shape[1])
                                    ]
                                )
                            ),
                        }
                    )
    del coordinates, split_hash
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def run_heldout(output_dir: Path, model_ids: tuple[str, ...], workers: int, split_hash: str) -> dict[str, int]:
    jobs = []
    for model_id in model_ids:
        for panel_name in THIRTY_NINE_BUCKET_PANELS:
            panel = load_panel(panel_name)
            for target in ("uncheatable", "table9"):
                for component_index in range(len(panel.group(target).components)):
                    jobs.append((model_id, panel_name, target, component_index))
    print(f"heldout fits: {len(jobs)}", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        statuses = Parallel(n_jobs=workers, verbose=5, batch_size=1)(
            delayed(fit_heldout_component)(*job, output_dir, split_hash) for job in jobs
        )
    return {
        "total": len(jobs),
        "cached": statuses.count("cached"),
        "fitted": statuses.count("fitted"),
        "failed": statuses.count("failed"),
    }


# ---------------------------------------------------------------------------------------------
# Report and command line
# ---------------------------------------------------------------------------------------------


def _markdown(frame: pd.DataFrame, columns: list[str] | None = None, floatfmt: str = ".5f") -> str:
    if frame.empty:
        return "_no rows_"
    selected = frame if columns is None else frame.loc[:, [column for column in columns if column in frame.columns]]
    return selected.to_markdown(index=False, floatfmt=floatfmt)


def write_report(
    output_dir: Path, tier: str, tables: dict[str, pd.DataFrame], counts: dict[str, int], panels: tuple[BenchPanel, ...]
) -> None:
    aggregate = tables["aggregate_metrics"]
    lines = [
        "# Single-phase Observatory benchmark",
        "",
        f"Tier `{tier}`. Shards: {counts.get('total', 0)} planned, {counts.get('cached', 0)} "
        "cached, {counts.get('fitted', 0)} fitted, {counts.get('failed', 0)} failed.",
        "",
        "Three kinds of evidence appear below and must not be mixed: fit-panel nested cross-validation "
        "(every leaderboard and ablation table), retrospective external validation on the coordinate-disjoint "
        "heldout registry (the selection tables, which are development evidence once they influence a choice), "
        "and fresh confirmation runs, of which there are none in this report.",
        "",
        "## Panels",
        "",
        "| panel | kind | rows | buckets | families | targets | repeat SD |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for panel in panels:
        if panel.kind == "curve":
            continue
        lines.append(
            f"| {panel.name} | {panel.kind} | {panel.rows} | {len(panel.buckets)} | "
            "{panel.features.families.description()} | {', '.join(group.name for group in "
            "panel.groups)} | {json.dumps({key: round(value, 6) for key, value in "
            "panel.repeat_sd.items()})} |"
        )
    curves = [panel for panel in panels if panel.kind == "curve"]
    if curves:
        lines.append(
            f"| StarCoder one-dimensional suite | curve | {sum(panel.rows for panel in curves)} "
            "observations | 2 | none | {STARCODER_TARGET} | fixed-model curves only |"
        )
    lines.extend(
        [
            "",
            "## Model inventory",
            "",
            "See `model_registry.csv` and `equivalence_classes.md`. Parents: "
            + ", ".join(f"`{entry.model_id}`" for entry in registry.PARENTS)
            + ".",
            "",
        ]
    )
    if not aggregate.empty:
        lines.extend(["## Reconstructed aggregates (fit-panel nested CV)", ""])
        for (panel_name, target), block in aggregate.groupby(["panel", "target"], sort=False):
            block = block.sort_values("rmse")
            lines.extend(
                [
                    f"### {panel_name} / {target}",
                    "",
                    _markdown(
                        block,
                        [
                            "model",
                            "role",
                            "rmse",
                            "rmse_over_repeat_sd",
                            "spearman",
                            "calibration_slope",
                            "regret_at_1",
                            "regret_at_top_k",
                            "mean_fold_regret_at_1",
                            "selection_optimism",
                            "basin_rmse",
                            "basin_spearman",
                        ],
                    ),
                    "",
                ]
            )
    component = tables["component_metrics"]
    if not component.empty:
        anchors = _anchor_components()
        anchor_ids = set(anchors["uncheatable"]) | set(anchors["table9"]) | set(MICHAEL_TASKS)
        selected = component[component["component"].isin(anchor_ids) & component["panel_kind"].eq("tabular")]
        if not selected.empty:
            lines.extend(["## Atomic anchors (fit-panel nested CV, pooled over folds)", ""])
            for (panel_name, comp), block in selected.groupby(["panel", "component"], sort=False):
                lines.extend(
                    [
                        f"### {panel_name} / {comp}",
                        "",
                        _markdown(
                            block.sort_values("rmse"),
                            [
                                "model",
                                "role",
                                "rmse",
                                "rmse_over_repeat_sd",
                                "spearman",
                                "calibration_slope",
                                "regret_at_1",
                                "regret_at_top_k",
                                "selection_optimism",
                                "basin_rmse",
                            ],
                        ),
                        "",
                    ]
                )
    contrasts = tables["paired_model_contrasts"]
    if not contrasts.empty:
        lines.extend(
            [
                "## Paired contrasts against canonical DSP and taskwise OLMix",
                "",
                "Nadeau-Bengio corrected intervals over the outer folds; screening uncertainty unless the "
                "finalist stage ran.",
                "",
            ]
        )
        aggregate_contrasts = contrasts[
            contrasts["level"].eq("aggregate")
            & contrasts["metric"].isin(["rmse", "regret_at_1"])
            & contrasts["comparator_kind"].eq("reference")
        ]
        if not aggregate_contrasts.empty:
            lines.extend(
                [
                    _markdown(
                        aggregate_contrasts.sort_values(["panel", "target", "metric", "model"]),
                        [
                            "panel",
                            "target",
                            "model",
                            "comparator",
                            "metric",
                            "mean_difference",
                            "corrected_se",
                            "ci_low",
                            "ci_high",
                            "n_folds",
                        ],
                    ),
                    "",
                ]
            )
    promotions = tables["ablation_promotions"]
    if not promotions.empty:
        lines.extend(
            [
                "## Ablations",
                "",
                PROMOTION_RULE,
                "",
                _markdown(
                    promotions[promotions["screened"]],
                    [
                        "ablation",
                        "parent",
                        "mechanism",
                        "role",
                        "promoted_by_frozen_rule",
                        "promoted_to_certify",
                        "posthoc_trigger",
                        "reason",
                    ],
                ),
                "",
            ]
        )
        pooled = tables["pooled_screen_contrasts"]
        if not pooled.empty:
            lines.extend(
                [
                    "### Pooled Screen contrasts against parents",
                    "",
                    _markdown(
                        pooled[pooled["comparator_kind"].eq("parent")].sort_values(["model", "metric"]),
                        [
                            "model",
                            "comparator",
                            "metric",
                            "units",
                            "mean",
                            "corrected_se",
                            "ci_low",
                            "ci_high",
                            "n_folds",
                        ],
                    ),
                    "",
                ]
            )
    heldout = tables["external_heldout_selection_metrics"]
    if not heldout.empty:
        lines.extend(
            [
                "## External heldout optimum selection (retrospective)",
                "",
                "Frozen fit-panel refits scored on the coordinate-disjoint registry. Regret at 1 is the "
                "primary comparison; a basin hit uses the frozen one-SD tolerance where repeat noise is "
                "identified.",
                "",
            ]
        )
        pooled_rows = heldout[heldout["stratum"].eq("pooled") & heldout["status"].eq("ok")]
        lines.extend(
            [
                _markdown(
                    pooled_rows.sort_values(["panel", "target", "regret_at_1"]),
                    [
                        "panel",
                        "target",
                        "model",
                        "bank_size",
                        "selected_measured_bpb",
                        "selected_measured_rank",
                        "regret_at_1",
                        "top5_regret",
                        "top10_regret",
                        "random_regret_at_1",
                        "random_best_of_5_regret",
                        "random_best_of_10_regret",
                        "rmse",
                        "spearman",
                        "basin_hit",
                    ],
                ),
                "",
            ]
        )
    curve_summary = tables["starcoder_family_summary"]
    if not curve_summary.empty:
        lines.extend(
            [
                "## StarCoder one-dimensional shape suite (out-of-fold)",
                "",
                "Per-family macro-averages and the equal-family macro; the four fixed-model curves are "
                "reported separately below.",
                "",
                _markdown(
                    curve_summary.sort_values(["family", "rmse"]),
                    [
                        "model",
                        "family",
                        "curves",
                        "rmse",
                        "spearman",
                        "calibration_slope",
                        "regret_at_1",
                        "regret_at_top_k",
                        "interior_expressed",
                    ],
                ),
                "",
            ]
        )
        tied = tables["starcoder_tied_diagonal_metrics"]
        if not tied.empty:
            lines.extend(
                [
                    "### Fixed-model token-ladder anchors",
                    "",
                    _markdown(
                        tied.sort_values(["curve_id", "rmse"]),
                        [
                            "model",
                            "curve_id",
                            "rmse",
                            "rmse_over_repeat_sd",
                            "spearman",
                            "selected_weight",
                            "sampled_min_weight",
                            "selected_in_one_sd_basin",
                            "regret_at_1",
                            "measured_interior_minimum",
                            "predicted_interior_minimum",
                        ],
                    ),
                    "",
                ]
            )
    runtime = tables["complexity_and_runtime"]
    if not runtime.empty:
        lines.extend(
            [
                "## Complexity and runtime",
                "",
                _markdown(
                    runtime.sort_values("fit_seconds_mean"),
                    [
                        "model",
                        "shards",
                        "ok",
                        "failed",
                        "fit_seconds_mean",
                        "fit_seconds_max",
                        "projected_certify_wall_clock_seconds",
                        "five_minute_target_met",
                        "eight_minute_gate_failed",
                        "converged_fraction",
                        "mean_boundary_hits",
                    ],
                    floatfmt=".2f",
                ),
                "",
            ]
        )
    failures = tables["failures"]
    lines.extend(
        [
            "## Failures",
            "",
            f"{len(failures)} failed or missing shards." if not failures.empty else "No failed or missing shards.",
            "",
        ]
    )
    if not failures.empty:
        lines.append(
            _markdown(failures.head(50), ["model", "panel", "target", "component", "repeat", "fold", "status", "error"])
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(entry.model_id for entry in registry.ALL_ENTRIES)
    if value == "parents":
        return tuple(entry.model_id for entry in registry.PARENTS + registry.REFERENCES)
    if value == "ablations":
        return tuple(entry.model_id for entry in registry.ABLATIONS + registry.ROW_SCRAMBLED_CONTROLS)
    if value == "successors":
        return tuple(entry.model_id for entry in registry.SUCCESSORS + registry.SUCCESSOR_ABLATIONS)
    if value == "finalists":
        return tuple(entry.model_id for entry in registry.PARENTS + registry.REFERENCES + registry.SUCCESSORS)
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [item for item in selected if item not in registry.ENTRY_BY_ID]
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    return selected


def protocol_payload(
    tier: str, plan: TierPlan, panels: tuple[BenchPanel, ...], split_hash: str, model_ids: tuple[str, ...], workers: int
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "tier": tier,
        "repeats": plan.repeats,
        "folds": list(plan.folds),
        "components": {
            panel: {target: list(components) for target, components in targets.items()}
            for panel, targets in plan.components.items()
        },
        "curves": list(plan.curves),
        "models": list(model_ids),
        "fit_protocol": {
            **fit_protocol_core(),
            "fit_path_hash": fit_path_hash(),
            "legacy_manifest_split_hash": split_hash,
        },
        "source_hashes": source_hashes(),
        "cache_key": (
            "models-module hash, fit-path source hash, per-split fingerprint, built-model "
            "configuration hash, panel input hashes, panel feature hash; legacy keys "
            "accepted only for (entry, panel) configurations unchanged since "
            "legacy_entry_descriptions.json"
        ),
        "input_hashes": {panel.name: panel.input_hashes for panel in panels if panel.kind == "tabular"},
        "starcoder_input_hashes": next((panel.input_hashes for panel in panels if panel.kind == "curve"), {}),
        "repeat_noise": {panel.name: panel.repeat_sd for panel in panels if panel.repeat_sd},
        "noise_sources": {
            panel.name: panel.metadata.get("noise_sources", []) for panel in panels if panel.kind == "tabular"
        },
        "basin_definition": (
            f"rows with observed aggregate at or below the {BASIN_FRACTION:.0%} quantile of the "
            "complete panel (minimum {BASIN_MIN_ROWS} rows)"
        ),
        "basin_tolerance": f"{BASIN_TOLERANCE_SD} x pooled same-mixture repeat SD",
        "top_k": TOP_K,
        "selection_optimism": "observed minus predicted BPB at the predicted-minimum row",
        "promotion_rule": PROMOTION_RULE,
        "promotion_rule_amendment": PROMOTION_AMENDMENT,
        "workers": workers,
        "uncheatable_aggregation_weights": {
            "legacy_60m_300m": canonical.LEGACY_UNCHEATABLE_WEIGHTS.tolist(),
            "delphi_3e18": canonical.DELPHI_UNCHEATABLE_WEIGHTS.tolist(),
        },
        "starcoder_exposure_rule": (
            "c_S from the inventory support rule; c_N = planned tokens / Nemotron source tokens for "
            "full support, 1.0 for simulated and matched supports"
        ),
        "michael_families": "manifest cluster ids cXX with quality bins qY; quality order undeclared",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tier", choices=TIERS, default="smoke")
    parser.add_argument("--models", default="parents")
    parser.add_argument("--stage", choices=("fit", "report", "heldout", "shared", "all"), default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 1) - 2)))
    parser.add_argument("--heldout-models", default="parents")
    parser.add_argument(
        "--report-models", default=None, help="Models included in the metric tables; defaults to --models"
    )
    parser.add_argument(
        "--report-subdir", default=None, help="Write metric tables and the report under this sub-directory"
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    model_ids = parse_models(args.models)
    report_ids = parse_models(args.report_models) if args.report_models else model_ids
    plan = tier_plan(args.tier)
    panels = tuple(load_panel(name) for name in plan.components) + tuple(
        load_panel(f"{STARCODER_PANEL_PREFIX}{curve}") for curve in plan.curves
    )
    tabular = tuple(panel for panel in panels if panel.kind == "tabular")
    curves = tuple(panel for panel in panels if panel.kind == "curve")
    manifest = pd.concat([split_manifest(tabular, plan.repeats), split_manifest(curves, 1)], ignore_index=True)
    # Shards written before per-split fingerprints existed carry the hash of the one-repeat manifest.
    legacy_manifest = pd.concat([split_manifest(tabular, 1), split_manifest(curves, 1)], ignore_index=True)
    split_hash = manifest_hash(legacy_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)
    protocol = protocol_payload(args.tier, plan, panels, split_hash, model_ids, args.workers)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    registry_table(tuple(panel for panel in panels if panel.kind == "tabular")).to_csv(
        args.output_dir / "model_registry.csv", index=False
    )
    (args.output_dir / "equivalence_classes.md").write_text(equivalence_markdown())

    counts: dict[str, int] = {}
    if args.stage in ("fit", "all"):
        fit_ids = tuple(model_id for model_id in model_ids if model_id not in registry.SHARED_SHAPE_UNITS)
        tasks = plan_tasks(plan, fit_ids)
        counts = run_tasks(tasks, args.output_dir, split_hash, args.workers)
    if args.stage in ("fit", "shared", "all"):
        counts.update(run_shared_stage(plan, model_ids, args.output_dir, split_hash, args.workers))
    if args.stage in ("heldout", "all") and args.tier in ("certify", "finalist"):
        heldout_ids = parse_models(args.heldout_models)
        counts["heldout"] = run_heldout(args.output_dir, heldout_ids, args.workers, split_hash)["fitted"]
    if args.stage in ("report", "all", "heldout"):
        tasks = plan_tasks(plan, report_ids)
        predictions, fits = collect_shards(tasks, args.output_dir, split_hash)
        component_fold = (
            grouped_metrics(predictions, [*COMPONENT_KEYS, "fold"]) if not predictions.empty else pd.DataFrame()
        )
        component = (
            add_noise_ratio(grouped_metrics(predictions, COMPONENT_KEYS), "component")
            if not predictions.empty
            else pd.DataFrame()
        )
        aggregate_rows = aggregate_predictions(predictions) if not predictions.empty else pd.DataFrame()
        aggregate_fold = (
            grouped_metrics(aggregate_rows, ["model", "role", "parent", "panel", "target", "repeat", "fold"])
            if not aggregate_rows.empty
            else pd.DataFrame()
        )
        aggregate_repeat = (
            grouped_metrics(aggregate_rows, ["model", "role", "parent", "panel", "target", "repeat"])
            if not aggregate_rows.empty
            else pd.DataFrame()
        )
        aggregate = aggregate_summary(aggregate_repeat, aggregate_fold)
        contrasts = (
            pd.concat(
                [
                    paired_contrasts(component_fold, fits, "component"),
                    paired_contrasts(aggregate_fold, fits, "aggregate"),
                ],
                ignore_index=True,
            )
            if not component_fold.empty
            else pd.DataFrame()
        )
        pooled = pooled_anchor_contrasts(component_fold, fits) if not component_fold.empty else pd.DataFrame()
        promotions = ablation_promotions(pooled) if not pooled.empty else pd.DataFrame()
        curve_metrics = starcoder_curve_metrics(predictions) if not predictions.empty else pd.DataFrame()
        family_summary = starcoder_family_summary(curve_metrics)
        tied = (
            curve_metrics[curve_metrics["curve_id"].isin(FIXED_MODEL_CURVES)]
            if not curve_metrics.empty
            else pd.DataFrame()
        )
        runtime = complexity_runtime(fits, args.workers)
        failures = fits[~fits["status"].eq("ok")] if not fits.empty else pd.DataFrame()
        heldout_predictions, heldout_metrics = (
            heldout_selection_metrics(args.output_dir, parse_models(args.heldout_models), split_hash)
            if args.tier in ("certify", "finalist")
            else (pd.DataFrame(), pd.DataFrame())
        )
        # Promotion is decided at Screen; other tiers keep their pooled contrasts under a scoped name so
        # the frozen Screen decision table is never overwritten.
        scope = "" if args.tier == "screen" else f"_{args.tier}_scope"
        tables = {
            "component_fold_metrics": component_fold,
            "component_metrics": component,
            "aggregate_fold_metrics": aggregate_fold,
            "aggregate_repeat_metrics": aggregate_repeat,
            "aggregate_metrics": aggregate,
            "paired_model_contrasts": contrasts,
            f"pooled_screen_contrasts{scope}": pooled,
            f"ablation_promotions{scope}": promotions,
            "starcoder_one_dimensional_curve_metrics": curve_metrics,
            "starcoder_family_summary": family_summary,
            "starcoder_tied_diagonal_metrics": tied,
            "complexity_and_runtime": runtime,
            "failures": failures,
            "external_heldout_predictions": heldout_predictions,
            "external_heldout_selection_metrics": heldout_metrics,
            "fit_diagnostics": fits,
        }
        report_dir = args.output_dir / args.report_subdir if args.report_subdir else args.output_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(report_dir / f"{name}.csv", index=False)
        predictions.to_csv(report_dir / "component_predictions.csv", index=False)
        screen_promotions = args.output_dir / "screen" / "ablation_promotions.csv"
        if scope and screen_promotions.is_file():
            (report_dir / "ablation_promotions.csv").write_bytes(screen_promotions.read_bytes())
            tables["ablation_promotions"] = pd.read_csv(screen_promotions)
            tables["pooled_screen_contrasts"] = pd.read_csv(args.output_dir / "screen" / "pooled_screen_contrasts.csv")
        write_report(report_dir, args.tier, tables, counts, panels)
        print(f"wrote {report_dir}", flush=True)


if __name__ == "__main__":
    # Dispatch through the importable module so worker processes unpickle every helper by
    # reference instead of by value from ``__main__``.
    from experiments.domain_phase_mix.exploratory.two_phase_many import (
        benchmark_single_phase_observatory_20260902 as importable,
    )

    importable.main()
