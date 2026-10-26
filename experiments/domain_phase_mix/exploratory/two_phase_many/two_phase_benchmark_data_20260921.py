# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Inspected local two-phase panels and outcome-independent nested spatial folds.

Only the explicitly enumerated archives below are read. No heldout discovery,
remote fetch, cache refresh, calibration anchor, or phase-boundary outcome enters
this loader. Tied rows are diagnostics and never enter either fitting partition.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_atomic_metrics as atomic,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_panel_20260728 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    swarm39_harness_20260725 as swarm,
)

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "reference_outputs"
TPP40 = REFERENCE / "tpp40_frontier_gap_20260909/refresh_20260920/inputs"
ATOMIC_COVERAGE = (
    REFERENCE / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811/coverage_observations.csv"
)
ATOMIC_METRICS = (
    REFERENCE / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811/full_pool_atomic_metric_observations.csv"
)
DENSE_ATOMIC_METRICS = starcoder.SURFACE_DIR / "wsd80_all_bpb_metrics.csv"
DELPHI_COMPONENTS = REFERENCE / "delphi_3e18_observed_components_20260724/observed_component_panel.csv"
LLAMA200_COMPONENTS = (
    REFERENCE / "two_phase_solver_gap_collaborator_packet_20260701/data/all_300m_checkpoint_metrics.csv"
)
OBJECTIVE_PROTOCOL = REFERENCE / "single_phase_observatory_final_model_20260907/protocol.json"
SUPPORTS = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
BASE_PANELS = ("tpp40", "delphi_3e18", "llama160", "llama200", "starcoder_wsd80_dense")
GROUP_DECIMALS = 10
WEIGHT_TOLERANCE = 1e-8
REALIZED_ALPHA = {"delphi_3e18": 0.7981376787495837, "llama160": 3648 / 4577, "llama200": 18304 / 22888}
ALLOWED_INPUTS = frozenset(
    [
        swarm.CATALOG,
        swarm.CANONICAL / "delphi_3e18_two_phase_fit.csv",
        swarm.CANONICAL / "300m_two_phase_fit.csv",
        swarm.SIXTY_M / "fit_two_phase.csv",
        starcoder.SURFACE_CSV,
        DENSE_ATOMIC_METRICS,
        ATOMIC_COVERAGE,
        ATOMIC_METRICS,
        DELPHI_COMPONENTS,
        LLAMA200_COMPONENTS,
        OBJECTIVE_PROTOCOL,
        *(TPP40 / name for name in ("outcomes.csv", "buckets.csv", "objectives.csv", "run_specs.json")),
    ]
)


@dataclass(frozen=True)
class BenchPanel:
    """Aligned policy coordinates, outcomes, physical exposures, and provenance."""

    name: str
    buckets: tuple[str, ...]
    phase0: np.ndarray
    phase1: np.ndarray
    alpha: float
    c0: np.ndarray
    c1: np.ndarray
    targets: dict[str, np.ndarray]
    rows: pd.DataFrame
    input_hashes: dict[str, str]
    metadata: dict

    @property
    def aggregate(self) -> np.ndarray:
        return self.alpha * self.phase0 + (1.0 - self.alpha) * self.phase1

    @property
    def grouping_aggregate(self) -> np.ndarray:
        alpha = float(self.metadata.get("grouping_alpha", self.alpha))
        return alpha * self.phase0 + (1.0 - alpha) * self.phase1

    @property
    def contrast(self) -> np.ndarray:
        return self.phase1 - self.phase0

    @property
    def phase0_epochs(self) -> np.ndarray:
        return np.asarray(self.metadata.get("realized_phase0_epochs", self.c0 * self.phase0), dtype=float)

    @property
    def phase1_epochs(self) -> np.ndarray:
        return np.asarray(self.metadata.get("realized_phase1_epochs", self.c1 * self.phase1), dtype=float)

    @property
    def epochs(self) -> np.ndarray:
        return self.phase0_epochs + self.phase1_epochs

    @property
    def tied(self) -> np.ndarray:
        return self.rows["tied"].to_numpy(bool)


def _input_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    unknown = set(paths) - ALLOWED_INPUTS
    if unknown:
        raise ValueError(f"Inputs are outside the inspected archive whitelist: {unknown}")
    return {str(path): swarm.sha256_of(path) for path in paths}


def _read_csv(path: Path) -> pd.DataFrame:
    if path not in ALLOWED_INPUTS:
        raise ValueError(f"Input is outside the inspected archive whitelist: {path}")
    frame = pd.read_csv(path)
    swarm.assert_sealed_absent(frame, str(path))
    return frame


def _coordinate_key(values: np.ndarray) -> str:
    rounded = np.round(values, GROUP_DECIMALS)
    rounded[rounded == 0] = 0.0
    return hashlib.sha256(rounded.astype("<f8").tobytes()).hexdigest()[:24]


def _rows(
    name: str,
    phase0: np.ndarray,
    phase1: np.ndarray,
    grouping_alpha: float,
    row_ids: np.ndarray,
    sources: np.ndarray,
    regions: np.ndarray,
) -> pd.DataFrame:
    aggregate = grouping_alpha * phase0 + (1.0 - grouping_alpha) * phase1
    return pd.DataFrame(
        {
            "row_id": row_ids.astype(str),
            "policy_id": [_coordinate_key(row) for row in np.column_stack([phase0, phase1])],
            "group_id": [_coordinate_key(row) for row in aggregate],
            "source": sources.astype(str),
            "region": regions.astype(str),
            "tied": np.max(np.abs(phase0 - phase1), axis=1) < WEIGHT_TOLERANCE,
        }
    )


def _validate(panel: BenchPanel) -> BenchPanel:
    n, m = panel.phase0.shape
    if panel.phase1.shape != (n, m) or m != len(panel.buckets) or len(panel.rows) != n:
        raise ValueError(f"{panel.name}: misaligned policy matrices or row metadata")
    for values in (panel.phase0, panel.phase1):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{panel.name}: invalid policy weights")
        if not np.allclose(values.sum(axis=1), 1.0, atol=WEIGHT_TOLERANCE, rtol=0):
            raise ValueError(f"{panel.name}: policy weights do not sum to one")
    if panel.rows.row_id.duplicated().any():
        raise ValueError(f"{panel.name}: duplicate observation identities")
    if panel.c0.shape != (m,) or panel.c1.shape != (m,) or not 0 < panel.alpha < 1:
        raise ValueError(f"{panel.name}: invalid exposure conversion")
    for phase in (panel.phase0_epochs, panel.phase1_epochs):
        if phase.shape != (n, m) or not np.isfinite(phase).all() or np.any(phase < 0):
            raise ValueError(f"{panel.name}: invalid realized exposures")
    for key, target in panel.targets.items():
        if target.shape != (n,) or np.isinf(target).any() or np.isfinite(target).sum() < 2:
            raise ValueError(f"{panel.name}/{key}: incomplete or misaligned endpoint target")
    return panel


def _swarm_components(
    name: str,
    rows: pd.DataFrame,
    buckets: tuple[str, ...],
    phase0: np.ndarray,
    phase1: np.ndarray,
    aggregate_targets: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, str], dict]:
    if name == "llama160":
        return {}, {}, {"component_objective_certification": "unavailable: audit export contains aggregates only"}
    path = DELPHI_COMPONENTS if name == "delphi_3e18" else LLAMA200_COMPONENTS
    hashes = _input_hashes((path, OBJECTIVE_PROTOCOL))
    # Metadata-only selection skips non-fit outcome rows before CSV parsing.
    if name == "delphi_3e18":
        selector = pd.read_csv(path, usecols=["panel"]).panel.eq("two_phase_fit")
        identity = "source_row_name"
        protocol_name = "delphi_3e18_39bucket"
        weight_name = "delphi_3e18"
    else:
        metadata = pd.read_csv(path, usecols=["split", "packet_panel"])
        selector = metadata.split.eq("train") & metadata.packet_panel.eq("augmented_fit_panel")
        identity = "run_name"
        protocol_name = "300m_39bucket"
        weight_name = "legacy_60m_300m"
    selected = pd.read_csv(path, skiprows=(np.flatnonzero(~selector.to_numpy()) + 1).tolist())
    swarm.assert_sealed_absent(selected, f"{name} fit components")
    selected = selected.set_index(identity, verify_integrity=True).loc[rows.row_id]
    for phase, expected in ((0, phase0), (1, phase1)):
        actual = selected[[f"phase_{phase}_{bucket}" for bucket in buckets]].to_numpy(float)
        if not np.allclose(actual, expected, atol=WEIGHT_TOLERANCE, rtol=0):
            raise ValueError(f"{name}: component rows do not match canonical policy weights")
    protocol = json.loads(OBJECTIVE_PROTOCOL.read_text())
    groups = protocol["components"][protocol_name]
    targets, aggregation, errors, mismatches = {}, {}, {}, {}
    for objective, target_name in (("uncheatable", swarm.UNCHEATABLE), ("table9", swarm.TABLE9)):
        components = groups[objective]
        for key in components:
            source_key = key.replace("/", "_") if name == "llama200" and objective == "uncheatable" else key
            targets[key] = selected[source_key].to_numpy(float)
        weights = (
            np.asarray(protocol["uncheatable_aggregation_weights"][weight_name])
            if objective == "uncheatable"
            else np.full(len(components), 1 / len(components))
        )
        reconstructed = np.column_stack([targets[key] for key in components]) @ weights
        error = np.abs(reconstructed - aggregate_targets[target_name])
        mismatch = error > 3e-6
        if np.any(mismatch & ~rows.tied.to_numpy(bool)):
            raise ValueError(f"{name}/{objective}: asymmetric components do not reconstruct the published aggregate")
        # A canonical tied baseline is a repeat mean, while the old component
        # packet has one constituent endpoint. Do not misrepresent its atoms
        # as components of the repeat mean used for the scalar diagnostic.
        if mismatch.any():
            mismatches[target_name] = rows.row_id.loc[mismatch].tolist()
            for key in components:
                targets[key][mismatch] = np.nan
        errors[target_name] = float(np.max(error[~mismatch]))
        aggregation[target_name] = {"components": components, "weights": weights.tolist()}
    return (
        targets,
        hashes,
        {
            "component_objective_certification": (
                "58 components; reconstruction verified where available; mismatching tied repeat means masked"
            ),
            "objective_components": aggregation,
            "aggregate_reconstruction_max_error": errors,
            "component_unavailable_tied_repeat_means": mismatches,
        },
    )


def _swarm_panel(name: str) -> BenchPanel:
    small = name == "llama160"
    dataset_id = "300m_two_phase_fit" if name == "llama200" else "delphi_3e18_two_phase_fit"
    path = swarm.SIXTY_M / "fit_two_phase.csv" if small else swarm.CANONICAL / f"{dataset_id}.csv"
    frame = _read_csv(path)
    buckets, base0, base1, family_index, family_names = swarm._exposure(dataset_id)
    inventory = base0 + base1
    alpha = REALIZED_ALPHA[name]
    grouping_alpha = alpha if name == "delphi_3e18" else 0.8
    prefix = "phase_{}_" if small else "phase_{}_weight::"
    phase0, phase1 = [frame[[prefix.format(phase) + b for b in buckets]].to_numpy(float) for phase in (0, 1)]
    rows = _rows(
        name,
        phase0,
        phase1,
        grouping_alpha,
        frame["observation_id" if small else "row_id"].to_numpy(str),
        frame["source_family" if small else "training_series"].to_numpy(str),
        np.full(len(frame), "archived"),
    )
    targets = {key: frame[key].to_numpy(float) for key in (swarm.UNCHEATABLE, swarm.TABLE9)}
    components, component_hashes, component_metadata = _swarm_components(name, rows, buckets, phase0, phase1, targets)
    targets.update(components)
    return _validate(
        BenchPanel(
            name,
            buckets,
            phase0,
            phase1,
            alpha,
            alpha * inventory,
            (1 - alpha) * inventory,
            targets,
            rows,
            {**_input_hashes((path, swarm.CATALOG)), **component_hashes},
            {
                "grouping_alpha": grouping_alpha,
                "group_rounding_decimals": GROUP_DECIMALS,
                "exposure_convention": "swarm simulated epochs anchored at proportional policy 0.905353",
                "alpha_source": "inspect_llama_uncheatable_optima_20260910.REALIZED_ALPHA / canonical Delphi schedule",
                "scope": "inspected fit archive only; no single-phase calibration or heldout archive",
                "family_index": family_index.tolist(),
                "family_names": list(family_names),
                "native_constant_exposure_adapter": True,
                **component_metadata,
            },
        )
    )


def _tpp40_panel() -> BenchPanel:
    paths = tuple(TPP40 / name for name in ("outcomes.csv", "buckets.csv", "objectives.csv", "run_specs.json"))
    hashes = _input_hashes(paths)
    frame, inventory_frame, objectives = [_read_csv(path) for path in paths[:3]]
    specs = {spec["run_name"]: spec for spec in json.loads(paths[3].read_text())}
    buckets = tuple(inventory_frame.bucket.astype(str))
    inventory = inventory_frame.epochs_per_unit_weight.to_numpy(float)
    fractions = {float(specs[run]["phase_0_fraction"]) for run in frame.run_name}
    if len(fractions) != 1:
        raise ValueError("TPP40 runs have inconsistent phase fractions")
    grouping_alpha = fractions.pop()
    alpha = 21856 / 27336
    phase0, phase1 = [
        np.asarray([[specs[run]["phase_weights"][f"phase_{phase}"][b] for b in buckets] for run in frame.run_name])
        for phase in (0, 1)
    ]
    targets = {}
    for key in atomic.METRIC_KEYS:
        column = key.replace("-llama3", "")
        if np.isfinite(frame[column].to_numpy(float)).sum() >= 2:
            targets[column] = frame[column].to_numpy(float)
    uncheatable = objectives.loc[objectives.objective.eq("uncheatable")]
    targets[swarm.UNCHEATABLE] = frame[uncheatable.component.tolist()].to_numpy(float) @ uncheatable.weight.to_numpy(
        float
    )
    reconstruction_error = float(np.max(np.abs(targets[swarm.UNCHEATABLE] - frame["eval/uncheatable_eval/bpb"])))
    if reconstruction_error > 3e-6:
        raise ValueError("TPP40 Uncheatable components do not reconstruct the native byte-weighted score")
    paloma = [key for key in targets if key.startswith("eval/paloma/")]
    targets["paloma_macro_bpb"] = np.column_stack([targets[key] for key in paloma]).mean(axis=1)
    rows = _rows(
        "tpp40",
        phase0,
        phase1,
        grouping_alpha,
        frame.run_name.to_numpy(str),
        frame.source.to_numpy(str),
        frame.region.to_numpy(str),
    )
    if not np.array_equal(rows.tied.to_numpy(), frame.tied.to_numpy()):
        raise ValueError("TPP40 tied metadata disagree with the frozen weights")
    _, _, _, family_index, family_names = swarm._exposure("delphi_3e18_two_phase_fit")
    hashes.update(_input_hashes((swarm.CATALOG,)))
    return _validate(
        BenchPanel(
            "tpp40",
            buckets,
            phase0,
            phase1,
            alpha,
            alpha * inventory,
            (1 - alpha) * inventory,
            targets,
            rows,
            hashes,
            {
                "grouping_alpha": grouping_alpha,
                "group_rounding_decimals": GROUP_DECIMALS,
                "snapshot": "refresh_20260920",
                "coverage": "197 completed policies of 280 designed; completion is not an IID sample",
                "exposure_convention": "frozen simulated-epoch inventory from inputs/buckets.csv",
                "missing_targets": [swarm.TABLE9],
                "target_coverage": {key: int(np.isfinite(values).sum()) for key, values in targets.items()},
                "native_constant_exposure_adapter": True,
                "alpha_source": "analyze_tpp40_completed_atomic_20260920.ALPHA_HIGH: 21856/27336",
                "family_index": family_index.tolist(),
                "family_names": list(family_names),
                "objective_components": {
                    swarm.UNCHEATABLE: {
                        "components": uncheatable.component.tolist(),
                        "weights": uncheatable.weight.tolist(),
                    },
                    "paloma_macro_bpb": {"components": paloma, "weights": [1 / len(paloma)] * len(paloma)},
                },
                "aggregate_reconstruction_max_error": {swarm.UNCHEATABLE: reconstruction_error},
            },
        )
    )


def _dense_panel() -> BenchPanel:
    source = starcoder.load_surface()
    metrics = _read_csv(DENSE_ATOMIC_METRICS)
    frame = source.frame.merge(metrics, on="wandb_run_id", validate="one_to_one", how="left")
    phase0, phase1 = source.phase_0, source.phase_1
    targets = {key: frame[key].to_numpy(float) for key in atomic.METRIC_KEYS}
    targets["wsd80_bpb"] = source.y
    rows = _rows(
        "starcoder_wsd80_dense",
        phase0,
        phase1,
        starcoder.PHASE_0_FRACTION,
        frame.wandb_run_id.to_numpy(str),
        frame.source.to_numpy(str),
        np.full(len(frame), "archived"),
    )
    return _validate(
        BenchPanel(
            "starcoder_wsd80_dense",
            starcoder.DOMAIN_NAMES,
            phase0,
            phase1,
            starcoder.REALIZED_PHASE_0_FRACTION,
            source.c0,
            source.c1,
            targets,
            rows,
            _input_hashes((starcoder.SURFACE_CSV, DENSE_ATOMIC_METRICS)),
            {
                "grouping_alpha": starcoder.PHASE_0_FRACTION,
                "group_rounding_decimals": GROUP_DECIMALS,
                "exposure_convention": "realized 3040/3814 fraction; nominal 0.8 aggregate groups",
                "scope": (
                    "current unique reference surface coordinates; repeated-seed and confirmation outcomes excluded"
                ),
                "family_index": [0, 1],
                "family_names": list(starcoder.DOMAIN_NAMES),
                "native_constant_exposure_adapter": True,
            },
        )
    )


@cache
def _atomic_observations() -> tuple[pd.DataFrame, dict[str, str]]:
    # Reproduce the canonical atomic loader's alias-aware local join. Its public
    # load_support also initializes GCS and rewrites the cache, so it is not a
    # read-only boundary even with refresh=False.
    coverage, metrics = _read_csv(ATOMIC_COVERAGE), _read_csv(ATOMIC_METRICS)
    coverage["metric_run_name"] = coverage.run_name.where(~coverage.is_alias, coverage.alias_of_run_name)
    if coverage.run_name.duplicated().any() or metrics.run_name.duplicated().any():
        raise ValueError("Atomic coverage/cache has duplicate run identities")
    frame = coverage.merge(
        metrics,
        left_on="metric_run_name",
        right_on="run_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metric"),
    ).drop(columns="run_name_metric")
    if frame[list(atomic.METRIC_KEYS)].isna().any().any():
        raise ValueError("Frozen atomic cache is incomplete; remote refresh is prohibited")
    return frame, _input_hashes((ATOMIC_COVERAGE, ATOMIC_METRICS))


def _atomic_panel(name: str) -> BenchPanel:
    support, index = name.removeprefix("starcoder_wsd80_").rsplit("_h", 1)
    observations, hashes = _atomic_observations()
    horizons = sorted(observations.materialized_tokens.unique())
    frame = (
        observations.loc[observations.support_id.eq(support) & observations.materialized_tokens.eq(horizons[int(index)])]
        .sort_values("coordinate_id")
        .reset_index(drop=True)
    )
    p0, p1 = frame.phase_0_starcoder.to_numpy(float), frame.phase_1_starcoder.to_numpy(float)
    phase0, phase1 = np.column_stack([1 - p0, p0]), np.column_stack([1 - p1, p1])
    fractions = frame.boundary_step.to_numpy(float) / frame.total_steps.to_numpy(float)
    if not np.allclose(fractions, fractions[0], atol=1e-12, rtol=0):
        raise ValueError(f"{name}: inconsistent schedule")
    alpha = float(fractions[0])
    capacity = frame.nemotron_max_total_epochs.to_numpy(float)
    physical0 = np.column_stack([alpha * capacity * (1 - p0), frame.starcoder_phase_0_epochs.to_numpy(float)])
    physical1 = np.column_stack([(1 - alpha) * capacity * (1 - p1), frame.starcoder_phase_1_epochs.to_numpy(float)])
    c0 = np.asarray([alpha * capacity[0], np.max(physical0[p0 == 1, 1])])
    c1 = np.asarray([(1 - alpha) * capacity[0], np.max(physical1[p1 == 1, 1])])
    if support == "full" and (np.max(physical0[:, 1] + physical1[:, 1]) >= 1 or frame.starcoder_support_wraps.any()):
        raise ValueError("Full-support no-repetition archive contains replay")
    rows = _rows(
        name,
        phase0,
        phase1,
        0.8,
        frame.run_name.to_numpy(str),
        frame.support_id.to_numpy(str),
        frame.cell_id.to_numpy(str),
    )
    rows["checkpoint_id"] = frame.metric_run_name.to_numpy(str)
    return _validate(
        BenchPanel(
            name,
            ("nemotron_full", "starcoder"),
            phase0,
            phase1,
            alpha,
            c0,
            c1,
            {key: frame[key].to_numpy(float) for key in atomic.METRIC_KEYS},
            rows,
            dict(hashes),
            {
                "grouping_alpha": 0.8,
                "group_rounding_decimals": GROUP_DECIMALS,
                "support": support,
                "horizon_tokens": int(horizons[int(index)]),
                "exposure_convention": (
                    "exact materialized per-row StarCoder epochs; "
                    "constant c0/c1 full-share rates cannot reproduce discrete exposures"
                ),
                "native_constant_exposure_adapter": False,
                "family_index": [0, 1],
                "family_names": ["nemotron_full", "starcoder"],
                "realized_phase0_epochs": physical0.tolist(),
                "realized_phase1_epochs": physical1.tolist(),
                "scope": "dense coverage only; confirmation repeats and phase-boundary outcomes excluded",
            },
        )
    )


def panel_names() -> tuple[str, ...]:
    """Return the explicit local panel inventory without opening outcome files."""
    return (*BASE_PANELS, *(f"starcoder_wsd80_{support}_h{index}" for support in SUPPORTS for index in range(4)))


def load_panels(names: tuple[str, ...] | None = None) -> list[BenchPanel]:
    """Load inspected local archives, failing rather than fetching missing outcomes."""
    requested = panel_names() if names is None else names
    if len(set(requested)) != len(requested) or set(requested) - set(panel_names()):
        raise ValueError(f"Unknown or duplicate benchmark panels: {requested}")
    result = []
    for name in requested:
        if name == "tpp40":
            result.append(_tpp40_panel())
        elif name == "starcoder_wsd80_dense":
            result.append(_dense_panel())
        elif name in REALIZED_ALPHA:
            result.append(_swarm_panel(name))
        else:
            result.append(_atomic_panel(name))
    return result


def _neighborhood_labels(panel: BenchPanel, eligible: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    groups = panel.rows.group_id.to_numpy(str)
    unique = np.unique(groups[eligible])
    if len(unique) < n_folds:
        raise ValueError(f"{panel.name}: {len(unique)} eligible aggregate groups cannot fill {n_folds} folds")
    aggregate = panel.grouping_aggregate
    points = np.asarray([aggregate[eligible[groups[eligible] == group]].mean(axis=0) for group in unique])
    # Domain-deletion ties can form isolated neighborhoods containing no fit
    # policies. Fit centers on asymmetric groups, then assign diagnostic ties
    # geometrically; a tied counterpart inherits its asymmetric group's center.
    asymmetric_groups = np.unique(groups[eligible[~panel.tied[eligible]]])
    fit_group = np.isin(unique, asymmetric_groups)
    if fit_group.sum() < n_folds:
        raise ValueError(f"{panel.name}: too few asymmetric aggregate groups for {n_folds} folds")
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=n_folds, n_init=20, random_state=seed, algorithm="lloyd").fit(
            np.sqrt(points[fit_group])
        )
        labels = model.predict(np.sqrt(points))
    if len(np.unique(labels)) != n_folds:
        raise ValueError(f"{panel.name}: degenerate aggregate neighborhoods")
    order = sorted(np.unique(labels), key=lambda label: min(unique[labels == label]))
    remap = {old: new for new, old in enumerate(order)}
    by_group = {group: remap[label] for group, label in zip(unique, labels, strict=True)}
    return np.asarray([by_group[group] for group in groups[eligible]], dtype=int)


def frozen_folds(
    panel: BenchPanel, outer_folds: int = 5, inner_folds: int = 3, seed: int = 20260921, repeat: int = 0
) -> pd.DataFrame:
    """Freeze nested aggregate neighborhoods without reading any target values.

    All asymmetric rows occur in each outer block. Only that neighborhood's tied
    rows occur as tied_test; other ties are absent. Inner neighborhoods are rebuilt
    exclusively from outer-training asymmetric groups. This produces exactly one
    test/tied_test prediction per policy per repeat, with no pinned calibration.
    """
    if outer_folds < 2 or inner_folds < 2 or repeat < 0:
        raise ValueError("Nested folds require at least two inner/outer folds and a nonnegative repeat")
    indices = np.arange(len(panel.rows))
    tied = panel.rows.tied.to_numpy(bool)
    outer_seed = seed + 1009 * repeat
    outer_labels = _neighborhood_labels(panel, indices, outer_folds, outer_seed)
    frames = []
    for outer in range(outer_folds):
        train = indices[(outer_labels != outer) & ~tied]
        test = indices[outer_labels == outer]
        if not np.any(~tied[test]):
            raise ValueError(f"{panel.name}: outer neighborhood {outer} has no asymmetric test policies")
        inner = _neighborhood_labels(panel, train, inner_folds, outer_seed + outer + 1)
        positions = np.concatenate([train, test])
        roles = np.concatenate([np.full(len(train), "train"), np.where(tied[test], "tied_test", "test")])
        frames.append(
            pd.DataFrame(
                {
                    "panel": panel.name,
                    "repeat": repeat,
                    "outer_fold": outer,
                    "row_id": panel.rows.row_id.iloc[positions].to_numpy(str),
                    "role": roles,
                    "inner_fold": np.concatenate([inner, np.full(len(test), -1)]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def split_coverage_audit(
    panel: BenchPanel,
    folds: pd.DataFrame,
    *,
    outer_folds: int,
    inner_folds: int,
    minimum_asymmetric_test_rows: int,
    minimum_asymmetric_test_groups: int,
    expected_group_count: int | None = None,
) -> dict:
    """Describe frozen split coverage and insufficiencies without changing folds.

    Thresholds belong to the protocol and must be supplied explicitly. Empty
    folds produce a missing-challenge record. Counts describe geometry, before
    any target-specific finite-label masks applied by the runner.
    """
    if min(outer_folds, inner_folds, minimum_asymmetric_test_rows, minimum_asymmetric_test_groups) < 1:
        raise ValueError("Split-audit fold counts and coverage thresholds must be positive")
    columns = ["panel", "repeat", "outer_fold", "row_id", "role", "inner_fold"]
    records = folds.loc[folds.panel.eq(panel.name), columns].copy()
    canonical = records.sort_values(["repeat", "outer_fold", "row_id"]).to_csv(index=False, lineterminator="\n")
    group_count = int(panel.rows.group_id.nunique())
    structural_errors = []
    grouping = pd.DataFrame(
        {
            "recorded": panel.rows.group_id.to_numpy(str),
            "coordinate": [_coordinate_key(row) for row in panel.grouping_aggregate],
        }
    )
    if not grouping.groupby("recorded").coordinate.nunique().eq(1).all():
        structural_errors.append("recorded_group_merges_distinct_aggregate_coordinates")
    if not grouping.groupby("coordinate").recorded.nunique().eq(1).all():
        structural_errors.append("aggregate_coordinate_split_across_recorded_groups")
    if expected_group_count is not None and group_count != expected_group_count:
        structural_errors.append("aggregate_group_count_differs_from_frozen_expectation")
    if records.duplicated(["repeat", "outer_fold", "row_id"]).any():
        structural_errors.append("duplicate_row_within_outer_fold")
    if not set(records.row_id).issubset(set(panel.rows.row_id)):
        structural_errors.append("unknown_row_identity")
    joined = records.merge(panel.rows, on="row_id", how="left", validate="many_to_one")
    if not set(records.role).issubset({"train", "test", "tied_test"}):
        structural_errors.append("unknown_split_role")
    if joined.loc[joined.role.eq("train"), "tied"].fillna(False).any():
        structural_errors.append("tied_row_in_training")
    if joined.loc[joined.role.eq("test"), "tied"].fillna(False).any():
        structural_errors.append("tied_row_in_asymmetric_test")
    if (~joined.loc[joined.role.eq("tied_test"), "tied"].fillna(True)).any():
        structural_errors.append("asymmetric_row_in_tied_test")
    if (joined.loc[joined.role.ne("train"), "inner_fold"] != -1).any():
        structural_errors.append("scored_row_has_inner_fold")
    epoch_error = max(
        float(np.max(np.abs(panel.phase0_epochs - panel.c0 * panel.phase0))),
        float(np.max(np.abs(panel.phase1_epochs - panel.c1 * panel.phase1))),
    )
    native_rpl = bool(panel.metadata.get("native_constant_exposure_adapter", False)) and epoch_error <= 1e-10
    cells = []
    repeats = sorted(int(value) for value in records.repeat.unique())
    for repeat in repeats:
        repeated = joined.loc[joined.repeat.eq(repeat)]
        scored = repeated.loc[repeated.role.ne("train")]
        if set(scored.row_id) != set(panel.rows.row_id) or not scored.row_id.value_counts().eq(1).all():
            structural_errors.append(f"repeat_{repeat}:rows_not_scored_exactly_once")
        if not scored.groupby("group_id").outer_fold.nunique().eq(1).all():
            structural_errors.append(f"repeat_{repeat}:aggregate_group_crosses_scoring_folds")
        if not scored.groupby("policy_id").outer_fold.nunique().eq(1).all():
            structural_errors.append(f"repeat_{repeat}:repeated_coordinate_crosses_scoring_folds")
        for outer in range(outer_folds):
            block = repeated.loc[repeated.outer_fold.eq(outer)]
            train = block.loc[block.role.eq("train")]
            counts = {}
            for role in ("train", "test", "tied_test", "joint_test"):
                menu = block.loc[block.role.ne("train")] if role == "joint_test" else block.loc[block.role.eq(role)]
                counts[role] = {
                    "rows": len(menu),
                    "groups": int(menu.group_id.nunique()),
                    "policies": int(menu.policy_id.nunique()),
                }
            reasons = []
            if counts["test"]["rows"] < minimum_asymmetric_test_rows:
                reasons.append("asymmetric_test_rows_below_minimum")
            if counts["test"]["groups"] < minimum_asymmetric_test_groups:
                reasons.append("asymmetric_test_groups_below_minimum")
            if set(train.group_id) & set(block.loc[block.role.ne("train"), "group_id"]):
                structural_errors.append(f"repeat_{repeat}/fold_{outer}:aggregate_group_crosses_train_test")
            if not train.inner_fold.isin(range(inner_folds)).all():
                structural_errors.append(f"repeat_{repeat}/fold_{outer}:invalid_inner_assignment")
            if not train.groupby("group_id").inner_fold.nunique().eq(1).all():
                structural_errors.append(f"repeat_{repeat}/fold_{outer}:aggregate_group_crosses_inner_folds")
            inner_cells = []
            for inner in range(inner_folds):
                validation = train.loc[train.inner_fold.eq(inner)]
                fitting = train.loc[train.inner_fold.ne(inner)]
                inner_cells.append(
                    {
                        "inner_fold": inner,
                        "fit_rows": len(fitting),
                        "fit_groups": int(fitting.group_id.nunique()),
                        "validation_rows": len(validation),
                        "validation_groups": int(validation.group_id.nunique()),
                    }
                )
                if fitting.empty or validation.empty:
                    reasons.append(f"inner_{inner}:empty_fit_or_validation")
            cells.append(
                {
                    "panel": panel.name,
                    "repeat": repeat,
                    "outer_fold": outer,
                    "menus": counts,
                    "inner_folds": inner_cells,
                    "tied_menu_has_two_rows": counts["tied_test"]["rows"] >= 2,
                    "native_rpl_feasible": native_rpl,
                    "status": "insufficient" if reasons else "sufficient",
                    "reasons": reasons,
                }
            )
    missing = []
    asymmetric_groups = int(panel.rows.loc[~panel.tied, "group_id"].nunique())
    if asymmetric_groups < outer_folds:
        missing.append("asymmetric_aggregate_groups_fewer_than_outer_folds")
    if not repeats:
        missing.append("no_frozen_folds")
    return {
        "panel": panel.name,
        "status": (
            "invalid"
            if structural_errors
            else "insufficient"
            if missing or any(c["reasons"] for c in cells)
            else "sufficient"
        ),
        "count_scope": "before target-specific finite-label masks",
        "geometry": {
            "rows": len(panel.rows),
            "policies": int(panel.rows.policy_id.nunique()),
            "aggregate_groups": group_count,
            "asymmetric_groups": asymmetric_groups,
            "tied_groups": int(panel.rows.loc[panel.tied, "group_id"].nunique()),
            "matched_asymmetric_tied_groups": int(panel.rows.groupby("group_id").tied.nunique().eq(2).sum()),
            "expected_aggregate_groups": expected_group_count,
            "grouping_alpha": float(panel.metadata.get("grouping_alpha", panel.alpha)),
            "rounding_decimals": int(panel.metadata.get("group_rounding_decimals", GROUP_DECIMALS)),
        },
        "algorithm": {
            "groups": "SHA256 prefix of little-endian float64 aggregate coordinates rounded to 10 decimals",
            "outer": "k-means on square roots of asymmetric aggregate-group mean coordinates; n_init=20; lloyd",
            "ties": "assigned to the nearest fitted centroid; scored only in that held neighborhood",
            "inner": "rebuilt from outer-training asymmetric groups only",
            "labels": "clusters renumbered by lexicographically first group hash",
            "seed": "outer=seed+1009*repeat; inner=outer_seed+outer_fold+1",
        },
        "requirements": {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "minimum_asymmetric_test_rows": minimum_asymmetric_test_rows,
            "minimum_asymmetric_test_groups": minimum_asymmetric_test_groups,
        },
        "fold_table_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "native_rpl": {
            "feasible": native_rpl,
            "maximum_constant_exposure_error": epoch_error,
            "reason": (
                "exact_constant_exposures" if native_rpl else "constant_multipliers_do_not_represent_materialization"
            ),
        },
        "structural_errors": sorted(set(structural_errors)),
        "missing_challenges": missing,
        "cells": cells,
    }


def _transfer_training_rows(
    registry: list[dict], candidates: list[int], held_rows: list[int]
) -> tuple[list[int], list[int], dict]:
    held_checkpoints = {registry[index]["checkpoint_id"] for index in held_rows}
    eligible = [index for index in candidates if registry[index]["checkpoint_id"] not in held_checkpoints]
    representatives = {}
    for index in eligible:
        representatives.setdefault(registry[index]["checkpoint_id"], index)
    fitting = list(representatives.values())
    return (
        eligible,
        fitting,
        {
            "candidate_rows": len(candidates),
            "removed_held_checkpoint_aliases": len(candidates) - len(eligible),
            "removed_training_checkpoint_duplicates": len(eligible) - len(fitting),
            "fit_rows": len(fitting),
            "fit_checkpoints": len(representatives),
        },
    )


def support_transfer_plan(panels: list[BenchPanel]) -> dict:
    """Plan fixed-horizon leave-finite-support-out fits without accessing labels.

    Physical checkpoint aliases are excluded before deduplication, including
    asymmetric aliases. Inner support folds use the outer-eligible candidates,
    then independently exclude their validation aliases and deduplicate. Tied
    rows never enter fitting; the shared zero-StarCoder tie is never scored.
    """
    finite_supports = SUPPORTS[1:]
    atomic_panels = [panel for panel in panels if panel.metadata.get("support") in SUPPORTS]
    horizons = sorted({int(panel.metadata["horizon_tokens"]) for panel in atomic_panels})
    registry = []
    by_cell = {}
    full_pool = []
    for panel in sorted(atomic_panels, key=lambda item: item.name):
        horizon = int(panel.metadata["horizon_tokens"])
        support = str(panel.metadata["support"])
        if support == "full":
            full_pool.append(panel.name)
            continue
        key = (horizon, support)
        if key in by_cell:
            raise ValueError(f"Duplicate support/horizon panel: {key}")
        if "checkpoint_id" not in panel.rows:
            raise ValueError(f"{panel.name}: transfer needs canonical physical checkpoint identities")
        starcoder_bucket = panel.buckets.index("starcoder")
        positions = np.argsort(panel.rows.row_id.to_numpy(str))
        row_indices = []
        for position in positions:
            row = panel.rows.iloc[position]
            index = len(registry)
            row_indices.append(index)
            registry.append(
                {
                    "index": index,
                    "panel": panel.name,
                    "row_id": str(row.row_id),
                    "checkpoint_id": str(row.checkpoint_id),
                    "group_id": str(row.group_id),
                    "support": support,
                    "horizon_tokens": horizon,
                    "tied": bool(row.tied),
                    "zero_starcoder": bool(
                        panel.phase0[position, starcoder_bucket] == 0 and panel.phase1[position, starcoder_bucket] == 0
                    ),
                }
            )
        by_cell[key] = row_indices
    cells, missing = [], []
    for horizon in horizons:
        absent = [support for support in finite_supports if (horizon, support) not in by_cell]
        if absent:
            missing.append({"horizon_tokens": horizon, "reason": "missing_finite_support_panels", "supports": absent})
            continue
        for held_support in finite_supports:
            held = by_cell[(horizon, held_support)]
            test = [index for index in held if not registry[index]["tied"]]
            tied = [index for index in held if registry[index]["tied"] and not registry[index]["zero_starcoder"]]
            candidates = [
                index
                for support in finite_supports
                if support != held_support
                for index in by_cell[(horizon, support)]
                if not registry[index]["tied"]
            ]
            eligible, fitting, counts = _transfer_training_rows(registry, candidates, test + tied)
            inner_cells = []
            for validation_support in finite_supports:
                if validation_support == held_support:
                    continue
                validation = [index for index in eligible if registry[index]["support"] == validation_support]
                inner_candidates = [index for index in eligible if registry[index]["support"] != validation_support]
                inner_eligible, inner_fitting, inner_counts = _transfer_training_rows(
                    registry, inner_candidates, validation
                )
                inner_cells.append(
                    {
                        "held_support": validation_support,
                        "train_candidate_rows": inner_eligible,
                        "train_rows": inner_fitting,
                        "validation_rows": validation,
                        "counts": {**inner_counts, "validation_rows": len(validation)},
                        "status": "ready" if inner_fitting and validation else "insufficient",
                    }
                )
            cells.append(
                {
                    "horizon_tokens": horizon,
                    "held_support": held_support,
                    "train_candidate_rows": eligible,
                    "train_rows": fitting,
                    "test_rows": test,
                    "tied_test_rows": tied,
                    "excluded_zero_starcoder_rows": [index for index in held if registry[index]["zero_starcoder"]],
                    "counts": {**counts, "test_rows": len(test), "tied_test_rows": len(tied)},
                    "inner_folds": inner_cells,
                    "status": (
                        "ready"
                        if fitting and test and all(c["status"] == "ready" for c in inner_cells)
                        else "insufficient"
                    ),
                }
            )
    if not atomic_panels:
        missing.append({"reason": "no_atomic_support_panels"})
    payload = {
        "schema": "two_phase_support_transfer_plan_v1",
        "status": "ready" if cells and not missing and all(c["status"] == "ready" for c in cells) else "insufficient",
        "challenge": "fixed-horizon leave-one-finite-support-out",
        "finite_supports": list(finite_supports),
        "required_model_inputs": ["phase weights", "realized phase exposures", "pool size or support budget", "horizon"],
        "full_pool_panels_separately_required": full_pool,
        "training_rule": (
            "asymmetric only; exclude all held physical checkpoint IDs; deduplicate remaining physical outcomes"
        ),
        "deduplication_rule": "first eligible row in fixed support order and lexicographic row identity",
        "inner_rule": (
            "leave-support-out among outer-eligible candidates, followed by validation-alias exclusion and deduplication"
        ),
        "zero_starcoder_rule": "excluded from transfer fitting and scoring; no pinned tied calibration",
        "count_scope": "before target-specific finite-label masks",
        "held_horizon_challenge": {
            "status": "pending",
            "construction": (
                "hold horizon and aggregate neighborhood; train other horizons outside that neighborhood, "
                "then exclude physical aliases"
            ),
        },
        "rows": registry,
        "panel_input_hashes": {panel.name: panel.input_hashes for panel in atomic_panels},
        "cells": cells,
        "missing_challenges": missing,
    }
    payload["plan_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload


if __name__ == "__main__":
    for loaded in load_panels():
        folds = frozen_folds(loaded)
        print(
            json.dumps(
                {
                    "panel": loaded.name,
                    "asymmetric": int((~loaded.tied).sum()),
                    "tied": int(loaded.tied.sum()),
                    "targets": list(loaded.targets),
                    "outer_test_sizes": folds.loc[folds.role.eq("test")].groupby("outer_fold").size().to_dict(),
                }
            )
        )
