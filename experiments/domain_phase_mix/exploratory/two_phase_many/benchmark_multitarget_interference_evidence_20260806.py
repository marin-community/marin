# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Does sharing one latent state across metrics identify it better than fitting each metric alone?

Every BPB metric on a checkpoint is a different readout of the same training run. If a surrogate's
nonlinear part really is a property of training rather than of the benchmark, then metrics that
individually cannot pin it down should still constrain it jointly. That is the whole claim under test
here, and it is a claim about identification, not about having more rows: joint fitting adds correlated
labels on policies that are already in the panel.

Correlated labels can only add information two ways -- the targets load on the shared state through
different directions, or their noise is not perfectly correlated. Both are measurable before anything
is fitted, so `audit` runs that measurement first and the round is willing to stop there.

The comparison is deliberately narrow. Joint and independent fits use the same state, the same head,
the same folds, and the same grids. The only difference is whether one `theta` serves every target.

Subcommands:
    audit     aggregation identities, cross-target information, algebraic properties, synthetic recovery
    wsd80     the 80/20 WSD StarCoder panel across 29 BPB metrics
    panel300m the 520-row 39-bucket high-TPP design across Uncheatable and Table-9 components
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    interference_evidence_model_20260806 as ile,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "multitarget_interference_evidence_20260806"
PROTOCOL_VERSION = "multitarget-ile-v1"

# Fold protocol, copied from the published repaired-RPL reference so the numbers are comparable.
WSD_OUTER_SPLITS = 3
WSD_INNER_SPLITS = 3
WSD_OUTER_SEED = 0
WSD_INNER_SEED_BASE = 31_000
PANEL_OUTER_SPLITS = 3
PANEL_INNER_SPLITS = 3
PANEL_OUTER_SEED = 7310
PANEL_INNER_SEED_BASE = 731_000

# Copied from `audit_wsd80_cross_metric_rpl_20260730` and `benchmark_wsd80_incumbents_20260728` rather
# than imported: those modules pull in a solver dependency chain this harness does not need, and the
# subset reused here is four constants and two fold builders.
BOUNDARY_MARGIN = 0.025
LOWER_TAIL_QUANTILE = 0.20
OPTIMUM_RADIUS = 0.15
OPTIMUM_GRID = 201
PRIMARY_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
POSITIVE_CONTROLS = (
    "eval/uncheatable_eval/github_python-llama3/bpb",
    "eval/uncheatable_eval/github_cpp-llama3/bpb",
)
NEGATIVE_CONTROLS = (
    "eval/paloma/c4_en-llama3/bpb",
    "eval/paloma/falcon-refinedweb-llama3/bpb",
)

# Frozen gates. Every threshold below is quoted in the preregistration and in the round charter.
OBSERVED_WSD_GAIN = 0.009594
TRUE_OPTIMUM = (0.100, 0.500)
WSD_GAIN_ERROR_LIMIT = 0.004439
WSD_OPTIMUM_DISTANCE_LIMIT = 0.05
WSD_NEGATIVE_GAIN_LIMIT = 0.005
RPL_PRIMARY_REGRET = 0.002842
REGRET_SLACK = 0.002
# The published HPR numbers, with the column each was computed on. The Uncheatable reference is the
# token-weighted `eval_uncheatable_eval_bpb`, NOT the flat-mean `eval_uncheatable_eval_macro_bpb`; the
# two differ by up to 0.004774 BPB on the shared rows, which is larger than the 0.002 Regret@1 slack.
# Any candidate compared against these must be scored on the same column.
HPR_REFERENCE = {
    "uncheatable": {
        "column": "eval_uncheatable_eval_bpb",
        "all_rmse": 0.006800,
        "regret_at_1": 0.002678,
        "pair_delta_rmse": 0.007850,
    },
    "table9": {
        "column": "table9_macro_bpb",
        "all_rmse": 0.013001,
        "regret_at_1": 0.003304,
        "pair_delta_rmse": 0.016902,
    },
}
CORE_RMSE_RATIO_LIMIT = 1.05

# Aggregation constants, verified numerically by `audit`.
NUM_TABLE9_COMPONENTS = 51
NUM_MMLU_BUCKETS = 4
DERIVED_MMLU_TARGET = "derived/table9_mmlu_bucket_mean"
UNCHEATABLE_MACRO = "eval_uncheatable_eval_macro_bpb"
TABLE9_MACRO = "table9_macro_bpb"

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260806


# --------------------------------------------------------------------------------------
# Multi-target containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTarget:
    """Several labels observed on one set of policies.

    `observed` is a boolean mask because the 300M panel does not measure every component on every row:
    Uncheatable components exist on the 280 two-phase rows only. `family_share` is the preregistered
    weighting that stops one large block of components from swamping a smaller one.
    """

    names: tuple[str, ...]
    values: np.ndarray
    observed: np.ndarray
    family: tuple[str, ...]
    family_share: np.ndarray

    @property
    def n_targets(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass(frozen=True)
class Selection:
    shape: ile.Shape
    ridge: float
    score: float


def random_folds(weights: np.ndarray, indices: np.ndarray, n_splits: int, seed: int):
    """Plain shuffled K-fold: the protocol the published repaired-RPL reference numbers were computed under."""
    return [
        (indices[train], indices[test])
        for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(indices)
    ]


def mixture_blocked_folds(weights: np.ndarray, indices: np.ndarray, n_splits: int, seed: int):
    """Hold out contiguous regions of mixture space rather than scattered points.

    The surface is densely sampled, so a randomly held-out coordinate almost always keeps a near
    neighbour in training and out-of-fold error measures interpolation rather than prediction.
    """
    coordinates = np.column_stack([weights[indices, 0, :], weights[indices, 1, :]])
    blocks = KMeans(n_clusters=n_splits, n_init=10, random_state=seed).fit_predict(coordinates)
    folds = []
    for block in np.unique(blocks):
        held = indices[blocks == block]
        if len(held) in (0, len(indices)):
            continue
        folds.append((np.setdiff1d(indices, held), held))
    assert len(folds) >= 2, f"mixture blocking produced {len(folds)} usable folds"
    return folds


def wsd80_folds(protocol: str, weights: np.ndarray, indices: np.ndarray, n_splits: int, seed: int):
    builder = random_folds if protocol == "random" else mixture_blocked_folds
    return tuple(
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in builder(weights, indices, n_splits, seed)
    )


def _target_scale(values: np.ndarray, mask: np.ndarray, rows: np.ndarray) -> float:
    """Dispersion used to put every metric on a comparable footing inside the joint objective.

    Restricted to the rows the folds actually cover, so the normalizer never sees the outer held fold.
    Metrics differ in absolute scale by more than an order of magnitude, and without this the joint
    objective would be dominated by whichever metric happens to have the largest spread.
    """
    usable = rows[mask[rows]]
    return float(np.var(values[usable])) if len(usable) > 1 else 1.0


def mask_groups(targets: MultiTarget, rows: np.ndarray) -> dict[bytes, list[int]]:
    """Group targets that are observed on exactly the same rows, so each group shares one factorization."""
    groups: dict[bytes, list[int]] = {}
    for j in range(targets.n_targets):
        key = targets.observed[rows, j].tobytes()
        groups.setdefault(key, []).append(j)
    return groups


def solve_all_heads(
    design: np.ndarray,
    targets: MultiTarget,
    geometry: ile.Geometry,
    ridge: float,
    rows: np.ndarray,
) -> list[ile.Head | None]:
    heads: list[ile.Head | None] = [None] * targets.n_targets
    for columns in mask_groups(targets, rows).values():
        mask = targets.observed[rows, columns[0]]
        if mask.sum() <= design.shape[1]:
            continue
        used = rows[mask]
        solved = ile.solve_heads_batch(design[used], targets.values[np.ix_(used, columns)], geometry, ridge)
        for j, head in zip(columns, solved, strict=True):
            heads[j] = head
    return heads


def fold_scores(
    designs: dict[ile.Shape, np.ndarray],
    targets: MultiTarget,
    geometry: ile.Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    ridge_grid: tuple[float, ...],
) -> np.ndarray:
    """Normalized out-of-fold MSE for every (shape, ridge, target) combination.

    Shape is the outer axis because the design matrix depends only on the shape, so it is built once and
    reused across every ridge and every target.
    """
    shapes = list(designs)
    covered = np.unique(np.concatenate([np.concatenate(fold) for fold in folds]))
    # Row masks depend only on the target and the fold, so resolve the grouping once instead of once per
    # (shape, ridge). This loop runs tens of thousands of times and Python overhead dominates it.
    fold_groups = [
        (
            train,
            test,
            [(columns, targets.observed[train, columns[0]]) for columns in mask_groups(targets, train).values()],
        )
        for train, test in folds
    ]
    scores = np.full((len(shapes), len(ridge_grid), targets.n_targets), np.nan)
    for si, shape in enumerate(shapes):
        design = designs[shape]
        for ri, ridge in enumerate(ridge_grid):
            squared = np.zeros(targets.n_targets)
            counts = np.zeros(targets.n_targets)
            for train, test, groups in fold_groups:
                for columns, mask in groups:
                    if mask.sum() <= design.shape[1]:
                        continue
                    used = train[mask]
                    coefficients = ile.solve_coefficients_batch(
                        design[used], targets.values[np.ix_(used, columns)], geometry, ridge
                    )
                    held = test[targets.observed[test, columns[0]]]
                    if not len(held):
                        continue
                    residual = design[held] @ coefficients - targets.values[np.ix_(held, columns)]
                    squared[columns] += np.einsum("ij,ij->j", residual, residual)
                    counts[columns] += len(held)
            for j in range(targets.n_targets):
                if counts[j] == 0:
                    continue
                scale = _target_scale(targets.values[:, j], targets.observed[:, j], covered)
                scores[si, ri, j] = (squared[j] / counts[j]) / max(scale, 1e-18)
    return scores


def select_joint(
    scores: np.ndarray,
    shapes: list[ile.Shape],
    ridge_grid: tuple[float, ...],
    targets: MultiTarget,
) -> tuple[ile.Shape, list[float]]:
    """One shape for every target; each target keeps its own ridge.

    Profiling ridge out per target before aggregating means the shared quantity really is only the
    nonlinear transition, which is what the joint-versus-independent claim is about.
    """
    per_shape = np.full(len(shapes), np.inf)
    for si in range(len(shapes)):
        best_by_target = np.nanmin(scores[si], axis=0)
        if np.all(np.isnan(best_by_target)):
            continue
        finite = np.isfinite(best_by_target)
        per_shape[si] = float(np.sum(targets.family_share[finite] * best_by_target[finite]))
    chosen = int(np.argmin(per_shape))
    ridges = [float(ridge_grid[int(np.nanargmin(scores[chosen, :, j]))]) for j in range(targets.n_targets)]
    return shapes[chosen], ridges


def select_independent(
    scores: np.ndarray,
    shapes: list[ile.Shape],
    ridge_grid: tuple[float, ...],
    targets: MultiTarget,
) -> list[Selection]:
    out = []
    for j in range(targets.n_targets):
        flat = scores[:, :, j]
        if np.all(np.isnan(flat)):
            out.append(Selection(shapes[0], ridge_grid[0], float("nan")))
            continue
        si, ri = np.unravel_index(int(np.nanargmin(flat)), flat.shape)
        out.append(Selection(shapes[si], float(ridge_grid[ri]), float(flat[si, ri])))
    return out


def build_designs(
    weights: np.ndarray,
    geometry: ile.Geometry,
    shapes: tuple[ile.Shape, ...],
) -> dict[ile.Shape, np.ndarray]:
    return {shape: ile.design_matrix(weights, geometry, shape) for shape in shapes}


def choose(
    scores: np.ndarray,
    shape_list: list[ile.Shape],
    ridge_grid: tuple[float, ...],
    targets: MultiTarget,
    mode: str,
) -> list[Selection]:
    if mode == "joint":
        shape, ridges = select_joint(scores, shape_list, ridge_grid, targets)
        return [Selection(shape, ridges[j], float("nan")) for j in range(targets.n_targets)]
    return select_independent(scores, shape_list, ridge_grid, targets)


def fit_on(
    designs: dict[ile.Shape, np.ndarray],
    targets: MultiTarget,
    geometry: ile.Geometry,
    chosen: list[Selection],
    rows: np.ndarray,
) -> list[ile.Head | None]:
    """Fit each target's head at its own selected shape and ridge, batching targets that agree on both."""
    heads: list[ile.Head | None] = [None] * targets.n_targets
    buckets: dict[tuple[ile.Shape, float], list[int]] = {}
    for j, selection in enumerate(chosen):
        buckets.setdefault((selection.shape, selection.ridge), []).append(j)
    for (shape, ridge), columns in buckets.items():
        design = designs[shape]
        for group in mask_groups(targets, rows).values():
            overlap = [j for j in group if j in columns]
            if not overlap:
                continue
            mask = targets.observed[rows, overlap[0]]
            if mask.sum() <= design.shape[1]:
                continue
            used = rows[mask]
            solved = ile.solve_heads_batch(design[used], targets.values[np.ix_(used, overlap)], geometry, ridge)
            for j, head in zip(overlap, solved, strict=True):
                heads[j] = head
    return heads


def nested_predictions(
    weights: np.ndarray,
    geometry: ile.Geometry,
    targets: MultiTarget,
    outer: tuple[tuple[np.ndarray, np.ndarray], ...],
    inner_folds_for,
    shapes: tuple[ile.Shape, ...],
    ridge_grid: tuple[float, ...],
    modes: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Nested out-of-fold predictions for several fitting modes at once.

    Joint and independent selection read the same inner-fold score tensor, so scoring runs once per
    outer fold and both modes are served from it. Every nonlinear quantity is still selected strictly
    inside the training fold.
    """
    designs = build_designs(weights, geometry, shapes)
    shape_list = list(designs)
    predictions = {mode: np.full(targets.values.shape, np.nan) for mode in modes}
    trace: list[dict[str, Any]] = []

    for fold_id, (train, test) in enumerate(outer):
        scores = fold_scores(designs, targets, geometry, inner_folds_for(fold_id, train), ridge_grid)
        for mode in modes:
            chosen = choose(scores, shape_list, ridge_grid, targets, mode)
            heads = fit_on(designs, targets, geometry, chosen, train)
            for j, head in enumerate(heads):
                if head is None:
                    continue
                rows = test[targets.observed[test, j]]
                predictions[mode][rows, j] = designs[chosen[j].shape][rows] @ ile.coefficient_vector(head)
                trace.append(
                    {
                        "mode": mode,
                        "fold": fold_id,
                        "target": targets.names[j],
                        "rho": chosen[j].shape.rho,
                        "interference": chosen[j].shape.interference,
                        "curvature": chosen[j].shape.curvature,
                        "ridge": chosen[j].ridge,
                    }
                )
    return predictions, trace


def full_fit(
    weights: np.ndarray,
    geometry: ile.Geometry,
    targets: MultiTarget,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    shapes: tuple[ile.Shape, ...],
    ridge_grid: tuple[float, ...],
    modes: tuple[str, ...],
) -> tuple[dict[str, list[ile.Model | None]], list[dict[str, Any]]]:
    designs = build_designs(weights, geometry, shapes)
    shape_list = list(designs)
    scores = fold_scores(designs, targets, geometry, folds, ridge_grid)
    all_rows = np.arange(len(weights))

    models: dict[str, list[ile.Model | None]] = {}
    trace = []
    for mode in modes:
        chosen = choose(scores, shape_list, ridge_grid, targets, mode)
        heads = fit_on(designs, targets, geometry, chosen, all_rows)
        models[mode] = [
            (
                None
                if head is None
                else ile.Model(shape=chosen[j].shape, geometry=geometry, head=head, ridge=chosen[j].ridge)
            )
            for j, head in enumerate(heads)
        ]
        trace.extend(
            {
                "mode": mode,
                "fold": "full",
                "target": targets.names[j],
                "rho": chosen[j].shape.rho,
                "interference": chosen[j].shape.interference,
                "curvature": chosen[j].shape.curvature,
                "ridge": chosen[j].ridge,
            }
            for j in range(targets.n_targets)
            if models[mode][j] is not None
        )
    return models, trace


# --------------------------------------------------------------------------------------
# Shared reporting helpers
# --------------------------------------------------------------------------------------


def write_json(path: Path, value: Any) -> None:
    def ready(item):
        if isinstance(item, (np.floating, np.integer)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, dict):
            return {str(k): ready(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [ready(v) for v in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(value), indent=2, sort_keys=True) + "\n")


def paired_bootstrap_difference(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    clusters: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Cluster bootstrap of RMSE(a) - RMSE(b) on the rows where both are defined."""
    valid = np.isfinite(residual_a) & np.isfinite(residual_b)
    residual_a, residual_b, clusters = residual_a[valid], residual_b[valid], clusters[valid]
    unique = np.unique(clusters)
    lookup = {key: np.flatnonzero(clusters == key) for key in unique}
    rng = np.random.default_rng(seed)
    point = float(np.sqrt(np.mean(residual_a**2)) - np.sqrt(np.mean(residual_b**2)))
    samples = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([lookup[key] for key in picked])
        samples[draw] = np.sqrt(np.mean(residual_a[rows] ** 2)) - np.sqrt(np.mean(residual_b[rows] ** 2))
    return {
        "difference": point,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        # Capped at one: when every resample gives an exact tie both inclusive tails are one and the
        # doubled value would be two, which is not a p-value.
        "p_two_sided": float(min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))),
    }


def protocol_hash(extra: dict[str, Any]) -> str:
    payload = {
        "version": PROTOCOL_VERSION,
        "rho_grid": list(ile.RHO_GRID),
        "mu_grid": list(ile.MU_GRID),
        "ridge_grid": list(ile.HEAD_RIDGE_GRID),
        "model_source": hashlib.sha256(Path(ile.__file__).read_bytes()).hexdigest(),
        "harness_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **extra,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "wsd80", "panel300m"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid", type=int, default=OPTIMUM_GRID)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The stage modules import this one for its shared helpers, so importing them at module scope would
    # be a cycle. The structural fix is to split those helpers into their own module; until then these
    # three imports are deferred, which is the exception the repository style guide allows.
    if args.command == "audit":
        from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
            multitarget_ile_audits_20260806 as audits,
        )

        audits.run(args.output_dir)
        return
    if args.command == "wsd80":
        from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
            multitarget_ile_wsd80_20260806 as wsd_run,
        )

        wsd_run.run(args.output_dir, grid=args.grid, draws=args.draws)
        return
    from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
        multitarget_ile_panel300m_20260806 as panel_run,
    )

    panel_run.run(args.output_dir, draws=args.draws)


if __name__ == "__main__":
    main()
