# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Exact single-phase restrictions of the Mixture Fit Observatory models.

Every Observatory model is written here as a function of the single-phase policy alone: the
materialized exposure ``E_b = c_b w_b`` and, where the original model uses it, the mixture weight
``w_b``. Phase-only columns that vanish or duplicate at a tied policy are removed, phase-only
nonlinear dimensions that no longer move predictions are removed, and every remaining shape,
ridge, link, and ensemble choice is refit inside the training rows the caller supplies.

The module has three layers. Heads and links are shared by every model. Design builders return
the linear feature block of one model at one nonlinear shape. Model classes combine a design
builder with a shape search (grid, continuous in-sample, or profiled inner-CV with an implicit
gradient through the nonnegative head) and expose one ``fit``/``predict`` contract.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.stats import qmc

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear

Shape = Mapping[str, float]
InnerFolds = tuple[tuple[np.ndarray, np.ndarray], ...]

EPSILON = 1e-12
LINK_CLIP = 30.0
NNLS_MAXITER_FACTOR = 200
DEFICIT_FLOOR_FRACTION = 0.95
HUBER_SCALE = 2.5
HUBER_ITERATIONS = 50
HUBER_TOLERANCE = 1e-3
MAD_TO_SIGMA = 1.4826
PAIR_SHRINKAGE = 1.0
# Canonical single-phase DSP search box and solver settings, identical to the profiled ladder.
DSP_LOG_RATE_BOUND = (float(np.log(1e-4)), float(np.log(2.0)))
DSP_THRESHOLD_BOUND = (-2.0, 8.0)
DSP_LOG_EXPONENT_BOUND = (float(np.log(0.2)), float(np.log(10.0)))
DSP_DAMAGE_KNEE = 105.0
DSP_LINEAR_REG = 1e-6
DSP_MAXITER = 36
DSP_RESTARTS = 2
DSP_SEED_BASE = 20_260_824
DSP_ACTIVE_TOL = 1e-10
# Compact retained state's one-phase search box and starts.
COMPACT_LOG_RATE_BOUNDS = (math.log(0.05), math.log(20.0))
COMPACT_POWER_BOUNDS = (0.2, 1.0)
COMPACT_MAXITER = 24
COMPACT_TOP_K = 2
COMPACT_RIDGE_GRID = (0.1, 1.0)
# Separate heads (one-phase asymmetric log bowl).
BOWL_MU_SHIFTS = tuple(float(value) for value in np.linspace(-2.0, 2.0, 9))
BOWL_MU_BOUND = (-2.0, 8.0)
BOWL_RIDGE_GRID = (0.03, 0.1, 0.3, 1.0, 1.5, 3.0)
# Family onset thresholds.
TAU_BOUNDS = (0.0, 7.0)
TAU_MAXITER = 60
TAU_SHRINK_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
# Hierarchical band.
BAND_RELATIVE_WIDTH = 0.15
BAND_MAX_MEMBERS = 24
# Retained power law.
RPL_TOP_SHAPES = 12
RPL_COLUMN_SCALE_FLOOR = 1e-12
# Positive log-link floor used by the taskwise linear epoch reference.
LOG_LINK_FLOOR_SPAN_FRACTION = 0.05
QUALITY_SUFFIXES = ("_high", "_low")
# scrambled_harm permutes the harm block's bucket columns (a no-op for one-column-per-bucket harms);
# row_scrambled_harm permutes its mixture rows, which is information-free for every harm form.
HARM_SCRAMBLE_SEED = 20_260_903
REFINE_EVALUATIONS = 80
LOG_SPACE_SHAPE_KEYS = frozenset({"rate", "saturation_epochs", "benefit_offset"})
QUALITY_SCRAMBLE_SEED = 20_260_904
# Bounded log-deficit link: the linear predictor is capped at the largest training log-deficit plus this margin.
LINK_CAP_MARGIN = 0.5
CC_PREFIX = "dolma3_cc/"


def softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(value, 0.0)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -value))


def _hash_arrays(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode())
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Families and features
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Families:
    """Declared domain/quality structure over buckets; a covering partition into families."""

    names: tuple[str, ...]
    members: tuple[np.ndarray, ...]
    quality: np.ndarray
    quality_ordered: bool

    def __post_init__(self) -> None:
        covered = np.sort(np.concatenate(self.members)) if self.members else np.empty(0, dtype=int)
        if not np.array_equal(covered, np.arange(len(self.quality))):
            raise ValueError("families must partition the buckets exactly once")

    @property
    def bucket_count(self) -> int:
        return len(self.quality)

    @property
    def nonsingleton(self) -> tuple[int, ...]:
        return tuple(index for index, members in enumerate(self.members) if len(members) > 1)

    @property
    def singleton_buckets(self) -> np.ndarray:
        return np.asarray([int(members[0]) for members in self.members if len(members) == 1], dtype=int)

    @property
    def multi_member_buckets(self) -> np.ndarray:
        blocks = [self.members[index] for index in self.nonsingleton]
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=int)

    @property
    def index(self) -> np.ndarray:
        result = np.empty(self.bucket_count, dtype=int)
        for family, members in enumerate(self.members):
            result[members] = family
        return result

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """(high, low) bucket pairs for two-member families with a declared quality order."""
        if not self.quality_ordered:
            return ()
        pairs = []
        for members in self.members:
            if len(members) != 2:
                continue
            first, second = (int(members[0]), int(members[1]))
            if self.quality[first] > self.quality[second]:
                pairs.append((first, second))
            else:
                pairs.append((second, first))
        return tuple(pairs)

    def totals(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack([values[:, members].sum(axis=1) for members in self.members])

    def means(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack([values[:, members].mean(axis=1) for members in self.members])

    def description(self) -> str:
        sizes = sorted(len(members) for members in self.members)
        return f"{len(self.members)} families, sizes {min(sizes)}-{max(sizes)}, {len(self.nonsingleton)} pooled"


def families_from_buckets(buckets: Sequence[str]) -> Families:
    """Families from declared naming only: CC high/low pairs, or manifest cluster/quality bins."""
    grouped: dict[str, list[int]] = {}
    quality = np.full(len(buckets), -1, dtype=int)
    ordered = False
    for index, bucket in enumerate(buckets):
        family = bucket
        if bucket.startswith(CC_PREFIX) and bucket.endswith(QUALITY_SUFFIXES):
            for rank, suffix in enumerate(reversed(QUALITY_SUFFIXES)):
                if bucket.endswith(suffix):
                    family = bucket[: -len(suffix)]
                    quality[index] = rank
                    ordered = True
        elif "_q" in bucket and bucket.startswith("c") and bucket.split("_q")[-1].isdigit():
            family, quality_text = bucket.rsplit("_q", 1)
            quality[index] = int(quality_text)
        grouped.setdefault(family, []).append(index)
    names = tuple(grouped)
    members = tuple(np.asarray(grouped[name], dtype=int) for name in names)
    return Families(names=names, members=members, quality=quality, quality_ordered=ordered)


def no_families(bucket_count: int) -> Families:
    return Families(
        names=tuple(f"bucket_{index}" for index in range(bucket_count)),
        members=tuple(np.asarray([index], dtype=int) for index in range(bucket_count)),
        quality=np.full(bucket_count, -1, dtype=int),
        quality_ordered=False,
    )


def shuffled_families(families: Families, seed: int) -> Families:
    """Matched-capacity control: permute the bucket-to-family assignment, keeping every family size."""
    permutation = np.random.default_rng(seed).permutation(families.bucket_count)
    members = tuple(np.sort(permutation[block]) for block in families.members)
    quality = np.full(families.bucket_count, -1, dtype=int)
    quality[permutation] = families.quality
    return Families(
        names=tuple(f"shuffled_{name}" for name in families.names),
        members=members,
        quality=quality,
        quality_ordered=families.quality_ordered,
    )


@dataclasses.dataclass(frozen=True)
class Features:
    """Single-phase policy inputs for one panel: exposures, weights, inventory, families."""

    exposures: np.ndarray
    weights: np.ndarray
    inventory: np.ndarray
    early_fraction: np.ndarray
    families: Families
    label: str
    buckets_names: tuple[str, ...] = ()
    component: str = ""

    def __post_init__(self) -> None:
        if self.exposures.shape != self.weights.shape or self.exposures.ndim != 2:
            raise ValueError("exposures and weights must be matrices of the same shape")
        if self.families.bucket_count != self.exposures.shape[1]:
            raise ValueError("family partition does not match the bucket count")

    @property
    def rows(self) -> int:
        return self.exposures.shape[0]

    @property
    def buckets(self) -> int:
        return self.exposures.shape[1]

    @functools.cached_property
    def cache_key(self) -> str:
        family_text = "|".join(
            f"{name}:{','.join(map(str, members))}"
            for name, members in zip(self.families.names, self.families.members, strict=True)
        )
        return _hash_arrays(self.exposures, self.weights, self.early_fraction, extra=f"{self.label}|{family_text}")

    def with_permuted_inventory(self, seed: int) -> Features:
        permutation = np.random.default_rng(seed).permutation(self.buckets)
        inventory = self.inventory[permutation]
        return dataclasses.replace(
            self,
            exposures=self.weights * inventory[None, :],
            inventory=inventory,
            label=f"{self.label}|permuted_inventory:{seed}",
        )

    def with_weight_coordinate(self) -> Features:
        return dataclasses.replace(
            self,
            exposures=self.weights.copy(),
            inventory=np.ones(self.buckets),
            label=f"{self.label}|weight_coordinate",
        )

    def with_families(self, families: Families, tag: str) -> Features:
        return dataclasses.replace(self, families=families, label=f"{self.label}|{tag}")


def features_from_panel(
    weights: np.ndarray,
    inventory: np.ndarray,
    buckets: Sequence[str],
    *,
    early_fraction: np.ndarray | None,
    label: str,
) -> Features:
    weights = np.asarray(weights, dtype=float)
    inventory = np.asarray(inventory, dtype=float)
    fraction = np.ones(len(buckets)) if early_fraction is None else np.asarray(early_fraction, dtype=float)
    return Features(
        exposures=weights * inventory[None, :],
        weights=weights,
        inventory=inventory,
        early_fraction=fraction,
        families=families_from_buckets(buckets),
        buckets_names=tuple(str(bucket) for bucket in buckets),
        label=label,
    )


# ---------------------------------------------------------------------------------------------
# Heads and links
# ---------------------------------------------------------------------------------------------


class HeadKind(StrEnum):
    NNLS = "nnls"
    LSTSQ = "lstsq"
    RIDGE = "ridge"
    HUBER_NNLS = "huber_nnls"


class LinkKind(StrEnum):
    IDENTITY = "identity"
    LOG_DEFICIT = "log_deficit"
    LOG_DEFICIT_BOUNDED = "log_deficit_bounded"
    LOG_FLOOR_MARGIN = "log_floor_margin"


@dataclasses.dataclass(frozen=True)
class HeadSpec:
    kind: HeadKind = HeadKind.NNLS
    scale_columns: bool = False
    link: LinkKind = LinkKind.IDENTITY
    floor_fraction: float = DEFICIT_FLOOR_FRACTION
    floor_margin: float = 0.0
    cap_margin: float = LINK_CAP_MARGIN
    huber_scale: float = HUBER_SCALE
    tie_pairs: tuple[tuple[int, int], ...] = ()
    # Solve the nonnegative least squares on the QR-reduced system when the row count exceeds the
    # column count. The minimizer is unchanged; the profiled DSP solver keeps the direct form so
    # its active-set gradient matches the reference ladder exactly.
    reduced_nnls: bool = True
    # Column scale for ``scale_columns``: root-mean-square (compact retained state and its
    # descendants) or maximum absolute value (the retained power law's own head).
    scale_rule: str = "rms"


@dataclasses.dataclass(frozen=True)
class FittedHead:
    intercept: float
    coefficients: np.ndarray
    floor: float
    active: int
    cap: float = float("inf")


@dataclasses.dataclass(frozen=True)
class Design:
    values: np.ndarray
    ridge: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.shape[1] != len(self.ridge) or len(self.names) != len(self.ridge):
            raise ValueError("design columns, ridge multipliers, and names must agree")


def link_floor(response: np.ndarray, spec: HeadSpec) -> float:
    if spec.link is LinkKind.IDENTITY:
        return float("nan")
    if spec.link in (LinkKind.LOG_DEFICIT, LinkKind.LOG_DEFICIT_BOUNDED):
        return spec.floor_fraction * float(np.min(response))
    low = float(np.min(response))
    span = float(np.max(response) - low)
    scale = max(span, LOG_LINK_FLOOR_SPAN_FRACTION * max(low, 1e-6))
    return low - spec.floor_margin * scale


def link_forward(response: np.ndarray, floor: float, spec: HeadSpec) -> np.ndarray:
    if spec.link is LinkKind.IDENTITY:
        return response
    return np.log(np.maximum(response - floor, 1e-9))


def link_inverse(linear: np.ndarray, floor: float, spec: HeadSpec, cap: float = float("inf")) -> np.ndarray:
    if spec.link is LinkKind.IDENTITY:
        return linear
    return floor + np.exp(np.clip(linear, -LINK_CLIP, min(LINK_CLIP, cap)))


def _tie_rows(width: int, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = np.zeros((len(pairs), width))
    for row, (first, second) in zip(rows, pairs, strict=True):
        row[first] = math.sqrt(PAIR_SHRINKAGE)
        row[second] = -math.sqrt(PAIR_SHRINKAGE)
    return rows


def _nonnegative_solve(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    multipliers: np.ndarray,
    spec: HeadSpec,
    row_weights: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Centered NNLS with a free intercept, optional column scaling, ridge rows, ties, and row weights."""
    width = design.shape[1]
    scale = np.ones(width)
    if spec.scale_columns and spec.scale_rule == "max_abs":
        scale = np.maximum(np.abs(design).max(axis=0), RPL_COLUMN_SCALE_FLOOR)
    elif spec.scale_columns:
        scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-8)
    scaled = design / scale[None, :]
    weights = np.ones(len(target)) if row_weights is None else row_weights
    total = float(weights.sum())
    design_mean = (weights[:, None] * scaled).sum(axis=0) / total
    target_mean = float((weights * target).sum() / total)
    root = np.sqrt(weights)[:, None]
    rows = root * (scaled - design_mean[None, :])
    rhs = np.sqrt(weights) * (target - target_mean)
    if ridge > 0.0:
        rows = np.vstack([rows, np.diag(np.sqrt(ridge * multipliers))])
        rhs = np.concatenate([rhs, np.zeros(width)])
    if spec.tie_pairs:
        rows = np.vstack([rows, _tie_rows(width, spec.tie_pairs)])
        rhs = np.concatenate([rhs, np.zeros(len(spec.tie_pairs))])
    if spec.reduced_nnls and rows.shape[0] > 2 * width:
        orthogonal, triangular = np.linalg.qr(rows, mode="reduced")
        rows, rhs = triangular, orthogonal.T @ rhs
    coefficients, _residual = nnls(rows, rhs, maxiter=NNLS_MAXITER_FACTOR * width)
    coefficients = coefficients / scale
    intercept = target_mean - float(design_mean @ (coefficients * scale))
    return intercept, coefficients


def fit_head(design: Design, response: np.ndarray, ridge: float, spec: HeadSpec) -> FittedHead:
    """Fit the linear head of one model on the supplied rows."""
    floor = link_floor(response, spec)
    target = link_forward(response, floor, spec)
    matrix = design.values
    if spec.kind is HeadKind.LSTSQ:
        center = matrix.mean(axis=0)
        target_mean = float(target.mean())
        coefficients, *_ = np.linalg.lstsq(matrix - center, target - target_mean, rcond=None)
        intercept = target_mean - float(center @ coefficients)
    elif spec.kind is HeadKind.RIDGE:
        center = matrix.mean(axis=0)
        centered = matrix - center
        scale = np.sqrt(np.mean(centered**2, axis=0))
        scale[scale < 1e-10] = 1.0
        normalized = centered / scale
        gram = normalized.T @ normalized + ridge * np.diag(design.ridge)
        target_mean = float(target.mean())
        coefficients = np.linalg.solve(gram, normalized.T @ (target - target_mean)) / scale
        intercept = target_mean - float(center @ coefficients)
    elif spec.kind is HeadKind.NNLS:
        intercept, coefficients = _nonnegative_solve(matrix, target, ridge, design.ridge, spec, None)
    elif spec.kind is HeadKind.HUBER_NNLS:
        intercept, coefficients = _nonnegative_solve(matrix, target, ridge, design.ridge, spec, None)
        for _ in range(HUBER_ITERATIONS):
            residual = intercept + matrix @ coefficients - target
            spread = MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
            if spread <= 0.0:
                break
            cut = spec.huber_scale * spread
            row_weights = np.minimum(1.0, cut / np.maximum(np.abs(residual), 1e-12))
            updated = _nonnegative_solve(matrix, target, ridge, design.ridge, spec, row_weights)
            shift = float(np.max(np.abs(matrix @ (updated[1] - coefficients) + (updated[0] - intercept))))
            intercept, coefficients = updated
            if shift < HUBER_TOLERANCE * spread:
                break
    else:
        raise ValueError(f"unsupported head {spec.kind}")
    active = int(
        np.count_nonzero(
            np.abs(coefficients) > DSP_ACTIVE_TOL * max(1.0, float(np.max(np.abs(coefficients), initial=0.0)))
        )
    )
    cap = float(np.max(target)) + spec.cap_margin if spec.link is LinkKind.LOG_DEFICIT_BOUNDED else float("inf")
    return FittedHead(
        intercept=float(intercept),
        coefficients=np.asarray(coefficients, dtype=float),
        floor=floor,
        active=active,
        cap=cap,
    )


def predict_head(head: FittedHead, matrix: np.ndarray, spec: HeadSpec) -> np.ndarray:
    return link_inverse(head.intercept + matrix @ head.coefficients, head.floor, spec, head.cap)


def effective_rank(matrix: np.ndarray) -> int:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.size == 0:
        return 0
    return int(np.linalg.matrix_rank(centered, tol=1e-8 * max(1.0, float(np.abs(centered).max()))))


# ---------------------------------------------------------------------------------------------
# Response primitives
# ---------------------------------------------------------------------------------------------


def power_response(exposure: np.ndarray, exponent: float) -> np.ndarray:
    return np.maximum(exposure, EPSILON) ** exponent


def weibull_response(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def saturation_response(exposure: np.ndarray, rate: np.ndarray | float) -> np.ndarray:
    return 1.0 - np.exp(-rate * exposure)


def softplus_harm(exposure: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    return softplus(np.log1p(np.maximum(exposure, 0.0)) - threshold) ** 2


def bounded_harm(exposure: np.ndarray, log_exponent: np.ndarray | float) -> np.ndarray:
    unit = np.maximum(exposure - 1.0, 0.0) / DSP_DAMAGE_KNEE
    powered = unit ** np.exp(log_exponent)
    return powered / (1.0 + powered)


def literal_replay(exposure: np.ndarray) -> np.ndarray:
    return np.maximum(exposure - 1.0, 0.0) ** 2


def retained_state(features: Features, late_multiplier: float, forgetting_rate: float) -> np.ndarray:
    """Tied image of the revisit-gated retained state: ``(a0 e^{-f(1-w)} + L a1) E``."""
    early = features.early_fraction[None, :]
    gate = np.exp(-forgetting_rate * (1.0 - np.clip(features.weights, 0.0, 1.0)))
    return np.maximum((gate * early + late_multiplier * (1.0 - early)) * features.exposures, 0.0)


def concentration(features: Features) -> np.ndarray:
    return np.sum(features.weights**2, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------
# Design builders for grid-searched models
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FamilyOptions:
    """Which family mechanisms a GRP-style design carries."""

    bucket_signal: bool = True
    family_signal: str = "none"  # none | sum | mean | pair_discount
    hierarchical: bool = False
    # Harm kinds: none, softplus_family, softplus_bucket, softplus_group_sum, literal_shared,
    # literal_family, overload_family, bounded_bucket.
    harm: str = "softplus_family"
    member_replay: bool = False
    benefit: str = "power"  # power | weibull | saturation | log1p | affine
    retention_gate: bool = False
    family_rate_divisor: bool = False
    families_in_benefit: str = "nonsingleton"  # nonsingleton | all
    # Capacity-matched control: the harm block reads exposures through a fixed bucket permutation,
    # keeping column count, ridge, and threshold search identical while removing bucket alignment.
    scrambled_harm: bool = False
    row_scrambled_harm: bool = False
    # Round-2 mechanisms: additive interaction columns, quality-axis pooling across families, and a
    # per-(component, bucket) ridge multiplier table keyed by bucket name.
    interaction: str = "none"
    quality_axis: str = "none"
    shuffled_quality: bool = False
    component_ridge: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()


def _benefit(values: np.ndarray, shape: Shape, options: FamilyOptions) -> np.ndarray:
    if options.benefit == "power":
        return power_response(values, float(shape["exponent"]))
    if options.benefit == "weibull":
        return weibull_response(values, float(shape["rate"]), float(shape["power"]))
    if options.benefit == "saturation":
        return saturation_response(values, float(shape["rate"]))
    if options.benefit == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if options.benefit == "affine":
        return np.maximum(values, 0.0)
    raise ValueError(f"unknown benefit {options.benefit}")


def family_design(features: Features, shape: Shape, options: FamilyOptions) -> Design:
    """GRP-style single-phase design: bucket benefit, family benefit, and repetition harm."""
    families = features.families
    exposure = features.exposures
    signal_input = (
        retained_state(features, float(shape.get("late_multiplier", 1.0)), float(shape.get("forgetting_rate", 0.0)))
        if options.retention_gate
        else exposure
    )
    pieces: list[np.ndarray] = []
    ridge: list[float] = []
    names: list[str] = []
    bucket_signal = _benefit(signal_input, shape, options)
    if options.scrambled_harm:
        exposure = exposure[:, np.random.default_rng(HARM_SCRAMBLE_SEED).permutation(features.buckets)]
    if options.row_scrambled_harm:
        exposure = exposure[np.random.default_rng(HARM_SCRAMBLE_SEED).permutation(features.rows)]
    family_total = families.totals(exposure)
    signal_total = families.totals(signal_input)
    pooled = families.nonsingleton

    if options.bucket_signal and options.hierarchical:
        singleton = families.singleton_buckets
        if len(singleton):
            pieces.append(-bucket_signal[:, singleton])
            ridge.extend([1.0] * len(singleton))
            names.extend(f"singleton_signal:{index}" for index in singleton)
        for family in pooled:
            members = families.members[family]
            pieces.append(-bucket_signal[:, members].sum(axis=1, keepdims=True))
            ridge.append(1.0)
            names.append(f"pooled_base_signal:{families.names[family]}")
        excess = families.multi_member_buckets
        if len(excess):
            pieces.append(-bucket_signal[:, excess])
            ridge.extend([float(shape.get("residual_shrink", 1.0))] * len(excess))
            names.extend(f"bucket_excess_signal:{index}" for index in excess)
    elif options.bucket_signal:
        pieces.append(-bucket_signal)
        ridge.extend([1.0] * features.buckets)
        names.extend(f"bucket_signal:{index}" for index in range(features.buckets))

    family_indices = tuple(range(len(families.members))) if options.families_in_benefit == "all" else pooled
    if options.family_signal == "sum" and family_indices:
        family_shape = dict(shape)
        if options.family_rate_divisor and "rate" in family_shape:
            family_shape["rate"] = float(shape["rate"]) / max(len(families.members), 1)
        pieces.append(-_benefit(signal_total[:, list(family_indices)], family_shape, options))
        ridge.extend([1.0] * len(family_indices))
        names.extend(f"family_signal:{families.names[index]}" for index in family_indices)
    elif options.family_signal == "mean" and family_indices:
        pieces.append(-_benefit(families.means(signal_input)[:, list(family_indices)], shape, options))
        ridge.extend([1.0] * len(family_indices))
        names.extend(f"family_signal:{families.names[index]}" for index in family_indices)
    elif options.family_signal == "pair_discount":
        discount = float(shape.get("quality_discount", 1.0))
        pairs = families.pairs
        paired = {index for pair in pairs for index in pair}
        for index in range(features.buckets):
            if index in paired:
                continue
            pieces.append(-_benefit(signal_input[:, [index]], shape, options))
            ridge.append(1.0)
            names.append(f"singleton_signal:{index}")
        for high, low in pairs:
            combined = signal_input[:, [high]] + discount * signal_input[:, [low]]
            pieces.append(-_benefit(combined, shape, options))
            ridge.append(1.0)
            names.append(f"pair_signal:{high}+{low}")

    if options.harm == "softplus_family":
        pieces.append(softplus_harm(family_total, float(shape["threshold"])))
        ridge.extend([1.0] * len(families.members))
        names.extend(f"family_overexposure:{name}" for name in families.names)
    elif options.harm == "softplus_bucket":
        pieces.append(softplus_harm(exposure, float(shape["threshold"])))
        ridge.extend([1.0] * features.buckets)
        names.extend(f"bucket_overexposure:{index}" for index in range(features.buckets))
    elif options.harm == "softplus_group_sum":
        groups = families.pairs
        paired = {index for pair in groups for index in pair}
        totals = [exposure[:, [index]] for index in range(features.buckets) if index not in paired]
        totals.extend(exposure[:, [high]] + exposure[:, [low]] for high, low in groups)
        stacked = np.hstack(totals)
        pieces.append(softplus_harm(stacked, float(shape["threshold"])).sum(axis=1, keepdims=True))
        ridge.append(1.0)
        names.append("group_overexposure_sum")
    elif options.harm == "literal_shared":
        pieces.append(literal_replay(exposure).sum(axis=1, keepdims=True))
        ridge.append(1.0)
        names.append("shared_literal_replay")
    elif options.harm == "literal_family":
        pieces.append(families.totals(literal_replay(exposure)))
        ridge.extend([1.0] * len(families.members))
        names.extend(f"family_literal_replay:{name}" for name in families.names)
    elif options.harm == "overload_family":
        overload = np.maximum(exposure - float(shape["overload_threshold"]), 0.0) ** 2
        pieces.append(families.totals(overload))
        ridge.extend([1.0] * len(families.members))
        names.extend(f"family_overload:{name}" for name in families.names)
    elif options.harm == "bounded_bucket":
        pieces.append(bounded_harm(exposure, float(shape["log_exponent"])))
        ridge.extend([1.0] * features.buckets)
        names.extend(f"bucket_bounded_harm:{index}" for index in range(features.buckets))
    elif options.harm != "none":
        raise ValueError(f"unknown harm {options.harm}")

    if options.member_replay:
        bucket_harm = softplus_harm(exposure, float(shape["threshold"]))
        pieces.append(families.means(bucket_harm))
        ridge.extend([1.0] * len(families.members))
        names.extend(f"family_member_replay:{name}" for name in families.names)

    if options.interaction == "total_square":
        total = bucket_signal.sum(axis=1, keepdims=True) ** 2
        pieces.extend([total, -total])
        ridge.extend([1.0, 1.0])
        names.extend(["interaction:total_square_plus", "interaction:total_square_minus"])
    elif options.interaction == "family_products":
        for high, low in families.pairs:
            product = bucket_signal[:, [high]] * bucket_signal[:, [low]]
            pieces.extend([product, -product])
            ridge.extend([1.0, 1.0])
            names.extend([f"interaction:pair_plus:{high}+{low}", f"interaction:pair_minus:{high}+{low}"])
    elif options.interaction != "none":
        raise ValueError(f"unknown interaction {options.interaction}")

    if options.quality_axis != "none":
        if options.quality_axis not in ("benefit", "harm", "both"):
            raise ValueError(f"unknown quality axis {options.quality_axis}")
        quality = np.asarray(families.quality, dtype=int).copy()
        if options.shuffled_quality:
            known = np.flatnonzero(quality >= 0)
            quality[known] = quality[np.random.default_rng(QUALITY_SCRAMBLE_SEED).permutation(known)]
        levels = sorted({int(level) for level in quality if level >= 0})
        if options.quality_axis in ("benefit", "both"):
            for level in levels:
                pieces.append(-bucket_signal[:, quality == level].sum(axis=1, keepdims=True))
                ridge.append(1.0)
                names.append(f"quality_benefit:{level}")
        if options.quality_axis in ("harm", "both"):
            quality_harm = softplus_harm(exposure, float(shape["threshold"]))
            for level in levels:
                pieces.append(quality_harm[:, quality == level].sum(axis=1, keepdims=True))
                ridge.append(1.0)
                names.append(f"quality_harm:{level}")

    if options.component_ridge:
        factors = dict(dict(options.component_ridge).get(features.component, ()))
        if factors:
            position = {name: index for index, name in enumerate(features.buckets_names)}
            for index, name in enumerate(names):
                if name.startswith(("bucket_signal:", "bucket_overexposure:")):
                    bucket = features.buckets_names[int(name.split(":")[1])] if features.buckets_names else None
                    if bucket in factors:
                        ridge[index] *= float(factors[bucket])
            del position

    return Design(np.hstack(pieces), np.asarray(ridge, dtype=float), tuple(names))


def crs_plus_design(features: Features, shape: Shape) -> Design:
    """Compact retained state plus family benefit and family overload, tied image."""
    families = features.families
    state = retained_state(features, float(shape["late_multiplier"]), float(shape["forgetting_rate"]))
    rate = 1.0 / float(shape["saturation_epochs"])
    power = float(shape["power"])
    family_rate = rate / max(len(families.members), 1)
    # Every family, singletons included: the original pools over all family members, and a singleton's
    # family column uses the family rate (rate / number of families), so it is not a duplicate of the
    # bucket column and must stay in the tied image.
    pieces = [-weibull_response(state, rate, power)]
    ridge = [1.0] * features.buckets
    names = [f"retained_benefit:{index}" for index in range(features.buckets)]
    pieces.append(-weibull_response(families.totals(state), family_rate, power))
    ridge.extend([1.0] * len(families.members))
    names.extend(f"family_benefit:{name}" for name in families.names)
    pieces.append(literal_replay(features.exposures).sum(axis=1, keepdims=True))
    ridge.append(1.0)
    names.append("shared_literal_replay")
    overload = np.maximum(features.exposures - float(shape["overload_threshold"]), 0.0) ** 2
    pieces.append(families.totals(overload))
    ridge.extend([1.0] * len(families.members))
    names.extend(f"family_overload:{name}" for name in families.names)
    return Design(np.hstack(pieces), np.asarray(ridge, dtype=float), tuple(names))


def retained_power_law_design(
    features: Features, shape: Shape, *, coordinate: str = "weight", hierarchical: bool = True, damage: bool = True
) -> Design:
    """Phase-blind retained power law: inverse-power benefit in share, power damage in epochs."""
    families = features.families
    basis_input = features.weights if coordinate == "weight" else features.exposures
    benefit = (basis_input + float(shape["benefit_offset"])) ** (-float(shape["benefit_exponent"]))
    excess = np.maximum(features.exposures - float(shape["damage_threshold"]), 0.0) ** float(shape["damage_exponent"])

    def block(values: np.ndarray, label: str) -> tuple[list[np.ndarray], list[float], list[str]]:
        if not hierarchical:
            return [values], [1.0] * features.buckets, [f"{label}_bucket:{index}" for index in range(features.buckets)]
        pooled = families.totals(values)
        pieces = [pooled]
        ridge = [0.0] * len(families.members)
        names = [f"{label}_family:{name}" for name in families.names]
        departures = families.multi_member_buckets
        if len(departures):
            pieces.append(values[:, departures])
            ridge.extend([1.0] * len(departures))
            names.extend(f"{label}_bucket_departure:{index}" for index in departures)
        return pieces, ridge, names

    pieces, ridge, names = block(benefit, "benefit")
    if damage:
        damage_pieces, damage_ridge, damage_names = block(excess, "damage")
        pieces.extend(damage_pieces)
        ridge.extend(damage_ridge)
        names.extend(damage_names)
    return Design(np.hstack(pieces), np.asarray(ridge, dtype=float), tuple(names))


def log_epoch_design(features: Features, shape: Shape) -> Design:
    del shape
    return Design(
        np.log1p(features.exposures),
        np.ones(features.buckets),
        tuple(f"log1p_epochs:{index}" for index in range(features.buckets)),
    )


def bowl_design(features: Features, mu: np.ndarray, *, symmetric: bool = False) -> Design:
    delta = np.log1p(features.exposures) - mu[None, :]
    if symmetric:
        return Design(delta**2, np.ones(features.buckets), tuple(f"bowl:{index}" for index in range(features.buckets)))
    values = np.hstack([np.minimum(delta, 0.0) ** 2, np.maximum(delta, 0.0) ** 2])
    names = tuple(f"bowl_under:{index}" for index in range(features.buckets)) + tuple(
        f"bowl_over:{index}" for index in range(features.buckets)
    )
    return Design(values, np.ones(2 * features.buckets), names)


def base_mu(exposure: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.where(exposure > 1e-8, exposure, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(logged, axis=0)
    return np.clip(np.where(np.isfinite(median), median, 0.0), *BOWL_MU_BOUND)


# ---------------------------------------------------------------------------------------------
# Fitted models and the common fit contract
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Fitted:
    """One fitted single-phase model: shape, ridge, head(s), and solver diagnostics."""

    shape: dict[str, float]
    ridge: float
    head: Any
    diagnostics: dict[str, float | int | bool | str]
    # Inner-CV RMSE for every (candidate shape, ridge) pair when the model searched a grid.
    cv_table: np.ndarray | None = None


class SinglePhaseModel(Protocol):
    model_id: str

    def fit(
        self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int
    ) -> Fitted: ...

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray: ...

    def nonlinear_dof(self, features: Features) -> int: ...


def _cv_rmse(design: Design, response: np.ndarray, ridge: float, spec: HeadSpec, inner: InnerFolds) -> float:
    error = 0.0
    count = 0
    for train, validation in inner:
        head = fit_head(Design(design.values[train], design.ridge, design.names), response[train], ridge, spec)
        prediction = predict_head(head, design.values[validation], spec)
        if not np.isfinite(prediction).all():
            return float("inf")
        error += float(np.sum((prediction - response[validation]) ** 2))
        count += len(validation)
    return math.sqrt(error / count)


def _grid_edges(shape: Shape, grid: Sequence[Shape]) -> int:
    hits = 0
    for key, value in shape.items():
        values = sorted({float(candidate[key]) for candidate in grid if key in candidate})
        if len(values) > 1 and (value <= values[0] or value >= values[-1]):
            hits += 1
    return hits


# Bumped when a design function's output changes for an unchanged configuration, so cached fits of
# models built on it are invalidated even though their described configuration is the same.
DESIGN_REVISIONS = {"crs_plus_design": 2}


def _describe_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Only non-default fields describe a configuration, so adding a defaulted option later does
        # not change the description of models that do not use it.
        described = {}
        for field in dataclasses.fields(value):
            current = getattr(value, field.name)
            default = field.default if field.default is not dataclasses.MISSING else dataclasses.MISSING
            if default is not dataclasses.MISSING and _describe_value(current) == _describe_value(default):
                continue
            described[field.name] = _describe_value(current)
        return described
    if isinstance(value, dict):
        return {str(key): _describe_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_describe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (StrEnum, str)):
        return str(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if callable(value):
        # A design builder is identified by its closure's configuration where one exists,
        # otherwise by its qualified name.
        closure = getattr(value, "__closure__", None) or ()
        cells = [_describe_value(cell.cell_contents) for cell in closure if not callable(cell.cell_contents)]
        return {"callable": getattr(value, "__qualname__", repr(value)), "closure": cells}
    return repr(value)


def describe_model(model: Any) -> dict[str, Any]:
    """Everything that determines a model's fits, as a JSON-serializable dict for cache keys."""
    payload = _describe_value(model)
    names: set[str] = set()
    for attribute in ("builder", "_builder"):
        function = getattr(model, attribute, None)
        code = getattr(function, "__code__", None)
        if code is not None:
            names.update(code.co_names)
    if isinstance(model, RetainedPowerLawModel):
        names.add("retained_power_law_design")
    revisions = {name: DESIGN_REVISIONS[name] for name in names if name in DESIGN_REVISIONS}
    if revisions:
        payload["design_revisions"] = revisions
    if isinstance(model, GridModel):
        payload["effective_search"] = (
            "exhaustive"
            if len(model.shapes) <= model.screen_top or len(model.ridge_grid) == 1
            else f"two_stage:{model.screen_top}:{model.screen_ridge_index}"
        )
        payload.pop("screen_top", None)
        payload.pop("screen_ridge_index", None)
    return payload


_DESIGN_CACHE: dict[tuple[str, str, str], Design] = {}
_DESIGN_CACHE_LIMIT = 6000


def _shape_key(shape: Shape) -> str:
    return "|".join(f"{key}={float(value):.17g}" for key, value in sorted(shape.items()))


def cached_design(
    model_id: str, features: Features, shape: Shape, builder: Callable[[Features, Shape], Design]
) -> Design:
    # The component is part of the identity because component-dependent designs (ridge priors) exist.
    key = (model_id, features.cache_key, features.component, _shape_key(shape))
    design = _DESIGN_CACHE.get(key)
    if design is None:
        design = builder(features, shape)
        if len(_DESIGN_CACHE) >= _DESIGN_CACHE_LIMIT:
            _DESIGN_CACHE.clear()
        _DESIGN_CACHE[key] = design
    return design


def _refine_shape(
    model: GridModel,
    features: Features,
    response: np.ndarray,
    ridge: float,
    inner: InnerFolds,
    shape: Shape,
    score: float,
    evaluations: int,
    hull: tuple[Shape, ...] | None = None,
    link: LinkKind | None = None,
) -> tuple[Shape, float, int]:
    """Nelder-Mead from the grid argmin on the inner-CV objective; returns the better of grid and refined.

    With ``hull`` the refined parameters are clipped to the candidate grid's range per key.
    """
    keys = [key for key, value in shape.items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not keys:
        return shape, score, 0
    bounds = None
    if hull is not None:
        bounds = {
            key: (min(float(c[key]) for c in hull), max(float(c[key]) for c in hull))
            for key in keys
            if all(key in c for c in hull)
        }

    def encode(values: Shape) -> np.ndarray:
        return np.asarray(
            [
                math.log(max(float(values[key]), 1e-12)) if key in LOG_SPACE_SHAPE_KEYS else float(values[key])
                for key in keys
            ]
        )

    def decode(vector: np.ndarray) -> Shape:
        decoded = dict(shape)
        for key, value in zip(keys, vector, strict=True):
            raw = float(math.exp(np.clip(value, -30.0, 30.0))) if key in LOG_SPACE_SHAPE_KEYS else float(value)
            if bounds is not None and key in bounds:
                raw = float(np.clip(raw, bounds[key][0], bounds[key][1]))
            decoded[key] = raw
        return decoded

    spec = model.head_for(shape, link)
    counter = 0

    def objective(vector: np.ndarray) -> float:
        nonlocal counter
        counter += 1
        candidate = decode(vector)
        if candidate.get("power", 1.0) <= 0.0 or candidate.get("threshold", 0.0) < 0.0:
            return float("inf")
        value = _cv_rmse(model.design(features, candidate), response, ridge, spec, inner)
        return value if math.isfinite(value) else float("inf")

    start = encode(shape)
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxfev": evaluations, "xatol": 1e-3, "fatol": 1e-7, "initial_simplex": _initial_simplex(start)},
    )
    if math.isfinite(result.fun) and result.fun < score:
        return decode(result.x), float(result.fun), counter
    return shape, score, counter


def _initial_simplex(start: np.ndarray) -> np.ndarray:
    """Simplex spanning about a quarter of a grid step in every direction."""
    simplex = np.tile(start, (len(start) + 1, 1))
    for index in range(len(start)):
        simplex[index + 1, index] += 0.25 if start[index] == 0.0 else 0.25 * max(abs(start[index]), 1.0)
    return simplex


@dataclasses.dataclass(frozen=True)
class GridModel:
    """A design builder with a finite shape grid; shape and ridge chosen by inner CV."""

    model_id: str
    builder: Callable[[Features, Shape], Design]
    shapes: tuple[Shape, ...]
    ridge_grid: tuple[float, ...]
    head: HeadSpec
    shape_dof: int
    dedupe_keys: tuple[str, ...] = ()
    # Every shape sees the full ridge grid by default. A two-stage screen (shapes ranked at one ridge,
    # then the ``screen_top`` best swept over every ridge) is available for experiments but is not
    # selection-equivalent to the exhaustive search: on crs_plus it chose a different shape and ridge
    # with inner RMSE 0.02900 against 0.02599, so it is never used for reported fits.
    screen_top: int = 1_000_000
    # Continuous refinement of the grid argmin: Nelder-Mead over the shape's numeric parameters on the
    # inner-CV objective at the selected ridge (rates and epoch scales move in log space).
    refine: bool = False
    refine_evaluations: int = REFINE_EVALUATIONS
    # Clip every refined parameter to the candidate grid's range (in log space for rates).
    refine_bounded: bool = False
    # Alternative links tried for every (shape, ridge) candidate; the inner-CV winner is stored on the fit.
    link_candidates: tuple[LinkKind, ...] = ()
    screen_ridge_index: int = 2

    def candidate_shapes(self, features: Features) -> tuple[Shape, ...]:
        if not self.dedupe_keys or features.early_fraction.min() < 1.0:
            return self.shapes
        # On phase-less panels the late multiplier is inert, so duplicate shapes are removed.
        seen: dict[str, Shape] = {}
        for shape in self.shapes:
            reduced = {key: 1.0 if key in self.dedupe_keys else value for key, value in shape.items()}
            seen.setdefault(_shape_key(reduced), reduced)
        return tuple(seen.values())

    def design(self, features: Features, shape: Shape) -> Design:
        return cached_design(self.model_id, features, shape, self.builder)

    def head_for(self, shape: Shape, link: LinkKind | None = None) -> HeadSpec:
        spec = self.head
        if "floor_margin" in shape:
            spec = dataclasses.replace(spec, floor_margin=float(shape["floor_margin"]))
        return spec if link is None else dataclasses.replace(spec, link=link)

    def _shortlist(
        self, features: Features, response: np.ndarray, inner: InnerFolds, candidates: tuple[Shape, ...]
    ) -> tuple[list[int], int]:
        if len(candidates) <= self.screen_top or len(self.ridge_grid) == 1:
            return list(range(len(candidates))), 0
        screen_ridge = self.ridge_grid[min(self.screen_ridge_index, len(self.ridge_grid) - 1)]
        scored = []
        for shape_index, shape in enumerate(candidates):
            scored.append(
                (
                    _cv_rmse(self.design(features, shape), response, screen_ridge, self.head_for(shape), inner),
                    shape_index,
                )
            )
        scored.sort()
        return [index for _, index in scored[: self.screen_top]], len(candidates)

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del seed
        candidates = self.candidate_shapes(features)
        shortlist, screened = self._shortlist(features, response, inner, candidates)
        best: tuple[float, int, int] | None = None
        table = np.full((len(candidates), len(self.ridge_grid)), np.inf)
        links: tuple[LinkKind | None, ...] = self.link_candidates or (None,)
        best_link: LinkKind | None = None
        for shape_index in shortlist:
            shape = candidates[shape_index]
            design = self.design(features, shape)
            for link in links:
                spec = self.head_for(shape, link)
                for ridge_index, ridge in enumerate(self.ridge_grid):
                    score = _cv_rmse(design, response, ridge, spec, inner)
                    table[shape_index, ridge_index] = min(table[shape_index, ridge_index], score)
                    candidate = (score, shape_index, ridge_index)
                    if best is None or candidate < best:
                        best = candidate
                        best_link = link
        if best is None or not math.isfinite(best[0]):
            raise ValueError(f"{self.model_id}: no finite inner-CV candidate")
        score, shape_index, ridge_index = best
        shape = dict(candidates[shape_index])
        ridge = self.ridge_grid[ridge_index]
        refined_evaluations = 0
        chosen_link = best_link
        if self.refine:
            shape, score, refined_evaluations = _refine_shape(
                self,
                features,
                response,
                ridge,
                inner,
                shape,
                score,
                self.refine_evaluations,
                candidates if self.refine_bounded else None,
                chosen_link,
            )
        design = self.design(features, shape)
        spec = self.head_for(shape, chosen_link)
        head = fit_head(Design(design.values[train], design.ridge, design.names), response[train], ridge, spec)
        rank = effective_rank(design.values[train])
        return Fitted(
            shape=shape,
            ridge=float(ridge),
            head=head,
            diagnostics={
                "inner_cv_rmse": score,
                "candidates": screened + len(shortlist) * len(self.ridge_grid) * len(links),
                "converged": True,
                "boundary_hits": (
                    _grid_edges(shape, candidates)
                    + int(ridge in (self.ridge_grid[0], self.ridge_grid[-1]) and len(self.ridge_grid) > 1)
                ),
                "effective_rank": rank,
                "columns": design.values.shape[1],
                "fitted_dof": head.active + 1 + self.shape_dof,
                "nonlinear_dof": self.shape_dof,
                "refine_evaluations": refined_evaluations,
                "link": str(spec.link),
            },
            cv_table=table,
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        design = self.design(features, fitted.shape)
        link = LinkKind(str(fitted.diagnostics["link"])) if "link" in fitted.diagnostics else None
        return predict_head(fitted.head, design.values[rows], self.head_for(fitted.shape, link))

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return self.shape_dof


@dataclasses.dataclass(frozen=True)
class FoldMeanModel:
    model_id: str = "fold_mean"

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del features, inner, seed
        return Fitted(
            {},
            0.0,
            float(response[train].mean()),
            {
                "converged": True,
                "boundary_hits": 0,
                "effective_rank": 0,
                "columns": 0,
                "fitted_dof": 1,
                "nonlinear_dof": 0,
                "inner_cv_rmse": float("nan"),
                "candidates": 1,
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        del features
        return np.full(len(rows), float(fitted.head))

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return 0


@dataclasses.dataclass(frozen=True)
class LinearWeightModel:
    """Minimum-norm affine fit in the single-phase policy coordinate."""

    model_id: str = "linear_weight"
    coordinate: str = "weight"

    def _matrix(self, features: Features) -> np.ndarray:
        return features.weights if self.coordinate == "weight" else features.exposures

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del inner, seed
        matrix = self._matrix(features)[train]
        design = Design(matrix, np.ones(matrix.shape[1]), tuple(f"weight:{index}" for index in range(matrix.shape[1])))
        head = fit_head(design, response[train], 0.0, HeadSpec(kind=HeadKind.LSTSQ))
        rank = effective_rank(matrix)
        return Fitted(
            {},
            0.0,
            head,
            {
                "converged": True,
                "boundary_hits": 0,
                "effective_rank": rank,
                "columns": matrix.shape[1],
                "fitted_dof": rank + 1,
                "nonlinear_dof": 0,
                "inner_cv_rmse": float("nan"),
                "candidates": 1,
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        return predict_head(fitted.head, self._matrix(features)[rows], HeadSpec(kind=HeadKind.LSTSQ))

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return 0


def fit_olmix_loglinear_analytic(
    weights: np.ndarray, targets: np.ndarray, *, delta: float, seed: int, n_starts: int
) -> olmix_loglinear.OlmixLoglinearFit:
    """The repository OLMix fit with the same starts and objective, driven by an analytic gradient.

    ``olmix_loglinear_fit.fit_olmix_loglinear_model`` differentiates numerically, which costs
    about forty function evaluations per gradient at 39 buckets. The objective, start bank, bounds,
    and selection rule are unchanged; only the gradient supplied to L-BFGS-B differs.
    """
    x = np.asarray(weights, dtype=float).reshape(len(weights), -1)
    y = np.asarray(targets, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("OLMix log-linear fitting requires positive targets")
    rng = np.random.default_rng(seed)
    limit = olmix_loglinear.MAX_LOG_MAGNITUDE

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        log_c = float(params[0])
        coefficients = params[1:]
        raw = x @ coefficients
        logits = np.clip(raw, -limit, limit)
        exp_logits = np.exp(logits)
        exp_c = np.exp(np.clip(log_c, -limit, limit))
        residual = exp_c + exp_logits - y
        magnitude = np.abs(residual)
        loss = float(np.where(magnitude <= delta, 0.5 * residual * residual, delta * (magnitude - 0.5 * delta)).sum())
        psi = np.where(magnitude <= delta, residual, delta * np.sign(residual))
        inside = (raw > -limit) & (raw < limit)
        gradient = np.empty(len(params))
        gradient[0] = float(psi.sum() * exp_c) if -limit < log_c < limit else 0.0
        gradient[1:] = x.T @ (psi * exp_logits * inside)
        return loss, gradient

    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    starts: list[np.ndarray] = []
    for log_c in log_c_candidates:
        starts.append(np.concatenate([[log_c], np.zeros(x.shape[1], dtype=float)]))
        for _ in range(max(n_starts // len(log_c_candidates) - 1, 0)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))
    best_params = None
    best_loss = float("inf")
    bounds = [(-limit, limit), *[(None, None)] * x.shape[1]]
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", jac=True, bounds=bounds)
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)
    if best_params is None:
        raise RuntimeError("OLMix log-linear fit failed")
    return olmix_loglinear.OlmixLoglinearFit(
        log_c=float(best_params[0]), coefficients=tuple(float(value) for value in best_params[1:]), huber_loss=best_loss
    )


@dataclasses.dataclass(frozen=True)
class OlmixTaskwiseModel:
    """The repository-exact OLMix positive log-linear law, fitted per atomic task with Huber loss."""

    model_id: str = "olmix_loglinear_taskwise"
    n_starts: int = olmix_loglinear.FIT_N_STARTS
    analytic_gradient: bool = True

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del inner
        solver = fit_olmix_loglinear_analytic if self.analytic_gradient else olmix_loglinear.fit_olmix_loglinear_model
        fit = solver(
            features.weights[train],
            response[train],
            delta=olmix_loglinear.DEFAULT_HUBER_DELTA,
            seed=seed,
            n_starts=self.n_starts,
        )
        return Fitted(
            {"log_c": fit.log_c},
            0.0,
            fit,
            {
                "converged": True,
                "boundary_hits": int(abs(fit.log_c) >= olmix_loglinear.MAX_LOG_MAGNITUDE - 1e-9),
                "effective_rank": effective_rank(features.weights[train]),
                "columns": features.buckets,
                "fitted_dof": features.buckets + 1,
                "nonlinear_dof": features.buckets + 1,
                "inner_cv_rmse": float("nan"),
                "candidates": self.n_starts,
                "huber_loss": fit.huber_loss,
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        return np.asarray(fitted.head.predict(features.weights[rows]), dtype=float)

    def nonlinear_dof(self, features: Features) -> int:
        return features.buckets + 1


@dataclasses.dataclass(frozen=True)
class BowlModel:
    """Separate heads collapsed to one phase: asymmetric bowls in log epochs around a selected center."""

    model_id: str = "asymmetric_log_bowl"
    symmetric: bool = False
    ridge_grid: tuple[float, ...] = BOWL_RIDGE_GRID
    head: HeadSpec = HeadSpec(kind=HeadKind.NNLS)

    def _select_mu(self, features: Features, response: np.ndarray, rows: np.ndarray, ridge: float) -> np.ndarray:
        median = base_mu(features.exposures[rows])
        best_rmse = float("inf")
        best = median
        subset = dataclasses.replace(
            features, exposures=features.exposures[rows], weights=features.weights[rows], label=features.label
        )
        for shift in BOWL_MU_SHIFTS:
            mu = np.clip(median + shift, *BOWL_MU_BOUND)
            design = bowl_design(subset, mu, symmetric=self.symmetric)
            head = fit_head(design, response[rows], ridge, self.head)
            rmse = float(np.sqrt(np.mean((predict_head(head, design.values, self.head) - response[rows]) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best = mu
        return best

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del seed
        best: tuple[float, int] | None = None
        for ridge_index, ridge in enumerate(self.ridge_grid):
            error = 0.0
            count = 0
            for inner_train, validation in inner:
                mu = self._select_mu(features, response, inner_train, ridge)
                design = bowl_design(features, mu, symmetric=self.symmetric)
                head = fit_head(
                    Design(design.values[inner_train], design.ridge, design.names),
                    response[inner_train],
                    ridge,
                    self.head,
                )
                error += float(
                    np.sum((predict_head(head, design.values[validation], self.head) - response[validation]) ** 2)
                )
                count += len(validation)
            candidate = (math.sqrt(error / count), ridge_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        ridge = self.ridge_grid[best[1]]
        mu = self._select_mu(features, response, train, ridge)
        design = bowl_design(features, mu, symmetric=self.symmetric)
        head = fit_head(Design(design.values[train], design.ridge, design.names), response[train], ridge, self.head)
        return Fitted(
            {"mu_shift": float(np.median(mu - base_mu(features.exposures[train])))},
            float(ridge),
            (head, mu),
            {
                "inner_cv_rmse": best[0],
                "candidates": len(self.ridge_grid) * len(BOWL_MU_SHIFTS),
                "converged": True,
                "boundary_hits": int(best[1] in (0, len(self.ridge_grid) - 1)),
                "effective_rank": effective_rank(design.values[train]),
                "columns": design.values.shape[1],
                "fitted_dof": head.active + 2,
                "nonlinear_dof": 1,
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        head, mu = fitted.head
        design = bowl_design(features, mu, symmetric=self.symmetric)
        return predict_head(head, design.values[rows], self.head)

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return 1


@dataclasses.dataclass(frozen=True)
class CompactRetainedModel:
    """Compact retained state, one-phase form: shared Weibull benefit and one literal replay column.

    The shape is fitted by an in-sample profile over the training rows, as the Observatory does,
    while the ridge is chosen by inner CV.
    """

    model_id: str = "weibull_shared_literal_replay"
    ridge_grid: tuple[float, ...] = COMPACT_RIDGE_GRID
    harm: str = "literal_shared"
    benefit: str = "weibull"
    scale_columns: bool = True
    rate_starts: tuple[float, ...] = (0.25, 1.0, 4.0)
    power_starts: tuple[float, ...] = (0.34, 0.67, 1.0)

    @property
    def head(self) -> HeadSpec:
        return HeadSpec(kind=HeadKind.NNLS, scale_columns=self.scale_columns)

    def _options(self) -> FamilyOptions:
        return FamilyOptions(bucket_signal=True, family_signal="none", harm=self.harm, benefit=self.benefit)

    def _shape(self, theta: np.ndarray) -> dict[str, float]:
        if self.benefit == "weibull":
            return {"rate": float(np.exp(theta[0])), "power": float(theta[1])}
        if self.benefit == "saturation":
            return {"rate": float(np.exp(theta[0]))}
        return {"exponent": float(theta[0])}

    def _bounds(self) -> list[tuple[float, float]]:
        if self.benefit == "weibull":
            return [COMPACT_LOG_RATE_BOUNDS, COMPACT_POWER_BOUNDS]
        if self.benefit == "saturation":
            return [COMPACT_LOG_RATE_BOUNDS]
        return [COMPACT_POWER_BOUNDS]

    def _starts(self) -> list[np.ndarray]:
        if self.benefit == "weibull":
            return [np.asarray([math.log(rate), power]) for rate in self.rate_starts for power in self.power_starts]
        if self.benefit == "saturation":
            return [np.asarray([math.log(rate)]) for rate in self.rate_starts]
        return [np.asarray([power]) for power in self.power_starts]

    def _design(self, features: Features, theta: np.ndarray) -> Design:
        return family_design(features, self._shape(theta), self._options())

    def _fit_shape(
        self, features: Features, response: np.ndarray, rows: np.ndarray, ridge: float
    ) -> tuple[np.ndarray, bool, int]:
        subset = dataclasses.replace(features, exposures=features.exposures[rows], weights=features.weights[rows])
        target = response[rows]

        def objective(theta: np.ndarray) -> float:
            design = self._design(subset, np.asarray(theta, dtype=float))
            head = fit_head(design, target, ridge, self.head)
            residual = predict_head(head, design.values, self.head) - target
            return float(np.mean(residual**2))

        scored = sorted(
            ((objective(start), index, start) for index, start in enumerate(self._starts())), key=lambda item: item[:2]
        )
        best_value, _index, best_theta = scored[0]
        converged = True
        boundary = 0
        for _value, _index, start in scored[:COMPACT_TOP_K]:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=self._bounds(),
                options={"maxiter": COMPACT_MAXITER, "ftol": 1e-10, "maxls": 30},
            )
            if np.isfinite(result.fun) and float(result.fun) < best_value:
                best_value = float(result.fun)
                best_theta = np.asarray(result.x, dtype=float)
                converged = bool(result.success)
        for value, (low, high) in zip(best_theta, self._bounds(), strict=True):
            boundary += int(value <= low + 1e-9 or value >= high - 1e-9)
        return best_theta, converged, boundary

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del seed
        best: tuple[float, int] | None = None
        for ridge_index, ridge in enumerate(self.ridge_grid):
            error = 0.0
            count = 0
            for inner_train, validation in inner:
                theta, _converged, _boundary = self._fit_shape(features, response, inner_train, ridge)
                design = self._design(features, theta)
                head = fit_head(
                    Design(design.values[inner_train], design.ridge, design.names),
                    response[inner_train],
                    ridge,
                    self.head,
                )
                error += float(
                    np.sum((predict_head(head, design.values[validation], self.head) - response[validation]) ** 2)
                )
                count += len(validation)
            candidate = (math.sqrt(error / count), ridge_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        ridge = self.ridge_grid[best[1]]
        theta, converged, boundary = self._fit_shape(features, response, train, ridge)
        design = self._design(features, theta)
        head = fit_head(Design(design.values[train], design.ridge, design.names), response[train], ridge, self.head)
        return Fitted(
            self._shape(theta),
            float(ridge),
            head,
            {
                "inner_cv_rmse": best[0],
                "candidates": len(self.ridge_grid) * len(self._starts()),
                "converged": converged,
                "boundary_hits": boundary + int(best[1] in (0, len(self.ridge_grid) - 1)),
                "effective_rank": effective_rank(design.values[train]),
                "columns": design.values.shape[1],
                "fitted_dof": head.active + 1 + len(theta),
                "nonlinear_dof": len(theta),
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        design = family_design(features, fitted.shape, self._options())
        return predict_head(fitted.head, design.values[rows], self.head)

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return len(self._bounds())


# ---------------------------------------------------------------------------------------------
# Profiled DSP family (continuous per-bucket or shared shapes, implicit gradient through NNLS)
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DspOptions:
    per_bucket: bool = True
    penalty: str = "canonical"  # canonical | bounded | none
    tie_pairs: bool = False
    concentration: bool = False
    maxiter: int = DSP_MAXITER
    restarts: int = DSP_RESTARTS
    linear_reg: float = DSP_LINEAR_REG


@dataclasses.dataclass(frozen=True)
class ProfiledDspModel:
    """Canonical single-phase DSP and its one-mechanism neighbours.

    The nonlinear shape (one saturation rate and one harm parameter per bucket, or one of each
    shared by all buckets) minimizes the blocked inner-CV loss of the ridge-NNLS head; the
    gradient flows through the head's active set exactly as in the profiled ladder solver.
    """

    model_id: str = "dsp_total_exposure"
    options: DspOptions = DspOptions()

    def _split(self, vector: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
        count = buckets if self.options.per_bucket else 1
        rate = vector[:count]
        harm = vector[count : 2 * count] if self.options.penalty != "none" else np.empty(0)
        if not self.options.per_bucket:
            rate = np.full(buckets, rate[0])
            harm = np.full(buckets, harm[0]) if harm.size else harm
        return rate, harm

    def design(self, features: Features, vector: np.ndarray) -> np.ndarray:
        log_rate, harm = self._split(vector, features.buckets)
        exposure = features.exposures
        pieces = [-saturation_response(exposure, np.exp(log_rate)[None, :])]
        if self.options.penalty == "canonical":
            pieces.append(softplus_harm(exposure, harm[None, :]))
        elif self.options.penalty == "bounded":
            pieces.append(bounded_harm(exposure, harm[None, :]))
        if self.options.concentration:
            pieces.append(concentration(features))
        return np.hstack(pieces)

    def _derivative(self, features: Features, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        buckets = features.buckets
        log_rate, harm = self._split(vector, buckets)
        exposure = features.exposures
        rate = np.exp(log_rate)[None, :]
        pieces = [-rate * exposure * np.exp(-rate * exposure)]
        if self.options.per_bucket:
            index = [np.arange(buckets)]
        else:
            index = [np.zeros(buckets, dtype=int)]
        if self.options.penalty == "canonical":
            shifted = np.log1p(exposure) - harm[None, :]
            pieces.append(-2.0 * softplus(shifted) * sigmoid(shifted))
            index.append(np.arange(buckets, 2 * buckets) if self.options.per_bucket else np.ones(buckets, dtype=int))
        elif self.options.penalty == "bounded":
            unit = np.maximum(exposure - 1.0, 0.0) / DSP_DAMAGE_KNEE
            exponent = np.exp(harm)[None, :]
            powered = np.zeros_like(unit)
            np.power(unit, exponent, out=powered, where=unit > 0.0)
            penalty = powered / (1.0 + powered)
            log_unit = np.zeros_like(unit)
            np.log(unit, out=log_unit, where=unit > 0.0)
            pieces.append(penalty * (1.0 - penalty) * exponent * log_unit)
            index.append(np.arange(buckets, 2 * buckets) if self.options.per_bucket else np.ones(buckets, dtype=int))
        if self.options.concentration:
            pieces.append(np.zeros((features.rows, 1)))
            index.append(np.asarray([-1]))
        return np.hstack(pieces), np.concatenate(index)

    def _head(self, features: Features) -> HeadSpec:
        pairs = features.families.pairs if self.options.tie_pairs else ()
        return HeadSpec(kind=HeadKind.NNLS, tie_pairs=pairs, reduced_nnls=False)

    def _regularizer(self, width: int, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
        regularizer = self.options.linear_reg * np.eye(width)
        for first, second in pairs:
            regularizer[first, first] += PAIR_SHRINKAGE
            regularizer[second, second] += PAIR_SHRINKAGE
            regularizer[first, second] -= PAIR_SHRINKAGE
            regularizer[second, first] -= PAIR_SHRINKAGE
        return regularizer

    def _objective(
        self, features: Features, response: np.ndarray, vector: np.ndarray, inner: InnerFolds
    ) -> tuple[float, np.ndarray]:
        design = self.design(features, vector)
        derivative, parameter_index = self._derivative(features, vector)
        spec = self._head(features)
        regularizer = self._regularizer(design.shape[1], spec.tie_pairs)
        total = 0.0
        gradient = np.zeros(len(vector))
        ridge_multipliers = np.ones(design.shape[1])
        for train, validation in inner:
            train_design = design[train]
            train_response = response[train]
            intercept, coefficients = _nonnegative_solve(
                train_design, train_response, self.options.linear_reg, ridge_multipliers, spec, None
            )
            residual = intercept + design[validation] @ coefficients - response[validation]
            if not np.isfinite(residual).all():
                return 1e6, np.zeros_like(gradient)
            total += float(residual @ residual)
            scale = max(1.0, float(np.max(coefficients, initial=0.0)))
            active = np.flatnonzero(coefficients > DSP_ACTIVE_TOL * scale)
            if len(active) == 0:
                continue
            design_mean = train_design.mean(axis=0)
            derivative_mean = derivative[train].mean(axis=0)
            centered_train = train_design - design_mean
            centered_derivative = derivative[train] - derivative_mean
            centered_validation = design[validation] - design_mean
            centered_validation_derivative = derivative[validation] - derivative_mean
            centered_response = train_response - train_response.mean()
            active_design = centered_train[:, active]
            active_derivative = centered_derivative[:, active]
            active_coefficients = coefficients[active]
            active_index = parameter_index[active]
            # Columns without a shape parameter (the concentration column) sit in the active
            # Hessian but carry no direct derivative, so they are masked out of the scatter.
            parametrized = np.flatnonzero(active_index >= 0)
            selector = np.zeros((len(active), len(vector)))
            selector[parametrized, active_index[parametrized]] = active_coefficients[parametrized]
            direct_train = active_derivative @ selector
            train_residual = centered_response - active_design @ active_coefficients
            feature_score = active_derivative.T @ train_residual
            right = np.zeros((len(active), len(vector)))
            right[parametrized, active_index[parametrized]] = feature_score[parametrized]
            right -= active_design.T @ direct_train
            hessian = active_design.T @ active_design + regularizer[np.ix_(active, active)]
            coefficient_derivative = np.linalg.solve(hessian, right)
            direct_validation = centered_validation_derivative[:, active] @ selector
            prediction_derivative = direct_validation + centered_validation[:, active] @ coefficient_derivative
            gradient += 2.0 * prediction_derivative.T @ residual
        return total, gradient

    def _bounds(self, buckets: int) -> list[tuple[float, float]]:
        count = buckets if self.options.per_bucket else 1
        harm_bound = DSP_THRESHOLD_BOUND if self.options.penalty == "canonical" else DSP_LOG_EXPONENT_BOUND
        bounds = [DSP_LOG_RATE_BOUND] * count
        if self.options.penalty != "none":
            bounds += [harm_bound] * count
        return bounds

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        box = self._bounds(features.buckets)
        lows = np.array([low for low, _ in box])
        highs = np.array([high for _, high in box])
        rng = np.random.default_rng(DSP_SEED_BASE + seed)
        starts = [0.5 * (lows + highs)]
        starts.extend(rng.uniform(lows, highs) for _ in range(self.options.restarts - 1))
        subset = dataclasses.replace(features, exposures=features.exposures[train], weights=features.weights[train])
        local = np.full(features.rows, -1)
        local[train] = np.arange(len(train))
        local_inner = tuple((local[inner_train], local[validation]) for inner_train, validation in inner)
        if any((block < 0).any() for pair in local_inner for block in pair):
            raise ValueError("inner folds must be subsets of the training rows")
        target = response[train]
        results = []
        for start in starts:
            result = minimize(
                lambda vector: self._objective(subset, target, np.asarray(vector, dtype=float), local_inner),
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=box,
                options={"maxiter": self.options.maxiter},
            )
            results.append((float(result.fun), np.asarray(result.x, dtype=float), bool(result.success), int(result.nit)))
        value, vector, success, iterations = min(results, key=lambda item: item[0])
        design_values = self.design(subset, vector)
        spec = self._head(features)
        design = Design(
            design_values,
            np.ones(design_values.shape[1]),
            tuple(f"col{index}" for index in range(design_values.shape[1])),
        )
        head = fit_head(design, target, self.options.linear_reg, spec)
        boundary = int(np.sum((vector <= lows + 1e-9) | (vector >= highs - 1e-9)))
        return Fitted(
            {f"theta_{index}": float(item) for index, item in enumerate(vector)},
            float(self.options.linear_reg),
            head,
            {
                "inner_cv_rmse": math.sqrt(value / len(train)),
                "candidates": len(starts),
                "converged": success,
                "iterations": iterations,
                "boundary_hits": boundary,
                "effective_rank": effective_rank(design_values),
                "columns": design_values.shape[1],
                "fitted_dof": head.active + 1 + len(vector),
                "nonlinear_dof": len(vector),
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        vector = np.asarray([fitted.shape[f"theta_{index}"] for index in range(len(fitted.shape))])
        design = self.design(features, vector)[rows]
        return predict_head(fitted.head, design, self._head(features))

    def nonlinear_dof(self, features: Features) -> int:
        return len(self._bounds(features.buckets))


# ---------------------------------------------------------------------------------------------
# Two-stage and ensemble models
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HierarchicalModel:
    """Hierarchical family replay: a bucket-resolved screen selects shapes, then structure is tuned."""

    model_id: str
    screen: GridModel
    builder: Callable[[Features, Shape], Design]
    residual_grid: tuple[float, ...]
    top_shapes: int
    head: HeadSpec = HeadSpec(kind=HeadKind.NNLS)

    def _stage_two_shapes(
        self, features: Features, response: np.ndarray, inner: InnerFolds
    ) -> tuple[tuple[Shape, ...], int]:
        candidates = self.screen.candidate_shapes(features)
        best_by_shape: dict[int, float] = {}
        for shape_index, shape in enumerate(candidates):
            design = self.screen.design(features, shape)
            for ridge in self.screen.ridge_grid:
                score = _cv_rmse(design, response, ridge, self.screen.head, inner)
                best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), score)
        ranked = sorted(best_by_shape.items(), key=lambda item: (item[1], item[0]))[: self.top_shapes]
        configs = tuple(
            {**candidates[index], "residual_shrink": shrink} for index, _ in ranked for shrink in self.residual_grid
        )
        return configs, len(candidates) * len(self.screen.ridge_grid)

    def _design(self, features: Features, shape: Shape) -> Design:
        return cached_design(self.model_id, features, shape, self.builder)

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del seed
        configs, screened = self._stage_two_shapes(features, response, inner)
        best: tuple[float, int, int] | None = None
        for config_index, config in enumerate(configs):
            design = self._design(features, config)
            for ridge_index, ridge in enumerate(self.screen.ridge_grid):
                candidate = (_cv_rmse(design, response, ridge, self.head, inner), config_index, ridge_index)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        score, config_index, ridge_index = best
        shape = dict(configs[config_index])
        ridge = self.screen.ridge_grid[ridge_index]
        design = self._design(features, shape)
        head = fit_head(Design(design.values[train], design.ridge, design.names), response[train], ridge, self.head)
        return Fitted(
            shape,
            float(ridge),
            head,
            {
                "inner_cv_rmse": score,
                "candidates": screened + len(configs) * len(self.screen.ridge_grid),
                "converged": True,
                "boundary_hits": _grid_edges(shape, configs),
                "effective_rank": effective_rank(design.values[train]),
                "columns": design.values.shape[1],
                "fitted_dof": head.active + 1 + self.screen.shape_dof + 1,
                "nonlinear_dof": self.screen.shape_dof + 1,
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        design = self._design(features, fitted.shape)
        return predict_head(fitted.head, design.values[rows], self.head)

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return self.screen.shape_dof + 1


def stack_weights(predictions: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Simplex-constrained stacking weights: NNLS on member differences, residual mass to member 0."""
    if predictions.shape[1] == 1:
        return np.ones(1)
    differences = predictions[:, 1:] - predictions[:, :1]
    coefficients, _ = nnls(differences, observed - predictions[:, 0], maxiter=NNLS_MAXITER_FACTOR * differences.shape[1])
    total = coefficients.sum()
    if total > 1.0:
        coefficients = coefficients / total
    weights = np.concatenate([[1.0 - coefficients.sum()], coefficients])
    weights = np.maximum(weights, 0.0)
    return weights / max(weights.sum(), EPSILON)


@dataclasses.dataclass(frozen=True)
class BandModel:
    """Hierarchical replay averaged over every configuration inside the unresolvable inner-CV band."""

    model_id: str
    base: HierarchicalModel
    relative_width: float = BAND_RELATIVE_WIDTH
    max_members: int = BAND_MAX_MEMBERS

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del seed
        configs, screened = self.base._stage_two_shapes(features, response, inner)
        scored: list[tuple[float, int, int]] = []
        oof: dict[tuple[int, int], np.ndarray] = {}
        for config_index, config in enumerate(configs):
            design = self.base._design(features, config)
            for ridge_index, ridge in enumerate(self.base.screen.ridge_grid):
                prediction = np.full(features.rows, np.nan)
                for inner_train, validation in inner:
                    head = fit_head(
                        Design(design.values[inner_train], design.ridge, design.names),
                        response[inner_train],
                        ridge,
                        self.base.head,
                    )
                    prediction[validation] = predict_head(head, design.values[validation], self.base.head)
                covered = np.concatenate([validation for _, validation in inner])
                rmse = float(np.sqrt(np.mean((prediction[covered] - response[covered]) ** 2)))
                scored.append((rmse, config_index, ridge_index))
                oof[(config_index, ridge_index)] = prediction
        scored.sort()
        best = scored[0][0]
        inside = [item for item in scored if item[0] <= best * (1.0 + self.relative_width)][: self.max_members]
        covered = np.concatenate([validation for _, validation in inner])
        stacked = np.column_stack([oof[(config_index, ridge_index)][covered] for _, config_index, ridge_index in inside])
        weights = stack_weights(stacked, response[covered])
        members = []
        for (rmse, config_index, ridge_index), weight in zip(inside, weights, strict=True):
            config = dict(configs[config_index])
            ridge = self.base.screen.ridge_grid[ridge_index]
            design = self.base._design(features, config)
            head = fit_head(
                Design(design.values[train], design.ridge, design.names), response[train], ridge, self.base.head
            )
            members.append((config, float(ridge), float(weight), float(rmse), head))
        top = max(members, key=lambda member: member[2])
        return Fitted(
            dict(top[0]),
            top[1],
            members,
            {
                "inner_cv_rmse": best,
                "candidates": screened + len(scored),
                "band_size": len(members),
                "active_members": int(sum(weight > 1e-6 for _, _, weight, _, _ in members)),
                "converged": True,
                "boundary_hits": 0,
                "effective_rank": effective_rank(self.base._design(features, top[0]).values[train]),
                "columns": self.base._design(features, top[0]).values.shape[1],
                "fitted_dof": sum(head.active for *_, head in members) + len(members),
                "nonlinear_dof": self.base.nonlinear_dof(features),
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        total = np.zeros(len(rows))
        for config, _ridge, weight, _rmse, head in fitted.head:
            if weight <= 0.0:
                continue
            design = self.base._design(features, config)
            total += weight * predict_head(head, design.values[rows], self.base.head)
        return total

    def nonlinear_dof(self, features: Features) -> int:
        return self.base.nonlinear_dof(features)


@dataclasses.dataclass(frozen=True)
class FamilyOnsetModel:
    """Power GRP with one learned replay onset per family, shrunk toward the shared onset."""

    model_id: str
    shared: GridModel
    tau_shrink_grid: tuple[float, ...] = TAU_SHRINK_GRID

    def _design(self, features: Features, shape: Shape, tau: np.ndarray) -> Design:
        families = features.families
        exposure = features.exposures
        pieces = [-power_response(exposure, float(shape["exponent"]))]
        names = [f"bucket_signal:{index}" for index in range(features.buckets)]
        pooled = list(families.nonsingleton)
        if pooled:
            pieces.append(-power_response(families.totals(exposure)[:, pooled], float(shape["exponent"])))
            names.extend(f"family_signal:{families.names[index]}" for index in pooled)
        pieces.append(softplus_harm(families.totals(exposure), tau[None, :]))
        names.extend(f"family_overexposure:{name}" for name in families.names)
        values = np.hstack(pieces)
        return Design(values, np.ones(values.shape[1]), tuple(names))

    def _fit_tau(
        self,
        features: Features,
        response: np.ndarray,
        rows: np.ndarray,
        shape: Shape,
        ridge: float,
        tau_shrink: float,
        *,
        multistart: bool,
    ) -> tuple[np.ndarray, bool]:
        families = features.families
        anchor = float(shape["threshold"])
        logged = np.log1p(families.totals(features.exposures)[rows])
        starts = [np.full(len(families.members), anchor)]
        quantiles = (0.5, 0.75, 0.9) if multistart else (0.75,)
        starts.extend(np.quantile(logged, quantile, axis=0) for quantile in quantiles)
        penalty_offset = features.buckets + len(families.nonsingleton)
        spec = self.shared.head
        target = response[rows]

        def objective(tau: np.ndarray) -> tuple[float, np.ndarray]:
            design = self._design(features, shape, tau)
            head = fit_head(Design(design.values[rows], design.ridge, design.names), target, ridge, spec)
            residual = predict_head(head, design.values[rows], spec) - target
            coefficients = head.coefficients[penalty_offset : penalty_offset + len(families.members)]
            delta = logged - tau[None, :]
            derivative = -2.0 * softplus(delta) * sigmoid(delta)
            data_gradient = 2.0 * np.mean(residual[:, None] * coefficients[None, :] * derivative, axis=0)
            displacement = tau - anchor
            loss = (
                float(np.mean(residual**2))
                + ridge * float(np.sum(head.coefficients**2)) / len(rows)
                + tau_shrink * float(np.mean(displacement**2))
            )
            return loss, data_gradient + 2.0 * tau_shrink * displacement / len(displacement)

        results = [
            minimize(
                objective,
                np.clip(start, *TAU_BOUNDS),
                method="L-BFGS-B",
                jac=True,
                bounds=[TAU_BOUNDS] * len(families.members),
                options={"maxiter": TAU_MAXITER, "ftol": 1e-12, "maxls": 30},
            )
            for start in starts
        ]
        finite = [result for result in results if np.isfinite(result.fun)]
        if not finite:
            raise ValueError(f"{self.model_id}: family onset optimization produced no finite objective")
        best = min(finite, key=lambda result: float(result.fun))
        return np.asarray(best.x, dtype=float), bool(best.success)

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        shared = self.shared.fit(features, response, train, inner, seed)
        shape = shared.shape
        ridge = shared.ridge
        spec = self.shared.head
        best: tuple[float, int] | None = None
        for shrink_index, tau_shrink in enumerate(self.tau_shrink_grid):
            error = 0.0
            count = 0
            for inner_train, validation in inner:
                tau, _converged = self._fit_tau(
                    features, response, inner_train, shape, ridge, tau_shrink, multistart=False
                )
                design = self._design(features, shape, tau)
                head = fit_head(
                    Design(design.values[inner_train], design.ridge, design.names), response[inner_train], ridge, spec
                )
                error += float(np.sum((predict_head(head, design.values[validation], spec) - response[validation]) ** 2))
                count += len(validation)
            candidate = (math.sqrt(error / count), shrink_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        tau_shrink = self.tau_shrink_grid[best[1]]
        tau, converged = self._fit_tau(features, response, train, shape, ridge, tau_shrink, multistart=True)
        design = self._design(features, shape, tau)
        head = fit_head(Design(design.values[train], design.ridge, design.names), response[train], ridge, spec)
        return Fitted(
            {**shape, "tau_shrink": float(tau_shrink)},
            ridge,
            (head, tau),
            {
                "inner_cv_rmse": best[0],
                "candidates": shared.diagnostics["candidates"] + len(self.tau_shrink_grid),
                "converged": converged,
                "boundary_hits": (
                    int(shared.diagnostics["boundary_hits"])
                    + int(np.sum((tau <= TAU_BOUNDS[0] + 1e-9) | (tau >= TAU_BOUNDS[1] - 1e-9)))
                ),
                "effective_rank": effective_rank(design.values[train]),
                "columns": design.values.shape[1],
                "fitted_dof": head.active + 1 + self.shared.shape_dof + len(tau),
                "nonlinear_dof": self.shared.shape_dof + len(tau),
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        head, tau = fitted.head
        design = self._design(features, fitted.shape, tau)
        return predict_head(head, design.values[rows], self.shared.head)

    def nonlinear_dof(self, features: Features) -> int:
        return self.shared.shape_dof + len(features.families.members)


@dataclasses.dataclass(frozen=True)
class RetainedPowerLawModel:
    """Phase-blind retained power law: least-squares shape screen, robust rescoring of the shortlist."""

    model_id: str
    shapes: tuple[Shape, ...]
    ridge_grid: tuple[float, ...]
    coordinate: str = "weight"
    hierarchical: bool = True
    damage: bool = True
    robust: bool = True
    top_shapes: int = RPL_TOP_SHAPES

    @property
    def screen_head(self) -> HeadSpec:
        return HeadSpec(kind=HeadKind.NNLS, scale_columns=True, scale_rule="max_abs")

    @property
    def robust_head(self) -> HeadSpec:
        return HeadSpec(
            kind=HeadKind.HUBER_NNLS if self.robust else HeadKind.NNLS, scale_columns=True, scale_rule="max_abs"
        )

    def _builder(self, features: Features, shape: Shape) -> Design:
        return retained_power_law_design(
            features, shape, coordinate=self.coordinate, hierarchical=self.hierarchical, damage=self.damage
        )

    def _design(self, features: Features, shape: Shape) -> Design:
        return cached_design(self.model_id, features, shape, self._builder)

    def fit(self, features: Features, response: np.ndarray, train: np.ndarray, inner: InnerFolds, seed: int) -> Fitted:
        del seed
        ranked: dict[int, float] = {}
        for shape_index, shape in enumerate(self.shapes):
            design = self._design(features, shape)
            if not np.isfinite(design.values).all():
                continue
            for ridge in self.ridge_grid:
                score = _cv_rmse(design, response, ridge, self.screen_head, inner)
                ranked[shape_index] = min(ranked.get(shape_index, float("inf")), score)
        shortlist = [
            index for index, _ in sorted(ranked.items(), key=lambda item: (item[1], item[0]))[: self.top_shapes]
        ]
        best: tuple[float, int, int] | None = None
        for shape_index in shortlist:
            design = self._design(features, self.shapes[shape_index])
            for ridge_index, ridge in enumerate(self.ridge_grid):
                candidate = (_cv_rmse(design, response, ridge, self.robust_head, inner), shape_index, ridge_index)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise ValueError(f"{self.model_id}: no finite design on this panel")
        score, shape_index, ridge_index = best
        shape = dict(self.shapes[shape_index])
        ridge = self.ridge_grid[ridge_index]
        design = self._design(features, shape)
        head = fit_head(
            Design(design.values[train], design.ridge, design.names), response[train], ridge, self.robust_head
        )
        return Fitted(
            shape,
            float(ridge),
            head,
            {
                "inner_cv_rmse": score,
                "candidates": len(ranked) * len(self.ridge_grid) + len(shortlist) * len(self.ridge_grid),
                "converged": True,
                "boundary_hits": _grid_edges(shape, self.shapes),
                "effective_rank": effective_rank(design.values[train]),
                "columns": design.values.shape[1],
                "fitted_dof": head.active + 1 + 3,
                "nonlinear_dof": 3,
            },
        )

    def predict(self, fitted: Fitted, features: Features, rows: np.ndarray) -> np.ndarray:
        design = self._design(features, fitted.shape)
        return predict_head(fitted.head, design.values[rows], self.robust_head)

    def nonlinear_dof(self, features: Features) -> int:
        del features
        return 3


# ---------------------------------------------------------------------------------------------
# Shape grids reproduced from the Observatory modules (single-phase images)
# ---------------------------------------------------------------------------------------------


def _sobol(dimension: int, count: int, seed: int) -> np.ndarray:
    sample_count = 1 << math.ceil(math.log2(max(count, 2)))
    return qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(int(math.log2(sample_count)))[:count]


def bucket_family_shapes(count: int = 24) -> tuple[Shape, ...]:
    """`fit_production_grp_quality_variants.shape_candidates` (bucket-resolved), phase-tied image."""
    unit = _sobol(4, count, 211)
    shapes: list[Shape] = [
        {"exponent": 0.33989885260566105, "threshold": 5.136810831800622},
    ]
    for row in unit:
        exponent = float(np.exp(np.log(0.08) + row[0] * (np.log(1.2) - np.log(0.08))))
        shapes.append({"exponent": exponent, "threshold": float(row[3] * 7.0)})
    return tuple(dict(item) for item in {_shape_key(shape): shape for shape in shapes}.values())


def retained_grp_shapes(variant: str, count: int = 32) -> tuple[Shape, ...]:
    """`benchmark_production_grp_retained_hybrids_20260713.shared_shape_candidates`, phase-tied image."""
    global_tau = variant.endswith("global_tau")
    dimension = 5 if global_tau else 4
    unit = _sobol(dimension, count, sum(ord(character) for character in variant))
    power = variant.startswith("power")
    shapes: list[Shape] = []
    if power:
        shapes.extend(
            [
                {"exponent": 0.33989885260566105, "threshold": 5.136810831800622},
                {"exponent": 0.36662675542192796, "threshold": 6.508345540612936},
            ]
        )
    else:
        shapes.extend(
            [
                {"rate": 2.79573, "power": 0.661316, "threshold": 5.136810831800622},
                {"rate": 0.25, "power": 0.67, "threshold": 4.0},
            ]
        )
    for row in unit:
        if power:
            shape: dict[str, float] = {"exponent": float(np.exp(np.log(0.08) + row[0] * (np.log(1.2) - np.log(0.08))))}
            offset = 1
        else:
            shape = {
                "rate": float(np.exp(np.log(0.05) + row[0] * (np.log(20.0) - np.log(0.05)))),
                "power": float(0.2 + 0.8 * row[1]),
            }
            offset = 2
        shape["threshold"] = float(row[offset + 2] * 7.0) if global_tau else 0.0
        shapes.append(shape)
    if not global_tau:
        shapes = [{key: value for key, value in shape.items() if key != "threshold"} for shape in shapes]
    return tuple(dict(item) for item in {_shape_key(shape): shape for shape in shapes}.values())


def crs_plus_shapes() -> tuple[Shape, ...]:
    return tuple(
        {
            "saturation_epochs": saturation,
            "power": power,
            "late_multiplier": late,
            "forgetting_rate": forgetting,
            "overload_threshold": overload,
        }
        for saturation in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0)
        for power in (0.4, 0.7, 1.0)
        for late in (0.5, 1.0, 2.0, 4.0)
        for forgetting in (0.0, 0.25, 1.0)
        for overload in (1.0, 2.0, 4.0)
    )


def crs_bounded_shapes() -> tuple[Shape, ...]:
    return tuple(
        {"rate": rate, "power": power, "late_multiplier": late, "forgetting_rate": forgetting}
        for rate in (0.25, 1.0)
        for power in (0.4, 0.7, 1.0)
        for late in (0.5, 1.0, 2.0, 4.0, 8.0)
        for forgetting in (0.0, 0.25, 1.0)
    )


RPL_BENEFIT_EXPONENTS = (0.25, 0.5, 1.0)
RPL_BENEFIT_OFFSETS = (0.01, 0.1)
RPL_LATE_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
RPL_DAMAGE_EXPONENTS = (1.5, 2.0, 3.0)
RPL_RIDGE_GRID = (1e-4, 1e-2, 1.0)


def retained_power_law_shapes(phase_0_fraction: float) -> tuple[Shape, ...]:
    """`retained_power_law_estimator_repair_20260731.phase_blind_shape_grid`."""
    offsets = sorted(
        {
            offset / (phase_0_fraction + late * (1.0 - phase_0_fraction))
            for offset in RPL_BENEFIT_OFFSETS
            for late in RPL_LATE_MULTIPLIERS
        }
    )
    return tuple(
        {"benefit_exponent": exponent, "benefit_offset": offset, "damage_exponent": damage, "damage_threshold": 0.0}
        for exponent in RPL_BENEFIT_EXPONENTS
        for offset in offsets
        for damage in RPL_DAMAGE_EXPONENTS
    )


GRP_PAIR_EXPONENTS = (
    0.1657586322714625,
    0.2076641777781618,
    0.25,
    0.33989885260566105,
    0.5,
    0.6462737477673589,
    0.85,
    1.0,
)
GRP_PAIR_DISCOUNTS = (0.2629059619755788, 0.5, 1.0)
GRP_PAIR_THRESHOLDS = (3.193090495213877, 4.0, 5.136810831800622, 6.2042610686315145, 7.0)
GRP_PAIR_RIDGE_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)


def grp_pair_shapes(*, discount: bool = True) -> tuple[Shape, ...]:
    discounts = GRP_PAIR_DISCOUNTS if discount else (1.0,)
    return tuple(
        {"exponent": exponent, "quality_discount": beta, "threshold": tau}
        for exponent in GRP_PAIR_EXPONENTS
        for beta in discounts
        for tau in GRP_PAIR_THRESHOLDS
    )


LOG_LINK_RIDGE_GRID = (0.0003, 0.003, 0.03, 0.3, 3.0, 30.0, 300.0, 3000.0)
LOG_LINK_FLOOR_MARGINS = (0.02, 0.08)


def log_link_shapes() -> tuple[Shape, ...]:
    return tuple({"floor_margin": margin} for margin in LOG_LINK_FLOOR_MARGINS)
