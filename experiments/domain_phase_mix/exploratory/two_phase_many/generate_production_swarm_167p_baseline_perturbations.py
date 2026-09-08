# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate lattice-executable baseline perturbations for the 167-partition swarm."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.design_production_swarm_167p import (
    DEFAULT_SOURCE_URL,
    MIXTURE_QUANTUM_DENOMINATOR,
    PHASE_FRACTIONS,
    PHASE_NAMES,
    TARGET_BUDGET,
    integer_simplex_counts,
    load_bucket_table,
    progress,
    write_interactive_sanity_dashboard,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "reference_outputs"
    / "production_swarm_mixture_design_167p_20260522_baseline_perturbations"
)
DEFAULT_D_OPTIMAL_CANDIDATE_CSV = (
    Path(__file__).resolve().parent
    / "reference_outputs"
    / "production_swarm_mixture_design_167p_20260523_cap100_pool262144"
    / "fisher_dsp_aligned"
    / "candidate_mixtures.csv"
)
DEFAULT_LOG_ODDS_MULTIPLIER = 2.0
DEFAULT_RANDOM_DIRECTION_COUNT = 64
DEFAULT_DIRECTION_SEED = 44
DEFAULT_ALPHA = 0.10
DEFAULT_UNIMAX_EPOCH_CAPS = (1.0, 4.0, 8.0, 16.0)
DEFAULT_D_OPTIMAL_MAX_EPOCH_CAP = 100.0


@dataclass(frozen=True)
class Candidate:
    """One materialized baseline perturbation."""

    name: str
    candidate_type: str
    counts: np.ndarray
    baseline_kind: str | None
    unimax_epoch_cap: float | None
    target_bucket: str | None
    target_bucket_index: int | None
    intervention_type: str
    tilt_sign: str
    alpha: float | None
    direction_id: str | None
    direction_index: int | None
    intended_direction: np.ndarray | None


def bucket_slug(bucket: str) -> str:
    """Return a stable candidate-name slug for a bucket name."""
    return bucket.replace("/", "_").replace("-", "_")


def partition_ablation_counts(proportional: np.ndarray, target_index: int, denominator: int) -> np.ndarray:
    """Return counts for deleting one partition and renormalizing the rest."""
    remaining = proportional.copy()
    remaining[target_index] = 0.0
    return integer_simplex_counts(remaining, denominator)


def unimax_counts(
    tokens: np.ndarray,
    *,
    phase_budget: float,
    max_epochs: float,
    denominator: int,
) -> np.ndarray:
    """Return lattice UniMax counts with an integer-safe epoch cap."""
    if phase_budget <= 0:
        raise ValueError(f"phase_budget must be positive, got {phase_budget}")
    if max_epochs <= 0:
        raise ValueError(f"max_epochs must be positive, got {max_epochs}")
    token_counts = np.asarray(tokens, dtype=float)
    if token_counts.ndim != 1:
        raise ValueError(f"Expected one-dimensional tokens, got shape {token_counts.shape}")
    if np.any(token_counts <= 0):
        raise ValueError("All token counts must be positive")

    n_domains = len(token_counts)
    sorted_indices = np.argsort(token_counts, kind="mergesort")
    caps = np.floor(max_epochs * token_counts * denominator / phase_budget + 1e-12).astype(np.int64)
    if caps.sum() < denominator:
        raise ValueError(
            f"UniMax cap={max_epochs} is infeasible on denominator={denominator}: " f"integer caps sum to {caps.sum()}"
        )

    allocations = np.zeros(n_domains, dtype=float)
    remaining_budget = phase_budget
    remaining_count = n_domains
    capped: set[int] = set()
    for idx in sorted_indices:
        uniform_share = remaining_budget / remaining_count
        max_allocation = max_epochs * token_counts[idx]
        if uniform_share > max_allocation:
            allocations[idx] = max_allocation
            remaining_budget -= max_allocation
            remaining_count -= 1
            capped.add(int(idx))
            continue
        for remaining_idx in sorted_indices:
            if int(remaining_idx) not in capped:
                allocations[remaining_idx] = remaining_budget / remaining_count
        break

    if allocations.sum() <= 0:
        raise ValueError("UniMax produced zero allocation")
    raw_counts = allocations / allocations.sum() * denominator
    counts = np.minimum(np.floor(raw_counts).astype(np.int64), caps)
    remainder = int(denominator - counts.sum())
    fractions = raw_counts - np.floor(raw_counts)
    while remainder > 0:
        eligible = np.flatnonzero(counts < caps)
        if len(eligible) == 0:
            raise ValueError(f"Could not distribute UniMax lattice remainder={remainder}")
        order = eligible[np.argsort(-fractions[eligible], kind="mergesort")]
        for idx in order:
            if remainder == 0:
                break
            counts[idx] += 1
            remainder -= 1
    if counts.sum() != denominator:
        raise ValueError(f"UniMax counts sum to {counts.sum()}, expected {denominator}")
    return counts


def integer_simplex_counts_with_min_count(
    probabilities: np.ndarray,
    *,
    denominator: int,
    min_count: int,
) -> np.ndarray:
    """Project probabilities to integer simplex counts with a per-cell lower bound."""
    if min_count < 0:
        raise ValueError(f"min_count must be nonnegative, got {min_count}")
    probs = np.asarray(probabilities, dtype=float)
    if min_count == 0:
        return integer_simplex_counts(probs, denominator)
    if probs.ndim != 1:
        raise ValueError(f"Expected a one-dimensional probability vector, got shape {probs.shape}")
    if min_count * len(probs) > denominator:
        raise ValueError(f"Cannot assign min_count={min_count} across {len(probs)} cells with denominator={denominator}")
    if np.any(probs < 0):
        raise ValueError("Probabilities must be nonnegative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability vector must have positive mass")

    normalized = probs / total
    remaining_denominator = denominator - min_count * len(probs)
    raw_counts = normalized * remaining_denominator
    counts = np.floor(raw_counts).astype(np.int64) + min_count
    remainder = int(denominator - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - np.floor(raw_counts)), kind="mergesort")
        counts[order[:remainder]] += 1
    if counts.sum() != denominator:
        raise ValueError(f"Integer simplex projection failed: counts sum to {counts.sum()}, expected {denominator}")
    return counts


def integer_simplex_counts_with_upper_bounds(
    probabilities: np.ndarray,
    *,
    denominator: int,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Project probabilities to integer simplex counts without exceeding per-cell caps."""
    probs = np.asarray(probabilities, dtype=float)
    caps = np.asarray(upper_bounds, dtype=np.int64)
    if probs.ndim != 1:
        raise ValueError(f"Expected a one-dimensional probability vector, got shape {probs.shape}")
    if caps.shape != probs.shape:
        raise ValueError(f"upper_bounds shape {caps.shape} does not match probabilities shape {probs.shape}")
    if np.any(caps < 0):
        raise ValueError("upper_bounds must be nonnegative")
    if caps.sum() < denominator:
        raise ValueError(f"upper_bounds sum to {caps.sum()}, below denominator={denominator}")
    if np.any(probs < 0):
        raise ValueError("Probabilities must be nonnegative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability vector must have positive mass")

    normalized = probs / total
    raw_counts = normalized * denominator
    counts = np.minimum(np.floor(raw_counts).astype(np.int64), caps)
    remainder = int(denominator - counts.sum())
    while remainder > 0:
        eligible = np.flatnonzero(counts < caps)
        if len(eligible) == 0:
            raise ValueError(f"Could not distribute bounded simplex remainder={remainder}")
        scores = raw_counts[eligible] - counts[eligible]
        order = eligible[np.argsort(-scores, kind="mergesort")]
        for idx in order:
            counts[idx] += 1
            remainder -= 1
            if remainder == 0:
                break
    if counts.sum() != denominator:
        raise ValueError(f"Bounded simplex projection failed: counts sum to {counts.sum()}, expected {denominator}")
    if np.any(counts > caps):
        raise ValueError("Bounded simplex projection exceeded a cap")
    return counts


def log_odds_tilt_counts(
    proportional: np.ndarray,
    target_index: int,
    *,
    log_odds_alpha: float,
    sign: int,
    denominator: int,
) -> np.ndarray:
    """Return counts for a centered partition log-odds tilt around proportional."""
    if sign not in {-1, 1}:
        raise ValueError(f"sign must be -1 or 1, got {sign}")
    logits = np.zeros_like(proportional)
    logits[target_index] = sign * log_odds_alpha
    tilted = proportional * np.exp(logits - logits.max())
    return integer_simplex_counts(tilted, denominator)


def sample_centered_fisher_sphere_directions(
    proportional: np.ndarray,
    *,
    n_directions: int,
    seed: int,
) -> np.ndarray:
    """Sample centered unit-norm directions on the Fisher/KL tangent sphere."""
    if n_directions <= 0:
        raise ValueError(f"n_directions must be positive, got {n_directions}")
    prop = np.asarray(proportional, dtype=float)
    if prop.ndim != 1:
        raise ValueError(f"Expected a one-dimensional proportional vector, got shape {prop.shape}")
    if np.any(prop <= 0):
        raise ValueError("Proportional weights must be strictly positive")
    prop = prop / prop.sum()
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_directions, len(prop))) / np.sqrt(prop)[None, :]
    centered = raw - (raw * prop[None, :]).sum(axis=1, keepdims=True)
    norms = np.sqrt((centered * centered * prop[None, :]).sum(axis=1))
    if np.any(norms <= 0):
        raise ValueError("Degenerate random Fisher direction")
    return centered / norms[:, None]


def central_logit_tilt_counts(
    proportional: np.ndarray,
    direction: np.ndarray,
    *,
    alpha: float,
    sign: int,
    denominator: int,
) -> np.ndarray:
    """Return counts for a central logit tilt along an intended Fisher direction."""
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if sign not in {-1, 1}:
        raise ValueError(f"sign must be -1 or 1, got {sign}")
    prop = np.asarray(proportional, dtype=float)
    if np.any(prop <= 0):
        raise ValueError("Proportional weights must be strictly positive")
    prop = prop / prop.sum()
    logits = sign * alpha * np.asarray(direction, dtype=float)
    tilted = prop * np.exp(logits - logits.max())
    return integer_simplex_counts_with_min_count(tilted, denominator=denominator, min_count=1)


def realized_logit_direction(
    *,
    proportional: np.ndarray,
    plus_counts: np.ndarray,
    minus_counts: np.ndarray,
    denominator: int,
    alpha: float,
) -> np.ndarray:
    """Return the centered realized logit direction from quantized plus/minus counts."""
    prop = np.asarray(proportional, dtype=float)
    prop = prop / prop.sum()
    plus = np.asarray(plus_counts, dtype=float) / float(denominator)
    minus = np.asarray(minus_counts, dtype=float) / float(denominator)
    if np.any(plus <= 0) or np.any(minus <= 0):
        raise ValueError("Realized logit direction requires strictly positive plus/minus weights")
    realized = (np.log(plus / prop) - np.log(minus / prop)) / (2.0 * alpha)
    return realized - float(np.sum(prop * realized))


def realized_logit_direction_cosine(
    *,
    proportional: np.ndarray,
    intended_direction: np.ndarray,
    plus_counts: np.ndarray,
    minus_counts: np.ndarray,
    denominator: int,
    alpha: float,
) -> float:
    """Return Fisher-inner-product cosine between intended and realized directions."""
    prop = np.asarray(proportional, dtype=float)
    prop = prop / prop.sum()
    intended = np.asarray(intended_direction, dtype=float)
    realized = realized_logit_direction(
        proportional=prop,
        plus_counts=plus_counts,
        minus_counts=minus_counts,
        denominator=denominator,
        alpha=alpha,
    )
    numerator = float(np.sum(prop * intended * realized))
    intended_norm = float(np.sqrt(np.sum(prop * intended * intended)))
    realized_norm = float(np.sqrt(np.sum(prop * realized * realized)))
    if intended_norm <= 0 or realized_norm <= 0:
        raise ValueError("Cannot compute cosine with zero-norm direction")
    return numerator / (intended_norm * realized_norm)


def build_baseline_candidates(
    *,
    buckets: tuple[str, ...],
    tokens: np.ndarray,
    proportional: np.ndarray,
    denominator: int,
    unimax_epoch_caps: tuple[float, ...],
    unimax_phase_budget: float,
) -> list[Candidate]:
    """Build fixed baseline rows before perturbation diagnostics."""
    proportional_counts = integer_simplex_counts(proportional, denominator)
    uniform_counts = integer_simplex_counts(np.ones(len(buckets), dtype=float), denominator)
    candidates = [
        Candidate(
            name="baseline_proportional",
            candidate_type="baseline_proportional",
            counts=np.stack([proportional_counts, proportional_counts]),
            baseline_kind="proportional",
            unimax_epoch_cap=None,
            target_bucket=None,
            target_bucket_index=None,
            intervention_type="baseline",
            tilt_sign="",
            alpha=None,
            direction_id=None,
            direction_index=None,
            intended_direction=None,
        ),
        Candidate(
            name="baseline_uniform",
            candidate_type="baseline_uniform",
            counts=np.stack([uniform_counts, uniform_counts]),
            baseline_kind="uniform",
            unimax_epoch_cap=None,
            target_bucket=None,
            target_bucket_index=None,
            intervention_type="baseline",
            tilt_sign="",
            alpha=None,
            direction_id=None,
            direction_index=None,
            intended_direction=None,
        ),
    ]
    for epoch_cap in unimax_epoch_caps:
        counts = unimax_counts(
            tokens,
            phase_budget=unimax_phase_budget,
            max_epochs=epoch_cap,
            denominator=denominator,
        )
        candidates.append(
            Candidate(
                name=f"baseline_unimax_epoch_cap_{int(epoch_cap):g}",
                candidate_type="baseline_unimax",
                counts=np.stack([counts, counts]),
                baseline_kind="unimax",
                unimax_epoch_cap=epoch_cap,
                target_bucket=None,
                target_bucket_index=None,
                intervention_type="baseline",
                tilt_sign="",
                alpha=None,
                direction_id=None,
                direction_index=None,
                intended_direction=None,
            )
        )
    return candidates


def build_candidates(
    *,
    buckets: tuple[str, ...],
    tokens: np.ndarray,
    proportional: np.ndarray,
    denominator: int,
    random_direction_count: int,
    direction_seed: int,
    alpha: float,
    unimax_epoch_caps: tuple[float, ...] = DEFAULT_UNIMAX_EPOCH_CAPS,
    unimax_phase_budget: float | None = None,
) -> tuple[list[Candidate], np.ndarray]:
    """Build baselines, partition ablations, and paired random central logit tilts."""
    phase_budget = float(PHASE_FRACTIONS[0] * np.asarray(tokens, dtype=float).sum())
    if unimax_phase_budget is not None:
        phase_budget = unimax_phase_budget
    candidates = build_baseline_candidates(
        buckets=buckets,
        tokens=tokens,
        proportional=proportional,
        denominator=denominator,
        unimax_epoch_caps=unimax_epoch_caps,
        unimax_phase_budget=phase_budget,
    )
    for bucket_idx, bucket in enumerate(buckets):
        slug = bucket_slug(bucket)
        ablation = partition_ablation_counts(proportional, bucket_idx, denominator)
        candidates.append(
            Candidate(
                name=f"abl_del_{slug}",
                candidate_type="partition_ablation",
                counts=np.stack([ablation, ablation]),
                baseline_kind=None,
                unimax_epoch_cap=None,
                target_bucket=bucket,
                target_bucket_index=bucket_idx,
                intervention_type="partition_ablation",
                tilt_sign="",
                alpha=None,
                direction_id=None,
                direction_index=None,
                intended_direction=None,
            )
        )
    directions = sample_centered_fisher_sphere_directions(
        proportional,
        n_directions=random_direction_count,
        seed=direction_seed,
    )
    for direction_idx, direction in enumerate(directions):
        direction_id = f"pcdir_{direction_idx:03d}"
        plus_counts = central_logit_tilt_counts(
            proportional,
            direction,
            alpha=alpha,
            sign=1,
            denominator=denominator,
        )
        minus_counts = central_logit_tilt_counts(
            proportional,
            direction,
            alpha=alpha,
            sign=-1,
            denominator=denominator,
        )
        for sign_name, counts in (("plus", plus_counts), ("minus", minus_counts)):
            candidates.append(
                Candidate(
                    name=f"{direction_id}_{sign_name}",
                    candidate_type=f"projected_controllability_{sign_name}",
                    counts=np.stack([counts, counts]),
                    baseline_kind=None,
                    unimax_epoch_cap=None,
                    target_bucket=None,
                    target_bucket_index=None,
                    intervention_type="projected_controllability_random_logit",
                    tilt_sign=sign_name,
                    alpha=alpha,
                    direction_id=direction_id,
                    direction_index=direction_idx,
                    intended_direction=direction.copy(),
                )
            )
    return candidates, directions


def candidate_weights(candidate: Candidate, denominator: int) -> np.ndarray:
    """Return candidate phase weights from integer counts."""
    return candidate.counts.astype(float) / float(denominator)


def write_candidate_csv(path: Path, candidates: list[Candidate], buckets: tuple[str, ...], denominator: int) -> None:
    """Write collaborator-shaped candidate weights."""
    rows: list[dict[str, float | str]] = []
    for candidate in candidates:
        weights = candidate_weights(candidate, denominator)
        row: dict[str, float | str] = {
            "candidate_name": candidate.name,
            "candidate_type": candidate.candidate_type,
        }
        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            for bucket_idx, bucket in enumerate(buckets):
                row[f"{phase_name}/{bucket}"] = float(weights[phase_idx, bucket_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def expected_candidate_columns(buckets: tuple[str, ...]) -> list[str]:
    """Return the collaborator candidate CSV schema."""
    return ["candidate_name", "candidate_type"] + [f"{phase}/{bucket}" for phase in PHASE_NAMES for bucket in buckets]


def quantize_candidate_frame(
    frame: pd.DataFrame,
    *,
    buckets: tuple[str, ...],
    denominator: int,
    tokens: np.ndarray | None = None,
    max_epoch_cap: float | None = None,
    target_budget: float = TARGET_BUDGET,
) -> pd.DataFrame:
    """Quantize every phase of a collaborator-shaped candidate frame to the lattice."""
    expected_columns = expected_candidate_columns(buckets)
    if list(frame.columns) != expected_columns:
        raise ValueError("Candidate frame does not match collaborator schema")
    quantized = frame.copy()
    token_counts = None if tokens is None else np.asarray(tokens, dtype=float)
    if token_counts is not None and token_counts.shape != (len(buckets),):
        raise ValueError(f"tokens shape {token_counts.shape} does not match {len(buckets)} buckets")
    for phase_idx, phase in enumerate(PHASE_NAMES):
        columns = [f"{phase}/{bucket}" for bucket in buckets]
        values = quantized[columns].to_numpy(dtype=float)
        if token_counts is not None and max_epoch_cap is not None:
            phase_budget = float(PHASE_FRACTIONS[phase_idx] * target_budget)
            upper_bounds = np.floor(max_epoch_cap * token_counts * denominator / phase_budget + 1e-12).astype(np.int64)
            quantized_values = np.asarray(
                [
                    integer_simplex_counts_with_upper_bounds(
                        row,
                        denominator=denominator,
                        upper_bounds=upper_bounds,
                    )
                    for row in values
                ],
                dtype=float,
            )
        else:
            quantized_values = np.asarray([integer_simplex_counts(row, denominator) for row in values], dtype=float)
        quantized.loc[:, columns] = quantized_values / float(denominator)
    return quantized


def final_candidate_frame(
    generated: pd.DataFrame,
    d_optimal: pd.DataFrame,
    *,
    buckets: tuple[str, ...],
    denominator: int,
    tokens: np.ndarray | None = None,
    max_epoch_cap: float | None = None,
    target_budget: float = TARGET_BUDGET,
) -> pd.DataFrame:
    """Return baselines/perturbations followed by quantized D-optimal candidates."""
    expected_columns = expected_candidate_columns(buckets)
    if list(generated.columns) != expected_columns:
        raise ValueError("Generated candidate frame does not match collaborator schema")
    d_optimal_quantized = quantize_candidate_frame(
        d_optimal,
        buckets=buckets,
        denominator=denominator,
        tokens=tokens,
        max_epoch_cap=max_epoch_cap,
        target_budget=target_budget,
    )
    final = pd.concat([generated, d_optimal_quantized], ignore_index=True)
    if final["candidate_name"].duplicated().any():
        duplicates = final.loc[final["candidate_name"].duplicated(), "candidate_name"].tolist()
        raise ValueError(f"Duplicate final candidate names: {duplicates[:5]}")
    return final


def final_dashboard_scopes(final_frame: pd.DataFrame, *, generated_row_count: int) -> dict[str, pd.DataFrame]:
    """Return dashboard scopes using the executable rows in the final candidate frame."""
    if generated_row_count < 0 or generated_row_count > len(final_frame):
        raise ValueError(f"generated_row_count={generated_row_count} is outside final frame length {len(final_frame)}")
    return {
        "Full swarm": final_frame,
        "D-optimal only": final_frame.iloc[generated_row_count:].copy(),
    }


def write_manifest_csv(
    path: Path,
    candidates: list[Candidate],
    buckets: tuple[str, ...],
    tokens: np.ndarray,
    proportional_counts: np.ndarray,
    denominator: int,
) -> pd.DataFrame:
    """Write per-candidate diagnostics and return the manifest frame."""
    base = proportional_counts.astype(float) / float(denominator)
    rows: list[dict[str, float | int | str | bool]] = []
    counts_by_direction: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        if candidate.direction_id is None:
            continue
        counts_by_direction.setdefault(candidate.direction_id, {})[candidate.tilt_sign] = candidate.counts[0]
    for candidate in candidates:
        weights = candidate_weights(candidate, denominator)
        epochs = weights * (PHASE_FRACTIONS[:, None] * TARGET_BUDGET / tokens[None, :])
        target_idx = candidate.target_bucket_index
        phase_tv_to_base = np.abs(weights - base[None, :]).sum(axis=1) / 2.0
        if target_idx is None:
            target_counts = np.array([np.nan, np.nan])
            base_count = None
            target_continuous_weight = None
            target_base_weight = None
        else:
            target_counts = candidate.counts[:, target_idx]
            base_count = int(proportional_counts[target_idx])
            target_continuous_weight = float(tokens[target_idx] / tokens.sum())
            target_base_weight = float(base[target_idx])
        realized_cosine = None
        if (
            candidate.direction_id is not None
            and candidate.intended_direction is not None
            and candidate.alpha is not None
        ):
            pair = counts_by_direction[candidate.direction_id]
            realized_cosine = realized_logit_direction_cosine(
                proportional=base,
                intended_direction=candidate.intended_direction,
                plus_counts=pair["plus"],
                minus_counts=pair["minus"],
                denominator=denominator,
                alpha=candidate.alpha,
            )
        rows.append(
            {
                "candidate_name": candidate.name,
                "candidate_type": candidate.candidate_type,
                "intervention_type": candidate.intervention_type,
                "baseline_kind": candidate.baseline_kind,
                "unimax_epoch_cap": candidate.unimax_epoch_cap,
                "target_bucket": candidate.target_bucket,
                "target_bucket_index": target_idx,
                "direction_id": candidate.direction_id,
                "direction_index": candidate.direction_index,
                "tilt_sign": candidate.tilt_sign,
                "alpha": candidate.alpha,
                "intended_direction_fisher_norm": (
                    float(np.sqrt(np.sum(base * candidate.intended_direction * candidate.intended_direction)))
                    if candidate.intended_direction is not None
                    else None
                ),
                "realized_direction_cosine": realized_cosine,
                "denominator": denominator,
                "target_continuous_proportional_weight": target_continuous_weight,
                "target_base_count": base_count,
                "target_phase0_count": int(target_counts[0]) if target_idx is not None else None,
                "target_phase1_count": int(target_counts[1]) if target_idx is not None else None,
                "target_phase0_count_delta": int(target_counts[0] - base_count) if target_idx is not None else None,
                "target_phase1_count_delta": int(target_counts[1] - base_count) if target_idx is not None else None,
                "target_base_weight": target_base_weight,
                "target_phase0_weight": float(weights[0, target_idx]) if target_idx is not None else None,
                "target_phase1_weight": float(weights[1, target_idx]) if target_idx is not None else None,
                "tv_to_base_phase0": float(phase_tv_to_base[0]),
                "tv_to_base_phase1": float(phase_tv_to_base[1]),
                "phase_tv": float(np.abs(weights[0] - weights[1]).sum() / 2.0),
                "support_phase0": int(np.sum(weights[0] > 0)),
                "support_phase1": int(np.sum(weights[1] > 0)),
                "max_weight": float(weights.max()),
                "max_epoch": float(epochs.max()),
                "phase0_max_epoch": float(epochs[0].max()),
                "phase1_max_epoch": float(epochs[1].max()),
                "lattice_valid": bool(
                    np.all(candidate.counts >= 0) and np.all(candidate.counts.sum(axis=1) == denominator)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def write_direction_csv(
    path: Path,
    candidates: list[Candidate],
    buckets: tuple[str, ...],
    proportional_counts: np.ndarray,
    denominator: int,
    alpha: float,
) -> None:
    """Write intended and realized projected-controllability directions."""
    base = proportional_counts.astype(float) / float(denominator)
    by_direction: dict[str, dict[str, Candidate]] = {}
    for candidate in candidates:
        if candidate.direction_id is None:
            continue
        by_direction.setdefault(candidate.direction_id, {})[candidate.tilt_sign] = candidate

    rows: list[dict[str, float | int | str]] = []
    for direction_id, pair in sorted(by_direction.items()):
        plus = pair["plus"]
        minus = pair["minus"]
        if plus.intended_direction is None:
            raise ValueError(f"Missing intended direction for {direction_id}")
        realized = realized_logit_direction(
            proportional=base,
            plus_counts=plus.counts[0],
            minus_counts=minus.counts[0],
            denominator=denominator,
            alpha=alpha,
        )
        row: dict[str, float | int | str] = {
            "direction_id": direction_id,
            "direction_index": int(plus.direction_index if plus.direction_index is not None else -1),
            "alpha": alpha,
            "realized_cosine": realized_logit_direction_cosine(
                proportional=base,
                intended_direction=plus.intended_direction,
                plus_counts=plus.counts[0],
                minus_counts=minus.counts[0],
                denominator=denominator,
                alpha=alpha,
            ),
            "plus_candidate_name": plus.name,
            "minus_candidate_name": minus.name,
            "plus_minus_tv": float(np.abs(plus.counts[0] - minus.counts[0]).sum() / (2.0 * denominator)),
        }
        for bucket_idx, bucket in enumerate(buckets):
            row[f"intended/{bucket}"] = float(plus.intended_direction[bucket_idx])
            row[f"realized/{bucket}"] = float(realized[bucket_idx])
            row[f"plus_weight/{bucket}"] = float(plus.counts[0, bucket_idx] / denominator)
            row[f"minus_weight/{bucket}"] = float(minus.counts[0, bucket_idx] / denominator)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_manifest(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact sanity-check summaries."""
    summaries: dict[str, object] = {
        "row_count": len(frame),
        "candidate_type_counts": frame["candidate_type"].value_counts().sort_index().to_dict(),
        "all_lattice_valid": bool(frame["lattice_valid"].all()),
        "max_phase_tv": float(frame["phase_tv"].max()),
    }
    by_type: dict[str, object] = {}
    for candidate_type, group in frame.groupby("candidate_type", sort=True):
        by_type[candidate_type] = {
            "n": len(group),
            "tv_to_base_phase0_min": float(group["tv_to_base_phase0"].min()),
            "tv_to_base_phase0_median": float(group["tv_to_base_phase0"].median()),
            "tv_to_base_phase0_q95": float(group["tv_to_base_phase0"].quantile(0.95)),
            "tv_to_base_phase0_max": float(group["tv_to_base_phase0"].max()),
            "max_epoch_median": float(group["max_epoch"].median()),
            "max_epoch_q95": float(group["max_epoch"].quantile(0.95)),
            "max_epoch_max": float(group["max_epoch"].max()),
            "max_weight_max": float(group["max_weight"].max()),
            "support_phase0_min": int(group["support_phase0"].min()),
            "support_phase1_min": int(group["support_phase1"].min()),
        }
    summaries["by_type"] = by_type
    return summaries


def validate_outputs(candidate_csv: Path, manifest: pd.DataFrame, buckets: tuple[str, ...], denominator: int) -> None:
    """Validate handoff shape and lattice invariants."""
    frame = pd.read_csv(candidate_csv)
    expected_columns = expected_candidate_columns(buckets)
    if list(frame.columns) != expected_columns:
        raise ValueError("Candidate CSV does not match collaborator schema")
    if len(frame) != len(manifest):
        raise ValueError(f"Candidate/manifest row mismatch: {len(frame)} != {len(manifest)}")
    if frame["candidate_name"].duplicated().any():
        raise ValueError("Candidate names must be unique")
    for phase in PHASE_NAMES:
        values = frame[[f"{phase}/{bucket}" for bucket in buckets]].to_numpy(dtype=float)
        if np.any(values < 0):
            raise ValueError(f"Negative weights in {phase}")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError(f"{phase} weights do not sum to one")
        if not np.allclose(values * denominator, np.round(values * denominator), atol=1e-9):
            raise ValueError(f"{phase} contains non-lattice weights")
    if not manifest["lattice_valid"].all():
        raise ValueError("Manifest contains invalid lattice rows")
    if manifest["phase_tv"].max() != 0.0:
        raise ValueError("Baseline perturbations should use identical phase_0 and phase_1 weights")


def validate_candidate_frame(frame: pd.DataFrame, buckets: tuple[str, ...], denominator: int) -> None:
    """Validate an in-memory collaborator-shaped candidate frame."""
    expected_columns = expected_candidate_columns(buckets)
    if list(frame.columns) != expected_columns:
        raise ValueError("Candidate frame does not match collaborator schema")
    if frame["candidate_name"].duplicated().any():
        raise ValueError("Candidate names must be unique")
    for phase in PHASE_NAMES:
        values = frame[[f"{phase}/{bucket}" for bucket in buckets]].to_numpy(dtype=float)
        if np.any(values < 0):
            raise ValueError(f"Negative weights in {phase}")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError(f"{phase} weights do not sum to one")
        if not np.allclose(values * denominator, np.round(values * denominator), atol=1e-9):
            raise ValueError(f"{phase} contains non-lattice weights")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--denominator", type=int, default=MIXTURE_QUANTUM_DENOMINATOR)
    parser.add_argument("--random-direction-count", type=int, default=DEFAULT_RANDOM_DIRECTION_COUNT)
    parser.add_argument("--direction-seed", type=int, default=DEFAULT_DIRECTION_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--d-optimal-candidate-csv", type=Path, default=DEFAULT_D_OPTIMAL_CANDIDATE_CSV)
    parser.add_argument("--d-optimal-max-epoch-cap", type=float, default=DEFAULT_D_OPTIMAL_MAX_EPOCH_CAP)
    parser.add_argument("--no-d-optimal", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate perturbation rows and sanity-check artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Loading bucket table from {args.source_url}")
    bucket_table = load_bucket_table(args.source_url)
    continuous_proportional = bucket_table.tokens / bucket_table.tokens.sum()
    proportional_counts = integer_simplex_counts(continuous_proportional, args.denominator)
    proportional = proportional_counts.astype(float) / float(args.denominator)
    progress(
        f"Building {2 + len(DEFAULT_UNIMAX_EPOCH_CAPS)} baselines, {len(bucket_table.buckets)} partition ablations, and "
        f"{2 * args.random_direction_count} projected-controllability random logit tilts "
        f"on denominator {args.denominator}"
    )
    candidates, directions = build_candidates(
        buckets=bucket_table.buckets,
        tokens=bucket_table.tokens,
        proportional=proportional,
        denominator=args.denominator,
        random_direction_count=args.random_direction_count,
        direction_seed=args.direction_seed,
        alpha=args.alpha,
        unimax_phase_budget=float(PHASE_FRACTIONS[0] * TARGET_BUDGET),
    )
    candidate_csv = args.output_dir / "production_swarm_167p_baseline_perturbation_candidate_mixtures.csv"
    final_candidate_csv = args.output_dir / "production_swarm_167p_final_candidate_mixtures.csv"
    manifest_csv = args.output_dir / "production_swarm_167p_baseline_perturbation_manifest.csv"
    directions_csv = args.output_dir / "production_swarm_167p_projected_controllability_directions.csv"
    summary_json = args.output_dir / "summary.json"
    write_candidate_csv(candidate_csv, candidates, bucket_table.buckets, args.denominator)
    manifest = write_manifest_csv(
        manifest_csv,
        candidates,
        bucket_table.buckets,
        bucket_table.tokens,
        proportional_counts,
        args.denominator,
    )
    write_direction_csv(
        directions_csv,
        candidates,
        bucket_table.buckets,
        proportional_counts,
        args.denominator,
        args.alpha,
    )
    validate_outputs(candidate_csv, manifest, bucket_table.buckets, args.denominator)
    final_row_count = None
    if not args.no_d_optimal:
        if not args.d_optimal_candidate_csv.exists():
            raise FileNotFoundError(
                f"D-optimal candidate CSV not found: {args.d_optimal_candidate_csv}. "
                "Pass --no-d-optimal to emit only baselines and perturbations."
            )
        d_optimal_frame = pd.read_csv(args.d_optimal_candidate_csv)
        generated_frame = pd.read_csv(candidate_csv)
        final_frame = final_candidate_frame(
            generated_frame,
            d_optimal_frame,
            buckets=bucket_table.buckets,
            denominator=args.denominator,
            tokens=bucket_table.tokens,
            max_epoch_cap=args.d_optimal_max_epoch_cap,
            target_budget=TARGET_BUDGET,
        )
        validate_candidate_frame(final_frame, bucket_table.buckets, args.denominator)
        final_frame.to_csv(final_candidate_csv, index=False)
        write_interactive_sanity_dashboard(
            args.output_dir / "production_swarm_167p_final_sanity_dashboard.html",
            scopes=final_dashboard_scopes(final_frame, generated_row_count=len(generated_frame)),
            buckets=bucket_table.buckets,
            tokens=bucket_table.tokens,
            proportional=proportional,
            denominator=args.denominator,
            title="Production Swarm 167p Final Sanity Dashboard",
            metadata={
                "baseline_and_perturbation_rows": len(generated_frame),
                "d_optimal_rows": len(d_optimal_frame),
                "final_rows": len(final_frame),
                "denominator": args.denominator,
                "d_optimal_max_epoch_cap": args.d_optimal_max_epoch_cap,
                "d_optimal_candidate_csv": str(args.d_optimal_candidate_csv),
            },
        )
        final_row_count = len(final_frame)
    summary = {
        "source_url": args.source_url,
        "source_sha256": bucket_table.source_sha256,
        "n_partitions": len(bucket_table.buckets),
        "total_tokens": int(bucket_table.tokens.sum()),
        "target_budget": TARGET_BUDGET,
        "phase_fractions": PHASE_FRACTIONS.tolist(),
        "denominator": args.denominator,
        "quantum": 1.0 / args.denominator,
        "random_direction_count": args.random_direction_count,
        "direction_seed": args.direction_seed,
        "alpha": args.alpha,
        "intended_direction_shape": list(directions.shape),
        "unimax_epoch_caps": list(DEFAULT_UNIMAX_EPOCH_CAPS),
        "unimax_phase_budget": float(PHASE_FRACTIONS[0] * TARGET_BUDGET),
        "d_optimal_candidate_csv": None if args.no_d_optimal else str(args.d_optimal_candidate_csv),
        "d_optimal_max_epoch_cap": None if args.no_d_optimal else args.d_optimal_max_epoch_cap,
        "final_candidate_csv": None if args.no_d_optimal else str(final_candidate_csv),
        "final_row_count": final_row_count,
        "proportional_lattice_l1_error": float(np.abs(proportional - continuous_proportional).sum()),
        "sanity": summarize_manifest(manifest),
        "candidate_csv": str(candidate_csv),
        "manifest_csv": str(manifest_csv),
        "directions_csv": str(directions_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
