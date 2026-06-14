# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Auxiliary perturbation rows for the Grug-MoE Fisher-DSP production swarm.

The live production swarm in ``swarm_fisher_dsp.py`` trains the 840 D-optimal
rows. This file appends diagnostic rows using the same model, optimizer, phase
split, output naming convention, W&B group, central2 datakit components, and
v4-8 resources:

* proportional, uniform, and UniMax baselines;
* one delete-and-renormalize ablation per live mixture bucket;
* paired random central logit tilts around proportional for projected
  controllability estimates.

Rows are quantized to the same 1/65536 lattice used by the production-swarm
mixture CSV. The training loader still uses ``_SWARM_BLOCK_SIZE=32768`` from
``swarm_fisher_dsp.py``; matching the original lattice keeps the generated
weights auditable while preserving the live executor semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.types import this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.datakit_moe_mix import COMPONENTS, PROPORTIONAL_WEIGHTS, TARGET_BUDGET, _TOKEN_COUNTS
from experiments.grug.moe.grug_moe_mix import run_grug_moe_mix
from experiments.grug.moe.swarm_fisher_dsp import (
    _BATCH,
    _BUDGET,
    _CANDIDATES as _PRODUCTION_SWARM_CANDIDATES,
    _EXPERIMENT_BUDGET,
    _HIDDEN_DIM,
    _MODEL,
    _OPTIMIZER,
    _PHASE1_START_STEP,
    _STEPS,
    _SWARM_BLOCK_SIZE,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.marin_models import marin_tokenizer

AUXILIARY_INDEX_START = len(_PRODUCTION_SWARM_CANDIDATES)
MIXTURE_QUANTUM_DENOMINATOR = 65536
RANDOM_DIRECTION_COUNT = 64
RANDOM_DIRECTION_SEED = 44
LOGIT_TILT_ALPHA = 0.10
UNIMAX_EPOCH_CAPS: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0)
PHASE0_FRACTION = 0.8
PHASE1_FRACTION = 0.2
MAX_CONCURRENT_AUXILIARY_STEPS = 240

_SWARM_WANDB_GROUP = "swarm_fisher_dsp_tau20_lam0p25_uscentral2"


@dataclass(frozen=True)
class AuxiliaryCandidate:
    """One central2 auxiliary mixture row."""

    index: int
    candidate_name: str
    candidate_type: str
    phase_0: dict[str, float]
    phase_1: dict[str, float]


def _integer_simplex_counts(probabilities: np.ndarray, denominator: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError(f"Expected one-dimensional probabilities, got {probs.shape}")
    if np.any(probs < 0):
        raise ValueError("Probabilities must be nonnegative")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Probabilities must have positive mass")

    normalized = probs / total
    raw_counts = normalized * denominator
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(denominator - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - np.floor(raw_counts)), kind="mergesort")
        counts[order[:remainder]] += 1
    if int(counts.sum()) != denominator:
        raise ValueError(f"Counts sum to {counts.sum()}, expected {denominator}")
    return counts


def _integer_simplex_counts_with_min_count(
    probabilities: np.ndarray,
    *,
    denominator: int,
    min_count: int,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if min_count < 0:
        raise ValueError(f"min_count must be nonnegative, got {min_count}")
    if min_count == 0:
        return _integer_simplex_counts(probs, denominator)
    if min_count * len(probs) > denominator:
        raise ValueError(f"min_count={min_count} infeasible for {len(probs)} cells and denominator={denominator}")

    counts = _integer_simplex_counts(probs, denominator - min_count * len(probs))
    counts = counts + min_count
    if int(counts.sum()) != denominator:
        raise ValueError(f"Counts sum to {counts.sum()}, expected {denominator}")
    return counts


def _weights_from_counts(buckets: tuple[str, ...], counts: np.ndarray) -> dict[str, float]:
    return {bucket: float(count) / MIXTURE_QUANTUM_DENOMINATOR for bucket, count in zip(buckets, counts, strict=True)}


def _delete_and_renormalize_counts(proportional: np.ndarray, target_index: int) -> np.ndarray:
    weights = proportional.copy()
    weights[target_index] = 0.0
    return _integer_simplex_counts(weights, MIXTURE_QUANTUM_DENOMINATOR)


def _unimax_weights(tokens: np.ndarray, *, phase_budget: float, epoch_cap: float) -> np.ndarray:
    if phase_budget <= 0:
        raise ValueError(f"phase_budget must be positive, got {phase_budget}")
    if epoch_cap <= 0:
        raise ValueError(f"epoch_cap must be positive, got {epoch_cap}")

    allocations = np.zeros_like(tokens, dtype=float)
    remaining_budget = float(phase_budget)
    remaining_count = len(tokens)
    sorted_indices = np.argsort(tokens, kind="mergesort")
    capped: set[int] = set()
    for idx in sorted_indices:
        uniform_share = remaining_budget / remaining_count
        cap = epoch_cap * float(tokens[idx])
        if uniform_share > cap:
            allocations[idx] = cap
            remaining_budget -= cap
            remaining_count -= 1
            capped.add(int(idx))
            continue
        for remaining_idx in sorted_indices:
            if int(remaining_idx) not in capped:
                allocations[remaining_idx] = uniform_share
        break
    if float(allocations.sum()) <= 0:
        raise ValueError("UniMax produced zero allocation")
    return allocations / allocations.sum()


def _sample_centered_fisher_directions(proportional: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_DIRECTION_SEED)
    raw = rng.normal(size=(RANDOM_DIRECTION_COUNT, len(proportional))) / np.sqrt(proportional)[None, :]
    centered = raw - (raw * proportional[None, :]).sum(axis=1, keepdims=True)
    norms = np.sqrt((centered * centered * proportional[None, :]).sum(axis=1))
    if np.any(norms <= 0):
        raise ValueError("Degenerate random Fisher direction")
    return centered / norms[:, None]


def _central_logit_tilt_counts(proportional: np.ndarray, direction: np.ndarray, *, sign: int) -> np.ndarray:
    if sign not in {-1, 1}:
        raise ValueError(f"sign must be -1 or 1, got {sign}")
    logits = sign * LOGIT_TILT_ALPHA * np.asarray(direction, dtype=float)
    tilted = proportional * np.exp(logits - logits.max())
    # Count 1 preserves the 1/65536 lattice support in the manifest. The live
    # loader's 32768 block size can still drop count-1 cells at materialization;
    # this matches the production swarm's documented lattice/block-size tradeoff.
    return _integer_simplex_counts_with_min_count(
        tilted,
        denominator=MIXTURE_QUANTUM_DENOMINATOR,
        min_count=1,
    )


def build_auxiliary_candidates() -> list[AuxiliaryCandidate]:
    """Build all auxiliary rows in stable training order."""
    buckets = tuple(COMPONENTS)
    tokens = np.asarray([_TOKEN_COUNTS[bucket] for bucket in buckets], dtype=float)
    proportional = np.asarray([PROPORTIONAL_WEIGHTS[bucket] for bucket in buckets], dtype=float)
    proportional = proportional / proportional.sum()

    candidates: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    proportional_counts = _integer_simplex_counts(proportional, MIXTURE_QUANTUM_DENOMINATOR)
    uniform_counts = _integer_simplex_counts(np.ones(len(buckets), dtype=float), MIXTURE_QUANTUM_DENOMINATOR)
    candidates.append(("baseline_proportional", "baseline_proportional", proportional_counts, proportional_counts))
    candidates.append(("baseline_uniform", "baseline_uniform", uniform_counts, uniform_counts))

    for epoch_cap in UNIMAX_EPOCH_CAPS:
        w0 = _unimax_weights(tokens, phase_budget=TARGET_BUDGET * PHASE0_FRACTION, epoch_cap=epoch_cap)
        w1 = _unimax_weights(tokens, phase_budget=TARGET_BUDGET * PHASE1_FRACTION, epoch_cap=epoch_cap)
        candidates.append(
            (
                f"baseline_unimax_epoch_cap_{int(epoch_cap):g}",
                "baseline_unimax",
                _integer_simplex_counts(w0, MIXTURE_QUANTUM_DENOMINATOR),
                _integer_simplex_counts(w1, MIXTURE_QUANTUM_DENOMINATOR),
            )
        )

    for bucket_idx, bucket in enumerate(buckets):
        counts = _delete_and_renormalize_counts(proportional, bucket_idx)
        candidates.append((f"abl_del_{bucket}", "partition_ablation", counts, counts))

    directions = _sample_centered_fisher_directions(proportional)
    for direction_idx, direction in enumerate(directions):
        plus_counts = _central_logit_tilt_counts(proportional, direction, sign=1)
        minus_counts = _central_logit_tilt_counts(proportional, direction, sign=-1)
        candidates.append(
            (
                f"pcdir_{direction_idx:03d}_plus",
                "projected_controllability_plus",
                plus_counts,
                plus_counts,
            )
        )
        candidates.append(
            (
                f"pcdir_{direction_idx:03d}_minus",
                "projected_controllability_minus",
                minus_counts,
                minus_counts,
            )
        )

    out: list[AuxiliaryCandidate] = []
    for offset, (candidate_name, candidate_type, phase0_counts, phase1_counts) in enumerate(candidates):
        out.append(
            AuxiliaryCandidate(
                index=AUXILIARY_INDEX_START + offset,
                candidate_name=candidate_name,
                candidate_type=candidate_type,
                phase_0=_weights_from_counts(buckets, phase0_counts),
                phase_1=_weights_from_counts(buckets, phase1_counts),
            )
        )
    return out


def _build_step(candidate: AuxiliaryCandidate) -> ExecutorStep:
    mixture_schedule: list[tuple[int, dict[str, float]]] = [
        (0, candidate.phase_0),
        (_PHASE1_START_STEP, candidate.phase_1),
    ]

    base_mixture = LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=COMPONENTS,
        train_weights=mixture_schedule,
        auto_build_caches=False,
        mixture_block_size=_SWARM_BLOCK_SIZE,
        target_budget=TARGET_BUDGET,
        experiment_budget=_EXPERIMENT_BUDGET,
    )
    data = add_validation_sets_to_mixture(base_mixture, default_validation_sets(tokenizer=marin_tokenizer))

    slug = f"d{_HIDDEN_DIM}_{candidate.index:06d}"
    return ExecutorStep(
        name=f"grug/swarm_fisher_dsp_{slug}",
        fn=run_grug_moe_mix,
        config=GrugMoeLaunchConfig(
            model=versioned(_MODEL),
            data=data,
            output_path=this_output_path(),
            run_id=f"swarm_fisher_dsp_{slug}",
            resources=versioned(ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=False)),
            steps=versioned(_STEPS),
            batch_size=versioned(_BATCH),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=[
                    "moe",
                    "swarm",
                    "fisher_dsp",
                    "tau20_lam0p25",
                    "auxiliary_perturbation",
                    candidate.candidate_type,
                    "uscentral2",
                    slug,
                ],
                group=_SWARM_WANDB_GROUP,
                name=None,
            ),
            optimizer=versioned(_OPTIMIZER),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=_BATCH,
                    steps_per_eval=1000,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


AUXILIARY_CANDIDATES: list[AuxiliaryCandidate] = build_auxiliary_candidates()
swarm_auxiliary_perturbation_steps: list[ExecutorStep] = [_build_step(c) for c in AUXILIARY_CANDIDATES]


if __name__ == "__main__":
    executor_main(
        steps=swarm_auxiliary_perturbation_steps,
        description=(
            "Grug MoE production-swarm auxiliary perturbations: baselines, one partition ablation per "
            "central2 datakit bucket, and paired random central logit tilts at D512 on ~100B tokens."
        ),
        max_concurrent=MAX_CONCURRENT_AUXILIARY_STEPS,
    )
