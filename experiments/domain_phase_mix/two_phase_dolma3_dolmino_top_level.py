# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-phase nextgen swarm over top-level Dolma 3 Pool and Dolmino Pool domains."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache, partial
import os

from levanter.optim import MuonHConfig
from fray.cluster import ResourceConfig

from experiments.domain_phase_mix.config import DatasetComponent, Domain, PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_PARTITIONS,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
    all_top_level_domain_names,
)
from experiments.domain_phase_mix.experiment import (
    DEFAULT_MUON_CONFIG,
    InitialFixedWeightConfig,
    MixtureExperiment,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.weight_sampler import (
    DirichletSamplingParams,
    SamplingStrategy,
    compute_unimax_weights,
)
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_PRO_5_SHOT
from experiments.marin_models import marin_tokenizer
from experiments.pretraining_datasets.dolma3_dolmino_pool import tokenize_dolmino_pool_subset
from experiments.pretraining_datasets.dolma3_pool import tokenize_dolma3_pool_subset
from marin.processing.tokenize import merge_tokenized_caches

NAME = "pinlin_calvin_xu/data_mixture/two_phase_dolma3_dolmino_top_level"
EXPERIMENT_BUDGET = 1_200_000_000
TARGET_BUDGET = TARGET_BUDGET_DOLMA3_COMMON_CRAWL
BATCH_SIZE = 128
SEQ_LEN = 2048
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN
NUM_TRAIN_STEPS = EXPERIMENT_BUDGET // TOKENS_PER_STEP
REALIZED_EXPERIMENT_BUDGET = NUM_TRAIN_STEPS * TOKENS_PER_STEP
PHASE_BOUNDARIES = [0.5]
PHASE_NAMES = ("phase_0", "phase_1")
DOMAIN_NAMES = tuple(all_top_level_domain_names())
EVAL_TASKS = (MMLU_5_SHOT, MMLU_PRO_5_SHOT)
EVAL_DATASETS_CACHE_PATH: str | None = None

SAMPLING_PARAMS = DirichletSamplingParams(
    strategy=SamplingStrategy.DIRICHLET,
    temp=0.5,
    min_strength=0.1,
    max_strength=0.5,
    min_weight=2e-4,
    min_config_distance=0.001,
)

WSD_BOUNDARY_ALIGNED_WARMUP = 0.01
WSD_BOUNDARY_ALIGNED_REWARMUP = 0.0

assert len(DOMAIN_NAMES) == 31, f"Expected 31 top-level domains, got {len(DOMAIN_NAMES)}"
assert TARGET_BUDGET == 6_325_183_647_689, TARGET_BUDGET
assert TOP_LEVEL_TOTAL_AVAILABLE_TOKENS == 8_813_462_126_096, TOP_LEVEL_TOTAL_AVAILABLE_TOKENS
assert REALIZED_EXPERIMENT_BUDGET == 1_199_833_088, REALIZED_EXPERIMENT_BUDGET


@dataclass(frozen=True)
class TwoPhaseWsdBoundarySchedule:
    """Resolved single-cycle WSD schedule aligned to the phase boundary."""

    total_steps: int
    boundary_step: int
    warmup_steps: int
    decay_steps: int


def _partition_step_fn(partition_name: str):
    if partition_name in TOP_LEVEL_DOMAIN_PARTITIONS:
        raise ValueError(f"Expected a raw partition name, got coarse domain {partition_name}")
    if partition_name.startswith(("common_crawl/", "stack_edu/")) or partition_name in {
        "arxiv",
        "finemath_3plus",
        "wikipedia",
    }:
        return partial(tokenize_dolma3_pool_subset, partition_name, tokenizer=marin_tokenizer)
    return partial(tokenize_dolmino_pool_subset, partition_name, tokenizer=marin_tokenizer)


@cache
def _top_level_domain_step(domain_name: str):
    if domain_name not in TOP_LEVEL_DOMAIN_PARTITIONS:
        raise ValueError(f"Unknown top-level domain {domain_name}")

    input_steps = {
        partition_name: _partition_step_fn(partition_name)()
        for partition_name in TOP_LEVEL_DOMAIN_PARTITIONS[domain_name]
    }
    return merge_tokenized_caches(
        os.path.join("dolma3_dolmino_top_level", domain_name),
        input_steps,
        tokenizer=marin_tokenizer,
        tags=["top_level_domain", domain_name],
    )


def build_top_level_domains() -> list[Domain]:
    """Build the 31 top-level domains as one merged cache each."""
    domains: list[Domain] = []
    for domain_name in DOMAIN_NAMES:
        components = [
            DatasetComponent(
                name=domain_name,
                step_fn=partial(_top_level_domain_step, domain_name),
                weight=TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name],
            )
        ]
        domains.append(
            Domain(
                name=domain_name,
                components=components,
                description=(
                    "Top-level domain backed by one merged tokenized cache assembled from "
                    f"{len(TOP_LEVEL_DOMAIN_PARTITIONS[domain_name])} partition caches."
                ),
            )
        )
    return domains


def build_top_level_domain_steps() -> dict[str, object]:
    """Return one merged-cache preparation step per top-level domain."""
    return {domain_name: _top_level_domain_step(domain_name) for domain_name in DOMAIN_NAMES}


def _constant_phase_weights(domain_weights: dict[str, float]) -> dict[str, dict[str, float]]:
    return {phase_name: dict(domain_weights) for phase_name in PHASE_NAMES}


def create_initial_fixed_weight_configs() -> tuple[InitialFixedWeightConfig, ...]:
    """Create the fixed proportional and UniMax baselines for an empty-state loop."""
    total_tokens = float(TOP_LEVEL_TOTAL_AVAILABLE_TOKENS)
    proportional = {
        domain_name: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] / total_tokens for domain_name in DOMAIN_NAMES
    }

    unimax_weights = compute_unimax_weights(
        [TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] for domain_name in DOMAIN_NAMES],
        budget=TARGET_BUDGET * PHASE_BOUNDARIES[0],
        max_epochs=1.0,
    )
    unimax = {domain_name: float(weight) for domain_name, weight in zip(DOMAIN_NAMES, unimax_weights, strict=True)}

    return (
        InitialFixedWeightConfig(
            run_name="baseline_proportional",
            weight_config=WeightConfig(run_id=0, phase_weights=_constant_phase_weights(proportional)),
        ),
        InitialFixedWeightConfig(
            run_name="baseline_unimax",
            weight_config=WeightConfig(run_id=1, phase_weights=_constant_phase_weights(unimax)),
        ),
    )


def resolve_two_phase_wsd_boundary_schedule(
    *,
    experiment_budget: int = EXPERIMENT_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    phase_schedule: PhaseSchedule | None = None,
    warmup_fraction: float = WSD_BOUNDARY_ALIGNED_WARMUP,
    mixture_block_size: int = 2048,
) -> TwoPhaseWsdBoundarySchedule:
    """Resolve the boundary-aligned single-cycle WSD schedule."""
    schedule = phase_schedule or PhaseSchedule.from_boundaries(boundaries=PHASE_BOUNDARIES, names=list(PHASE_NAMES))
    total_steps = experiment_budget // (batch_size * seq_len)
    boundary_step = schedule.phases[1].get_start_step_aligned(total_steps, batch_size, mixture_block_size)
    warmup_steps = int(total_steps * warmup_fraction)
    decay_steps = total_steps - boundary_step

    if decay_steps <= 0:
        raise ValueError(f"Invalid WSD decay length: total_steps={total_steps}, boundary_step={boundary_step}")
    if warmup_steps >= boundary_step:
        raise ValueError(
            "Warmup must end before the phase boundary for boundary-aligned WSD: "
            f"warmup_steps={warmup_steps}, boundary_step={boundary_step}"
        )

    return TwoPhaseWsdBoundarySchedule(
        total_steps=total_steps,
        boundary_step=boundary_step,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )


def create_two_phase_dolma3_dolmino_top_level_optimizer_config(
    *,
    experiment_budget: int = EXPERIMENT_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    phase_schedule: PhaseSchedule | None = None,
) -> MuonHConfig:
    """Create the boundary-aligned WSD optimizer config for this topology."""
    schedule = resolve_two_phase_wsd_boundary_schedule(
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        phase_schedule=phase_schedule,
    )
    return replace(
        DEFAULT_MUON_CONFIG,
        warmup=schedule.warmup_steps,
        decay=schedule.decay_steps,
        rewarmup=WSD_BOUNDARY_ALIGNED_REWARMUP,
        lr_schedule="cosine",
        cycles=None,
        cycle_length=None,
        haps=None,
    )


def create_two_phase_dolma3_dolmino_top_level_experiment(
    *,
    name: str = NAME,
    experiment_budget: int = EXPERIMENT_BUDGET,
    target_budget: int = TARGET_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    eval_datasets_cache_path: str | None = EVAL_DATASETS_CACHE_PATH,
    resources: ResourceConfig | None = None,
) -> MixtureExperiment:
    """Create the top-level Dolma 3 + Dolmino two-phase nextgen experiment."""
    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=PHASE_BOUNDARIES,
        names=list(PHASE_NAMES),
    )
    optimizer_config = create_two_phase_dolma3_dolmino_top_level_optimizer_config(
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        phase_schedule=phase_schedule,
    )
    return MixtureExperiment(
        name=name,
        domains=build_top_level_domains(),
        phase_schedule=phase_schedule,
        model_config=regmix_60m_proxy,
        batch_size=batch_size,
        seq_len=seq_len,
        num_train_steps=experiment_budget // (batch_size * seq_len),
        target_budget=target_budget,
        resources=resources or ResourceConfig.with_tpu("v5p-8"),
        eval_harness_tasks=EVAL_TASKS,
        sampling_params=SAMPLING_PARAMS,
        optimizer_config=optimizer_config,
        eval_datasets_cache_path=eval_datasets_cache_path,
        initial_fixed_weight_configs=create_initial_fixed_weight_configs(),
    )
