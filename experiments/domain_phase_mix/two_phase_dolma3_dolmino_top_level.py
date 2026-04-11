# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-phase nextgen swarm over top-level Dolma 3 Pool and Dolmino Pool domains."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache, partial

from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig
from fray.cluster import ResourceConfig

from experiments.domain_phase_mix.config import DatasetComponent, Domain, PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_PARTITIONS,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
    all_top_level_domain_names,
    top_level_domain_partition_counts,
)
from experiments.domain_phase_mix.experiment import (
    DEFAULT_MUON_CONFIG,
    InitialFixedWeightConfig,
    MixtureExperiment,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_PHASE_WEIGHTS,
    OLMIX_LOGLINEAR_RUN_NAME,
    create_olmix_loglinear_weight_config,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.weight_sampler import (
    DirichletSamplingParams,
    SamplingStrategy,
    compute_unimax_weights,
)
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_PRO_5_SHOT, MMLU_SL_VERB_5_SHOT
from marin.evaluation.evaluation_config import EvalTaskConfig
from experiments.marin_models import marin_tokenizer
from experiments.pretraining_datasets.dolma3_dolmino_pool import tokenize_dolmino_pool_subset
from experiments.pretraining_datasets.dolma3_pool import tokenize_dolma3_pool_subset
from marin.processing.tokenize.data_configs import ExistingTokenizedCacheConfig
from marin.processing.tokenize.merge_tokenized_caches import merge_tokenized_caches

NAME = "pinlin_calvin_xu/data_mixture/two_phase_dolma3_dolmino_top_level"
EXPERIMENT_BUDGET = 1_200_000_000
TARGET_BUDGET = TARGET_BUDGET_DOLMA3_COMMON_CRAWL
BATCH_SIZE = 128
SEQ_LEN = 2048
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN
NUM_TRAIN_STEPS = EXPERIMENT_BUDGET // TOKENS_PER_STEP
REALIZED_EXPERIMENT_BUDGET = NUM_TRAIN_STEPS * TOKENS_PER_STEP
PHASE_BOUNDARIES = [0.8]
PHASE_NAMES = ("phase_0", "phase_1")
DOMAIN_NAMES = tuple(all_top_level_domain_names())
EVAL_TASKS = (MMLU_5_SHOT, MMLU_SL_VERB_5_SHOT, MMLU_PRO_5_SHOT)
EVAL_DATASETS_CACHE_PATH: str | None = None
INITIAL_BASELINE_RUNS = 3
STRATIFIED_RUN_ID = 3
STRATIFIED_RUN_NAME = "baseline_stratified"
MIN_RECOMMENDED_SWARM_RUNS = 6 * len(DOMAIN_NAMES)
MIN_RECOMMENDED_SAMPLED_RUNS = MIN_RECOMMENDED_SWARM_RUNS - INITIAL_BASELINE_RUNS
DEFAULT_RUNTIME_CACHE_REGION = "us-east5"
MERGED_CC_DOMAIN_NAMES = tuple(name for name in DOMAIN_NAMES if name.startswith("dolma3_cc/"))
PREFERRED_MERGED_RUNTIME_DOMAIN_NAMES = (
    *MERGED_CC_DOMAIN_NAMES,
    "dolma3_stack_edu",
    "dolmino_stem_heavy_crawl",
)
PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION = {
    "us-central1": {
        "dolma3_stack_edu": (
            "gs://marin-us-central1/tokenized/merged/dolma3_dolmino_top_level/" "dolma3_stack_edu-a7297b"
        ),
        "dolmino_stem_heavy_crawl": (
            "gs://marin-us-central1/tokenized/merged/dolma3_dolmino_top_level/" "dolmino_stem_heavy_crawl-e1ec3b"
        ),
    },
    "us-east5": {
        "dolma3_stack_edu": "gs://marin-us-east5/tokenized/merged/dolma3_dolmino_top_level/" "dolma3_stack_edu-a7297b",
        "dolmino_stem_heavy_crawl": (
            "gs://marin-us-east5/tokenized/merged/dolma3_dolmino_top_level/" "dolmino_stem_heavy_crawl-e1ec3b"
        ),
    },
}

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

assert len(DOMAIN_NAMES) == 39, f"Expected 39 top-level domains, got {len(DOMAIN_NAMES)}"
assert len(MERGED_CC_DOMAIN_NAMES) == 26, f"Expected 26 merged CC domains, got {len(MERGED_CC_DOMAIN_NAMES)}"
assert TARGET_BUDGET == 6_325_183_647_689, TARGET_BUDGET
assert TOP_LEVEL_TOTAL_AVAILABLE_TOKENS == 6_986_431_605_135, TOP_LEVEL_TOTAL_AVAILABLE_TOKENS
assert REALIZED_EXPERIMENT_BUDGET == 1_199_833_088, REALIZED_EXPERIMENT_BUDGET
assert MIN_RECOMMENDED_SWARM_RUNS == 234, MIN_RECOMMENDED_SWARM_RUNS
assert MIN_RECOMMENDED_SAMPLED_RUNS == 231, MIN_RECOMMENDED_SAMPLED_RUNS


def _resolved_runtime_cache_region(
    *, runtime_cache_region: str | None = None, resources: ResourceConfig | None = None
) -> str:
    if runtime_cache_region is not None:
        return runtime_cache_region
    if resources is not None:
        if resources.zone:
            return resources.zone.rsplit("-", 1)[0]
        if resources.regions:
            return resources.regions[0]
    return DEFAULT_RUNTIME_CACHE_REGION


def _prebuilt_merged_runtime_cache_paths(runtime_cache_region: str) -> dict[str, str]:
    return PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION.get(runtime_cache_region, {})


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
def _existing_merged_runtime_cache_config(domain_name: str, runtime_cache_region: str) -> ExistingTokenizedCacheConfig:
    cache_path = _prebuilt_merged_runtime_cache_paths(runtime_cache_region)[domain_name]
    return ExistingTokenizedCacheConfig(
        cache_path=cache_path,
        tokenizer=marin_tokenizer,
        tags=[domain_name, "top_level_runtime_cache"],
    )


@cache
def _merged_top_level_domain_step(domain_name: str):
    partition_counts = top_level_domain_partition_counts(domain_name)
    input_steps = {
        partition_name: _partition_step_fn(partition_name)()
        for partition_name in TOP_LEVEL_DOMAIN_PARTITIONS[domain_name]
    }
    return merge_tokenized_caches(
        output_cache_path_name=os.path.join("dolma3_dolmino_top_level", domain_name),
        input_steps=input_steps,
        tokenizer=marin_tokenizer,
        tags=[domain_name, *partition_counts],
    )


def build_top_level_domains(*, runtime_cache_region: str = DEFAULT_RUNTIME_CACHE_REGION) -> list[Domain]:
    """Build top-level domains with hybrid runtime loading.

    The runtime uses:
    - shared merged caches for the 26 Dolma 3 CC high/low domains,
    - prebuilt merged caches for Stack-Edu and Dolmino STEM crawl when available in the selected region,
    - otherwise merge steps for those same two domains,
    - direct single-partition caches for singleton domains,
    - hierarchical loading for the remaining multi-partition Dolmino domains.
    """
    domains: list[Domain] = []
    prebuilt_merged_runtime_cache_paths = _prebuilt_merged_runtime_cache_paths(runtime_cache_region)
    for domain_name in DOMAIN_NAMES:
        partition_counts = top_level_domain_partition_counts(domain_name)
        if domain_name in prebuilt_merged_runtime_cache_paths:
            domains.append(
                Domain(
                    name=domain_name,
                    components=[
                        DatasetComponent(
                            name=domain_name,
                            step_fn=partial(_existing_merged_runtime_cache_config, domain_name, runtime_cache_region),
                            weight=TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name],
                        )
                    ],
                    description=f"Top-level domain reusing an existing finished {runtime_cache_region} merged cache.",
                )
            )
            continue

        if domain_name in PREFERRED_MERGED_RUNTIME_DOMAIN_NAMES:
            domains.append(
                Domain(
                    name=domain_name,
                    components=[
                        DatasetComponent(
                            name=domain_name,
                            step_fn=partial(_merged_top_level_domain_step, domain_name),
                            weight=TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name],
                        )
                    ],
                    description=(
                        "Top-level domain backed by a shared merged cache over "
                        f"{len(TOP_LEVEL_DOMAIN_PARTITIONS[domain_name])} source partition caches."
                    ),
                )
            )
            continue

        if len(partition_counts) == 1:
            partition_name = next(iter(partition_counts))
            domains.append(
                Domain(
                    name=domain_name,
                    components=[
                        DatasetComponent(
                            name=partition_name,
                            step_fn=_partition_step_fn(partition_name),
                            weight=partition_counts[partition_name],
                        )
                    ],
                    description="Top-level singleton domain backed directly by its original tokenized cache.",
                )
            )
            continue

        components = [
            DatasetComponent(
                name=partition_name,
                step_fn=_partition_step_fn(partition_name),
                weight=partition_counts[partition_name],
            )
            for partition_name in TOP_LEVEL_DOMAIN_PARTITIONS[domain_name]
        ]
        domains.append(
            Domain(
                name=domain_name,
                components=components,
                description=(
                    "Top-level domain with runtime hierarchical loading over "
                    f"{len(TOP_LEVEL_DOMAIN_PARTITIONS[domain_name])} source partition cache(s)."
                ),
            )
        )
    return domains


def build_top_level_domain_steps(*, runtime_cache_region: str = DEFAULT_RUNTIME_CACHE_REGION) -> dict[str, object]:
    """Build the shared prep steps for domains that should be physically merged in this region."""
    prebuilt_merged_runtime_cache_paths = _prebuilt_merged_runtime_cache_paths(runtime_cache_region)
    return {
        domain_name: _merged_top_level_domain_step(domain_name)
        for domain_name in PREFERRED_MERGED_RUNTIME_DOMAIN_NAMES
        if domain_name not in prebuilt_merged_runtime_cache_paths
    }


def _constant_phase_weights(domain_weights: dict[str, float]) -> dict[str, dict[str, float]]:
    return {phase_name: dict(domain_weights) for phase_name in PHASE_NAMES}


def create_stratified_domain_weights() -> dict[str, float]:
    """Return a static equal-weight allocation across the top-level domains."""
    uniform_weight = 1.0 / len(DOMAIN_NAMES)
    return {domain_name: uniform_weight for domain_name in DOMAIN_NAMES}


def create_stratified_weight_config(run_id: int = STRATIFIED_RUN_ID) -> WeightConfig:
    """Return the explicit stratified baseline used in the Olmix paper."""
    return WeightConfig(run_id=run_id, phase_weights=_constant_phase_weights(create_stratified_domain_weights()))


def create_initial_fixed_weight_configs() -> tuple[InitialFixedWeightConfig, ...]:
    """Create the fixed proportional, UniMax, and Olmix baselines for an empty-state loop."""
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

    if set(OLMIX_LOGLINEAR_PHASE_WEIGHTS) != set(PHASE_NAMES):
        raise ValueError("Olmix baseline phase names do not match the two-phase topology")
    for phase_name, phase_weights in OLMIX_LOGLINEAR_PHASE_WEIGHTS.items():
        if set(phase_weights) != set(DOMAIN_NAMES):
            raise ValueError(f"Olmix baseline domains do not match topology domains for {phase_name}")

    return (
        InitialFixedWeightConfig(
            run_name="baseline_proportional",
            weight_config=WeightConfig(run_id=0, phase_weights=_constant_phase_weights(proportional)),
        ),
        InitialFixedWeightConfig(
            run_name="baseline_unimax",
            weight_config=WeightConfig(run_id=1, phase_weights=_constant_phase_weights(unimax)),
        ),
        InitialFixedWeightConfig(
            run_name=OLMIX_LOGLINEAR_RUN_NAME,
            weight_config=create_olmix_loglinear_weight_config(),
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
    optimizer_config: MuonHConfig | None = None,
) -> MuonHConfig:
    """Create the boundary-aligned WSD optimizer config for this topology."""
    schedule = resolve_two_phase_wsd_boundary_schedule(
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        phase_schedule=phase_schedule,
    )
    base_optimizer_config = optimizer_config or DEFAULT_MUON_CONFIG
    return replace(
        base_optimizer_config,
        warmup=schedule.warmup_steps,
        decay=schedule.decay_steps,
        rewarmup=WSD_BOUNDARY_ALIGNED_REWARMUP,
        lr_schedule="linear",
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
    model_config: LmConfig | None = None,
    optimizer_config: MuonHConfig | None = None,
    resources: ResourceConfig | None = None,
    eval_harness_tasks: Sequence[EvalTaskConfig] | None = None,
    runtime_cache_region: str | None = None,
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
        optimizer_config=optimizer_config,
    )
    resolved_runtime_cache_region = _resolved_runtime_cache_region(
        runtime_cache_region=runtime_cache_region,
        resources=resources,
    )
    return MixtureExperiment(
        name=name,
        domains=build_top_level_domains(runtime_cache_region=resolved_runtime_cache_region),
        phase_schedule=phase_schedule,
        model_config=model_config or regmix_60m_proxy,
        batch_size=batch_size,
        seq_len=seq_len,
        num_train_steps=experiment_budget // (batch_size * seq_len),
        target_budget=target_budget,
        resources=resources or ResourceConfig.with_tpu("v5p-8"),
        eval_harness_tasks=tuple(eval_harness_tasks) if eval_harness_tasks is not None else EVAL_TASKS,
        sampling_params=SAMPLING_PARAMS,
        optimizer_config=optimizer_config,
        eval_datasets_cache_path=eval_datasets_cache_path,
        initial_fixed_weight_configs=create_initial_fixed_weight_configs(),
        hierarchical_runtime_domains=True,
    )
