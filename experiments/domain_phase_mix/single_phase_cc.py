# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-phase Common Crawl experiment for RegMix over topic-level domains.

This experiment explores data mixture weights across 18 Common Crawl topic domains
from Dolma 3 Pool, each containing multiple quality tier partitions.

Domains (18 CC topics):
    adult_content, art_and_design, crime_and_law, education_and_jobs,
    electronics_and_hardware, entertainment, fashion_and_beauty,
    finance_and_business, food_and_dining, games, health,
    history_and_geography, home_and_hobbies, industrial, literature,
    politics, religion, science_math_and_technology

Two settings:
    - epoch: Simulates full-data training where smaller topics are epoched.
      Sets target_budget to total CC token count (~8.1T).
    - no_epoch: No partition is exhausted during simulated epoching.
      Sets target_budget = experiment_budget (1B << any topic's data).

Usage:
    # Run training swarm (no_epoch, default)
    python -m experiments.domain_phase_mix.single_phase_cc [--n_runs N] [--seed SEED]

    # Run training swarm (epoch)
    python -m experiments.domain_phase_mix.single_phase_cc --epoch

    # Run predefined baseline runs
    python -m experiments.domain_phase_mix.single_phase_cc --baseline_runs [--epoch]

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.single_phase_cc --analyze [--epoch]
"""

import json
import logging
import os
from functools import partial

import fsspec

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.domain_phase_mix.config import DatasetComponent, Domain, PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.weight_sampler import DirichletSamplingParams, SamplingStrategy, compute_unimax_weights
from experiments.evals.task_configs import CORE_TASKS, MMLU_TASKS, convert_to_task_metrics
from experiments.marin_models import marin_tokenizer
from experiments.pretraining_datasets.dolma3_pool import (
    COMMON_CRAWL_TOPICS,
    DOLMA3_POOL_TOKEN_COUNTS_B,
    get_common_crawl_partitions_by_topic,
    tokenize_dolma3_pool_subset,
)

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME_BASE = "pinlin_calvin_xu/data_mixture/single_phase_v2"

# Token budget: 1B tokens (as specified in RegMix paper)
EXPERIMENT_BUDGET = 1_000_000_000

# Batch and sequence configuration
BATCH_SIZE = 128
SEQ_LEN = 2048

# Domain names: 18 CC topics
DOMAIN_NAMES = list(COMMON_CRAWL_TOPICS)

# Evaluation tasks: general benchmarks + MMLU (no code-specific tasks)
EVAL_TASKS = CORE_TASKS + MMLU_TASKS

ANALYSIS_METRICS = [
    "eval/loss",
    # CORE_TASKS metrics
    *convert_to_task_metrics(CORE_TASKS, "acc"),
    *convert_to_task_metrics(CORE_TASKS, "acc_norm"),
    *convert_to_task_metrics(CORE_TASKS, "bpb"),
    *convert_to_task_metrics(CORE_TASKS, "choice_logprob"),
    # MMLU metrics
    *convert_to_task_metrics(MMLU_TASKS, "acc"),
    *convert_to_task_metrics(MMLU_TASKS, "bpb"),
    # lm_eval averages
    "lm_eval/averages/macro_avg_acc",
    "lm_eval/averages/macro_avg_acc_norm",
    "lm_eval/averages/macro_avg_bpb",
    # Paloma BPB (from use_default_validation)
    "eval/paloma/c4_en/bpb",
    "eval/paloma/dolma-v1_5/bpb",
    "eval/paloma/m2d2_wikipedia_unsplit/bpb",
    # Uncheatable eval BPB (from use_default_validation)
    "eval/uncheatable_eval/wikipedia_english/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/ao3_english/bpb",
]

# GCS paths for pre-cached data to avoid HuggingFace rate limiting
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/core-tasks"
TOKENIZER_CACHE_BASE = "gs://marin-us-central1/raw/tokenizers"

# Tokenizer used by regmix_60m_proxy and all domains
TOKENIZER_NAME = marin_tokenizer  # "marin-community/marin-tokenizer"

# Dirichlet sampling with lower min_weight for 18 domains.
# strength is per-domain concentration: 0.2 = sparse (few active), 2.0 = dense (most active).
SAMPLING_PARAMS = DirichletSamplingParams(
    strategy=SamplingStrategy.DIRICHLET,
    min_weight=0.01,
    min_config_distance=0.001,
    min_strength=3.0,
    max_strength=10.0,
    temp=0.5,
)


# ============================================================================
# BASELINE CONFIGURATIONS
# ============================================================================

# Predefined baselines for single-phase, 18-domain experiment.
# Each entry is a list of weights for the 18 CC topics (in COMMON_CRAWL_TOPICS order).

# UniMax baseline (Chung et al., 2023): uniform budget with epoch cap N=5.
# With N=5, no CC domain is capped (uniform share ~453B << 5*365B), so UniMax = uniform.
_domain_sizes = [
    sum(DOLMA3_POOL_TOKEN_COUNTS_B[p] for p in get_common_crawl_partitions_by_topic(topic))
    for topic in COMMON_CRAWL_TOPICS
]
_unimax_weights = compute_unimax_weights(_domain_sizes, sum(_domain_sizes), max_epochs=5.0)

# Proportional baseline: weights ∝ domain token count.
_total_domain_size = sum(_domain_sizes)
_proportional_weights = [s / _total_domain_size for s in _domain_sizes]

BASELINES: list[list[float]] = [
    # Uniform across all topics
    # [1 / len(COMMON_CRAWL_TOPICS)] * len(COMMON_CRAWL_TOPICS),
    # UniMax (Chung et al., 2023)
    _unimax_weights,
    _proportional_weights,
]


# ============================================================================
# EXPERIMENT DEFINITION
# ============================================================================


def get_cc_topic_domains() -> list[Domain]:
    """Create Domain objects for each CC topic from Dolma 3 Pool.

    Each domain groups all quality tier partitions for a single topic.
    Uses llama3_tokenizer to match the regmix_60m_proxy model config.

    Returns:
        List of 18 Domain objects, one per CC topic.
    """
    domains = []
    for topic in COMMON_CRAWL_TOPICS:
        partitions = get_common_crawl_partitions_by_topic(topic)
        components = [
            DatasetComponent(
                name=p,
                step_fn=partial(tokenize_dolma3_pool_subset, p, tokenizer=marin_tokenizer),
                weight=DOLMA3_POOL_TOKEN_COUNTS_B[p],
            )
            for p in partitions
        ]
        domains.append(
            Domain(
                name=topic,
                components=components,
                description=f"Common Crawl {topic} ({len(partitions)} quality tiers)",
            )
        )
    return domains


def _get_total_cc_tokens() -> int:
    """Compute total token count across all CC topic partitions (in raw tokens, not billions)."""
    domains = get_cc_topic_domains()
    return int(sum(d.total_weight for d in domains) * 1_000_000_000)


def get_name(epoch: bool) -> str:
    """Get experiment name with epoch/no_epoch suffix."""
    return f"{NAME_BASE}_{'epoch' if epoch else 'no_epoch'}"


def get_target_budget(epoch: bool) -> int:
    """Get target budget based on epoch setting."""
    if epoch:
        return _get_total_cc_tokens()
    return EXPERIMENT_BUDGET


def create_single_phase_experiment(
    epoch: bool = False,
    name: str | None = None,
    experiment_budget: int = EXPERIMENT_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> MixtureExperiment:
    """Create the single-phase CC topic experiment.

    Args:
        epoch: If True, use simulated epoching (target_budget = total CC tokens).
            If False, no epoching (target_budget = experiment_budget).
        name: Experiment name. Defaults to NAME_BASE + _epoch/_no_epoch.
        experiment_budget: Actual token budget for training.
        batch_size: Training batch size.
        seq_len: Sequence length.

    Returns:
        MixtureExperiment configured for single-phase CC experiment.
    """
    name = name or get_name(epoch)
    target_budget = get_target_budget(epoch)

    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=[],
        names=["phase_0"],
    )

    domains = get_cc_topic_domains()

    return MixtureExperiment(
        name=name,
        domains=domains,
        phase_schedule=phase_schedule,
        model_config=regmix_60m_proxy,
        batch_size=batch_size,
        seq_len=seq_len,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        eval_harness_tasks=EVAL_TASKS,
        sampling_params=SAMPLING_PARAMS,
        eval_datasets_cache_path=EVAL_DATASETS_CACHE_PATH,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def create_baseline_weight_configs(
    baselines: list[list[float]] | None = None,
    phase_names: list[str] | None = None,
    domain_names: list[str] | None = None,
) -> list[WeightConfig]:
    """Create WeightConfig objects from predefined baseline weights.

    Args:
        baselines: List of baseline weight vectors (one per phase_0).
        phase_names: Names of phases. Defaults to ["phase_0"].
        domain_names: Names of domains. Defaults to DOMAIN_NAMES.

    Returns:
        List of WeightConfig objects with unique run_ids starting from 90000.
    """
    baselines = baselines or BASELINES
    phase_names = phase_names or ["phase_0"]
    domain_names = domain_names or DOMAIN_NAMES

    BASELINE_RUN_ID_START = 90000
    configs = []
    for i, weights in enumerate(baselines):
        phase_weights = {
            phase_names[0]: dict(zip(domain_names, weights, strict=True)),
        }
        configs.append(WeightConfig(run_id=BASELINE_RUN_ID_START + i, phase_weights=phase_weights))

    return configs


def _load_original_weight_configs(name_prefix: str) -> list[WeightConfig]:
    """Load weight configs saved by the original training swarm from GCS.

    Args:
        name_prefix: Experiment name prefix.

    Returns:
        List of WeightConfig objects from the original training run.
    """
    prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")
    pattern = f"{name_prefix}/weight_configs-*/weight_configs.json"

    fs, base = fsspec.core.url_to_fs(prefix)
    matches = fs.glob(f"{base}/{pattern}")

    if not matches:
        raise FileNotFoundError(
            f"No weight_configs found at {prefix}/{pattern}. " "Run the training swarm first before running --analyze."
        )

    if len(matches) > 1:
        logger.warning(f"Found multiple weight_configs: {matches}. Using the first one.")

    path = f"{fs.protocol}://{matches[0]}" if isinstance(fs.protocol, str) else f"{fs.protocol[0]}://{matches[0]}"
    logger.info(f"Loading original weight configs from {path}")

    with fsspec.open(path) as f:
        data = json.load(f)

    return [WeightConfig.from_dict(c) for c in data["configs"]]


def run_baselines(
    epoch: bool = False,
    name_prefix: str | None = None,
    baselines: list[list[float]] | None = None,
):
    """Run predefined baseline trial runs.

    Args:
        epoch: Epoch setting (controls target_budget and name suffix).
        name_prefix: Prefix for run names.
        baselines: List of baseline configurations. If None, uses BASELINES.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    baselines = baselines or BASELINES
    name_prefix = name_prefix or get_name(epoch)
    experiment = create_single_phase_experiment(epoch=epoch, name=name_prefix)

    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = TOKENIZER_CACHE_BASE

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(TOKENIZER_CACHE_BASE, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=name_prefix,
    )

    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=EVAL_DATASETS_CACHE_PATH,
        name_prefix=name_prefix,
    )

    weight_configs = create_baseline_weight_configs(baselines)

    logger.info(f"Running {len(weight_configs)} baseline configurations:")
    for config in weight_configs:
        logger.info(f"  baseline_run_{config.run_id}: {config.phase_weights}")

    training_steps = []
    for config in weight_configs:
        step = experiment.create_training_step(
            config,
            name_prefix=name_prefix,
            run_name=f"base_{config.run_id:05d}",
        )
        training_steps.append(step)

    executor_main(
        steps=[cache_tokenizer_step, cache_eval_datasets_step, *training_steps],
        description=f"Baseline runs for {name_prefix}",
    )


def main(
    n_runs: int = 50,
    seed: int = 42,
    name_prefix: str | None = None,
    analyze: bool = False,
    baseline_runs: bool = False,
    epoch: bool = False,
):
    """Main entry point for running the swarm experiment.

    Args:
        n_runs: Number of training runs.
        seed: Random seed for weight sampling.
        name_prefix: Prefix for run names. Auto-set from epoch flag if None.
        analyze: If True, only run analysis step (collect results from W&B).
        baseline_runs: If True, run predefined baseline trial runs.
        epoch: If True, use simulated epoching. If False (default), no epoching.

    Note:
        Additional executor options like --max_concurrent and --force_run_failed
        are handled by executor_main via draccus CLI parsing.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    name_prefix = name_prefix or get_name(epoch)

    if baseline_runs:
        run_baselines(epoch=epoch, name_prefix=name_prefix)
        return

    experiment = create_single_phase_experiment(epoch=epoch, name=name_prefix)

    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = TOKENIZER_CACHE_BASE

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(TOKENIZER_CACHE_BASE, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=name_prefix,
    )

    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=EVAL_DATASETS_CACHE_PATH,
        name_prefix=name_prefix,
    )

    weight_configs_step, training_steps = experiment.create_swarm_steps(
        n_runs=n_runs, seed=seed, name_prefix=name_prefix
    )

    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=name_prefix,
        metrics=ANALYSIS_METRICS,
    )

    if analyze:
        logger.info("Running analysis only (collecting results from W&B)")
        original_configs = _load_original_weight_configs(name_prefix)
        existing_ids = {c.run_id for c in original_configs}

        baseline_weight_configs = create_baseline_weight_configs(BASELINES)
        new_baselines = [c for c in baseline_weight_configs if c.run_id not in existing_ids]
        all_configs = original_configs + new_baselines

        logger.info(
            f"Loaded {len(original_configs)} original configs, "
            f"appending {len(new_baselines)} new baselines ({len(all_configs)} total)"
        )

        weight_configs_step_for_analysis = experiment.create_weight_configs_step(
            configs=all_configs,
            summary={},
            seed=seed,
            name_prefix=f"{name_prefix}_analysis",
        )
        analysis_step_for_analysis = create_analysis_step(
            weight_configs_step=weight_configs_step_for_analysis,
            name_prefix=name_prefix,
            metrics=ANALYSIS_METRICS,
        )
        all_steps = [weight_configs_step_for_analysis, analysis_step_for_analysis]
        executor_main(
            steps=all_steps,
            description=f"Analysis for {name_prefix}",
        )
        return

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    total_steps = EXPERIMENT_BUDGET // tokens_per_step
    target_budget = get_target_budget(epoch)

    logger.info(
        f"Created {len(training_steps)} training steps + 1 tokenizer cache step + "
        f"1 eval datasets cache step + 1 weight configs step + 1 analysis step"
    )
    logger.info(f"Total tokens per run: {EXPERIMENT_BUDGET:,}")
    logger.info(f"Total steps per run: {total_steps:,}")
    logger.info(f"Epoch mode: {epoch}")
    logger.info(f"Target budget (simulated epoching): {target_budget:,}")
    logger.info(f"Domains: {len(DOMAIN_NAMES)} CC topics")

    all_steps = [
        cache_tokenizer_step,
        cache_eval_datasets_step,
        weight_configs_step,
        *training_steps,
        analysis_step,
    ]
    executor_main(
        steps=all_steps,
        description=f"Single-phase CC experiment ({'epoch' if epoch else 'no_epoch'}): {n_runs} runs",
    )


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Single-phase CC topic experiment for RegMix.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=50,
        help="Number of training runs (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight sampling (default: 42).",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default=None,
        help="Prefix for run names. Defaults to NAME_BASE + _epoch/_no_epoch.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis only (collect results from W&B and export CSV).",
    )
    parser.add_argument(
        "--baseline_runs",
        action="store_true",
        help="Run predefined baseline trial runs instead of random sampling.",
    )
    parser.add_argument(
        "--epoch",
        action="store_true",
        help="Use simulated epoching (target_budget = total CC tokens). Default is no_epoch.",
    )

    return parser.parse_known_args()


if __name__ == "__main__":
    import sys

    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    main(
        n_runs=args.n_runs,
        seed=args.seed,
        name_prefix=args.name_prefix,
        analyze=args.analyze,
        baseline_runs=args.baseline_runs,
        epoch=args.epoch,
    )
