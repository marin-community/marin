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

"""Three-phase starcoder experiment extending the two-phase setup.

Same two domains (Nemotron + StarCoder) but with three training phases,
giving a (N=3, M=2) configuration to complement the existing (N=2, M=2) data.

- Two domains: Nemotron (common web, ~5.7T tokens) and StarCoder (code, ~217B tokens)
- Three phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0]
- 1.5B token budget
- Evaluates on code-related benchmarks (CODE_TASKS + starcoder validation)
- Uses Dirichlet sampling for weight exploration

Usage:
    # Run training with random weight sampling (Dirichlet)
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment [--n_runs N] [--seed SEED]

    # Run predefined baseline runs
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment --baseline_runs

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment --analyze
"""

import json
import logging
import os

import fsspec

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.llama import llama3_tokenizer
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.domains import (
    NEMOTRON_FULL_DOMAIN,
    STARCODER_DOMAIN,
)
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.weight_sampler import DirichletSamplingParams, SamplingStrategy, compute_unimax_weights
from experiments.evals.task_configs import CORE_TASKS, CODE_TASKS, convert_to_task_metrics

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1"

# Token budget: 1.5B tokens
EXPERIMENT_BUDGET = 1_500_000_000

# Target budget: set to size of Nemotron (~5.7T tokens)
# StarCoder can be epoched at most ~26x (5.7T / 217B)
TARGET_BUDGET = 5_729_908_864_777  # Nemotron full token count

# Batch and sequence configuration
BATCH_SIZE = 128
SEQ_LEN = 2048

# Phase boundaries (fractions of total training)
PHASE_BOUNDARIES = [0.33, 0.67]  # Creates 3 phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0]

PHASE_NAMES = ["phase_0", "phase_1", "phase_2"]

# Domain names (must match names from get_nemotron_starcoder_domains())
DOMAIN_NAMES = ["nemotron_full", "starcoder"]

# Combine CORE_TASKS + CODE_TASKS for evaluation
EVAL_TASKS = CORE_TASKS + CODE_TASKS

# Metrics to collect from W&B for analysis.
ANALYSIS_METRICS = [
    "eval/loss",
    # CORE_TASKS: acc and acc_norm for multiple-choice tasks
    *convert_to_task_metrics(CORE_TASKS, "acc"),
    *convert_to_task_metrics(CORE_TASKS, "acc_norm"),
    *convert_to_task_metrics(CORE_TASKS, "bpb"),
    *convert_to_task_metrics(CORE_TASKS, "choice_logprob"),
    # code2text tasks: BLEU score
    "lm_eval/code2text_go_0shot/smoothed_bleu_4",
    "lm_eval/code2text_java_0shot/smoothed_bleu_4",
    "lm_eval/code2text_javascript_0shot/smoothed_bleu_4",
    "lm_eval/code2text_php_0shot/smoothed_bleu_4",
    "lm_eval/code2text_python_0shot/smoothed_bleu_4",
    "lm_eval/code2text_ruby_0shot/smoothed_bleu_4",
    # jsonschema_bench tasks: validity and compliance
    "lm_eval/jsonschema_bench_easy_2shot/json_validity",
    "lm_eval/jsonschema_bench_easy_2shot/schema_compliance",
    "lm_eval/jsonschema_bench_medium_2shot/json_validity",
    "lm_eval/jsonschema_bench_medium_2shot/schema_compliance",
    "lm_eval/jsonschema_bench_hard_2shot/json_validity",
    "lm_eval/jsonschema_bench_hard_2shot/schema_compliance",
    # humaneval: pass@1
    "lm_eval/humaneval_0shot/pass@1,create_test",
    # Averages
    "lm_eval/averages/macro_avg_acc",
    "lm_eval/averages/macro_avg_acc_norm",
    "lm_eval/averages/macro_avg_bpb",
    "lm_eval/averages/macro_avg_smoothed_bleu_4",
]

# GCS paths for pre-cached data to avoid HuggingFace rate limiting
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/code-tasks"
TOKENIZER_CACHE_BASE = "gs://marin-us-central1/raw/tokenizers"

# Tokenizer used by regmix_60m_proxy and all domains
TOKENIZER_NAME = llama3_tokenizer  # "meta-llama/Meta-Llama-3.1-8B"

# Use Dirichlet sampling strategy for weight exploration
SAMPLING_PARAMS = DirichletSamplingParams(
    strategy=SamplingStrategy.DIRICHLET,
    min_weight=0.05,
    min_config_distance=0.05,
    min_strength=0.1,
    max_strength=0.5,
    temp=0.5,
)


# ============================================================================
# BASELINE CONFIGURATIONS
# ============================================================================

# Predefined baseline configurations for baseline runs
# Each entry is a tuple of phase weights: [[phase_0], [phase_1], [phase_2]]
# Weights correspond to domains in order: [nemotron_full, starcoder]
BASELINES: list[tuple[list[float], list[float], list[float]]] = [
    # Gradual transition: Nemotron -> balanced -> StarCoder
    ([1, 0], [0.5, 0.5], [0, 1]),
    # Balanced throughout
    ([0.5, 0.5], [0.5, 0.5], [0.5, 0.5]),
    # Two-stage-inspired: web-heavy first two phases, code-focused last phase
    ([0.99, 0.01], [0.99, 0.01], [0.2, 0.8]),
    # Gradual code ramp
    ([0.99, 0.01], [0.5, 0.5], [0.2, 0.8]),
    ([0.95, 0.05], [0.7, 0.3], [0.2, 0.8]),
    # More balanced phase 3
    ([0.99, 0.01], [0.8, 0.2], [0.5, 0.5]),
    # Single-domain baselines
    ([1, 0], [1, 0], [1, 0]),  # Nemotron only (no code)
    ([0, 1], [0, 1], [0, 1]),  # StarCoder only (no web)
    # Code in the middle phase only
    ([1, 0], [0.2, 0.8], [1, 0]),
    # Code ramp then taper
    ([0.2, 0.8], [0.5, 0.5], [0.99, 0.01]),
    # Late code burst
    ([1, 0], [1, 0], [0, 1]),
    # Early code then web
    ([0, 1], [1, 0], [1, 0]),
    # Constant moderate code
    ([0.8, 0.2], [0.8, 0.2], [0.8, 0.2]),
    ([0.7, 0.3], [0.7, 0.3], [0.7, 0.3]),
]

# UniMax baseline (Chung et al., 2023): uniform budget with epoch cap N=1
_unimax_phase_budget = TARGET_BUDGET * (PHASE_BOUNDARIES[0])  # Per-phase budget (~33%)
_unimax_domain_sizes = [NEMOTRON_FULL_DOMAIN.total_weight, STARCODER_DOMAIN.total_weight]
_unimax_weights = compute_unimax_weights(_unimax_domain_sizes, _unimax_phase_budget, max_epochs=1.0)
BASELINES.append((_unimax_weights, _unimax_weights, _unimax_weights))


# ============================================================================
# EXPERIMENT DEFINITION
# ============================================================================


def get_nemotron_starcoder_domains():
    """Get domains for the experiment.

    Uses Nemotron (large common web data, ~5.7T tokens) and StarCoder (rare code data, ~217B tokens).

    Returns:
        List of [NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN]
    """
    return [NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN]


def create_three_phase_experiment(
    name: str = NAME,
    experiment_budget: int = EXPERIMENT_BUDGET,
    target_budget: int = TARGET_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> MixtureExperiment:
    """Create the three-phase starcoder experiment.

    This sets up:
    - 2 domains: Nemotron (common web, ~5.7T tokens) and StarCoder (code, ~217B tokens)
    - 3 phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0]
    - RegMix 60M proxy model
    - Simulated epoching with max ~26x epoching on smallest dataset (StarCoder)
    """
    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=PHASE_BOUNDARIES,
        names=PHASE_NAMES,
    )

    domains = get_nemotron_starcoder_domains()

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
    baselines: list[tuple[list[float], list[float], list[float]]] = BASELINES,
    phase_names: list[str] | None = None,
    domain_names: list[str] | None = None,
) -> list[WeightConfig]:
    """Create WeightConfig objects from predefined baseline weights.

    Args:
        baselines: List of baseline configurations. Each is a tuple of 3 lists,
            one per phase, with weights for each domain.
        phase_names: Names of phases. Defaults to PHASE_NAMES.
        domain_names: Names of domains. Defaults to DOMAIN_NAMES.

    Returns:
        List of WeightConfig objects with unique run_ids starting from 90000.
    """
    phase_names = phase_names or PHASE_NAMES
    domain_names = domain_names or DOMAIN_NAMES

    BASELINE_RUN_ID_START = 90000
    configs = []
    for i, phases in enumerate(baselines):
        phase_weights = {
            pname: dict(zip(domain_names, pweights, strict=True))
            for pname, pweights in zip(phase_names, phases, strict=True)
        }
        configs.append(WeightConfig(run_id=BASELINE_RUN_ID_START + i, phase_weights=phase_weights))

    return configs


def _load_original_weight_configs(name_prefix: str) -> list[WeightConfig]:
    """Load weight configs saved by the original training swarm from GCS."""
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
    name_prefix: str = NAME,
    baselines: list[tuple[list[float], list[float], list[float]]] | None = None,
):
    """Run predefined baseline trial runs."""
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    baselines = baselines or BASELINES
    experiment = create_three_phase_experiment(name=name_prefix)

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
    name_prefix: str = NAME,
    analyze: bool = False,
    baseline_runs: bool = False,
):
    """Main entry point for running the swarm experiment."""
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    if baseline_runs:
        run_baselines(name_prefix=name_prefix)
        return

    experiment = create_three_phase_experiment(name=name_prefix)

    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = TOKENIZER_CACHE_BASE
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

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
        depends_on=list(training_steps),
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
    phase1_end = int(total_steps * PHASE_BOUNDARIES[0])
    phase2_end = int(total_steps * PHASE_BOUNDARIES[1])

    logger.info(
        f"Created {len(training_steps)} training steps + 1 tokenizer cache step + "
        f"1 eval datasets cache step + 1 weight configs step + 1 analysis step"
    )
    logger.info(f"Total tokens per run: {EXPERIMENT_BUDGET:,}")
    logger.info(f"Total steps per run: {total_steps:,}")
    logger.info(f"Phase boundaries: step {phase1_end} (33%), step {phase2_end} (67%)")
    logger.info(f"Target budget (simulated epoching): {TARGET_BUDGET:,}")
    logger.info("Max epoching on smallest dataset (StarCoder): ~26x")

    all_steps = [
        cache_tokenizer_step,
        cache_eval_datasets_step,
        weight_configs_step,
        *training_steps,
        analysis_step,
    ]
    executor_main(
        steps=all_steps,
        description=f"Three-phase starcoder experiment: {n_runs} runs",
    )


def run_analysis_local(
    name_prefix: str = NAME,
    output_dir: str | None = None,
):
    """Collect results from W&B locally (no executor/GCS needed for output)."""
    from experiments.domain_phase_mix.analysis import (
        build_results_dataframe,
        match_runs_to_configs,
        query_wandb_runs,
    )

    print(f"Loading original weight configs for {name_prefix}...")
    original_configs = _load_original_weight_configs(name_prefix)
    existing_ids = {c.run_id for c in original_configs}

    baseline_weight_configs = create_baseline_weight_configs(BASELINES)
    new_baselines = [c for c in baseline_weight_configs if c.run_id not in existing_ids]
    all_configs = original_configs + new_baselines

    print(f"  {len(original_configs)} original configs + " f"{len(new_baselines)} baselines = {len(all_configs)} total")

    domains = DOMAIN_NAMES
    phases = PHASE_NAMES
    configs_dicts = [c.to_dict() for c in all_configs]

    tags = [name_prefix]
    print(f"Querying W&B for runs with tags: {tags}...")
    runs = query_wandb_runs(
        entity="marin-community",
        project="marin",
        tags=tags,
        metrics=ANALYSIS_METRICS,
    )
    print(f"  Found {len(runs)} W&B runs")

    matched = match_runs_to_configs(runs, configs_dicts, experiment_name=name_prefix)
    n_matched = sum(1 for m in matched if m.get("wandb_run_id"))
    n_completed = sum(1 for m in matched if m.get("status") == "completed")
    print(f"  Matched {n_matched} runs ({n_completed} completed)")

    df = build_results_dataframe(matched, domains, phases)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "exploratory")

    csv_name = name_prefix.rsplit("/", 1)[-1] + ".csv"
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {len(df)} rows to {csv_path}")

    missing = df[df["status"] != "completed"]
    if len(missing) > 0:
        print(f"\n  {len(missing)} runs not completed:")
        for _, row in missing.iterrows():
            print(f"    run_id={int(row['run_id'])}: status={row['status']}")

    return df


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Three-phase starcoder experiment (N=3, M=2).")
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
        default=NAME,
        help="Prefix for run names.",
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
        "--analyze-local",
        action="store_true",
        help="Run analysis locally (query W&B, write CSV to exploratory/). No executor/cluster needed.",
    )

    return parser.parse_known_args()


if __name__ == "__main__":
    import sys

    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.analyze_local:
        run_analysis_local(name_prefix=args.name_prefix)
    else:
        main(
            n_runs=args.n_runs,
            seed=args.seed,
            name_prefix=args.name_prefix,
            analyze=args.analyze,
            baseline_runs=args.baseline_runs,
        )
