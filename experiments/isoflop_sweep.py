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

"""Generate ISOFlop sweep steps for varying model sizes on a target dataset.

This script constructs `ExecutorStep` objects that train models of different
sizes while keeping the total training FLOPs roughly constant.
"""

import logging
import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace

from levanter.data.text import LMMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import EvalTaskConfig
from experiments.common_pile.tokenize_common_pile import comma_main_mixture
from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple import downloads
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.processing.tokenize import get_vocab_size_for_tokenizer, lm_mixture_data_config
from marin.scaling_laws import (
    CandidateConfig,
    FitScalingLawsResult,
    IsoFlopRecord,
    ScalingRecipe,
    fit_scaling_laws,
    generate_isoflop_train_args,
    pick_v5p_type,
    round_flops_to_bucket,
    solve_for_batch_size,
    solve_for_train_steps,
)
from marin.scaling_laws.eval_metrics_reader import read_raw_records
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger(__name__)

DEFAULT_BUDGETS: tuple[float, ...] = (1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20)
LEGACY_BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)
DEFAULT_SEQ_LEN: int = 4096
DEFAULT_STEPS_PER_RUN: int = 2**16
DEFAULT_FLOP_TOLERANCE: float = 0.01

# ---------------- Levanter WandB Metric Keys ----------------
# These keys correspond to the metrics logged by Levanter's training callbacks.
THROUGHPUT_TOKENS_KEY = "throughput/total_tokens"
THROUGHPUT_GFLOPS_KEY = "throughput/total_gflops"
PARAMETER_COUNT_KEY = "parameter_count"
MODEL_CONFIG_KEY = "model"
TRAINER_CONFIG_KEY = "trainer"
DEFAULT_METRIC_KEY = "eval/paloma/c4_en/bpb"


# ---------------- Levanter Metrics Transform ----------------


def parse_isoflop_run_name(run_name: str) -> str | None:
    """Parse experiment name from isoflop run name.

    Supports two formats:
    - New: isoflop-{budget}-N{params}-B{batch}-{experiment_name}
      E.g., 'isoflop-1e+18-N1e+08-B128-nemo-wider-depth-adapt'
    - Legacy: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
      E.g., 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt'

    Optionally with a trailing -<hash> which is ignored.

    Returns experiment_name or None if parsing fails.
    """
    # Strip optional -<hash> suffix
    run_name = re.sub(r"-[0-9a-fA-F]{6}$", "", run_name)

    # New format: isoflop-{budget}-N{params}-B{batch}-{experiment_name}
    new_pattern = r"isoflop-(?:[0-9.e+]+)-N(?:[0-9.e+]+)-B(?:\d+)-(.+)"
    match = re.match(new_pattern, run_name)
    if match:
        return match.group(1)

    # Legacy format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    legacy_pattern = r"isoflop-(?:[0-9.e+]+)-d(?:\d+)-L(?:\d+)-B(?:\d+)-(.+)"
    match = re.match(legacy_pattern, run_name)
    if match:
        return match.group(1)

    return None


def transform_levanter_metrics(
    raw_records: list[dict],
    metric_key: str = DEFAULT_METRIC_KEY,
    label_map: dict[str, str] | None = None,
    min_flops: float = 1e18,
) -> list[IsoFlopRecord]:
    """Transform raw Levanter metrics into IsoFlopRecord list.

    Args:
        raw_records: Raw records from read_raw_records(), each containing
            'config', 'summary', and 'run_path' keys.
        metric_key: Which metric to use (default: eval/paloma/c4_en/bpb).
        label_map: Optional mapping from experiment_name -> display label.
        min_flops: Minimum FLOP threshold to include (default: 1e18).

    Returns:
        List of IsoFlopRecord for records that have all required fields.
        Records missing required fields are logged and skipped.
    """
    records = []

    for raw in raw_records:
        run_path = raw.get("run_path", "")
        run_name = os.path.basename(run_path.rstrip("/"))

        summary = raw.get("summary", {}) or {}

        # Extract tokens
        tokens = summary.get(THROUGHPUT_TOKENS_KEY)
        if tokens is None:
            logger.warning(f"Missing {THROUGHPUT_TOKENS_KEY} for run {run_name}, skipping")
            continue

        # Extract FLOPs (convert GFLOPs to FLOPs and bucket)
        total_gflops = summary.get(THROUGHPUT_GFLOPS_KEY)
        if total_gflops is None:
            logger.warning(f"Missing {THROUGHPUT_GFLOPS_KEY} for run {run_name}, skipping")
            continue
        flops = round_flops_to_bucket(total_gflops * 1e9)

        if flops < min_flops:
            continue

        # Extract metric
        metric = summary.get(metric_key)
        if metric is None:
            logger.warning(f"Missing metric {metric_key} for run {run_name}, skipping")
            continue

        # Extract params (required)
        params = summary.get(PARAMETER_COUNT_KEY)
        if params is None:
            logger.warning(f"Missing {PARAMETER_COUNT_KEY} for run {run_name}, skipping")
            continue

        # Determine label from run name
        exp_name = parse_isoflop_run_name(run_name) or run_name
        if label_map and exp_name in label_map:
            label = label_map[exp_name]
        else:
            label = exp_name

        records.append(
            IsoFlopRecord(
                tokens=float(tokens),
                metric=float(metric),
                flops=float(flops),
                params=float(params),
                label=label,
            )
        )

    logger.info(f"Transformed {len(records)} records from {len(raw_records)} raw records")
    return records


def _round_to_power_of_two(x: float) -> int:
    """Round x UP to the nearest power of 2."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _format_run_name(
    budget: float,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    experiment_name: str,
) -> str:
    """Format run name using architecture details (hidden size and layers).

    Format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    """
    return f"isoflop-{budget:.0e}-d{hidden_size}-L{num_layers}-B{batch_size}-{experiment_name}"


@dataclass(frozen=True)
class Marin2025Recipe:
    """Marin 2025 scaling recipe with all hyperparameters and formulas.

    This recipe implements all the Marin-specific decisions for scaling experiments.
    The vocab_size is derived from the tokenizer, making the recipe self-contained
    for all model configuration decisions.
    """

    name: str = "marin-2025"
    tokenizer: str = "stanford-crfm/marin-tokenizer"
    """Tokenizer to use. vocab_size is derived from this."""

    @property
    def vocab_size(self) -> int:
        """Vocabulary size derived from the tokenizer."""
        return get_vocab_size_for_tokenizer(self.tokenizer)

    # --- Learning rate scaling ---
    # lr = lr_constant * sqrt(batch_size) / hidden_dim
    lr_constant: float = 0.33

    # --- Beta2 scaling for Adam ---
    # beta2 = beta2_base ** (batch_size / beta2_batch_divisor)
    beta2_base: float = 0.98
    beta2_batch_divisor: float = 128

    # --- Optimizer hyperparameters ---
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    beta1: float = 0.95
    epsilon: float = 1e-15
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"
    decay: float = 0.2

    # --- Architecture ratios ---
    mlp_ratio: int = 4
    hidden_head_ratio: int = 128

    # --- Architecture formula for depth-to-width scaling ---
    base_hidden_layer_ratio: int = 64
    layer_scaling_factor: float = 4.0
    layer_formula_offset: int = 9

    # --- Constraints ---
    max_learning_rate: float = 0.01
    min_batch_size: int = 8

    # --- Search bounds for isoflop sweeps ---
    min_hidden_pow: int = 9
    max_hidden_pow: int = 12
    small_budget_step_size: int = 128
    large_budget_step_size: int = 256
    budget_step_threshold: float = 9e18

    def _compute_learning_rate(self, batch_size: int, hidden_dim: int) -> float:
        """Compute learning rate from batch size and hidden dim."""
        return (self.lr_constant * math.sqrt(batch_size)) / hidden_dim

    def _compute_beta2(self, batch_size: int) -> float:
        """Compute beta2 from batch size."""
        return self.beta2_base ** (batch_size / self.beta2_batch_divisor)

    def compute_num_layers(self, hidden_size: int) -> int:
        """Compute number of layers from hidden size using the depth-width formula."""
        hs_pow = math.log2(hidden_size)
        return round(
            hidden_size
            / (self.base_hidden_layer_ratio + (hs_pow * self.layer_scaling_factor) - self.layer_formula_offset)
        )

    def _get_step_size(self, budget: float) -> int:
        """Get hidden_size search step size based on budget."""
        if budget > self.budget_step_threshold:
            return self.large_budget_step_size
        return self.small_budget_step_size

    def _build_model_config_from_hidden_size(self, hidden_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Build model config from hidden_size directly."""
        if hidden_size % self.hidden_head_ratio != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by hidden_head_ratio ({self.hidden_head_ratio}). "
                f"Got remainder {hidden_size % self.hidden_head_ratio}."
            )
        num_layers = self.compute_num_layers(hidden_size)
        intermediate_dim = hidden_size * self.mlp_ratio
        n_heads = max(1, hidden_size // self.hidden_head_ratio)

        return Qwen3Config(
            hidden_dim=hidden_size,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            max_seq_len=seq_len,
            rope=Llama3RotaryEmbeddingsConfig(),
        )

    def estimate_memory_bytes(
        self,
        candidate: CandidateConfig,
        seq_len: int = DEFAULT_SEQ_LEN,
        optim_mult: int = 3,
        dtype_size: int = 4,
        fudge_factor: float = 2.0,
    ) -> int:
        """Estimate float32 memory usage in bytes for training."""
        model_config = candidate.model_config
        batch_size, _ = self.compute_training_schedule(candidate, seq_len)

        param_count = model_config.total_trainable_params(self.vocab_size)
        param_bytes = param_count * optim_mult * dtype_size
        act_bytes = (batch_size * model_config.max_seq_len) * (
            (model_config.hidden_dim * model_config.num_layers) + self.vocab_size * fudge_factor
        )
        total_bytes = param_bytes + act_bytes
        return int(total_bytes * fudge_factor)

    def compute_training_schedule(self, candidate: CandidateConfig, seq_len: int = DEFAULT_SEQ_LEN) -> tuple[int, int]:
        """Compute training schedule (batch_size, train_steps) for a candidate."""
        hidden_size = candidate.model_config.hidden_dim

        # Start with batch_size that gives us ~DEFAULT_STEPS_PER_RUN steps for the tokens
        target_steps = DEFAULT_STEPS_PER_RUN
        batch_exact = candidate.tokens / (target_steps * seq_len)
        batch_size = _round_to_power_of_two(batch_exact)

        # Adjust batch_size to respect learning rate constraints
        lr = self._compute_learning_rate(batch_size, hidden_size)
        while lr > self.max_learning_rate and batch_size >= self.min_batch_size * 2:
            batch_size //= 2
            lr = self._compute_learning_rate(batch_size, hidden_size)

        # Ensure minimum batch size
        if batch_size < self.min_batch_size:
            batch_size = self.min_batch_size

        # Compute train_steps to achieve target tokens
        train_steps = round(candidate.tokens / (batch_size * seq_len))

        return (batch_size, train_steps)

    def build_optimizer_config(self, candidate: CandidateConfig, seq_len: int = DEFAULT_SEQ_LEN) -> OptimizerConfig:
        """Build optimizer config for a candidate."""
        batch_size, _ = self.compute_training_schedule(candidate, seq_len)
        hidden_size = candidate.model_config.hidden_dim
        learning_rate = self._compute_learning_rate(batch_size, hidden_size)
        beta2 = self._compute_beta2(batch_size)

        return CautiousConfig(
            learning_rate=learning_rate,
            weight_decay=self.weight_decay,
            min_lr_ratio=self.min_lr_ratio,
            warmup=self.warmup,
            beta1=self.beta1,
            beta2=beta2,
            epsilon=self.epsilon,
            max_grad_norm=self.max_grad_norm,
            adamc_weight_decay=True,
            lr_schedule=self.lr_schedule,
            decay=self.decay,
        )

    def candidate_configs(
        self,
        budget: float,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> Iterator[CandidateConfig]:
        """Yield candidate configurations within the FLOP budget.

        Iterates over feasible model architectures, computes tokens to hit the
        FLOP budget, and yields CandidateConfigs with the model_config directly.
        """
        step_size = self._get_step_size(budget)
        min_hidden = 2**self.min_hidden_pow
        max_hidden = 2**self.max_hidden_pow

        for hidden_size in range(min_hidden, max_hidden + 1, step_size):
            model_config = self._build_model_config_from_hidden_size(hidden_size, seq_len)

            # Compute batch_size to hit FLOP budget
            batch_exact = solve_for_batch_size(model_config, self.vocab_size, budget, steps_per_run, seq_len)
            batch_size = _round_to_power_of_two(batch_exact)

            # Adjust batch_size to respect learning rate constraints
            lr = self._compute_learning_rate(batch_size, hidden_size)
            while lr > self.max_learning_rate:
                batch_size //= 2
                lr = self._compute_learning_rate(batch_size, hidden_size)

            if batch_size < self.min_batch_size:
                continue

            train_steps = round(solve_for_train_steps(model_config, self.vocab_size, budget, batch_size, seq_len))

            # Validate achieved FLOPs are within tolerance
            achieved_flops = (
                3 * model_config.flops_per_token(self.vocab_size, seq_len) * batch_size * train_steps * seq_len
            )
            if abs(achieved_flops - budget) / budget > flop_tolerance:
                continue

            tokens = batch_size * train_steps * seq_len

            yield CandidateConfig(
                model_config=model_config,
                tokens=tokens,
                flops_budget=budget,
            )


MARIN_2025_RECIPE = Marin2025Recipe()
"""Default Marin scaling recipe."""


# ---------------- IsoFlop Analysis ----------------


@dataclass(frozen=True, kw_only=True)
class IsoFlopAnalysisConfig:
    """Configuration for IsoFLOP scaling law analysis.

    The training_runs field creates blocking dependencies on the training jobs.
    This config is for use with ExecutorStep.
    """

    training_runs: tuple[str, ...]
    """Training run output paths (executor resolves InputName to str at runtime)."""

    output_path: str
    """Where to write analysis outputs."""

    recipe: ScalingRecipe
    """Scaling recipe for computing optimal hyperparameters."""

    metric_key: str = DEFAULT_METRIC_KEY
    """Metric to use for loss (default: eval/paloma/c4_en/bpb)."""

    label_map: tuple[tuple[str, str], ...] | None = None
    """Optional mapping from experiment_name -> display label as tuple of pairs."""

    metrics_filename: str = "tracker_metrics.jsonl"
    """Name of the metrics file within each checkpoint directory."""

    wandb_entity_project: str = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    """WandB entity/project to query for backfill (format: 'entity/project')."""


def run_isoflop_analysis_step(config: IsoFlopAnalysisConfig) -> FitScalingLawsResult:
    """Execute IsoFLOP scaling law analysis.

    This is the experiment step function that:
    1. Reads raw metrics from training runs
    2. Transforms them using Levanter schema knowledge
    3. Runs the scaling law analysis
    4. Saves results to output_path

    Args:
        config: Analysis config with training_runs and analysis settings

    Returns:
        FitScalingLawsResult with fitted scaling laws
    """
    import json

    import fsspec

    # Read raw records from training runs
    raw_records = read_raw_records(config)

    if not raw_records:
        logger.warning("No eval metrics found in training runs")
        return FitScalingLawsResult(minima_records=[], scaling_fits={}, fit_curves={})

    # Transform to typed records using Levanter schema knowledge
    label_map = dict(config.label_map) if config.label_map else None
    records = transform_levanter_metrics(raw_records, config.metric_key, label_map)

    if not records:
        logger.warning("No valid isoflop data after transformation")
        return FitScalingLawsResult(minima_records=[], scaling_fits={}, fit_curves={})

    logger.info(f"Loaded {len(records)} runs for scaling law analysis")
    labels = list(dict.fromkeys(r.label for r in records))
    flops_budgets = sorted(set(r.flops for r in records))
    logger.info(f"Labels found: {labels}")
    logger.info(f"FLOP budgets: {flops_budgets}")

    # Run scaling law analysis
    result = fit_scaling_laws(records)

    logger.info(f"Found {len(result.minima_records)} optimal configurations")
    for label, scaling_fit in result.scaling_fits.items():
        logger.info(f"  {label}: D* = {scaling_fit.A:.2e} * C^{scaling_fit.alpha:.3f}")

    # Save results
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    result_path = os.path.join(config.output_path, "isoflop_analysis_result.json")
    result_dict = {
        "minima_records": [
            {
                "label": r.label,
                "flops": r.flops,
                "optimal_tokens": r.optimal_tokens,
                "loss_at_optimal": r.loss_at_optimal,
                "optimal_params": r.optimal_params,
                "scaling_alpha": r.scaling_alpha,
                "scaling_A": r.scaling_A,
            }
            for r in result.minima_records
        ],
        "scaling_fits": {k: list(v) for k, v in result.scaling_fits.items()},
    }
    with fs.open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"Saved results to {result_path}")

    # Save fit curves for downstream plotting
    fit_curves_path = os.path.join(config.output_path, "fit_curves.json")
    fit_curves_json = {f"{label}|{flops}": list(coeffs) for (label, flops), coeffs in result.fit_curves.items()}
    with fs.open(fit_curves_path, "w") as f:
        json.dump(fit_curves_json, f, indent=2)
    logger.info(f"Saved fit curves to {fit_curves_path}")

    return result


def create_isoflop_sweep_steps(
    tokenized: InputName | str | LMMixtureDatasetConfig,
    experiment_name: str,
    recipe: ScalingRecipe,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    eval_tasks: tuple[EvalTaskConfig, ...] | None = None,
    seq_len: int = 4096,
) -> tuple[list[ExecutorStep], list[CandidateConfig]]:
    """Create ExecutorSteps for an ISOFlop sweep.

    This function creates ExecutorSteps directly in experiment code, using
    `generate_isoflop_train_args()` from the library to compute configs.

    Args:
        tokenized: Tokenized dataset to train on.
        experiment_name: Name suffix for the experiment (e.g., 'nemo', 'dclm').
        recipe: ScalingRecipe with hyperparameters (includes vocab_size).
        budgets: FLOP budgets to sweep over.
        eval_tasks: Optional evaluation tasks to run after training.
        seq_len: Sequence length for training.

    Returns:
        A tuple of:
        - steps: Training and evaluation ExecutorSteps for the sweep.
        - candidates: CandidateConfig for each training run with full config details.
    """
    # Library provides the training arguments (model configs, optimizer configs, etc.)
    # vocab_size is owned by the recipe
    train_args_list = generate_isoflop_train_args(
        budgets=budgets,
        recipe=recipe,
    )

    # Base config for training runs (values overridden per-candidate via optimizer_config)
    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=1,
        num_train_steps=50_000,
        learning_rate=1.0,  # Overridden via optimizer_config
    )

    train_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    candidates: list[CandidateConfig] = []

    # Create ExecutorSteps for each candidate configuration
    for args in train_args_list:
        candidate = args.candidate

        # Model config is on the candidate; build optimizer config using the recipe
        model_config = candidate.model_config
        optimizer_config = recipe.build_optimizer_config(candidate, seq_len)
        tpu_type = pick_v5p_type(candidate, seq_len, recipe)

        # Compute training schedule from recipe
        batch_size, num_steps = recipe.compute_training_schedule(candidate, seq_len)

        # Use local naming with architecture details for backward compatibility
        run_name = _format_run_name(
            candidate.flops_budget,
            model_config.hidden_dim,
            model_config.num_layers,
            batch_size,
            experiment_name,
        )
        output_path = f"checkpoints/isoflop/{run_name}"

        train_cfg = replace(
            base_train_config,
            train_batch_size=batch_size,
            learning_rate=optimizer_config.learning_rate,
            num_train_steps=num_steps,
            resources=ResourceConfig.with_tpu(tpu_type),
            optimizer_config=optimizer_config,
        )

        # Create training step
        train_step = default_train(
            name=run_name,
            tokenized=tokenized,
            model_config=model_config,
            train_config=train_cfg,
            eval_harness_tasks=[],
            tags=args.tags,
        )

        # Pin to static output path for checkpoint reuse
        train_step = train_step.with_output_path(output_path)
        train_steps.append(train_step)
        candidates.append(candidate)

        # Create evaluation step if eval tasks specified
        if eval_tasks:
            eval_step = default_eval(
                train_step,
                resource_config=train_cfg.resources,
                evals=eval_tasks,
            )
            eval_steps.append(eval_step)

    all_steps: list[ExecutorStep] = [*train_steps, *eval_steps]
    return all_steps, candidates


# --- Tokenized Datasets ---

dclm_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=downloads["dclm_baseline"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dclm_baseline-0206f1/")

dclm_mix = lm_mixture_data_config(
    components={"dclm": dclm_tokenized},
    weights={"dclm": 1.0},
    num_validation_sequences={"dclm": 1024},
)

dolma3_mix_tokenized = default_tokenize(
    name="dolma3_mix-150B-1025",
    dataset=downloads["dolma3_mix_150b_1025"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dolma3_mix-150B-1025-15d04ee/")

dolma3_mix = lm_mixture_data_config(
    components={"dolma3_mix-150B-1025": dolma3_mix_tokenized},
    weights={"dolma3_mix-150B-1025": 1.0},
    num_validation_sequences={"dolma3_mix-150B-1025": 1024},
)


MARIN_SCALING_SUITES = {
    "nemotron": create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="nemo-wider-depth-adapt",
        recipe=MARIN_2025_RECIPE,
        budgets=LEGACY_BUDGETS,
    ),
    "common_pile": create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(permutation_type="linear"),
        experiment_name="comma-mix",
        recipe=MARIN_2025_RECIPE,
        budgets=LEGACY_BUDGETS,
    ),
    "common_pile_feistel": create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(permutation_type="feistel"),
        experiment_name="comma-mix-feistel",
        recipe=MARIN_2025_RECIPE,
        budgets=LEGACY_BUDGETS,
    ),
    "dclm-default": create_isoflop_sweep_steps(
        tokenized=dclm_mix,
        experiment_name="dclm-default",
        recipe=MARIN_2025_RECIPE,
        budgets=LEGACY_BUDGETS,
    ),
    "dolma3_mix_150b": create_isoflop_sweep_steps(
        tokenized=dolma3_mix,
        experiment_name="dolma3-mix-150b-1025",
        recipe=MARIN_2025_RECIPE,
        budgets=LEGACY_BUDGETS,
    ),
}

if __name__ == "__main__":
    steps, _ = MARIN_SCALING_SUITES["dolma3_mix_150b"]
    executor_main(steps=steps)
