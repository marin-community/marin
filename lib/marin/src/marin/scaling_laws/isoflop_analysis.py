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

"""IsoFLOP analysis for finding compute-optimal training configurations.

Primary usage - create ExecutorSteps for your pipeline:

    from marin.scaling_laws import (
        isoflop_analysis_step,
        isoflop_plots_step,
        upload_isoflop_plots_to_wandb_step,
    )

    # Step 1: Compute metrics and fit scaling laws
    analysis = isoflop_analysis_step(
        name="my-scaling-analysis",
        training_runs=my_training_steps,  # list of ExecutorStep
    )

    # Step 2: Generate HTML plots (optional)
    plots = isoflop_plots_step(
        name="my-scaling-plots",
        analysis_step=analysis,
    )

    # Step 3: Upload to WandB (optional)
    upload = upload_isoflop_plots_to_wandb_step(
        name="upload-scaling-plots",
        analysis_step=analysis,
    )

The analysis step will:
1. Read eval metrics from completed training runs
2. Fit scaling laws to find compute-optimal token counts
3. Save results to JSON/parquet files

For programmatic use, see `run_isoflop_analysis()` which returns a `IsoFlopAnalysisResult`.
"""

import json
import logging
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from typing import NamedTuple

import fsspec
import jax.numpy as jnp
import pandas as pd
from jaxopt import ScipyMinimize

from levanter.models.llama import LlamaConfig
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path
from marin.scaling_laws.eval_metrics_reader import (
    EvalMetricsAnalysisConfig,
    extract_run_name_from_path,
    read_metrics_dataframe,
)
from marin.scaling_laws.recipe import MARIN_2025_RECIPE, ScalingRecipe
from marin.scaling_laws.tpu_utils import (
    pick_v5p_type,
)
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger(__name__)

# ---------------- Constants ----------------

# Paloma is a standard LLM evaluation benchmark. C4-en BPB (bits-per-byte) is a
# common loss metric that measures model perplexity on the C4 English dataset.
# See: https://arxiv.org/abs/2312.10523
DEFAULT_EVAL_METRIC_KEY = "eval/paloma/c4_en/bpb"
SEQ_LEN = 4096

# Marin tokenizer vocab size (stanford-crfm/marin-tokenizer)
MARIN_TOKENIZER_VOCAB_SIZE = 128256

# ---------------- IsoFLOP Sweep Constants ----------------
# Budgets in training FLOPs (includes 3x multiplier for forward + backward pass).
# This matches how FLOPs are tracked in WandB via Levanter's log_performance_stats.
DEFAULT_BUDGETS: tuple[float, ...] = (1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20)

# Derived from Kaiyue's hyperparameter sweep: optimal_LR * hidden_size * sqrt(batch_size)
LR_CONSTANT = 0.33

# ---------------- WandB Metric Keys ----------------
# These keys correspond to the metrics logged by Levanter's training callbacks.
THROUGHPUT_TOKENS_KEY = "throughput/total_tokens"
THROUGHPUT_GFLOPS_KEY = "throughput/total_gflops"
PARAMETER_COUNT_KEY = "parameter_count"
MODEL_CONFIG_KEY = "model"
TRAINER_CONFIG_KEY = "trainer"


# ---------------- Typed Tuples ----------------


class ScalingFit(NamedTuple):
    """Scaling law fit parameters for N* ~ A * C^alpha."""

    alpha: float
    """Exponent in scaling law."""

    A: float
    """Coefficient in scaling law."""


class QuadraticFitCoeffs(NamedTuple):
    """Quadratic fit coefficients for loss = a * log10(tokens)^2 + b * log10(tokens) + c."""

    a: float
    """Quadratic coefficient."""

    b: float
    """Linear coefficient."""

    c: float
    """Constant term."""

    token_min: float
    """Minimum token count used for fitting."""

    token_max: float
    """Maximum token count used for fitting."""


# ---------------- IsoFLOP Sweep Config ----------------
@dataclass(frozen=True)
class IsoFlopSweepConfig:
    """Configuration for generating ISOFlop sweep candidate configs.

    This config controls the FLOP budgets and training parameters.
    Architecture decisions (num_layers formula, hidden_pow bounds, etc.)
    are controlled by the ScalingRecipe.
    """

    recipe: ScalingRecipe = MARIN_2025_RECIPE
    """Scaling recipe with all opinionated hyperparameters:
    - Architecture formula (num_layers from hidden_size)
    - Architecture ratios (mlp_ratio, hidden_head_ratio)
    - Search bounds (min/max hidden_pow, step_size)
    - Constraints (max_learning_rate, min_batch_size)
    - Optimizer settings (weight_decay, warmup, etc.)
    """

    tokenizer: str = "stanford-crfm/marin-tokenizer"
    """Tokenizer to use (needed for vocab size)."""

    budgets: tuple[float, ...] = DEFAULT_BUDGETS
    """Tuple of FLOP budgets to generate configs for."""

    seq_len: int = SEQ_LEN
    """Sequence length for training."""

    steps_per_run: int = 2**16
    """Number of training steps used for FLOP budget calculation and hyperparameter tuning.

    This is the reference step count that other hyperparameters (LR, beta2) are tuned for.
    The actual training steps may differ based on batch size to hit the target FLOP budget.
    Default of 2^16 (65,536) steps is used because the LR_CONSTANT and other tuned values
    were optimized for this step count.
    """

    flop_tolerance: float = 0.01
    """Tolerance for matching FLOP budget (relative error)."""


# ---------------- Candidate Config ----------------


@dataclass
class CandidateConfig:
    """A candidate model/training configuration from the isoflop sweep.

    This dataclass contains all the information needed to create a training run.
    Callers are responsible for converting this to their specific config format
    (e.g., SimpleTrainConfig, Qwen3Config).
    """

    hidden_size: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    batch_size: int
    train_steps: int
    learning_rate: float
    beta2: float
    tokens: float  # total tokens = batch_size * train_steps * seq_len
    flops_budget: float = 0.0  # the FLOP budget this config was generated for


@dataclass
class IsoFlopTrainArgs:
    """Arguments needed to set up an isoflop training run.

    This dataclass contains the parameters needed for training. The caller is
    responsible for constructing the model config from candidate parameters,
    allowing flexibility in model type (Qwen3Config, LlamaConfig, etc.).

    Example:
        >>> args = generate_isoflop_train_args(config, "my-exp", vocab_size)[0]
        >>> # Caller constructs the model config
        >>> model_config = Qwen3Config(
        ...     hidden_dim=args.candidate.hidden_size,
        ...     intermediate_dim=args.candidate.intermediate_dim,
        ...     num_layers=args.candidate.num_layers,
        ...     num_heads=args.candidate.num_heads,
        ...     num_kv_heads=args.candidate.num_kv_heads,
        ...     max_seq_len=4096,
        ...     rope=Llama3RotaryEmbeddingsConfig(),
        ... )
    """

    candidate: CandidateConfig
    """The candidate configuration with model/training hyperparameters."""

    optimizer_config: OptimizerConfig
    """Levanter optimizer configuration with learning_rate and beta2 set."""

    tpu_type: str
    """TPU slice type (e.g., 'v5p-8', 'v5p-32')."""

    run_name: str
    """Name for the training run."""

    tags: tuple[str, ...]
    """Tags for tracking/filtering runs."""

    output_path: str
    """Static output path for checkpoints."""


# ---------------- Typed Records ----------------


@dataclass
class MinimaRecord:
    """Record of optimal configuration found at a specific (label, flops) point."""

    label: str
    flops: float
    optimal_tokens: float
    loss_at_optimal: float
    hidden_dim: int
    num_layers: int
    batch_size: int
    optimal_params: float
    scaling_alpha: float | None = None
    scaling_A: float | None = None


@dataclass
class FitScalingLawsResult:
    """Result from fit_scaling_laws containing minima, scaling fits, and fit curves."""

    minima_records: list[MinimaRecord]
    """List of optimal configurations found at each (label, flops) point."""

    scaling_fits: dict[str, ScalingFit]
    """Per-label scaling fits: {label: ScalingFit} for N* ~ A * C^alpha."""

    fit_curves: dict[tuple[str, float], QuadraticFitCoeffs]
    """Quadratic fit coefficients {(label, flops): QuadraticFitCoeffs} for plotting."""


# ---------------- Candidate Config Generation ----------------


def round_to_power_of_two(x: float) -> int:
    """Round ``x`` to the nearest power of two."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def round_flops_to_bucket(flops: float) -> float:
    """Round FLOP count to 1 significant figure (XeYY format).

    This ensures runs with slightly different achieved FLOPs are grouped
    together for analysis when they were targeting the same budget.
    Using 1 significant figure creates buckets at 1e19, 2e19, 3e19, etc.,
    which matches the typical spacing of isoflop budget targets.

    Note: This means 1.5e19 and 2.4e19 both map to 2e19. For finer granularity,
    consider using 2 significant figures (round to nearest 0.1 mantissa).

    Examples:
        1.05e19 → 1e19
        1.5e19  → 2e19
        2.8e19  → 3e19
        9.5e19  → 1e20
    """
    if flops <= 0:
        return flops

    exponent = math.floor(math.log10(flops))
    mantissa = flops / (10**exponent)
    rounded_mantissa = round(mantissa)

    if rounded_mantissa == 10:
        return 1.0 * (10 ** (exponent + 1))
    return float(rounded_mantissa) * (10**exponent)


def compute_training_flops(
    model_config: "LlamaConfig",
    vocab_size: int,
    batch_size: int,
    train_steps: int,
    seq_len: int,
) -> float:
    """Compute total training FLOPs using the model config's own method.

    This returns training FLOPs which includes forward pass (1x) + backward pass (2x) = 3x.
    This matches the FLOP accounting in Levanter's log_performance_stats callback
    (see train_lm.py) and standard ML conventions (e.g., Chinchilla paper).

    Args:
        model_config: Levanter model config with flops_per_token method (LlamaConfig or subclass).
        vocab_size: Vocabulary size.
        batch_size: Training batch size.
        train_steps: Number of training steps.
        seq_len: Sequence length.

    Returns:
        Total training FLOPs (including 3x multiplier for forward + backward pass).
    """
    flops_per_token = model_config.flops_per_token(vocab_size, seq_len)
    # Multiply by 3 for training: forward (1x) + backward (2x)
    return 3 * flops_per_token * batch_size * train_steps * seq_len


def solve_for_batch_size(
    model_config: "LlamaConfig",
    vocab_size: int,
    target_flops: float,
    train_steps: int,
    seq_len: int,
) -> float:
    """Solve for batch size needed to hit a target FLOP budget.

    Given: total_flops = 3 * flops_per_token * batch * steps * seq_len
    Solve: batch = total_flops / (3 * flops_per_token * steps * seq_len)

    Args:
        model_config: Levanter model config with flops_per_token method.
        vocab_size: Vocabulary size.
        target_flops: Target total training FLOPs.
        train_steps: Number of training steps.
        seq_len: Sequence length.

    Returns:
        Exact batch size (float) - caller decides how to round.
    """
    flops_per_token = model_config.flops_per_token(vocab_size, seq_len)
    return target_flops / (3 * flops_per_token * train_steps * seq_len)


def solve_for_train_steps(
    model_config: "LlamaConfig",
    vocab_size: int,
    target_flops: float,
    batch_size: int,
    seq_len: int,
) -> float:
    """Solve for training steps needed to hit a target FLOP budget.

    Given: total_flops = 3 * flops_per_token * batch * steps * seq_len
    Solve: steps = total_flops / (3 * flops_per_token * batch * seq_len)

    Args:
        model_config: Levanter model config with flops_per_token method.
        vocab_size: Vocabulary size.
        target_flops: Target total training FLOPs.
        batch_size: Training batch size.
        seq_len: Sequence length.

    Returns:
        Exact training steps (float) - caller decides how to round.
    """
    flops_per_token = model_config.flops_per_token(vocab_size, seq_len)
    return target_flops / (3 * flops_per_token * batch_size * seq_len)


def compute_transformer_params(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    vocab_size: int,
    tie_embeddings: bool = False,
) -> int:
    """Compute parameter count for a standard transformer (Llama/Qwen architecture).

    This matches the formula used in Levanter's LlamaConfig.total_trainable_params(),
    allowing parameter estimation without constructing a model config.

    Args:
        hidden_dim: Model hidden dimension.
        intermediate_dim: MLP intermediate dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        vocab_size: Vocabulary size.
        tie_embeddings: Whether embeddings are tied (default False).

    Returns:
        Total parameter count.
    """
    token_embedding = vocab_size * hidden_dim
    head_size = hidden_dim // num_heads

    # Attention: Q, K, V projections + output projection
    q_proj = hidden_dim * head_size * num_heads
    kv_proj = 2 * hidden_dim * head_size * num_kv_heads
    o_proj = head_size * num_heads * hidden_dim
    attn = q_proj + kv_proj + o_proj

    # MLP: gate, up, down projections (SwiGLU uses 3 matrices)
    mlp = 3 * hidden_dim * intermediate_dim

    # Per-layer: attention + mlp + 2 RMSNorm
    transformer_layer = attn + mlp + 2 * hidden_dim

    # Full transformer: layers + final RMSNorm
    transformer = num_layers * transformer_layer + hidden_dim

    # LM head (separate unless tied)
    lm_head = 0 if tie_embeddings else token_embedding

    return transformer + token_embedding + lm_head


def candidate_configs(
    cfg: IsoFlopSweepConfig,
    budget: float,
    vocab_size: int,
) -> Iterator[CandidateConfig]:
    """Yield candidate model configurations within the FLOP budget.

    This function uses the recipe for all opinionated choices:
    - Architecture formula (num_layers from hidden_size)
    - Architecture ratios (mlp_ratio, hidden_head_ratio)
    - Search bounds (min/max hidden_pow, step_size)
    - Constraints (max_learning_rate, min_batch_size)

    The mechanics layer (solve_for_batch_size, solve_for_train_steps, compute_training_flops)
    handles the pure FLOP math.

    Args:
        cfg: IsoFlopSweepConfig with recipe and other search parameters
        budget: Target FLOP budget
        vocab_size: Vocabulary size for the tokenizer

    Yields:
        CandidateConfig objects for each valid configuration
    """
    recipe = cfg.recipe

    # RECIPE: Get search parameters
    step_size = recipe.get_step_size(budget)
    min_hidden = 2**recipe.min_hidden_pow
    max_hidden = 2**recipe.max_hidden_pow

    for hidden_size in range(min_hidden, max_hidden + 1, step_size):
        # RECIPE: Build model config (makes all architecture decisions)
        model_config = recipe.build_model_config(hidden_size, cfg.seq_len)

        # MECHANICS: Solve for batch size to hit budget with target steps
        batch_exact = solve_for_batch_size(model_config, vocab_size, budget, cfg.steps_per_run, cfg.seq_len)
        batch_size = round_to_power_of_two(batch_exact)

        # RECIPE: Apply LR constraint
        lr = recipe.compute_learning_rate(batch_size, hidden_size)
        while lr > recipe.max_learning_rate:
            batch_size //= 2
            lr = recipe.compute_learning_rate(batch_size, hidden_size)

        # RECIPE: Apply min batch constraint
        if batch_size < recipe.min_batch_size:
            continue

        # MECHANICS: Solve for steps to hit budget with chosen batch
        train_steps = round(solve_for_train_steps(model_config, vocab_size, budget, batch_size, cfg.seq_len))

        # MECHANICS: Verify we hit the budget within tolerance
        achieved_flops = compute_training_flops(model_config, vocab_size, batch_size, train_steps, cfg.seq_len)
        if abs(achieved_flops - budget) / budget > cfg.flop_tolerance:
            continue

        # RECIPE: Compute optimizer hyperparameters
        beta2 = recipe.compute_beta2(batch_size)
        tokens = batch_size * train_steps * cfg.seq_len

        yield CandidateConfig(
            hidden_size=hidden_size,
            intermediate_dim=model_config.intermediate_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            batch_size=batch_size,
            train_steps=train_steps,
            learning_rate=lr,
            beta2=beta2,
            tokens=tokens,
            flops_budget=budget,
        )


def _minima_to_candidates(
    minima_records: list[MinimaRecord],
    recipe: ScalingRecipe = MARIN_2025_RECIPE,
) -> list[CandidateConfig]:
    """Convert minima records to CandidateConfig objects.

    This is used by both run_isoflop_analysis_step() and run_isoflop_analysis()
    to convert the fitted minima into usable candidate configs.

    Args:
        minima_records: List of optimal configurations from scaling law fits.
        recipe: ScalingRecipe with architecture and hyperparameter settings.
    """
    configs = []
    for rec in minima_records:
        if rec.hidden_dim == 0:
            continue
        configs.append(
            CandidateConfig(
                hidden_size=rec.hidden_dim,
                intermediate_dim=rec.hidden_dim * recipe.mlp_ratio,
                num_layers=rec.num_layers,
                num_heads=max(1, rec.hidden_dim // recipe.hidden_head_ratio),
                num_kv_heads=max(1, rec.hidden_dim // recipe.hidden_head_ratio),
                batch_size=rec.batch_size,
                train_steps=int(rec.optimal_tokens / (rec.batch_size * SEQ_LEN)),
                learning_rate=recipe.compute_learning_rate(rec.batch_size, rec.hidden_dim),
                beta2=recipe.compute_beta2(rec.batch_size),
                tokens=rec.optimal_tokens,
                flops_budget=rec.flops,
            )
        )
    return configs


# ---------------- Training Args Generation ----------------


def generate_isoflop_train_args(
    sweep_config: IsoFlopSweepConfig,
    experiment_name: str,
    vocab_size: int,
    base_optimizer_config: OptimizerConfig | None = None,
) -> list[IsoFlopTrainArgs]:
    """Generate training arguments for each candidate in an isoflop sweep.

    This function generates all the arguments needed to call default_train() for
    each candidate configuration in the sweep. The caller is responsible for
    constructing the experiment-specific SimpleTrainConfig.

    Args:
        sweep_config: Configuration for the sweep (budgets, seq_len, etc.)
        experiment_name: Name suffix for run names (e.g., 'nemo', 'dclm')
        vocab_size: Vocabulary size for the tokenizer
        base_optimizer_config: Base optimizer config to modify. If None, uses CautiousConfig defaults.

    Returns:
        List of IsoFlopTrainArgs, one per candidate config across all budgets.

    Example:
        >>> from marin.scaling_laws import IsoFlopSweepConfig, generate_isoflop_train_args
        >>> from levanter.models.qwen import Qwen3Config
        >>> config = IsoFlopSweepConfig(budgets=(1e18, 1e19))
        >>> train_args = generate_isoflop_train_args(config, "my-experiment", vocab_size=128256)
        >>> for args in train_args:
        ...     # Caller constructs the model config from candidate parameters
        ...     model_config = Qwen3Config(
        ...         hidden_dim=args.candidate.hidden_size,
        ...         intermediate_dim=args.candidate.intermediate_dim,
        ...         num_layers=args.candidate.num_layers,
        ...         # ... etc
        ...     )
        ...     # Then use model_config with default_train()
    """
    recipe = sweep_config.recipe
    if base_optimizer_config is None:
        base_optimizer_config = CautiousConfig(
            learning_rate=1.0,  # Placeholder, will be overridden
            weight_decay=recipe.weight_decay,
            min_lr_ratio=recipe.min_lr_ratio,
            warmup=recipe.warmup,
            beta1=recipe.beta1,
            beta2=0.98,  # Placeholder, will be overridden
            epsilon=recipe.epsilon,
            max_grad_norm=recipe.max_grad_norm,
            adamc_weight_decay=True,
            lr_schedule=recipe.lr_schedule,
            decay=recipe.decay,
        )

    results: list[IsoFlopTrainArgs] = []

    for budget in sweep_config.budgets:
        for candidate in candidate_configs(sweep_config, budget, vocab_size):
            # Pick TPU type based on candidate parameters
            tpu_type = pick_v5p_type(candidate, vocab_size, sweep_config.seq_len)

            # Build optimizer config with candidate-specific LR and beta2
            optimizer_cfg = replace(
                base_optimizer_config,
                learning_rate=candidate.learning_rate,
                beta2=candidate.beta2,
            )

            # Generate run name and tags
            run_name = (
                f"isoflop-{budget:.0e}-d{candidate.hidden_size}-"
                f"L{candidate.num_layers}-B{candidate.batch_size}-{experiment_name}"
            )

            tags = (
                f"FLOPs={budget:.1e}",
                f"d={candidate.hidden_size}",
                f"L={candidate.num_layers}",
                f"B={candidate.batch_size}",
                f"steps={candidate.train_steps}",
                f"tpu={tpu_type}",
            )

            # Static output path for checkpoint reuse
            output_path = os.path.join("checkpoints", "isoflop", run_name)

            results.append(
                IsoFlopTrainArgs(
                    candidate=candidate,
                    optimizer_config=optimizer_cfg,
                    tpu_type=tpu_type,
                    run_name=run_name,
                    tags=tags,
                    output_path=output_path,
                )
            )

    return results


# ---------------- Helpers ----------------


def _resolve_run_paths(runs: Sequence[ExecutorStep | InputName | str]) -> list[InputName | str]:
    """Convert mixed ExecutorStep/InputName/path inputs to executor-ready paths.

    This helper reduces duplication across functions that accept either
    ExecutorSteps or string paths.

    Args:
        runs: Sequence of ExecutorStep, InputName, or path strings.

    Returns:
        List of InputName or string paths.
    """
    return [output_path_of(run) if isinstance(run, ExecutorStep) else run for run in runs]


def parse_isoflop_run_name(run_name: str) -> str | None:
    """Parse experiment name from isoflop run name.

    Expected format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    Optionally with a trailing -<hash> which is ignored.
    E.g., 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt'
    or 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt-a1b2c3'

    Returns experiment_name or None if parsing fails.
    """
    # Strip optional -<hash> suffix
    run_name = re.sub(r"-[0-9a-fA-F]{6}$", "", run_name)

    pattern = r"isoflop-(?:[0-9.e+]+)-d(?:\d+)-L(?:\d+)-B(?:\d+)-(.+)"
    match = re.match(pattern, run_name)
    if not match:
        return None

    return match.group(1)


def robust_quad_logx(x: jnp.ndarray, y: jnp.ndarray, delta: float = 1.0) -> tuple[float, float, float]:
    """Fit a robust quadratic in log10(x) space using Huber loss.

    Log10 space is used because FLOP budgets and token counts span many orders of
    magnitude (e.g., 1e18 to 1e21+). Fitting in linear space would be numerically
    unstable and dominated by the largest values. Log space provides better
    conditioning and more interpretable coefficients.

    The Huber loss provides robustness to outliers compared to standard least squares.

    Args:
        x: Input array (e.g., token counts). Must be positive.
        y: Output array (e.g., loss values).
        delta: Huber loss threshold. Residuals larger than delta use linear loss.

    Returns:
        Tuple (a, b, c) of coefficients for: loss = a * log10(x)^2 + b * log10(x) + c
    """
    L = jnp.log10(x)

    def huber(residual):
        abs_r = jnp.abs(residual)
        quad = 0.5 * residual**2
        linear = delta * (abs_r - 0.5 * delta)
        return jnp.where(abs_r <= delta, quad, linear)

    def objective(params):
        a, b, c = params
        pred = a * L**2 + b * L + c
        residuals = y - pred
        return jnp.sum(huber(residuals))

    opt = ScipyMinimize(fun=objective, method="BFGS", value_and_grad=False)
    init = jnp.array(jnp.polyfit(L, y, 2)) if len(L) >= 3 else jnp.array([0.0, *jnp.polyfit(L, y, 1)])
    result = opt.run(init_params=init).params
    return float(result[0]), float(result[1]), float(result[2])


# ---------------- Core Analysis ----------------


def fit_scaling_laws(
    df: pd.DataFrame,
) -> FitScalingLawsResult:
    """
    Fit scaling laws and extract optimal configurations.

    Args:
        df: DataFrame with columns: tokens, loss, flops, params, name, label

    Returns:
        FitScalingLawsResult containing minima_records, scaling_fits, and fit_curves.
    """
    if df is None or df.empty:
        return FitScalingLawsResult(minima_records=[], scaling_fits={}, fit_curves={})

    datasets = list(dict.fromkeys(df["label"].tolist()))

    buckets = sorted(df.flops.unique())

    minima_records: list[MinimaRecord] = []
    fit_curves: dict[tuple[str, float], QuadraticFitCoeffs] = {}

    # Fit quadratic for each (label, budget) and find minima
    for lab in datasets:
        for C in buckets:
            sub = df[(df.flops == C) & (df.label == lab)].sort_values("tokens")
            if sub.empty:
                continue

            # Robust quadratic fit in log10(tokens)
            # Use float64 to avoid int32 overflow for token counts > 2^31
            tokens_array = jnp.array(sub.tokens.values, dtype=jnp.float64)
            a, b, c = robust_quad_logx(
                tokens_array,
                jnp.array(sub.loss.values, dtype=jnp.float64),
            )
            # Store coefficients along with token range used for fitting
            fit_curves[(lab, C)] = QuadraticFitCoeffs(a, b, c, float(tokens_array.min()), float(tokens_array.max()))

            if a == 0:
                continue

            L_opt = -b / (2 * a)
            N_star = float(10**L_opt)
            loss_opt = float(a * L_opt**2 + b * L_opt + c)

            idx = (sub.tokens - N_star).abs().argmin()
            nearest_row = sub.iloc[idx]

            minima_records.append(
                MinimaRecord(
                    label=lab,
                    flops=float(C),
                    optimal_tokens=N_star,
                    loss_at_optimal=loss_opt,
                    hidden_dim=int(nearest_row["hidden_dim"]),
                    num_layers=int(nearest_row["num_layers"]),
                    batch_size=int(nearest_row["batch_size"]),
                    optimal_params=float(nearest_row.get("params") or C / (6 * N_star)),
                )
            )

    # Fit scaling law N* ~ A * C^alpha per dataset
    scaling_fits: dict[str, ScalingFit] = {}
    by_lab: dict[str, list[MinimaRecord]] = {}
    for rec in minima_records:
        by_lab.setdefault(rec.label, []).append(rec)

    for lab in datasets:
        recs = by_lab.get(lab, [])
        if len(recs) < 2:
            continue

        recs = sorted(recs, key=lambda r: r.flops)
        Cs = jnp.array([r.flops for r in recs])
        Ns = jnp.array([r.optimal_tokens for r in recs])

        alpha, logA = jnp.polyfit(jnp.log10(Cs), jnp.log10(Ns), 1)
        A = float(10**logA)
        alpha = float(alpha)
        scaling_fits[lab] = ScalingFit(alpha, A)

        # Augment minima records with scaling fit params
        for rec in recs:
            rec.scaling_alpha = alpha
            rec.scaling_A = A

    return FitScalingLawsResult(
        minima_records=minima_records,
        scaling_fits=scaling_fits,
        fit_curves=fit_curves,
    )


def transform_metrics_for_isoflop(
    df: pd.DataFrame,
    metric_key: str,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Transform raw metrics DataFrame into isoflop analysis format.

    Takes the generic metrics DataFrame from read_metrics_dataframe() and
    transforms it into the format expected by the analysis:
    columns: tokens, loss, flops, params, name, label

    The DataFrame contains nested 'config' and 'summary' dicts from tracker_metrics.jsonl.

    Args:
        df: Raw metrics DataFrame from read_metrics_dataframe()
        metric_key: Which metric column to use for loss (e.g., 'eval/paloma/c4_en/bpb')
        label_map: Optional mapping from experiment_name -> display label

    Returns:
        Transformed DataFrame ready for fit_scaling_laws()
    """
    if df.empty:
        return pd.DataFrame(columns=["tokens", "loss", "flops", "params", "name", "label"])

    records = []
    for _, row in df.iterrows():
        run_path = row["run_path"]
        run_name = extract_run_name_from_path(run_path)

        # Extract config and summary dicts
        config = row.get("config", {}) or {}
        summary = row.get("summary", {}) or {}
        model_config = config.get(MODEL_CONFIG_KEY, {}) or {}
        trainer_config = config.get(TRAINER_CONFIG_KEY, {}) or {}

        # Get tokens directly from summary
        tokens = summary.get(THROUGHPUT_TOKENS_KEY)
        if tokens is None or pd.isna(tokens):
            logger.warning(f"Missing {THROUGHPUT_TOKENS_KEY} in summary for run {run_name}")
            continue

        # Get total FLOPs from summary (convert GFLOPs to FLOPs)
        total_gflops = summary.get(THROUGHPUT_GFLOPS_KEY)
        if total_gflops is None or pd.isna(total_gflops):
            logger.warning(f"Missing {THROUGHPUT_GFLOPS_KEY} in summary for run {run_name}")
            continue
        flops = round_flops_to_bucket(total_gflops * 1e9)

        if flops < 1e18:
            continue

        # Get loss from summary[metric_key]
        loss = summary.get(metric_key)
        if loss is None or pd.isna(loss):
            logger.warning(f"Missing metric {metric_key} for run {run_name}")
            continue

        # Get parameter count from summary
        params = summary.get(PARAMETER_COUNT_KEY)
        if params is None or pd.isna(params):
            params = None

        # Get model architecture from config
        hidden_dim = model_config.get("hidden_dim")
        num_layers = model_config.get("num_layers")
        batch_size = trainer_config.get("train_batch_size")

        # Determine experiment name and label from run name
        exp_name = parse_isoflop_run_name(run_name) or run_name
        if label_map and exp_name in label_map:
            label = label_map[exp_name]
        else:
            label = exp_name

        records.append(
            dict(
                tokens=tokens,
                loss=loss,
                flops=flops,
                params=params,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                batch_size=batch_size,
                name=run_name,
                label=label,
            )
        )

    return pd.DataFrame.from_records(records)


# ---------------- Predict Optimal Config ----------------


def predict_optimal_config(
    scaling_fits: dict[str, ScalingFit],
    target_flops: float,
    label: str,
    sweep_config: IsoFlopSweepConfig | None = None,
    vocab_size: int = MARIN_TOKENIZER_VOCAB_SIZE,
) -> CandidateConfig | None:
    """Predict optimal training config for a target compute budget using fitted scaling laws.

    This implements IsoFLOP Approach 2 from the Chinchilla paper:
    1. D_opt (optimal tokens) is found empirically at each compute budget by fitting
       parabolas to actual loss values and finding the minimum.
    2. D_opt ~ A * C^alpha is fitted from those empirical minima.
    3. Given D_opt and C, N_opt (optimal params) is derived as C/(6D), so no
       separate alpha fit for params is needed.

    This approach works regardless of whether the scaling exponents for params
    and tokens are equal (alpha == beta), unlike Approach 3 which fits a
    parametric loss surface.

    This function:
    1. Uses the scaling fit (N* ~ A * C^alpha) to predict optimal tokens for target_flops
    2. Generates candidate configs for the target budget using candidate_configs()
    3. Selects the candidate whose token count is closest to the predicted optimal

    Args:
        scaling_fits: Dict of {label: ScalingFit} from scaling ladder result.
        target_flops: Target compute budget in FLOPs.
        label: Dataset/experiment label to use for scaling fit.
        sweep_config: Optional IsoFlopSweepConfig. If None, uses defaults.
        vocab_size: Vocabulary size (default: MARIN_TOKENIZER_VOCAB_SIZE for marin tokenizer).

    Returns:
        CandidateConfig for the predicted optimal, or None if label not in fits
        or no valid candidates found.
    """
    if label not in scaling_fits:
        logger.warning(f"Label '{label}' not found in scaling fits")
        return None

    alpha, A = scaling_fits[label]
    optimal_tokens = A * (target_flops**alpha)

    logger.info(f"Predicted optimal tokens for {target_flops:.2e} FLOPs: {optimal_tokens:.2e}")

    if sweep_config is None:
        sweep_config = IsoFlopSweepConfig()

    candidates = list(candidate_configs(sweep_config, target_flops, vocab_size))

    if not candidates:
        logger.warning(f"No valid candidates found for budget {target_flops:.2e}")
        return None

    best = min(candidates, key=lambda c: abs(c.tokens - optimal_tokens))

    logger.info(
        f"Selected config: d={best.hidden_size}, L={best.num_layers}, "
        f"B={best.batch_size}, tokens={best.tokens:.2e} (optimal: {optimal_tokens:.2e})"
    )

    return best


def predict_optimal_configs_for_budgets(
    scaling_fits: dict[str, ScalingFit],
    target_budgets: list[float],
    label: str,
    sweep_config: IsoFlopSweepConfig | None = None,
    vocab_size: int = MARIN_TOKENIZER_VOCAB_SIZE,
) -> list[CandidateConfig]:
    """Predict optimal configs for multiple target compute budgets.

    Args:
        scaling_fits: Dict of {label: ScalingFit} from scaling ladder result.
        target_budgets: List of target compute budgets in FLOPs.
        label: Dataset/experiment label to use for scaling fit.
        sweep_config: Optional IsoFlopSweepConfig. If None, uses defaults.
        vocab_size: Vocabulary size.

    Returns:
        List of CandidateConfig for each budget.

    Raises:
        RuntimeError: If any budget cannot be predicted (to prevent silent failures).
    """
    configs = []
    for budget in target_budgets:
        config = predict_optimal_config(scaling_fits, budget, label, sweep_config, vocab_size)
        if config is None:
            raise RuntimeError(
                f"Failed to predict optimal config for budget {budget:.2e} FLOPs "
                f"with label '{label}'. Check that the label exists in scaling_fits "
                f"and that the budget is within a valid range."
            )
        configs.append(config)
    return configs


# ---------------- Result Dataclass ----------------


@dataclass
class IsoFlopAnalysisResult:
    """Result from scaling ladder analysis containing optimal configs and analysis data."""

    configs: list[CandidateConfig]
    """List of optimal CandidateConfig for each (label, flops_budget) pair."""

    scaling_fits: dict[str, ScalingFit]
    """Per-label scaling fits: {label: ScalingFit} for N* ~ A * C^alpha."""

    isoflop_df: pd.DataFrame
    """Transformed dataframe used for analysis."""

    minima_records: list[MinimaRecord]
    """Raw minima records with detailed info for each optimum."""

    fit_curves: dict[tuple[str, float], QuadraticFitCoeffs]
    """Quadratic fit coefficients {(label, flops): QuadraticFitCoeffs} for plotting."""

    def to_json_dict(self) -> dict:
        """Convert result to JSON-serializable dict (excludes DataFrame and fit_curves)."""
        return {
            "configs": [asdict(c) for c in self.configs],
            "scaling_fits": {k: list(v) for k, v in self.scaling_fits.items()},
            "minima_records": [asdict(r) for r in self.minima_records],
        }


def _parse_fit_curve_coeffs(coeffs: Sequence[float]) -> QuadraticFitCoeffs:
    if len(coeffs) != 5:
        raise ValueError(f"Expected 5 fit curve coefficients, got {len(coeffs)}")
    a, b, c, token_min, token_max = coeffs
    return QuadraticFitCoeffs(float(a), float(b), float(c), float(token_min), float(token_max))


# ---------------- ExecutorStep Config ----------------


@dataclass(frozen=True)
class IsoFlopAnalysisConfig(EvalMetricsAnalysisConfig):
    """Configuration for scaling ladder analysis ExecutorStep."""

    recipe: ScalingRecipe = MARIN_2025_RECIPE
    """Scaling recipe for computing optimal hyperparameters."""

    metric_key: str = DEFAULT_EVAL_METRIC_KEY
    """Metric to use for loss (default: eval/paloma/c4_en/bpb - Paloma benchmark on C4 English)."""

    label_map: tuple[tuple[str, str], ...] | None = None
    """Optional mapping from experiment_name -> display label as tuple of pairs."""


@dataclass(frozen=True)
class IsoFlopPlotsConfig:
    """Configuration for isoflop plots ExecutorStep."""

    analysis_output_path: str
    """Path to the isoflop analysis output (containing isoflop_analysis_result.json)."""

    output_path: str
    """Path to save the HTML plots."""


@dataclass(frozen=True)
class UploadPlotsToWandbConfig:
    """Configuration for uploading plots to WandB."""

    plots_path: str
    """Path to the directory containing HTML plots."""

    wandb_entity: str = WANDB_ENTITY
    """WandB entity for uploads."""

    wandb_project: str = f"{WANDB_PROJECT}-analysis"
    """WandB project for uploads."""

    wandb_run_name: str = "scaling-ladder-analysis"
    """Name for the WandB run."""


def run_isoflop_analysis_step(config: IsoFlopAnalysisConfig) -> None:
    """Execute scaling ladder analysis (called by ExecutorStep)."""
    raw_df = read_metrics_dataframe(config)

    if raw_df.empty:
        logger.warning("No eval metrics found in training runs")
        return

    label_map = dict(config.label_map) if config.label_map else None
    isoflop_df = transform_metrics_for_isoflop(raw_df, config.metric_key, label_map)

    if isoflop_df.empty:
        logger.warning("No valid isoflop data after transformation")
        return

    logger.info(f"Loaded {len(isoflop_df)} runs for scaling ladder analysis")
    logger.info(f"Labels found: {isoflop_df['label'].unique().tolist()}")
    logger.info(f"FLOP budgets: {sorted(isoflop_df['flops'].unique())}")

    fit_result = fit_scaling_laws(isoflop_df)

    logger.info(f"Found {len(fit_result.minima_records)} optimal configurations")
    for label, (alpha, A) in fit_result.scaling_fits.items():
        logger.info(f"  {label}: N* = {A:.2e} * C^{alpha:.3f}")

    configs = _minima_to_candidates(fit_result.minima_records, config.recipe)

    result = IsoFlopAnalysisResult(
        configs=configs,
        scaling_fits=fit_result.scaling_fits,
        isoflop_df=isoflop_df,
        minima_records=fit_result.minima_records,
        fit_curves=fit_result.fit_curves,
    )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    result_path = os.path.join(config.output_path, "isoflop_analysis_result.json")
    with fs.open(result_path, "w") as f:
        json.dump(result.to_json_dict(), f, indent=2)
    logger.info(f"Saved results to {result_path}")

    # Also save the full dataframe and fit curves for downstream plotting
    df_path = os.path.join(config.output_path, "isoflop_df.parquet")
    isoflop_df.to_parquet(df_path)
    logger.info(f"Saved dataframe to {df_path}")

    fit_curves_path = os.path.join(config.output_path, "fit_curves.json")
    # Convert tuple keys to strings for JSON serialization
    fit_curves_json = {f"{label}|{flops}": list(coeffs) for (label, flops), coeffs in result.fit_curves.items()}
    with fs.open(fit_curves_path, "w") as f:
        json.dump(fit_curves_json, f, indent=2)
    logger.info(f"Saved fit curves to {fit_curves_path}")


def _run_isoflop_plots_step(config: IsoFlopPlotsConfig) -> None:
    """Generate and save isoflop plots (called by ExecutorStep)."""
    from marin.scaling_laws.scaling_plots import (
        create_isoflop_plot,
        create_scaling_plot,
        save_plots,
    )

    fs, _, _ = fsspec.get_fs_token_paths(config.analysis_output_path)

    # Load the analysis results
    result_path = os.path.join(config.analysis_output_path, "isoflop_analysis_result.json")
    with fs.open(result_path, "r") as f:
        result_dict = json.load(f)

    # Load the dataframe
    df_path = os.path.join(config.analysis_output_path, "isoflop_df.parquet")
    isoflop_df = pd.read_parquet(df_path)

    # Load fit curves and reconstruct tuple keys
    fit_curves_path = os.path.join(config.analysis_output_path, "fit_curves.json")
    with fs.open(fit_curves_path, "r") as f:
        fit_curves_json = json.load(f)
    fit_curves: dict[tuple[str, float], QuadraticFitCoeffs] = {}
    for key_str, coeffs in fit_curves_json.items():
        label, flops = key_str.rsplit("|", 1)
        fit_curves[(label, float(flops))] = _parse_fit_curve_coeffs(coeffs)

    # Reconstruct minima records
    minima_records = [MinimaRecord(**r) for r in result_dict["minima_records"]]
    scaling_fits = {k: ScalingFit(*v) for k, v in result_dict["scaling_fits"].items()}

    # Create plots
    fig_isoflop = create_isoflop_plot(isoflop_df, minima_records, fit_curves)
    fig_scaling = create_scaling_plot(minima_records, scaling_fits)

    # Save plots
    save_plots(fig_isoflop, fig_scaling, config.output_path)


def _run_upload_plots_to_wandb_step(config: UploadPlotsToWandbConfig) -> None:
    """Upload plots to WandB (called by ExecutorStep)."""
    from marin.scaling_laws.scaling_plots import (
        create_isoflop_plot,
        create_scaling_plot,
        upload_plots_to_wandb,
    )

    fs, _, _ = fsspec.get_fs_token_paths(config.plots_path)

    # Load the analysis results to regenerate plots
    result_path = os.path.join(config.plots_path, "isoflop_analysis_result.json")
    with fs.open(result_path, "r") as f:
        result_dict = json.load(f)

    # Load the dataframe
    df_path = os.path.join(config.plots_path, "isoflop_df.parquet")
    isoflop_df = pd.read_parquet(df_path)

    # Load fit curves and reconstruct tuple keys
    fit_curves_path = os.path.join(config.plots_path, "fit_curves.json")
    with fs.open(fit_curves_path, "r") as f:
        fit_curves_json = json.load(f)
    fit_curves: dict[tuple[str, float], QuadraticFitCoeffs] = {}
    for key_str, coeffs in fit_curves_json.items():
        label, flops = key_str.rsplit("|", 1)
        fit_curves[(label, float(flops))] = _parse_fit_curve_coeffs(coeffs)

    # Reconstruct minima records
    minima_records = [MinimaRecord(**r) for r in result_dict["minima_records"]]
    scaling_fits = {k: ScalingFit(*v) for k, v in result_dict["scaling_fits"].items()}

    # Create plots
    fig_isoflop = create_isoflop_plot(isoflop_df, minima_records, fit_curves)
    fig_scaling = create_scaling_plot(minima_records, scaling_fits)

    upload_plots_to_wandb(
        fig_isoflop,
        fig_scaling,
        entity=config.wandb_entity,
        project=config.wandb_project,
        run_name=config.wandb_run_name,
    )


# ---------------- Primary Export: ExecutorStep Factory ----------------


def isoflop_analysis_step(
    name: str,
    training_runs: Sequence[ExecutorStep | InputName],
    metric_key: str = DEFAULT_EVAL_METRIC_KEY,
    label_map: dict[str, str] | None = None,
    recipe: ScalingRecipe = MARIN_2025_RECIPE,
) -> ExecutorStep:
    """Create an ExecutorStep for scaling ladder analysis.

    This step computes scaling law fits and saves results to JSON/parquet files.
    For plotting, use `isoflop_plots_step()`. For WandB upload, use
    `upload_isoflop_plots_to_wandb_step()`.

    Args:
        name: Name for this executor step
        training_runs: Training run ExecutorSteps or InputNames to analyze
        metric_key: Which metric to use for loss (default: eval/paloma/c4_en/bpb)
        label_map: Optional mapping from experiment_name -> display label
        recipe: ScalingRecipe with hyperparameters

    Returns:
        ExecutorStep configured to run the analysis

    Example:
        >>> from marin.scaling_laws import isoflop_analysis_step, isoflop_plots_step
        >>> analysis = isoflop_analysis_step(
        ...     name="my-scaling-analysis",
        ...     training_runs=my_training_steps,
        ... )
        >>> plots = isoflop_plots_step(
        ...     name="my-scaling-plots",
        ...     analysis_step=analysis,
        ... )
    """
    run_paths = _resolve_run_paths(training_runs)

    config = IsoFlopAnalysisConfig(
        training_runs=run_paths,
        output_path=this_output_path(),
        recipe=recipe,
        metric_key=metric_key,
        label_map=tuple(label_map.items()) if label_map else None,
    )

    return ExecutorStep(
        name=name,
        fn=run_isoflop_analysis_step,
        config=config,
        description=f"Scaling ladder analysis for {len(training_runs)} training runs",
    )


def isoflop_plots_step(
    name: str,
    analysis_step: ExecutorStep | InputName,
) -> ExecutorStep:
    """Create an ExecutorStep to generate isoflop HTML plots.

    This step reads the output from an isoflop_analysis_step and generates
    HTML plots for the isoflop curves and scaling fits.

    Args:
        name: Name for this executor step
        analysis_step: The isoflop_analysis_step to read results from

    Returns:
        ExecutorStep configured to generate plots

    Example:
        >>> analysis = isoflop_analysis_step(name="analysis", training_runs=runs)
        >>> plots = isoflop_plots_step(name="plots", analysis_step=analysis)
    """
    analysis_path = output_path_of(analysis_step) if isinstance(analysis_step, ExecutorStep) else analysis_step

    config = IsoFlopPlotsConfig(
        analysis_output_path=analysis_path,
        output_path=this_output_path(),
    )

    return ExecutorStep(
        name=name,
        fn=_run_isoflop_plots_step,
        config=config,
        description="Generate isoflop HTML plots",
    )


def upload_isoflop_plots_to_wandb_step(
    name: str,
    analysis_step: ExecutorStep | InputName,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = f"{WANDB_PROJECT}-analysis",
    wandb_run_name: str | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep to upload isoflop plots to WandB.

    This step reads the analysis results and uploads interactive plots to WandB.

    Args:
        name: Name for this executor step
        analysis_step: The isoflop_analysis_step to read results from
        wandb_entity: WandB entity for uploads
        wandb_project: WandB project for uploads
        wandb_run_name: Name for WandB run (defaults to step name)

    Returns:
        ExecutorStep configured to upload plots to WandB

    Example:
        >>> analysis = isoflop_analysis_step(name="analysis", training_runs=runs)
        >>> upload = upload_isoflop_plots_to_wandb_step(
        ...     name="upload-plots",
        ...     analysis_step=analysis,
        ... )
    """
    analysis_path = output_path_of(analysis_step) if isinstance(analysis_step, ExecutorStep) else analysis_step

    config = UploadPlotsToWandbConfig(
        plots_path=analysis_path,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or name,
    )

    return ExecutorStep(
        name=name,
        fn=_run_upload_plots_to_wandb_step,
        config=config,
        description="Upload isoflop plots to WandB",
    )


# ---------------- Programmatic Interface ----------------


def run_isoflop_analysis(
    training_runs: Sequence[ExecutorStep] | Sequence[str],
    metric_key: str = DEFAULT_EVAL_METRIC_KEY,
    label_map: dict[str, str] | None = None,
    recipe: ScalingRecipe = MARIN_2025_RECIPE,
) -> IsoFlopAnalysisResult:
    """Analyze isoflop training runs and return optimal training configurations.

    This is the programmatic interface for scaling ladder analysis. For pipeline
    usage, prefer `isoflop_analysis_step()` which returns an ExecutorStep.

    Args:
        training_runs: List of ExecutorSteps or path strings to training runs
        metric_key: Which metric to use for loss (default: eval/paloma/c4_en/bpb)
        label_map: Optional mapping from experiment_name -> display label
        recipe: ScalingRecipe with hyperparameter settings

    Returns:
        IsoFlopAnalysisResult with configs, scaling_fits, and analysis data
    """
    run_paths = _resolve_run_paths(training_runs)

    config = EvalMetricsAnalysisConfig(
        training_runs=run_paths,
        output_path="analysis/scaling_ladder",
    )
    raw_df = read_metrics_dataframe(config)

    if raw_df.empty:
        logger.warning("No eval metrics found")
        return IsoFlopAnalysisResult(
            configs=[],
            scaling_fits={},
            isoflop_df=pd.DataFrame(),
            minima_records=[],
            fit_curves={},
        )

    isoflop_df = transform_metrics_for_isoflop(raw_df, metric_key, label_map)

    if isoflop_df.empty:
        logger.warning("No valid isoflop data after transformation")
        return IsoFlopAnalysisResult(
            configs=[],
            scaling_fits={},
            isoflop_df=pd.DataFrame(),
            minima_records=[],
            fit_curves={},
        )

    logger.info(f"Transformed {len(isoflop_df)} runs for scaling ladder analysis")

    fit_result = fit_scaling_laws(isoflop_df)
    configs = _minima_to_candidates(fit_result.minima_records, recipe)

    return IsoFlopAnalysisResult(
        configs=configs,
        scaling_fits=fit_result.scaling_fits,
        isoflop_df=isoflop_df,
        minima_records=fit_result.minima_records,
        fit_curves=fit_result.fit_curves,
    )
