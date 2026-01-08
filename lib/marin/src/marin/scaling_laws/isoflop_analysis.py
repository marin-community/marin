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

This module provides functions and configs for IsoFLOP scaling law analysis:
1. Read eval metrics from completed training runs
2. Fit scaling laws to find compute-optimal token counts
3. Save results to JSON/parquet files

For programmatic use (without ExecutorStep), see `run_isoflop_analysis()`.
"""

import json
import logging
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import fsspec
import jax.numpy as jnp
import pandas as pd
from jaxopt import ScipyMinimize

from levanter.models.llama import LlamaConfig

from marin.scaling_laws.eval_metrics_reader import (
    EvalMetricsAnalysisConfig,
    extract_run_name_from_path,
    read_metrics_dataframe,
)
from marin.scaling_laws.recipe import ScalingRecipe

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
    """Scaling law fit parameters for D* ~ A * C^alpha (optimal tokens ~ compute^alpha)."""

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


# ---------------- IsoFLOP Sweep Defaults ----------------
DEFAULT_SEQ_LEN = SEQ_LEN
DEFAULT_STEPS_PER_RUN = 2**16  # Reference step count for hyperparameter tuning
DEFAULT_FLOP_TOLERANCE = 0.01  # Relative error tolerance for FLOP budget


# ---------------- Candidate Config ----------------


@dataclass
class CandidateConfig:
    """Model-agnostic compute allocation from scaling law analysis.

    Contains only the fundamental parameters that scaling laws reason about:
    - How much compute (flops_budget)
    - How to allocate it between model size (target_params) and data (tokens)
    - Training batch configuration (batch_size, train_steps)

    All model-specific details (architecture, optimizer hyperparameters) are
    computed by the ScalingRecipe from these values.
    """

    batch_size: int
    train_steps: int
    tokens: float  # = batch_size * train_steps * seq_len
    target_params: int  # Optimal parameter count for this flops_budget
    flops_budget: float  # Compute budget this config was generated for


@dataclass
class IsoFlopTrainArgs:
    """Arguments needed to set up an isoflop training run.

    This dataclass contains the model-agnostic parameters needed for training.
    The ScalingRecipe is responsible for converting these to model-specific
    configs (model architecture, optimizer hyperparameters).

    Example:
        >>> args = generate_isoflop_train_args(budgets, "my-exp", vocab_size, recipe)[0]
        >>> # Recipe converts candidate to model-specific configs
        >>> model_config = recipe.build_model_config(args.candidate.target_params, vocab_size)
        >>> optimizer_config = recipe.build_optimizer_config(args.candidate)
    """

    candidate: CandidateConfig
    """Model-agnostic compute allocation (batch_size, train_steps, tokens, target_params)."""

    run_name: str
    """Name for the training run."""

    tags: tuple[str, ...]
    """Tags for tracking/filtering runs."""

    output_path: str
    """Static output path for checkpoints."""


# ---------------- Typed Records ----------------


@dataclass
class MinimaRecord:
    """Model-agnostic record of optimal configuration found at a specific (label, flops) point."""

    label: str
    flops: float
    optimal_tokens: float
    loss_at_optimal: float
    optimal_params: float
    batch_size: int
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
    budget: float,
    vocab_size: int,
    recipe: ScalingRecipe,
    seq_len: int = DEFAULT_SEQ_LEN,
    steps_per_run: int = DEFAULT_STEPS_PER_RUN,
    flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
) -> Iterator[CandidateConfig]:
    """Yield candidate model configurations within the FLOP budget.

    This is a convenience function that delegates to recipe.candidate_configs().
    The recipe encapsulates all model-specific decisions (architecture formula,
    search bounds, constraints), while this function provides backward compatibility.

    Args:
        budget: Target FLOP budget.
        vocab_size: Vocabulary size for the tokenizer.
        recipe: ScalingRecipe with architecture/hyperparameter settings.
        seq_len: Sequence length for training.
        steps_per_run: Reference step count for FLOP budget calculation.
        flop_tolerance: Tolerance for matching FLOP budget (relative error).

    Yields:
        CandidateConfig objects for each valid configuration.
    """
    yield from recipe.candidate_configs(budget, vocab_size, seq_len, steps_per_run, flop_tolerance)


def _minima_to_candidates(
    minima_records: list[MinimaRecord],
) -> list[CandidateConfig]:
    """Convert minima records to model-agnostic CandidateConfig objects.

    This is used by both run_isoflop_analysis_step() and run_isoflop_analysis()
    to convert the fitted minima into usable candidate configs.

    Args:
        minima_records: List of optimal configurations from scaling law fits.
    """
    configs = []
    for rec in minima_records:
        if rec.optimal_params == 0:
            continue
        configs.append(
            CandidateConfig(
                batch_size=rec.batch_size,
                train_steps=int(rec.optimal_tokens / (rec.batch_size * SEQ_LEN)),
                tokens=rec.optimal_tokens,
                target_params=int(rec.optimal_params),
                flops_budget=rec.flops,
            )
        )
    return configs


# ---------------- Training Args Generation ----------------


def generate_isoflop_train_args(
    budgets: Sequence[float],
    experiment_name: str,
    vocab_size: int,
    recipe: ScalingRecipe,
    seq_len: int = DEFAULT_SEQ_LEN,
    steps_per_run: int = DEFAULT_STEPS_PER_RUN,
    flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
) -> list[IsoFlopTrainArgs]:
    """Generate model-agnostic training arguments for each candidate in an isoflop sweep.

    This is a convenience function that delegates to recipe.generate_isoflop_train_args().
    Returns IsoFlopTrainArgs containing model-agnostic CandidateConfig objects.
    Use recipe.build_model_config() and recipe.build_optimizer_config() to get
    model-specific configs.

    Args:
        budgets: Sequence of FLOP budgets to generate configs for.
        experiment_name: Name suffix for run names (e.g., 'nemo', 'dclm').
        vocab_size: Vocabulary size for the tokenizer.
        recipe: ScalingRecipe with architecture/hyperparameter settings.
        seq_len: Sequence length for training.
        steps_per_run: Reference step count for FLOP budget calculation.
        flop_tolerance: Tolerance for matching FLOP budget.

    Returns:
        List of IsoFlopTrainArgs, one per candidate config across all budgets.

    Example:
        >>> from marin.scaling_laws import generate_isoflop_train_args, DEFAULT_BUDGETS, ScalingRecipe
        >>> recipe = ScalingRecipe(name="my-recipe")
        >>> train_args = generate_isoflop_train_args(
        ...     budgets=DEFAULT_BUDGETS,
        ...     experiment_name="my-experiment",
        ...     vocab_size=128256,
        ...     recipe=recipe,
        ... )
        >>> for args in train_args:
        ...     # Recipe converts model-agnostic candidate to model-specific configs
        ...     model_config = recipe.build_model_config(args.candidate.target_params, vocab_size)
        ...     optimizer_config = recipe.build_optimizer_config(args.candidate, vocab_size)
    """
    return recipe.generate_isoflop_train_args(
        budgets, experiment_name, vocab_size, seq_len, steps_per_run, flop_tolerance
    )


# ---------------- Helpers ----------------


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

            log_D_opt = -b / (2 * a)
            D_star = float(10**log_D_opt)
            loss_opt = float(a * log_D_opt**2 + b * log_D_opt + c)

            idx = (sub.tokens - D_star).abs().argmin()
            nearest_row = sub.iloc[idx]

            minima_records.append(
                MinimaRecord(
                    label=lab,
                    flops=float(C),
                    optimal_tokens=D_star,
                    loss_at_optimal=loss_opt,
                    optimal_params=float(nearest_row.get("params") or C / (6 * D_star)),
                    batch_size=int(nearest_row["batch_size"]),
                )
            )

    # Fit scaling law D* ~ A * C^alpha per dataset (optimal tokens ~ compute^alpha)
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
        Ds = jnp.array([r.optimal_tokens for r in recs])

        alpha, logA = jnp.polyfit(jnp.log10(Cs), jnp.log10(Ds), 1)
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
    vocab_size: int,
    recipe: ScalingRecipe,
    seq_len: int = DEFAULT_SEQ_LEN,
    steps_per_run: int = DEFAULT_STEPS_PER_RUN,
    flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
) -> CandidateConfig | None:
    """Predict optimal training config for a target compute budget using fitted scaling laws.

    This is a convenience function that delegates to recipe.predict_optimal_config().
    The recipe encapsulates all model-specific decisions, while this function provides
    backward compatibility.

    This implements IsoFLOP Approach 2 from the Chinchilla paper:
    1. D_opt (optimal tokens) is found empirically at each compute budget by fitting
       parabolas to actual loss values and finding the minimum.
    2. D_opt ~ A * C^alpha is fitted from those empirical minima.
    3. Given D_opt and C, N_opt (optimal params) is derived as C/(6D), so no
       separate alpha fit for params is needed.

    Args:
        scaling_fits: Dict of {label: ScalingFit} from scaling ladder result.
        target_flops: Target compute budget in FLOPs.
        label: Dataset/experiment label to use for scaling fit.
        vocab_size: Vocabulary size.
        recipe: ScalingRecipe with architecture/hyperparameter settings.
        seq_len: Sequence length for training.
        steps_per_run: Reference step count for FLOP budget calculation.
        flop_tolerance: Tolerance for matching FLOP budget.

    Returns:
        CandidateConfig for the predicted optimal, or None if label not in fits
        or no valid candidates found.
    """
    # Convert ScalingFit NamedTuples to plain tuples for recipe method
    fits_as_tuples = {k: (v.alpha, v.A) for k, v in scaling_fits.items()}
    return recipe.predict_optimal_config(
        fits_as_tuples, target_flops, label, vocab_size, seq_len, steps_per_run, flop_tolerance
    )


def predict_optimal_configs_for_budgets(
    scaling_fits: dict[str, ScalingFit],
    target_budgets: list[float],
    label: str,
    vocab_size: int,
    recipe: ScalingRecipe,
    seq_len: int = DEFAULT_SEQ_LEN,
    steps_per_run: int = DEFAULT_STEPS_PER_RUN,
    flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
) -> list[CandidateConfig]:
    """Predict optimal configs for multiple target compute budgets.

    Args:
        scaling_fits: Dict of {label: ScalingFit} from scaling ladder result.
        target_budgets: List of target compute budgets in FLOPs.
        label: Dataset/experiment label to use for scaling fit.
        vocab_size: Vocabulary size.
        recipe: ScalingRecipe with architecture/hyperparameter settings.
        seq_len: Sequence length for training.
        steps_per_run: Reference step count for FLOP budget calculation.
        flop_tolerance: Tolerance for matching FLOP budget.

    Returns:
        List of CandidateConfig for each budget.

    Raises:
        RuntimeError: If any budget cannot be predicted (to prevent silent failures).
    """
    configs = []
    for budget in target_budgets:
        config = predict_optimal_config(
            scaling_fits, budget, label, vocab_size, recipe, seq_len, steps_per_run, flop_tolerance
        )
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


# ---------------- ExecutorStep Config ----------------


@dataclass(frozen=True)
class IsoFlopAnalysisConfig(EvalMetricsAnalysisConfig):
    """Configuration for scaling ladder analysis ExecutorStep."""

    recipe: ScalingRecipe = field(kw_only=True)
    """Scaling recipe for computing optimal hyperparameters."""

    metric_key: str = field(default=DEFAULT_EVAL_METRIC_KEY, kw_only=True)
    """Metric to use for loss (default: eval/paloma/c4_en/bpb - Paloma benchmark on C4 English)."""

    label_map: tuple[tuple[str, str], ...] | None = field(default=None, kw_only=True)
    """Optional mapping from experiment_name -> display label as tuple of pairs."""


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

    configs = _minima_to_candidates(fit_result.minima_records)

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


# ---------------- Programmatic Interface ----------------


def run_isoflop_analysis(
    training_runs: Sequence[str],
    recipe: ScalingRecipe,
    metric_key: str = DEFAULT_EVAL_METRIC_KEY,
    label_map: dict[str, str] | None = None,
) -> IsoFlopAnalysisResult:
    """Analyze isoflop training runs and return optimal training configurations.

    This is the programmatic interface for scaling ladder analysis, useful for
    notebooks or scripts. For ExecutorStep-based pipelines, use
    `run_isoflop_analysis_step()` with `IsoFlopAnalysisConfig`.

    Args:
        training_runs: List of path strings to training run output directories
        metric_key: Which metric to use for loss (default: eval/paloma/c4_en/bpb)
        label_map: Optional mapping from experiment_name -> display label
        recipe: ScalingRecipe with hyperparameter settings

    Returns:
        IsoFlopAnalysisResult with configs, scaling_fits, and analysis data
    """
    config = EvalMetricsAnalysisConfig(
        training_runs=training_runs,
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
    configs = _minima_to_candidates(fit_result.minima_records)

    return IsoFlopAnalysisResult(
        configs=configs,
        scaling_fits=fit_result.scaling_fits,
        isoflop_df=isoflop_df,
        minima_records=fit_result.minima_records,
        fit_curves=fit_result.fit_curves,
    )
