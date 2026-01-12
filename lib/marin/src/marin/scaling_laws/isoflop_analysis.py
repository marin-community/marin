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

This module provides the core data types and analysis functions for IsoFLOP
scaling law analysis. It is intentionally schema-agnostic - experiment code
should transform raw metrics into IsoFlopRecord before calling these functions.

Key types:
- IsoFlopRecord: The contract for a single training run's metrics
- FitScalingLawsResult: Output from fit_scaling_laws()
- CandidateConfig: Model-agnostic compute allocation from scaling law analysis

Key functions:
- fit_scaling_laws(records): Fit scaling laws from typed records
- predict_optimal_config(): Predict optimal training config for a target budget
- generate_isoflop_train_args(): Generate training args for an isoflop sweep
"""

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import jax.numpy as jnp
from jaxopt import ScipyMinimize

from levanter.optim.config import OptimizerConfig

logger = logging.getLogger(__name__)

# ---------------- Constants ----------------

# Paloma is a standard LLM evaluation benchmark. C4-en BPB (bits-per-byte) is a
# common loss metric that measures model perplexity on the C4 English dataset.
# See: https://arxiv.org/abs/2312.10523
DEFAULT_EVAL_METRIC_KEY = "eval/paloma/c4_en/bpb"
SEQ_LEN = 4096

# ---------------- IsoFLOP Sweep Constants ----------------
# Budgets in training FLOPs (includes 3x multiplier for forward + backward pass).
# This matches how FLOPs are tracked in WandB via Levanter's log_performance_stats.
DEFAULT_BUDGETS: tuple[float, ...] = (1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20)


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


# ---------------- IsoFlopRecord ----------------


@dataclass
class IsoFlopRecord:
    """A single training run record for isoflop analysis.

    This is the contract between experiment code (which knows how to extract
    these fields from raw metrics) and the analysis code (which just does math).
    """

    tokens: float
    """Total tokens trained on."""

    metric: float
    """Evaluation metric value (e.g., bits-per-byte from Paloma)."""

    flops: float
    """Total training FLOPs (bucketed)."""

    params: float
    """Parameter count."""

    label: str
    """Experiment label for grouping (e.g., 'nemo', 'dclm')."""


# ---------------- IsoFLOP Sweep Defaults ----------------
DEFAULT_SEQ_LEN = SEQ_LEN
DEFAULT_STEPS_PER_RUN = 2**16  # Reference step count for hyperparameter tuning
DEFAULT_FLOP_TOLERANCE = 0.01  # Relative error tolerance for FLOP budget


# ---------------- Model Configuration Protocol ----------------


class ModelConfiguration(Protocol):
    """Protocol for model configs used in scaling law calculations.

    Any model config that implements these methods can be used with the
    scaling law functions. This allows the library to be model-agnostic
    while still working with LlamaConfig, QwenConfig, etc.
    """

    def flops_per_token(self, vocab_size: int, seq_len: int) -> float:
        """Return FLOPs per token for this model configuration."""
        ...

    def total_trainable_params(self, vocab_size: int) -> int:
        """Return total trainable parameter count for this model configuration."""
        ...


# ---------------- Candidate Config ----------------


@dataclass
class CandidateConfig:
    """Compute allocation from scaling law analysis.

    Contains the model configuration and training parameters:
    - model_config: The actual model architecture (satisfies ModelConfiguration protocol)
    - tokens: How many tokens to train on
    - flops_budget: The compute budget this config was generated for

    Parameter count is derived from model_config.total_trainable_params(vocab_size).
    Training schedule (batch_size, train_steps) is computed by the ScalingRecipe
    at training time via compute_training_schedule().
    """

    model_config: ModelConfiguration
    """Model configuration for this candidate."""

    tokens: float
    """Total tokens to train on."""

    flops_budget: float
    """Compute budget this config was generated for."""


class ScalingRecipe(Protocol):
    """Protocol defining the interface for scaling law recipes.

    Concrete implementations (e.g., Marin2025Recipe) should implement these
    model-specific methods. Orchestration logic (generating training args,
    predicting optimal configs) is handled by library functions that use
    these core methods.

    The recipe owns the vocab_size, which is derived from the tokenizer choice.
    This ensures consistency and simplifies the API by not requiring vocab_size
    to be threaded through every function call.
    """

    name: str
    """Name identifying this recipe (e.g., 'marin-2025')."""

    vocab_size: int
    """Vocabulary size for the tokenizer used with this recipe."""

    def estimate_memory_bytes(self, candidate: CandidateConfig, seq_len: int = DEFAULT_SEQ_LEN) -> int:
        """Estimate memory usage in bytes for training a candidate configuration.

        The implementation can access candidate.model_config, candidate.tokens, and
        candidate.flops_budget to compute memory requirements. This allows the recipe
        to compute the actual batch_size (from tokens) when estimating memory.
        """
        ...

    def build_optimizer_config(self, candidate: CandidateConfig, seq_len: int = DEFAULT_SEQ_LEN) -> OptimizerConfig:
        """Build optimizer config for a candidate."""
        ...

    def candidate_configs(
        self,
        budget: float,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> Iterator[CandidateConfig]:
        """Yield candidate configurations within the FLOP budget.

        Each candidate includes the model_config directly. A typical implementation
        will iterate over feasible model architectures, compute the tokens needed
        to hit the FLOP budget, and yield configs where the relative FLOP error
        is within tolerance.

        The implementation should handle model-specific constraints like:
        - Hidden size increments (e.g., multiples of 64 or 128)
        - Memory constraints affecting maximum batch size
        """
        ...

    def compute_training_schedule(self, candidate: CandidateConfig, seq_len: int = DEFAULT_SEQ_LEN) -> tuple[int, int]:
        """Compute training schedule (batch_size, train_steps) for a candidate."""
        ...


@dataclass
class IsoFlopTrainArgs:
    """Arguments needed to set up an isoflop training run.

    This dataclass contains the parameters needed for training.
    The ScalingRecipe is responsible for computing training schedules
    and optimizer hyperparameters from the candidate.

    Naming (run_name, output_path) is intentionally not included here - that's
    the responsibility of experiment code which may have its own conventions.

    Example:
        >>> args = generate_isoflop_train_args(budgets, recipe)[0]
        >>> model_config = args.candidate.model_config  # Model config is on the candidate
        >>> batch_size, train_steps = recipe.compute_training_schedule(args.candidate)
        >>> optimizer_config = recipe.build_optimizer_config(args.candidate)
    """

    candidate: CandidateConfig
    """Compute allocation (model_config, tokens, flops_budget)."""

    tags: tuple[str, ...]
    """Tags for tracking/filtering runs."""


# ---------------- Typed Records ----------------


@dataclass
class MinimaRecord:
    """Model-agnostic record of optimal configuration found at a specific (label, flops) point."""

    label: str
    flops: float
    optimal_tokens: float
    loss_at_optimal: float
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


def round_flops_to_bucket(flops: float, base: float = 1.1) -> float:
    """Round FLOP count to the nearest power of base.

    Args:
        flops: FLOP count to round.
        base: Base for the power buckets (default 1.1 for ~10% buckets).
    """
    if flops <= 0:
        return flops

    k = math.log(flops) / math.log(base)
    return base ** round(k)


def compute_training_flops(
    model_config: ModelConfiguration,
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
        model_config: Model config with flops_per_token method.
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
    model_config: ModelConfiguration,
    vocab_size: int,
    target_flops: float,
    train_steps: int,
    seq_len: int,
) -> float:
    """Solve for batch size needed to hit a target FLOP budget.

    Given: total_flops = 3 * flops_per_token * batch * steps * seq_len
    Solve: batch = total_flops / (3 * flops_per_token * steps * seq_len)

    Args:
        model_config: Model config with flops_per_token method.
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
    model_config: ModelConfiguration,
    vocab_size: int,
    target_flops: float,
    batch_size: int,
    seq_len: int,
) -> float:
    """Solve for training steps needed to hit a target FLOP budget.

    Given: total_flops = 3 * flops_per_token * batch * steps * seq_len
    Solve: steps = total_flops / (3 * flops_per_token * batch * seq_len)

    Args:
        model_config: Model config with flops_per_token method.
        vocab_size: Vocabulary size.
        target_flops: Target total training FLOPs.
        batch_size: Training batch size.
        seq_len: Sequence length.

    Returns:
        Exact training steps (float) - caller decides how to round.
    """
    flops_per_token = model_config.flops_per_token(vocab_size, seq_len)
    return target_flops / (3 * flops_per_token * batch_size * seq_len)


# ---------------- Training Args Generation ----------------


def generate_isoflop_train_args(
    budgets: Sequence[float],
    recipe: ScalingRecipe,
    seq_len: int = DEFAULT_SEQ_LEN,
    steps_per_run: int = DEFAULT_STEPS_PER_RUN,
    flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
) -> list[IsoFlopTrainArgs]:
    """Generate training arguments for each candidate in an isoflop sweep.

    Returns IsoFlopTrainArgs containing CandidateConfig objects with model configs.
    Use recipe.build_optimizer_config() to get optimizer configs.
    Naming (run_name, output_path) is left to the caller.

    Args:
        budgets: Sequence of FLOP budgets to generate configs for.
        recipe: ScalingRecipe with architecture/hyperparameter settings (includes vocab_size).
        seq_len: Sequence length for training.
        steps_per_run: Reference step count for FLOP budget calculation.
        flop_tolerance: Tolerance for matching FLOP budget.

    Returns:
        List of IsoFlopTrainArgs, one per candidate config across all budgets.

    Example:
        >>> from marin.scaling_laws import generate_isoflop_train_args, DEFAULT_BUDGETS
        >>> # Use a concrete recipe implementation (e.g., from experiments/isoflop_sweep.py)
        >>> # recipe = Marin2025Recipe()  # vocab_size is a property of the recipe
        >>> train_args = generate_isoflop_train_args(
        ...     budgets=DEFAULT_BUDGETS,
        ...     recipe=recipe,
        ... )
        >>> for args in train_args:
        ...     model_config = args.candidate.model_config  # Model config is on the candidate
        ...     batch_size, train_steps = recipe.compute_training_schedule(args.candidate)
        ...     optimizer_config = recipe.build_optimizer_config(args.candidate)
    """
    results: list[IsoFlopTrainArgs] = []

    for budget in budgets:
        for candidate in recipe.candidate_configs(budget, seq_len, steps_per_run, flop_tolerance):
            # Compute training schedule from recipe (for tags)
            batch_size, train_steps = recipe.compute_training_schedule(candidate, seq_len)
            params = candidate.model_config.total_trainable_params(recipe.vocab_size)

            tags = (
                f"FLOPs={budget:.1e}",
                f"N={params:.1e}",
                f"B={batch_size}",
                f"steps={train_steps}",
                f"tokens={candidate.tokens:.1e}",
            )

            results.append(
                IsoFlopTrainArgs(
                    candidate=candidate,
                    tags=tags,
                )
            )

    return results


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
    records: list[IsoFlopRecord],
) -> FitScalingLawsResult:
    """Fit scaling laws and extract optimal configurations.

    Args:
        records: List of IsoFlopRecord with tokens, metric, flops, params, label, batch_size.

    Returns:
        FitScalingLawsResult containing minima_records, scaling_fits, and fit_curves.
    """
    if not records:
        return FitScalingLawsResult(minima_records=[], scaling_fits={}, fit_curves={})

    # Get unique labels preserving order of first appearance
    datasets = list(dict.fromkeys(r.label for r in records))

    # Get unique flop buckets
    buckets = sorted(set(r.flops for r in records))

    minima_records: list[MinimaRecord] = []
    fit_curves: dict[tuple[str, float], QuadraticFitCoeffs] = {}

    # Fit quadratic for each (label, budget) and find minima
    for lab in datasets:
        for C in buckets:
            sub = sorted(
                [r for r in records if r.flops == C and r.label == lab],
                key=lambda r: r.tokens,
            )
            if not sub:
                continue

            # Robust quadratic fit in log10(tokens)
            # Use float64 to avoid int32 overflow for token counts > 2^31
            tokens_array = jnp.array([r.tokens for r in sub], dtype=jnp.float64)
            a, b, c = robust_quad_logx(
                tokens_array,
                jnp.array([r.metric for r in sub], dtype=jnp.float64),
            )
            # Store coefficients along with token range used for fitting
            fit_curves[(lab, C)] = QuadraticFitCoeffs(a, b, c, float(tokens_array.min()), float(tokens_array.max()))

            if a == 0:
                continue

            log_D_opt = -b / (2 * a)
            D_star = float(10**log_D_opt)
            metric_opt = float(a * log_D_opt**2 + b * log_D_opt + c)

            # Find record with tokens closest to optimal
            nearest_record = min(sub, key=lambda r: abs(r.tokens - D_star))

            minima_records.append(
                MinimaRecord(
                    label=lab,
                    flops=float(C),
                    optimal_tokens=D_star,
                    loss_at_optimal=metric_opt,
                    optimal_params=nearest_record.params,
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


# ---------------- Predict Optimal Config ----------------


def predict_optimal_config(
    scaling_fits: dict[str, ScalingFit],
    target_flops: float,
    label: str,
    recipe: ScalingRecipe,
    seq_len: int = DEFAULT_SEQ_LEN,
    steps_per_run: int = DEFAULT_STEPS_PER_RUN,
    flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
) -> CandidateConfig | None:
    """Predict optimal training config for a target compute budget using fitted scaling laws.

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
        recipe: ScalingRecipe with architecture/hyperparameter settings (includes vocab_size).
        seq_len: Sequence length for training.
        steps_per_run: Reference step count for FLOP budget calculation.
        flop_tolerance: Tolerance for matching FLOP budget.

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

    candidates = list(recipe.candidate_configs(target_flops, seq_len, steps_per_run, flop_tolerance))

    if not candidates:
        logger.warning(f"No valid candidates found for budget {target_flops:.2e}")
        return None

    # Find candidate with tokens >= optimal_tokens, closest to optimal
    best = min(candidates, key=lambda c: c.tokens - optimal_tokens if c.tokens >= optimal_tokens else float("inf"))
    if best.tokens < optimal_tokens:
        best = max(candidates, key=lambda c: c.tokens)

    params = best.model_config.total_trainable_params(recipe.vocab_size)
    logger.info(f"Selected config: N={params:.2e}, tokens={best.tokens:.2e} (optimal: {optimal_tokens:.2e})")

    return best


def predict_optimal_configs_for_budgets(
    scaling_fits: dict[str, ScalingFit],
    target_budgets: list[float],
    label: str,
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
        recipe: ScalingRecipe with architecture/hyperparameter settings (includes vocab_size).
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
        config = predict_optimal_config(scaling_fits, budget, label, recipe, seq_len, steps_per_run, flop_tolerance)
        if config is None:
            raise RuntimeError(
                f"Failed to predict optimal config for budget {budget:.2e} FLOPs "
                f"with label '{label}'. Check that the label exists in scaling_fits "
                f"and that the budget is within a valid range."
            )
        configs.append(config)
    return configs
