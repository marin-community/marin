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

Primary usage - create an ExecutorStep for your pipeline:

    from marin.scaling_laws import isoflop_analysis_step

    analysis = isoflop_analysis_step(
        name="my-scaling-analysis",
        training_runs=my_training_steps,  # list of ExecutorStep
    )

The step will:
1. Read eval metrics from completed training runs
2. Fit scaling laws to find compute-optimal token counts
3. Save plots and results to the output path

For programmatic use, see `run_isoflop_analysis()` which returns a `IsoFlopAnalysisResult`.
"""

import json
import logging
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field

import fsspec
import jax.numpy as jnp
import pandas as pd
from jaxopt import ScipyMinimize
from levanter.utils.flop_utils import lm_flops_per_token

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path
from marin.scaling_laws.eval_metrics_reader import (
    EvalMetricsAnalysisConfig,
    extract_run_name_from_path,
    read_metrics_dataframe,
)


logger = logging.getLogger(__name__)

# ---------------- Constants ----------------
DEFAULT_METRIC_KEY = "eval/paloma/c4_en/bpb"
SEQ_LEN = 4096
CANON_LABELS = ["nemo", "comma", "dclm"]

# ---------------- IsoFLOP Sweep Constants ----------------
DEFAULT_BUDGETS = [1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20]
MLP_RATIO = 4

# TPU v5p hardware constants for memory estimation
HBM_PER_CHIP_GIB = 95
CORES_PER_CHIP = 2
V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]


# ---------------- IsoFLOP Sweep Config ----------------
@dataclass
class IsoFlopSweepConfig:
    """Configuration for generating ISOFlop sweep candidate configs.

    This config controls the model architecture search space and training
    hyperparameters for isoflop experiments.
    """

    tokenizer: str = "stanford-crfm/marin-tokenizer"
    """Tokenizer to use (needed for vocab size)."""

    budgets: list[float] = field(default_factory=lambda: DEFAULT_BUDGETS.copy())
    """List of FLOP budgets to generate configs for."""

    seq_len: int = 4096
    """Sequence length for training."""

    steps_per_run: int = 2**16
    """Target number of training steps per run."""

    flop_tolerance: float = 0.01
    """Tolerance for matching FLOP budget (relative error)."""

    base_hidden_layer_ratio: int = 64
    """Base ratio for hidden_dim to num_layers calculation."""

    hidden_head_ratio: int = 128
    """Ratio for hidden_dim to num_heads calculation."""

    lr_constant: float = 0.33
    """Constant for learning rate calculation: lr = (lr_constant * sqrt(batch)) / hidden_dim."""

    min_hidden_pow: int = 9
    """Minimum hidden dimension as power of 2 (2^9 = 512)."""

    max_hidden_pow: int = 12
    """Maximum hidden dimension as power of 2 (2^12 = 4096)."""


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


# ---------------- Candidate Config Generation ----------------


def round_to_power_of_two(x: float) -> int:
    """Round ``x`` to the nearest power of two."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def compute_total_flops(
    batch: int,
    num_layers: int,
    hidden: int,
    intermediate: int,
    num_kv_heads: int,
    num_heads: int,
    steps: int,
    seq_len: int,
    vocab_size: int,
) -> float:
    """Compute total training FLOPs using Levanter utilities."""
    flops_per_token = lm_flops_per_token(
        hidden,
        intermediate,
        num_layers,
        num_kv_heads,
        num_heads,
        seq_len,
        vocab_size,
        glu=True,
    )
    return flops_per_token * batch * steps * seq_len


def estimate_memory_bytes(
    param_count: int,
    hidden_dim: int,
    num_layers: int,
    batch: int,
    seq_len: int,
    vocab: int,
    optim_mult: int = 3,
    dtype_size: int = 4,
    fudge_factor: float = 2,
) -> int:
    """
    Estimate float32 memory usage (in bytes) for one training step.

    Parameters:
    - param_count: number of model parameters
    - hidden_dim: model hidden size
    - num_layers: number of Transformer layers
    - batch, seq_len: training batch size and sequence length
    - vocab: vocabulary size
    - optim_mult: optimizer memory multiplier (e.g., 3x for Adam + states)
    - dtype_size: bytes per float (4 for float32)
    - fudge_factor: safety margin for extra memory

    Returns:
    - total estimated memory in bytes
    """
    param_bytes = param_count * optim_mult * dtype_size
    act_bytes = (batch * seq_len) * ((hidden_dim * num_layers) + vocab * fudge_factor)
    total_bytes = param_bytes + act_bytes
    return int(total_bytes * fudge_factor)


def pick_v5p_type(
    param_count: int,
    hidden: int,
    layers: int,
    batch: int,
    seq_len: int,
    vocab: int,
) -> str:
    """
    Select the smallest TPU v5p slice that fits the model in float32.

    Returns:
    - TPU slice name, e.g., "v5p-8" or "v5p-32"
    """
    need_bytes = estimate_memory_bytes(param_count, hidden, layers, batch, seq_len, vocab)
    chip_bytes = HBM_PER_CHIP_GIB * 1024**3
    chips = math.ceil(need_bytes / chip_bytes)
    cores_req = chips * CORES_PER_CHIP

    valid = [c for c in V5P_CORE_OPTIONS if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available v5p slices (need {cores_req} cores).")

    return f"v5p-{min(valid)}"


def candidate_configs(
    cfg: IsoFlopSweepConfig,
    budget: float,
    vocab_size: int,
) -> Iterator[CandidateConfig]:
    """Yield candidate model configurations within the FLOP budget.

    Args:
        cfg: IsoFlopSweepConfig with search parameters
        budget: Target FLOP budget
        vocab_size: Vocabulary size for the tokenizer

    Yields:
        CandidateConfig objects for each valid configuration
    """
    if budget > 9e18:
        step_size = 256
    else:
        step_size = 128

    for hidden_size in range(2**cfg.min_hidden_pow, (2**cfg.max_hidden_pow) + 1, step_size):
        hs_pow = math.log2(hidden_size)
        intermediate_dim = hidden_size * MLP_RATIO
        num_layers = round(hidden_size / (cfg.base_hidden_layer_ratio + (hs_pow * 4) - cfg.min_hidden_pow))
        n_heads = max(1, hidden_size // cfg.hidden_head_ratio)
        n_kv_heads = n_heads

        batch_exact = budget / compute_total_flops(
            1,
            num_layers,
            hidden_size,
            intermediate_dim,
            n_kv_heads,
            n_heads,
            cfg.steps_per_run,
            cfg.seq_len,
            vocab_size,
        )

        batch_size = round_to_power_of_two(batch_exact)
        lr = (cfg.lr_constant * math.sqrt(batch_size)) / hidden_size
        while lr > 0.01:
            batch_size //= 2
            lr = (cfg.lr_constant * math.sqrt(batch_size)) / hidden_size
        b2 = 0.98 ** (batch_size / 128)  # https://arxiv.org/pdf/2507.07101

        if batch_size < 8:
            continue

        steps_exact = budget / compute_total_flops(
            batch_size,
            num_layers,
            hidden_size,
            intermediate_dim,
            n_kv_heads,
            n_heads,
            1,
            cfg.seq_len,
            vocab_size,
        )
        train_steps = round(steps_exact)

        achieved_flops = compute_total_flops(
            batch_size,
            num_layers,
            hidden_size,
            intermediate_dim,
            n_kv_heads,
            n_heads,
            train_steps,
            cfg.seq_len,
            vocab_size,
        )

        if abs(achieved_flops - budget) / budget > cfg.flop_tolerance:
            continue

        tokens = batch_size * train_steps * cfg.seq_len

        yield CandidateConfig(
            hidden_size=hidden_size,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            batch_size=batch_size,
            train_steps=train_steps,
            learning_rate=lr,
            beta2=b2,
            tokens=tokens,
            flops_budget=budget,
        )


# ---------------- Helpers ----------------


def parse_isoflop_run_name(run_name: str) -> dict | None:
    """Parse metadata from isoflop run name.

    Expected format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    Optionally with a trailing -<hash> which is ignored.
    E.g., 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt'
    or 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt-a1b2c3'

    Returns dict with: flops, d, L, B, experiment_name or None if parsing fails.
    """
    # Strip optional -<hash> suffix
    run_name = re.sub(r"-[0-9a-fA-F]{6}$", "", run_name)

    pattern = r"isoflop-([0-9.e+]+)-d(\d+)-L(\d+)-B(\d+)-(.+)"
    match = re.match(pattern, run_name)
    if not match:
        return None

    flops_str, d, L, B, exp_name = match.groups()
    return {
        "flops": float(flops_str),
        "d": int(d),
        "L": int(L),
        "B": int(B),
        "experiment_name": exp_name,
    }


def robust_quad_logx(x: jnp.ndarray, y: jnp.ndarray, delta: float = 1.0) -> tuple[float, float, float]:
    """Fit a robust quadratic in log10(x) space using Huber loss.

    Returns (a, b, c) coefficients for: loss = a * log10(x)^2 + b * log10(x) + c
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


def _compute_optimal_params(flops: float, tokens: float) -> float:
    """Compute optimal parameters from C = 6 * N * P approximation."""
    return flops / (6 * tokens)


def _find_nearest_config(df: pd.DataFrame, flops: float, tokens: float) -> dict:
    """Find the nearest actual config from the dataframe to use as template."""
    sub = df[df.flops == flops]
    if sub.empty:
        sub = df
    idx = (sub.tokens - tokens).abs().argmin()
    row = sub.iloc[idx]

    run_name = row["name"]
    meta = parse_isoflop_run_name(run_name)

    return {
        "hidden_dim": meta["d"] if meta else 0,
        "num_layers": meta["L"] if meta else 0,
        "batch_size": meta["B"] if meta else 0,
        "params": row.get("params", _compute_optimal_params(flops, tokens)),
    }


# ---------------- Core Analysis ----------------


def fit_scaling_laws(
    df: pd.DataFrame,
) -> tuple[list[dict], dict[str, tuple[float, float]], dict[tuple[str, float], tuple[float, float, float]]]:
    """
    Fit scaling laws and extract optimal configurations.

    Args:
        df: DataFrame with columns: tokens, loss, flops, params, name, label

    Returns:
        - minima_records: List of dicts with optimal config info per (label, flops)
        - scaling_fits: Dict of {label: (alpha, A)} for N* ~ A * C^alpha
        - fit_curves: Dict of {(label, flops): (a, b, c)} quadratic coefficients for plotting
    """
    if df is None or df.empty:
        return [], {}, {}

    present = list(dict.fromkeys(df["label"].tolist()))
    datasets = [lab for lab in CANON_LABELS if lab in present] + [lab for lab in present if lab not in CANON_LABELS]

    buckets = sorted(df.flops.unique())

    minima_records = []
    fit_curves = {}

    # Fit quadratic for each (label, budget) and find minima
    for lab in datasets:
        for C in buckets:
            sub = df[(df.flops == C) & (df.label == lab)].sort_values("tokens")
            if sub.empty:
                continue

            # Robust quadratic fit in log10(tokens)
            a, b, c = robust_quad_logx(jnp.array(sub.tokens.values), jnp.array(sub.loss.values))
            fit_curves[(lab, C)] = (a, b, c)

            if a == 0:
                continue

            # Compute minimum
            L_opt = -b / (2 * a)
            N_star = float(10**L_opt)
            loss_opt = float(a * L_opt**2 + b * L_opt + c)

            # Find nearest actual config for template
            nearest = _find_nearest_config(sub, C, N_star)

            minima_records.append(
                {
                    "label": lab,
                    "flops": float(C),
                    "optimal_tokens": N_star,
                    "loss_at_optimal": loss_opt,
                    "hidden_dim": nearest["hidden_dim"],
                    "num_layers": nearest["num_layers"],
                    "batch_size": nearest["batch_size"],
                    "optimal_params": float(nearest["params"]),
                }
            )

    # Fit scaling law N* ~ A * C^alpha per dataset
    scaling_fits = {}
    by_lab = {}
    for rec in minima_records:
        by_lab.setdefault(rec["label"], []).append(rec)

    for lab in datasets:
        recs = by_lab.get(lab, [])
        if len(recs) < 2:
            continue

        recs = sorted(recs, key=lambda r: r["flops"])
        Cs = jnp.array([r["flops"] for r in recs])
        Ns = jnp.array([r["optimal_tokens"] for r in recs])

        alpha, logA = jnp.polyfit(jnp.log10(Cs), jnp.log10(Ns), 1)
        A = float(10**logA)
        alpha = float(alpha)
        scaling_fits[lab] = (alpha, A)

        # Augment minima records with scaling fit params
        for rec in recs:
            rec["scaling_alpha"] = alpha
            rec["scaling_A"] = A

    return minima_records, scaling_fits, fit_curves


def transform_metrics_for_isoflop(
    df: pd.DataFrame,
    metric_key: str,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Transform raw metrics DataFrame into isoflop analysis format.

    Takes the generic metrics DataFrame from read_metrics_dataframe() and
    transforms it into the format expected by the analysis:
    columns: tokens, loss, flops, params, name, label

    Args:
        df: Raw metrics DataFrame from read_metrics_dataframe()
        metric_key: Which metric column to use for loss
        label_map: Optional mapping from experiment_name -> display label

    Returns:
        Transformed DataFrame ready for fit_scaling_laws()
    """
    if df.empty:
        return pd.DataFrame(columns=["tokens", "loss", "flops", "params", "name", "label"])

    # Get final metrics for each run (max step)
    final_metrics = df.loc[df.groupby("run_path")["step"].idxmax()].copy()

    records = []
    for _, row in final_metrics.iterrows():
        run_path = row["run_path"]
        run_name = extract_run_name_from_path(run_path)

        # Parse metadata from run name
        meta = parse_isoflop_run_name(run_name)
        if meta is None:
            logger.warning(f"Could not parse metadata from run name: {run_name}")
            continue

        flops = meta["flops"]
        if flops < 1e18:
            continue

        # Calculate tokens = steps * batch * seq_len
        steps = row["step"]
        batch = meta["B"]
        tokens = steps * batch * SEQ_LEN

        # Get loss from the metric column
        loss = row.get(metric_key)
        if loss is None or pd.isna(loss):
            logger.warning(f"Missing metric {metric_key} for run {run_name}")
            continue

        params = row.get("parameter_count")
        if params is None or pd.isna(params):
            params = None

        # Determine label
        exp_name = meta["experiment_name"]
        if label_map and exp_name in label_map:
            label = label_map[exp_name]
        else:
            label = exp_name
            for canon in CANON_LABELS:
                if canon in exp_name.lower():
                    label = canon
                    break

        records.append(
            dict(
                tokens=tokens,
                loss=loss,
                flops=flops,
                params=params,
                name=run_name,
                label=label,
            )
        )

    return pd.DataFrame.from_records(records)


# ---------------- Predict Optimal Config ----------------


def predict_optimal_config(
    scaling_fits: dict[str, tuple[float, float]],
    target_flops: float,
    label: str,
    sweep_config: IsoFlopSweepConfig | None = None,
    vocab_size: int = 128256,
) -> CandidateConfig | None:
    """Predict optimal training config for a target compute budget using fitted scaling laws.

    This function:
    1. Uses the scaling fit (N* ~ A * C^alpha) to predict optimal tokens for target_flops
    2. Generates candidate configs for the target budget using candidate_configs()
    3. Selects the candidate whose token count is closest to the predicted optimal

    Args:
        scaling_fits: Dict of {label: (alpha, A)} from scaling ladder result.
        target_flops: Target compute budget in FLOPs.
        label: Dataset/experiment label to use for scaling fit.
        sweep_config: Optional IsoFlopSweepConfig. If None, uses defaults.
        vocab_size: Vocabulary size (default: 128256 for marin tokenizer).

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

    # Use default config if none provided
    if sweep_config is None:
        sweep_config = IsoFlopSweepConfig()

    # Generate candidates for this budget
    candidates = list(candidate_configs(sweep_config, target_flops, vocab_size))

    if not candidates:
        logger.warning(f"No valid candidates found for budget {target_flops:.2e}")
        return None

    # Find candidate closest to optimal token count
    best = min(candidates, key=lambda c: abs(c.tokens - optimal_tokens))

    logger.info(
        f"Selected config: d={best.hidden_size}, L={best.num_layers}, "
        f"B={best.batch_size}, tokens={best.tokens:.2e} (optimal: {optimal_tokens:.2e})"
    )

    return best


def predict_optimal_configs_for_budgets(
    scaling_fits: dict[str, tuple[float, float]],
    target_budgets: list[float],
    label: str,
    sweep_config: IsoFlopSweepConfig | None = None,
    vocab_size: int = 128256,
) -> list[CandidateConfig]:
    """Predict optimal configs for multiple target compute budgets.

    Args:
        scaling_fits: Dict of {label: (alpha, A)} from scaling ladder result.
        target_budgets: List of target compute budgets in FLOPs.
        label: Dataset/experiment label to use for scaling fit.
        sweep_config: Optional IsoFlopSweepConfig. If None, uses defaults.
        vocab_size: Vocabulary size.

    Returns:
        List of CandidateConfig for each budget (skips budgets with no valid config).
    """
    configs = []
    for budget in target_budgets:
        config = predict_optimal_config(scaling_fits, budget, label, sweep_config, vocab_size)
        if config is not None:
            configs.append(config)
    return configs


# ---------------- Result Dataclass ----------------


@dataclass
class IsoFlopAnalysisResult:
    """Result from scaling ladder analysis containing optimal configs and analysis data."""

    configs: list[CandidateConfig]
    """List of optimal CandidateConfig for each (label, flops_budget) pair."""

    scaling_fits: dict[str, tuple[float, float]]
    """Per-label scaling fits: {label: (alpha, A)} for N* ~ A * C^alpha."""

    isoflop_df: pd.DataFrame
    """Transformed dataframe used for analysis."""

    minima_records: list[dict]
    """Raw minima records with detailed info for each optimum."""

    fit_curves: dict[tuple[str, float], tuple[float, float, float]]
    """Quadratic fit coefficients {(label, flops): (a, b, c)} for plotting."""

    def to_json_dict(self) -> dict:
        """Convert result to JSON-serializable dict (excludes DataFrame and fit_curves)."""
        return {
            "configs": [asdict(c) for c in self.configs],
            "scaling_fits": {k: list(v) for k, v in self.scaling_fits.items()},
            "minima_records": self.minima_records,
        }


# ---------------- ExecutorStep Config ----------------


@dataclass(frozen=True)
class IsoFlopAnalysisConfig(EvalMetricsAnalysisConfig):
    """Configuration for scaling ladder analysis ExecutorStep."""

    metric_key: str = DEFAULT_METRIC_KEY
    """Metric to use for loss (default: eval/paloma/c4_en/bpb)."""

    label_map: tuple[tuple[str, str], ...] | None = None
    """Optional mapping from experiment_name -> display label as tuple of pairs."""

    save_plots: bool = True
    """Whether to save HTML plots to output_path."""

    upload_to_wandb: bool = True
    """Whether to upload plots to WandB."""

    wandb_entity: str = "marin-community"
    """WandB entity for uploads."""

    wandb_project: str = "marin-analysis"
    """WandB project for uploads."""

    wandb_run_name: str = "scaling-ladder-analysis"
    """Name for the WandB run."""


def _run_isoflop_analysis_step(config: IsoFlopAnalysisConfig) -> None:
    """Execute scaling ladder analysis (called by ExecutorStep)."""
    # Read metrics from training runs
    raw_df = read_metrics_dataframe(config)

    if raw_df.empty:
        logger.warning("No eval metrics found in training runs")
        return

    # Convert label_map tuple to dict if provided
    label_map = dict(config.label_map) if config.label_map else None

    # Transform to isoflop analysis format
    isoflop_df = transform_metrics_for_isoflop(raw_df, config.metric_key, label_map)

    if isoflop_df.empty:
        logger.warning("No valid isoflop data after transformation")
        return

    logger.info(f"Loaded {len(isoflop_df)} runs for scaling ladder analysis")
    logger.info(f"Labels found: {isoflop_df['label'].unique().tolist()}")
    logger.info(f"FLOP budgets: {sorted(isoflop_df['flops'].unique())}")

    # Fit scaling laws
    minima_records, scaling_fits, fit_curves = fit_scaling_laws(isoflop_df)

    logger.info(f"Found {len(minima_records)} optimal configurations")
    for label, (alpha, A) in scaling_fits.items():
        logger.info(f"  {label}: N* = {A:.2e} * C^{alpha:.3f}")

    # Convert minima to CandidateConfigs
    configs = []
    for rec in minima_records:
        if rec["hidden_dim"] == 0:
            continue
        candidate = CandidateConfig(
            hidden_size=rec["hidden_dim"],
            intermediate_dim=rec["hidden_dim"] * MLP_RATIO,
            num_layers=rec["num_layers"],
            num_heads=max(1, rec["hidden_dim"] // 128),
            num_kv_heads=max(1, rec["hidden_dim"] // 128),
            batch_size=rec["batch_size"],
            train_steps=int(rec["optimal_tokens"] / (rec["batch_size"] * SEQ_LEN)),
            learning_rate=(0.33 * math.sqrt(rec["batch_size"])) / rec["hidden_dim"],
            beta2=0.98 ** (rec["batch_size"] / 128),
            tokens=rec["optimal_tokens"],
            flops_budget=rec["flops"],
        )
        configs.append(candidate)

    result = IsoFlopAnalysisResult(
        configs=configs,
        scaling_fits=scaling_fits,
        isoflop_df=isoflop_df,
        minima_records=minima_records,
        fit_curves=fit_curves,
    )

    # Save outputs
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    # Save result JSON
    result_path = os.path.join(config.output_path, "isoflop_analysis_result.json")
    with fs.open(result_path, "w") as f:
        json.dump(result.to_json_dict(), f, indent=2)
    logger.info(f"Saved results to {result_path}")

    # Save plots if enabled
    if config.save_plots:
        from marin.scaling_laws.scaling_plots import (
            create_isoflop_plot,
            create_scaling_plot,
            save_plots,
        )

        fig_isoflop = create_isoflop_plot(isoflop_df, minima_records, fit_curves)
        fig_scaling = create_scaling_plot(minima_records, scaling_fits)
        save_plots(fig_isoflop, fig_scaling, config.output_path)

        # Upload to WandB if enabled
        if config.upload_to_wandb:
            from marin.scaling_laws.scaling_plots import upload_plots_to_wandb

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
    metric_key: str = DEFAULT_METRIC_KEY,
    label_map: dict[str, str] | None = None,
    save_plots: bool = True,
    upload_to_wandb: bool = True,
    wandb_entity: str = "marin-community",
    wandb_project: str = "marin-analysis",
    wandb_run_name: str | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep for scaling ladder analysis.

    This is the primary interface for using scaling ladder analysis in a pipeline.
    The step will:
    1. Wait for all training runs to complete
    2. Read eval metrics from the training runs
    3. Fit scaling laws to find compute-optimal configurations
    4. Save plots and results to the output path

    Args:
        name: Name for this executor step
        training_runs: Training run ExecutorSteps or InputNames to analyze
        metric_key: Which metric to use for loss (default: eval/paloma/c4_en/bpb)
        label_map: Optional mapping from experiment_name -> display label
        save_plots: Whether to save HTML plots (default: True)
        upload_to_wandb: Whether to upload plots to WandB (default: True)
        wandb_entity: WandB entity for uploads
        wandb_project: WandB project for uploads
        wandb_run_name: Name for WandB run (defaults to step name)

    Returns:
        ExecutorStep configured to run the analysis

    Example:
        >>> from marin.scaling_laws import isoflop_analysis_step
        >>> analysis = scaling_ladder_step(
        ...     name="my-scaling-analysis",
        ...     training_runs=my_training_steps,
        ... )
    """
    run_paths = [output_path_of(run) if isinstance(run, ExecutorStep) else run for run in training_runs]

    config = IsoFlopAnalysisConfig(
        training_runs=run_paths,
        output_path=this_output_path(),
        metric_key=metric_key,
        label_map=tuple(label_map.items()) if label_map else None,
        save_plots=save_plots,
        upload_to_wandb=upload_to_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or name,
    )

    return ExecutorStep(
        name=name,
        fn=_run_isoflop_analysis_step,
        config=config,
        description=f"Scaling ladder analysis for {len(training_runs)} training runs",
    )


# ---------------- Programmatic Interface ----------------


def run_isoflop_analysis(
    training_runs: Sequence[ExecutorStep] | Sequence[str],
    metric_key: str = DEFAULT_METRIC_KEY,
    label_map: dict[str, str] | None = None,
) -> IsoFlopAnalysisResult:
    """Analyze isoflop training runs and return optimal training configurations.

    This is the programmatic interface for scaling ladder analysis. For pipeline
    usage, prefer `isoflop_analysis_step()` which returns an ExecutorStep.

    Args:
        training_runs: List of ExecutorSteps or path strings to training runs
        metric_key: Which metric to use for loss (default: eval/paloma/c4_en/bpb)
        label_map: Optional mapping from experiment_name -> display label

    Returns:
        IsoFlopAnalysisResult with configs, scaling_fits, and analysis data
    """
    # Convert to paths
    run_paths = []
    for run in training_runs:
        if isinstance(run, ExecutorStep):
            run_paths.append(output_path_of(run))
        else:
            run_paths.append(run)

    # Read metrics
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

    # Transform to isoflop format
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

    # Fit scaling laws and extract optima
    minima_records, scaling_fits, fit_curves = fit_scaling_laws(isoflop_df)

    # Convert minima records to CandidateConfig objects
    configs = []
    for rec in minima_records:
        if rec["hidden_dim"] == 0:
            continue
        candidate = CandidateConfig(
            hidden_size=rec["hidden_dim"],
            intermediate_dim=rec["hidden_dim"] * MLP_RATIO,
            num_layers=rec["num_layers"],
            num_heads=max(1, rec["hidden_dim"] // 128),
            num_kv_heads=max(1, rec["hidden_dim"] // 128),
            batch_size=rec["batch_size"],
            train_steps=int(rec["optimal_tokens"] / (rec["batch_size"] * SEQ_LEN)),
            learning_rate=(0.33 * math.sqrt(rec["batch_size"])) / rec["hidden_dim"],
            beta2=0.98 ** (rec["batch_size"] / 128),
            tokens=rec["optimal_tokens"],
            flops_budget=rec["flops"],
        )
        configs.append(candidate)

    return IsoFlopAnalysisResult(
        configs=configs,
        scaling_fits=scaling_fits,
        isoflop_df=isoflop_df,
        minima_records=minima_records,
        fit_curves=fit_curves,
    )
