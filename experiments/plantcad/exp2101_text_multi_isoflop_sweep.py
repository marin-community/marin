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

"""Generate ISOFlop sweep steps for model sizes, architectures, and epochs on pre-tokenized text."""

import os
import math
import logging
import dataclasses
import numpy as np
import pandas as pd
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from levanter.data.text import LMMixtureDatasetConfig, UrlDatasetSourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

from experiments.defaults import default_train
from experiments.evals.task_configs import EvalTaskConfig
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main
from fray.cluster import ResourceConfig

logger = logging.getLogger("ray")

# TPU v5p hardware constants for memory estimation
# Constants for TPU v5p
HBM_PER_CHIP_GIB = 95
CORES_PER_CHIP = 2
V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]  # TPU slices

ModelConfig = LlamaConfig | Qwen3Config

# Maximum number of configs per (budget, architecture) combination before downsampling
MAX_SWEEP_CONFIGS = 10


def simulated_epoch_train(
    name: str,
    tokenized: LMMixtureDatasetConfig,
    model_config: ModelConfig,
    train_config: "SimpleTrainConfig",
    train_tokens: int,
    dataset_tokens: int,
    epoch_count: int = 1,
    tags: Sequence[str] = (),
    eval_harness_tasks: Sequence[EvalTaskConfig] = (),
) -> ExecutorStep:
    """Train with simulated epoching. When epoch_count=1, uses full dataset."""
    if not isinstance(epoch_count, int) or epoch_count < 1:
        raise ValueError(f"epoch_count must be int >= 1, got {epoch_count}")

    pretraining_data = tokenized

    if epoch_count == 1:
        return default_train(
            name,
            tokenized=pretraining_data,
            model_config=model_config,
            train_config=train_config,
            tags=tags,
            eval_harness_tasks=eval_harness_tasks,
            use_default_validation=False,
        )

    # To use simulated epoching in Levanter, we need to first address the fact that
    # we are already limiting training to less than 1 true epoch in each run.
    #
    # The Levanter formula for this feature takes two arguments, experiment_budget and target_budget,
    # and then uses this formula to determine how to slice each epoch:
    #
    # simulated_data_ratio = experiment_budget / target_budget
    # simulated_length_of_dataset = int(true_length_of_dataset * simulated_data_ratio)
    # sliced_datasets[name] = ds.slice_dataset(end_index=simulated_length_of_dataset)
    #
    # See: https://github.com/marin-community/marin/blob/eb4acbdd185a34202da16052c46c74eb570e69a5/lib/levanter/src/levanter/data/text.py#L1273-L1280
    #
    # This means that `simulated_data_ratio` must become equal to `train_tokens / dataset_tokens / epoch_count`
    # in order for the simulated epochs to work on top of a partial epoch.
    # We accomplish this here by setting:
    # - experiment_budget = train_tokens
    # - target_budget = dataset_tokens * epoch_count
    experiment_budget, target_budget = train_tokens, dataset_tokens * epoch_count
    simulated_pretraining_data = dataclasses.replace(
        pretraining_data, target_budget=target_budget, experiment_budget=experiment_budget
    )

    return default_train(
        name,
        tokenized=simulated_pretraining_data,
        model_config=model_config,
        train_config=train_config,
        tags=tags,
        eval_harness_tasks=eval_harness_tasks,
        use_default_validation=False,
    )


def format_num(n: int | float) -> str:
    """Format numbers in T/B/M/K notation (e.g., 1.5T, 100B, 5.2M, 1.0K)."""
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"
    elif n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 10_000_000:
        return f"{int(n / 1_000_000)}M"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


@dataclass(frozen=True)
class IsoFlopDataConfig:
    # Use pretokenized DCLM baseline from GCS (control experiment)
    # Absolute path in same region used to avoid re-downloading when running with a different MARIN_PREFIX
    # (i.e. marin-dna-us-central1)
    tokenized_path: str = "gs://marin-us-central1/tokenized/dclm_baseline-0206f1"
    tokenizer: str = llama3_tokenizer
    vocab_size: int = llama3_tokenizer_vocab_size
    seq_len: int = 4096
    # Keep consistent with PlantCAD total token count even though DCLM has far, far more
    total_token_count: int = 2_600_000_000_000  # 2.6T


@dataclass(frozen=True)
class IsoFlopSweepParams:
    """Configuration for a specific compute range in the ISOFlop sweep."""

    experiment_name: str
    compute_range_name: str
    budgets: list[float]
    steps_per_run: int
    hidden_step_size: int = 128
    hidden_head_ratio: int = 128


# Predefined compute range configurations
ISOFLOP_SWEEPS = {
    # low-compute: 1.6e17 to 2e18, steps=16_384 (v2.12)
    "low": IsoFlopSweepParams(
        experiment_name="plantcad_isoflop_v2.12",
        compute_range_name="low",
        budgets=list(np.logspace(np.log10(1.6e17), np.log10(2e18), 5)),
        steps_per_run=16_384,
        hidden_step_size=128,
    ),
    # # mid-compute: 3.2e17 to 4e18, steps=32_768 (v2.13)
    "mid": IsoFlopSweepParams(
        experiment_name="plantcad_isoflop_v2.13",
        compute_range_name="mid",
        budgets=list(np.logspace(np.log10(3.2e17), np.log10(4e18), 5)),
        steps_per_run=32_768,
    ),
    # # high-compute: 6.4e17 to 8e18, steps=65_536 (v2.9)
    "high": IsoFlopSweepParams(
        experiment_name="plantcad_isoflop_v2.9",
        compute_range_name="high",
        # i in [0, 1, 4]
        budgets=list(np.logspace(np.log10(6.4e17), np.log10(8e18), 5)),
        steps_per_run=65_536,
    ),
}


@dataclass(frozen=True)
class IsoFlopSweepConfig:
    tokenized_dataset: LMMixtureDatasetConfig
    vocab_size: int
    seq_len: int
    total_token_count: int
    experiment_name: str
    compute_range_name: str
    budgets: list[float]
    steps_per_run: int
    hidden_step_size: int
    hidden_head_ratio: int

    epochs: list[int] = dataclasses.field(default_factory=lambda: [1])
    min_hidden_pow: int = 9
    max_hidden_pow: int = 12
    mlp_ratio: int = 4
    base_hidden_layer_ratio: int = 64
    lr_max: float | None = 0.03
    flop_tolerance: float = 0.01
    architectures: list[str] = dataclasses.field(default_factory=lambda: ["qwen"])
    # TODO: adjust eval example count to account for num tpus in the even that v5p-8 is not used
    per_device_eval_parallelism: int = 8
    max_eval_batches: int = 1024
    num_evals: int = 3
    lr_constant: float = 0.33
    base_optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: CautiousConfig(
            learning_rate=1.0,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=0.1,
            beta1=0.95,
            beta2=0.98,
            epsilon=1e-15,
            max_grad_norm=1,
            adamc_weight_decay=True,
            lr_schedule="linear",
            decay=0.2,
        ),
    )
    base_train_config: SimpleTrainConfig = dataclasses.field(
        default_factory=lambda: SimpleTrainConfig(
            resources=ResourceConfig.with_tpu("v5p-8"),
            train_batch_size=1,
            num_train_steps=50_000,
            learning_rate=1.0,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            decay=0.2,
        )
    )


def estimate_bytes(
    param_count: int,
    hidden_dim: int,
    num_layers: int,
    batch: int,
    seq_len: int,
    vocab: int,
    optim_mult: int = 3,
    dtype_size: int = 4,
    # TODO: determine why this needs to be higher than the original 2 to avoid OOMs on DCLM
    fudge_factor: float = 4,
) -> int:
    """Estimate memory usage (in bytes) for one training step."""
    param_bytes = param_count * optim_mult * dtype_size

    act_bytes = (batch * seq_len) * ((hidden_dim * num_layers) + vocab * fudge_factor)

    total_bytes = param_bytes + act_bytes
    return int(total_bytes) * fudge_factor


def pick_v5p_type(
    config: ModelConfig,
    hidden: int,
    layers: int,
    batch: int,
    seq_len: int,
    vocab: int,
) -> str:
    """Select the smallest TPU v5p slice that fits the model in float32."""
    _, _, param_count = compute_param_count(config, vocab)
    need_bytes = estimate_bytes(param_count, hidden, layers, batch, seq_len, vocab)
    chip_bytes = HBM_PER_CHIP_GIB * 1024**3
    chips = math.ceil(need_bytes / chip_bytes)
    cores_req = chips * CORES_PER_CHIP

    valid = [c for c in V5P_CORE_OPTIONS if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available v5p slices (need {cores_req} cores).")

    return f"v5p-{min(valid)}"


def round_to_power_of_two(x: float) -> int:
    """Round ``x`` to the nearest power of two."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def compute_param_count(config: LlamaConfig, vocab_size: int) -> tuple[int, int, int]:
    # Copied from compute_num_parameters in experiments/llama.py and modified to
    # return multiple results; see:
    # https://github.com/marin-community/marin/blob/bc58ab8ee62ba5e38ce4f1e2d7d64271431be160/experiments/llama.py#L249-L267
    head_size = config.hidden_dim // config.num_heads
    q_params = config.num_heads * head_size * config.hidden_dim
    k_params = config.num_kv_heads * head_size * config.hidden_dim
    v_params = config.num_kv_heads * head_size * config.hidden_dim
    o_params = config.num_heads * head_size * config.hidden_dim
    attention_params = q_params + k_params + v_params + o_params

    layer_norm_params = 2 * config.hidden_dim

    gate_params = config.hidden_dim * config.intermediate_dim
    up_params = config.hidden_dim * config.intermediate_dim
    down_params = config.intermediate_dim * config.hidden_dim
    mlp_params = gate_params + up_params + down_params

    nonembedding_params = config.num_layers * (attention_params + mlp_params + layer_norm_params)
    embedding_params = 2 * vocab_size * config.hidden_dim

    return embedding_params, nonembedding_params, nonembedding_params + embedding_params


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


@dataclass
class IsoFlopRunConfig:
    experiment_name: str
    compute_range_name: str
    steps_per_run: int
    hidden_step_size: int
    architecture: str
    hidden_size: int
    intermediate_dim: int
    num_layers: int
    n_heads: int
    n_kv_heads: int
    batch_size: int
    batch_target: int  # Original batch size before LR-based halving
    train_steps: int
    lr: float
    beta2: float
    budget: float
    steps_per_eval: int
    train_tokens: int
    dataset_tokens: int
    num_params: int
    embed_params: int
    epoch_count: int
    model_config: ModelConfig


def generate_run_configs(cfg: IsoFlopSweepConfig, budget: float) -> Iterator[IsoFlopRunConfig]:
    """Generate ISOFlop run configurations within the FLOP budget."""

    dataset_tokens = cfg.total_token_count

    # Loop over architecture as the primary dimension of the search space
    for architecture in cfg.architectures:

        # Loop through hidden size on a grid, which will determine the model
        # size and therefore token count for each run config
        for hidden_size in range(2**cfg.min_hidden_pow, (2**cfg.max_hidden_pow) + 1, cfg.hidden_step_size):
            hs_pow = math.log2(hidden_size)
            intermediate_dim = hidden_size * cfg.mlp_ratio
            num_layers = round(hidden_size / (cfg.base_hidden_layer_ratio + (hs_pow * 4) - cfg.min_hidden_pow))
            assert (
                hidden_size % cfg.hidden_head_ratio == 0
            ), f"hidden_size ({hidden_size}) must be evenly divisible by hidden_head_ratio ({cfg.hidden_head_ratio})"
            n_heads = max(1, hidden_size // cfg.hidden_head_ratio)
            n_kv_heads = n_heads

            # Calculate batch size to meet budget with fixed steps
            batch_exact_val = budget / compute_total_flops(
                batch=1,
                num_layers=num_layers,
                hidden=hidden_size,
                intermediate=intermediate_dim,
                num_kv_heads=n_kv_heads,
                num_heads=n_heads,
                steps=cfg.steps_per_run,
                seq_len=cfg.seq_len,
                vocab_size=cfg.vocab_size,
            )

            batch_size = round_to_power_of_two(batch_exact_val)
            batch_target = batch_size  # Store original before LR-based halving

            # Scale LR with sqrt(batch) and hidden size
            # Reference: https://arxiv.org/pdf/2203.03466 (Section 10 Related Works)
            lr = (cfg.lr_constant * math.sqrt(batch_size)) / hidden_size

            # Halve batch size until LR is stable
            if cfg.lr_max is not None:
                while lr > cfg.lr_max:
                    logger.warning(
                        f"Halving batch size for ({architecture=}, {hidden_size=}): "
                        f"{batch_size} -> {batch_size // 2} (lr={lr:.4f}, lr_max={cfg.lr_max})"
                    )
                    batch_size //= 2
                    lr = (cfg.lr_constant * math.sqrt(batch_size)) / hidden_size

            # Set beta2 based on https://arxiv.org/pdf/2507.07101
            b2 = 0.98 ** (batch_size / 128)

            if batch_size < 8:
                logger.warning(
                    f"Skipping config for ({architecture=}, {hidden_size=}) "
                    f"with batch size {batch_size} (less than 8)"
                )
                continue

            # Recompute exact steps based on adjusted batch size
            steps_exact = budget / compute_total_flops(
                batch=batch_size,
                num_layers=num_layers,
                hidden=hidden_size,
                intermediate=intermediate_dim,
                num_kv_heads=n_kv_heads,
                num_heads=n_heads,
                steps=1,
                seq_len=cfg.seq_len,
                vocab_size=cfg.vocab_size,
            )
            train_steps = round(steps_exact)

            # Ensure actual flops still within range
            achieved_flops = compute_total_flops(
                batch=batch_size,
                num_layers=num_layers,
                hidden=hidden_size,
                intermediate=intermediate_dim,
                num_kv_heads=n_kv_heads,
                num_heads=n_heads,
                steps=train_steps,
                seq_len=cfg.seq_len,
                vocab_size=cfg.vocab_size,
            )
            if abs(achieved_flops - budget) / budget > cfg.flop_tolerance:
                logger.warning(
                    f"Skipping config for ({architecture=}, {hidden_size=}) with achieved flops {achieved_flops} "
                    f"(not within {cfg.flop_tolerance} of budget {budget})"
                )
                continue

            train_tokens = train_steps * batch_size * cfg.seq_len
            # Subtract 1 from num_evals to account for the first evaluation
            num_evals = max(1, cfg.num_evals - 1)
            steps_per_eval = max(1, train_steps // num_evals)

            if train_tokens > dataset_tokens:
                logger.warning(
                    f"Skipping config for ({architecture=}, {hidden_size=}) with train tokens {train_tokens} "
                    f"(greater than dataset tokens {dataset_tokens})"
                )
                continue

            if architecture == "llama":
                model_cfg = LlamaConfig(
                    max_seq_len=cfg.seq_len,
                    hidden_dim=hidden_size,
                    intermediate_dim=intermediate_dim,
                    num_heads=n_heads,
                    num_kv_heads=n_kv_heads,
                    num_layers=num_layers,
                )
            elif architecture == "qwen":
                model_cfg = Qwen3Config(
                    max_seq_len=cfg.seq_len,
                    hidden_dim=hidden_size,
                    intermediate_dim=intermediate_dim,
                    num_heads=n_heads,
                    num_kv_heads=n_kv_heads,
                    num_layers=num_layers,
                    rope=Llama3RotaryEmbeddingsConfig(),
                )
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

            embed_params, _, num_params = compute_param_count(model_cfg, cfg.vocab_size)

            for epoch_count in cfg.epochs:
                yield IsoFlopRunConfig(
                    experiment_name=cfg.experiment_name,
                    compute_range_name=cfg.compute_range_name,
                    steps_per_run=cfg.steps_per_run,
                    hidden_step_size=cfg.hidden_step_size,
                    architecture=architecture,
                    hidden_size=hidden_size,
                    intermediate_dim=intermediate_dim,
                    num_layers=num_layers,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    batch_size=batch_size,
                    batch_target=batch_target,
                    train_steps=train_steps,
                    lr=lr,
                    beta2=b2,
                    budget=budget,
                    steps_per_eval=steps_per_eval,
                    train_tokens=train_tokens,
                    dataset_tokens=dataset_tokens,
                    num_params=num_params,
                    embed_params=embed_params,
                    epoch_count=epoch_count,
                    model_config=model_cfg,
                )


def downsample_configs(configs: list[IsoFlopRunConfig], max_configs: int = MAX_SWEEP_CONFIGS) -> list[IsoFlopRunConfig]:
    """Downsample configs to at most max_configs, keeping first and last, evenly sampling middle.

    Configs are assumed to be sorted by hidden_size (ascending). The downsampling preserves
    the first and last configs (smallest and largest hidden_size) and takes every Nth config
    from the middle, where N is chosen to keep the total count <= max_configs.

    Args:
        configs: List of IsoFlopRunConfig (assumed sorted by hidden_size ascending)
        max_configs: Maximum number of configs to return

    Returns:
        Downsampled list with at most max_configs configs, preserving first and last
    """
    if len(configs) <= max_configs:
        return configs

    if max_configs < 2:
        raise ValueError("max_configs must be at least 2 to keep first and last")

    # Keep first and last, downsample middle
    first = configs[0]
    last = configs[-1]
    middle = configs[1:-1]

    # Calculate step size N to get at most (max_configs - 2) middle elements
    # N is the minimum step that keeps len(downsampled_middle) <= max_middle
    max_middle = max_configs - 2
    n = math.ceil(len(middle) / max_middle)

    # Take every Nth element from middle
    downsampled_middle = middle[::n]

    return [first, *downsampled_middle, last]


def _log_isoflop_run_configs(all_configs: list[IsoFlopRunConfig]):
    """Log summary of generated ISOFlop configurations."""
    if all_configs:
        df = pd.DataFrame([dataclasses.asdict(c) for c in all_configs])

        # Format large numbers for readability
        if "num_params" in df.columns:
            df["num_params"] = df["num_params"].apply(format_num)
        if "embed_params" in df.columns:
            df["embed_params"] = df["embed_params"].apply(format_num)
        if "train_tokens" in df.columns:
            df["train_tokens"] = df["train_tokens"].apply(format_num)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        logger.info("\n" + "=" * 80)
        logger.info("Configuration Summary Dataframe")
        logger.info("=" * 80)
        logger.info(
            "\n"
            + str(df.drop(columns=["model_config", "intermediate_dim", "n_kv_heads", "dataset_tokens", "steps_per_run"]))
        )

        logger.info("\n" + "=" * 50)
        logger.info("Configs per Budget")
        logger.info("=" * 50)
        logger.info("\n" + str(df.groupby("budget").size()))

        logger.info("\n" + "=" * 50)
        logger.info("Configs per Architecture")
        logger.info("=" * 50)
        logger.info("\n" + str(df.groupby("architecture").size()))

        # Create table of unique param count mappings
        logger.info("\n" + "=" * 50)
        logger.info("Param Count Mapping")
        logger.info("=" * 50)
        raw_df = pd.DataFrame([dataclasses.asdict(c) for c in all_configs])
        param_mapping = raw_df[["num_params", "embed_params"]].drop_duplicates()
        param_mapping["nonembedding_params"] = param_mapping["num_params"] - param_mapping["embed_params"]
        param_mapping["pct_embedding_params"] = (
            param_mapping["embed_params"] / param_mapping["num_params"] * 100
        ).round(2)
        param_mapping = param_mapping[
            ["num_params", "embed_params", "nonembedding_params", "pct_embedding_params"]
        ].sort_values("num_params")
        param_mapping = param_mapping.rename(
            columns={
                "num_params": "Total Params",
                "embed_params": "Embedding Params",
                "nonembedding_params": "Non-Embedding Params",
                "pct_embedding_params": "% Embedding",
            }
        )
        logger.info("\n" + param_mapping.to_string(index=False))

        logger.info("=" * 50 + "\n")
    else:
        logger.warning("No configurations generated!")


def generate_isoflop_steps(config: IsoFlopSweepConfig) -> list[ExecutorStep]:
    """Generate executor steps for an ISOFlop sweep."""

    # Collect all run configs first, downsampling per (budget, architecture) combination
    all_configs: list[IsoFlopRunConfig] = []

    for budget in config.budgets:
        # Collect configs for this budget, grouped by architecture
        budget_configs = list(generate_run_configs(config, budget))

        # Group by architecture
        configs_by_arch: dict[str, list[IsoFlopRunConfig]] = {}
        for c in budget_configs:
            if c.architecture not in configs_by_arch:
                configs_by_arch[c.architecture] = []
            configs_by_arch[c.architecture].append(c)

        # Downsample each architecture group and add to all_configs
        for arch, arch_configs in configs_by_arch.items():
            downsampled = downsample_configs(arch_configs)
            if len(arch_configs) > len(downsampled):
                logger.info(
                    f"Downsampled {len(arch_configs)} -> {len(downsampled)} configs "
                    f"for budget={budget:.1e}, architecture={arch}"
                )
            all_configs.extend(downsampled)

    _log_isoflop_run_configs(all_configs)

    # Generate executor steps from validated configs
    steps: list[ExecutorStep] = []
    for c in all_configs:
        # Use the pre-computed model config
        model_cfg = c.model_config

        tpu_type = pick_v5p_type(
            config=model_cfg,
            hidden=c.hidden_size,
            layers=c.num_layers,
            batch=c.batch_size,
            seq_len=config.seq_len,
            vocab=config.vocab_size,
        )
        optimizer_cfg = replace(config.base_optimizer_config, learning_rate=c.lr, beta2=c.beta2)
        train_cfg = replace(
            config.base_train_config,
            train_batch_size=c.batch_size,
            learning_rate=c.lr,
            num_train_steps=c.train_steps,
            steps_per_eval=c.steps_per_eval,
            per_device_eval_parallelism=config.per_device_eval_parallelism,
            max_eval_batches=config.max_eval_batches,
            resources=ResourceConfig.with_tpu(tpu_type),
            optimizer_config=optimizer_cfg,
        )

        param_count = c.num_params
        step = simulated_epoch_train(
            name="-".join(
                [
                    config.experiment_name,
                    f"A_{c.architecture}",
                    f"F{c.budget:.1e}",
                    f"P{format_num(param_count)}",
                    f"T{format_num(c.train_tokens)}",
                    f"E{c.epoch_count}",
                ]
            ),
            tokenized=config.tokenized_dataset,
            model_config=model_cfg,
            train_config=train_cfg,
            train_tokens=c.train_tokens,
            dataset_tokens=c.dataset_tokens,
            epoch_count=c.epoch_count,
            eval_harness_tasks=[],
            tags=(
                f"architecture={c.architecture}",
                f"flops_budget={c.budget:.1e}",
                f"hidden_size={c.hidden_size}",
                f"num_layers={c.num_layers}",
                f"batch_size={c.batch_size}",
                f"steps={c.train_steps}",
                f"tpu={tpu_type}",
                f"params={param_count}",
                f"params_embed={c.embed_params}",
                f"params_nonembed={c.num_params - c.embed_params}",
                f"tokens={c.train_tokens}",
                f"epochs={c.epoch_count}",
            ),
        )
        steps.append(step)

    return steps


def generate_isoflop_sweeps(
    tokenized: LMMixtureDatasetConfig,
    vocab_size: int,
    seq_len: int,
    total_token_count: int,
) -> list[ExecutorStep]:
    """Generate executor steps for all ISOFlop sweeps."""
    all_steps: list[ExecutorStep] = []
    for sweep_params in ISOFLOP_SWEEPS.values():
        sweep_cfg = IsoFlopSweepConfig(
            tokenized_dataset=tokenized,
            vocab_size=vocab_size,
            seq_len=seq_len,
            total_token_count=total_token_count,
            experiment_name=sweep_params.experiment_name,
            compute_range_name=sweep_params.compute_range_name,
            budgets=sweep_params.budgets,
            steps_per_run=sweep_params.steps_per_run,
            hidden_step_size=sweep_params.hidden_step_size,
            hidden_head_ratio=sweep_params.hidden_head_ratio,
        )
        steps = generate_isoflop_steps(sweep_cfg)
        all_steps.extend(steps)
    return all_steps


def get_data_config() -> tuple[LMMixtureDatasetConfig, IsoFlopDataConfig]:
    """Use pretokenized DCLM baseline dataset from GCS.

    Returns:
        A tuple of (LMMixtureDatasetConfig, IsoFlopDataConfig with dataset metadata).
    """
    data_config = IsoFlopDataConfig()
    # Create LMMixtureDatasetConfig directly with absolute GCS path (no download needed)
    mixture_config = LMMixtureDatasetConfig(
        configs={
            "dclm_baseline": UrlDatasetSourceConfig(
                cache_dir=data_config.tokenized_path,
            )
        },
        train_weights={"dclm_baseline": 1.0},
        tokenizer=data_config.tokenizer,
        # TODO: reduce this after ruling out eval noise (back to ~1024)
        num_validation_sequences={"dclm_baseline": 131_072},
    )
    return mixture_config, data_config


def main():
    mixture_config, data_config = get_data_config()

    # Generate sweep steps
    plantcad_sweep = generate_isoflop_sweeps(
        tokenized=mixture_config,
        vocab_size=data_config.vocab_size,
        seq_len=data_config.seq_len,
        total_token_count=data_config.total_token_count,
    )

    # Execute in batches of 16 sweep steps to avoid head node OOM errors.
    # See: https://discord.com/channels/1354881461060243556/1442689455554171071/1447914001957785620
    # TODO: Figure out how to run on compute nodes instead w/o reverting back to this PR:
    #       https://discord.com/channels/1354881461060243556/1442689455554171071/1447920947402375291
    batch_size = 16
    batches = [plantcad_sweep[i : i + batch_size] for i in range(0, len(plantcad_sweep), batch_size)]
    batch_index: int | None = int(os.environ["SWEEP_BATCH_INDEX"]) if "SWEEP_BATCH_INDEX" in os.environ else None
    if batch_index is not None:
        logger.info(f"SWEEP_BATCH_INDEX={batch_index}; running batch {batch_index + 1} of {len(batches)}")
        batches = [batches[int(batch_index)]]
    for i, batch in enumerate(batches):
        logger.info(f"Running batch {i + 1}/{len(batches)} with {len(batch)} sweep steps")
        if os.environ.get("DRY_RUN"):
            logger.info("DRY RUN; skipping execution")
            continue
        executor_main(steps=batch)


if __name__ == "__main__":
    main()
