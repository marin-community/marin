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

"""Generate ISOFlop sweep steps for varying model sizes, architectures and epochs on a target plant DNA dataset."""

import os
import math
import logging
import dataclasses
import numpy as np
import pandas as pd
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.defaults import default_train, _prepare_data_config
from experiments.evals.task_configs import EvalTaskConfig
from experiments.llama import compute_num_parameters
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path, versioned
from marin.processing.tokenize.tokenize import HfTokenizeConfig, tokenize
from fray.cluster import ResourceConfig

logger = logging.getLogger("ray")

# TPU v5p hardware constants for memory estimation
# Constants for TPU v5p
HBM_PER_CHIP_GIB = 95
CORES_PER_CHIP = 2
V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]  # TPU slices

ModelConfig = LlamaConfig | Qwen3Config


def simulated_epoch_train(
    name: str,
    tokenized: InputName | ExecutorStep,
    model_config: ModelConfig,
    train_config: "SimpleTrainConfig",
    train_tokens: int,
    dataset_tokens: int,
    epoch_count: int = 1,
    tags: Sequence[str] = (),
    use_default_validation: bool = False,
    eval_harness_tasks: Sequence[EvalTaskConfig] = (),
    complement_map: tuple[int, ...] | None = None,
) -> ExecutorStep:
    """Train with simulated epoching. When epoch_count=1, uses full dataset."""
    if not isinstance(epoch_count, int) or epoch_count < 1:
        raise ValueError(f"epoch_count must be int >= 1, got {epoch_count}")

    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    # Enable reverse-complement augmentation if complement_map is provided
    if complement_map is not None:
        pretraining_data = dataclasses.replace(pretraining_data, complement_map=complement_map)

    if epoch_count == 1:
        return default_train(
            name,
            tokenized=pretraining_data,
            model_config=model_config,
            train_config=train_config,
            tags=tags,
            use_default_validation=use_default_validation,
            eval_harness_tasks=eval_harness_tasks,
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
        use_default_validation=use_default_validation,
        eval_harness_tasks=eval_harness_tasks,
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
    dataset_name: str = versioned("plantcad/opengenome2-metagenomes-plantcad2-c4096")
    dataset_revision: str | None = versioned("913963b5ffe4354a23f9200d39984c0565ef45f9")
    seq_len: int = 4096
    total_token_count: int = 10_807_934_976


@dataclass(frozen=True)
class IsoFlopTokenizeConfig(HfTokenizeConfig):
    tokenizer: str = versioned("kuleshov-group/PlantCAD2-Small-l24-d0768")
    format: TextLmDatasetFormat = dataclasses.field(default_factory=lambda: TextLmDatasetFormat(text_key="text"))
    vocab_size: int = 7
    # DNA complement map for reverse-complement augmentation.
    # Maps token IDs to their complements: A↔T (3↔6), C↔G (4↔5), special tokens unchanged.
    # From: https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768/blob/main/config.json
    # Set to enable: (0, 1, 2, 6, 5, 4, 3, 7)
    complement_map: tuple[int, ...] | None = None


@dataclass(frozen=True)
class IsoFlopSweepConfig:
    tokenized_dataset: InputName | str
    vocab_size: int
    seq_len: int
    total_token_count: int
    experiment_name: str = "plantcad_isoflop_v1.8"
    complement_map: tuple[int, ...] | None = None
    budgets: list[float] = dataclasses.field(
        default_factory=lambda: list(np.logspace(np.log10(3.3e16), np.log10(2.03e17), 5))
    )
    epochs: list[int] = dataclasses.field(default_factory=lambda: [1])
    steps_per_run: int = 8_192
    min_hidden_pow: int = 8
    max_hidden_pow: int = 10
    hidden_step_size: int = 128
    mlp_ratio: int = 4
    base_hidden_layer_ratio: int = 64
    hidden_head_ratio: int = 128
    lr_max: float | None = 0.02
    flop_tolerance: float = 0.01
    architectures: list[str] = dataclasses.field(default_factory=lambda: ["qwen"])
    per_device_eval_parallelism: int = 512
    max_eval_batches: int = 64
    num_evals: int = 3

    lr_constant: float = 0.33
    base_optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: CautiousConfig(
            learning_rate=1.0,  # Placeholder
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=0.1,
            beta1=0.95,
            beta2=0.98,  # Placeholder
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
            num_train_steps=50_000,  # Placeholder
            learning_rate=1.0,  # Placeholder
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
    fudge_factor: float = 2,
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
    param_count = compute_num_parameters(config, vocab)
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
    architecture: str
    hidden_size: int
    intermediate_dim: int
    num_layers: int
    n_heads: int
    n_kv_heads: int
    batch_size: int
    train_steps: int
    lr: float
    beta2: float
    budget: float
    steps_per_eval: int
    train_tokens: int
    dataset_tokens: int
    num_params: int
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
            n_heads = max(1, hidden_size // cfg.hidden_head_ratio)
            n_kv_heads = n_heads

            # Calculate batch size to meet budget with fixed steps
            batch_exact = budget / compute_total_flops(
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

            batch_size = round_to_power_of_two(batch_exact)

            # Scale LR with sqrt(batch) and hidden size
            # Reference: https://arxiv.org/pdf/2203.03466 (Section 10 Related Works)
            lr = (cfg.lr_constant * math.sqrt(batch_size)) / hidden_size

            # Halve batch size until LR is stable
            if cfg.lr_max is not None:
                while lr > cfg.lr_max:
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
                )
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

            num_params = compute_num_parameters(model_cfg, cfg.vocab_size)

            for epoch_count in cfg.epochs:
                yield IsoFlopRunConfig(
                    architecture=architecture,
                    hidden_size=hidden_size,
                    intermediate_dim=intermediate_dim,
                    num_layers=num_layers,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    batch_size=batch_size,
                    train_steps=train_steps,
                    lr=lr,
                    beta2=b2,
                    budget=budget,
                    steps_per_eval=steps_per_eval,
                    train_tokens=train_tokens,
                    dataset_tokens=dataset_tokens,
                    num_params=num_params,
                    epoch_count=epoch_count,
                    model_config=model_cfg,
                )


def _log_isoflop_run_configs(all_configs: list[IsoFlopRunConfig]):
    """Log summary of generated ISOFlop configurations."""
    if all_configs:
        df = pd.DataFrame([dataclasses.asdict(c) for c in all_configs])

        # Format large numbers for readability
        if "num_params" in df.columns:
            df["num_params"] = df["num_params"].apply(format_num)
        if "train_tokens" in df.columns:
            df["train_tokens"] = df["train_tokens"].apply(format_num)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        logger.info("\n" + "=" * 80)
        logger.info("Configuration Summary Dataframe")
        logger.info("=" * 80)
        logger.info("\n" + str(df.drop(columns=["model_config"])))

        logger.info("\n" + "=" * 50)
        logger.info("Configs per Budget")
        logger.info("=" * 50)
        logger.info("\n" + str(df.groupby("budget").size()))

        logger.info("\n" + "=" * 50)
        logger.info("Configs per Architecture")
        logger.info("=" * 50)
        logger.info("\n" + str(df.groupby("architecture").size()))

        logger.info("=" * 50 + "\n")
    else:
        logger.warning("No configurations generated!")


def generate_isoflop_steps(config: IsoFlopSweepConfig) -> list[ExecutorStep]:
    """Generate executor steps for an ISOFlop sweep."""

    # Collect all run configs first
    all_configs: list[IsoFlopRunConfig] = []

    for budget in config.budgets:
        for c in generate_run_configs(config, budget):
            all_configs.append(c)

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
            use_default_validation=False,
            tags=(
                f"architecture={c.architecture}",
                f"flops_budget={c.budget:.1e}",
                f"hidden_size={c.hidden_size}",
                f"num_layers={c.num_layers}",
                f"batch_size={c.batch_size}",
                f"steps={c.train_steps}",
                f"tpu={tpu_type}",
                f"params={param_count}",
                f"tokens={c.train_tokens}",
                f"epochs={c.epoch_count}",
                f"complement_map={config.complement_map is not None}",
            ),
            complement_map=config.complement_map,
        )
        steps.append(step)

    return steps


def generate_isoflop_sweep(
    tokenized: ExecutorStep,
    **kwargs,
) -> list[ExecutorStep]:
    sweep_cfg = IsoFlopSweepConfig(tokenized_dataset=tokenized, **kwargs)
    steps = generate_isoflop_steps(sweep_cfg)

    return steps


def tokenize_plantcad() -> tuple[ExecutorStep, IsoFlopDataConfig]:
    """Tokenize the PlantCAD dataset directly from HuggingFace.

    Returns:
        A tuple of (tokenized ExecutorStep, IsoFlopDataConfig with dataset metadata).
    """
    data_config = IsoFlopDataConfig()
    step = ExecutorStep(
        name="tokenized/plantcad2",
        fn=tokenize,
        config=IsoFlopTokenizeConfig(
            id=data_config.dataset_name,
            revision=data_config.dataset_revision,
            cache_path=this_output_path(),
        ),
    )
    return step, data_config


def main():
    plantcad_tokenized, data_config = tokenize_plantcad()

    # Generate sweep steps
    plantcad_sweep = generate_isoflop_sweep(
        tokenized=plantcad_tokenized,
        vocab_size=plantcad_tokenized.config.vocab_size,
        seq_len=data_config.seq_len,
        total_token_count=data_config.total_token_count,
        complement_map=plantcad_tokenized.config.complement_map,
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
        executor_main(steps=[plantcad_tokenized, *batch])


if __name__ == "__main__":
    main()
