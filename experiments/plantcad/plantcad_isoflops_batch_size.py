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

"""Generate ISOFlop sweep steps for varying model and batch sizes on a target plant DNA dataset."""

import dataclasses
import math
import logging
import os
import jax
from collections.abc import Iterator
import pandas as pd
from dataclasses import dataclass, replace

from levanter.data.text import TextLmDatasetFormat
from levanter.data.sharded_datasource import WrappedHFDataSource
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.defaults import default_train
from experiments.llama import compute_num_parameters
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path
from marin.resources import TpuPodConfig
from marin.processing.tokenize import TokenizeConfig, tokenize
from zephyr import Dataset, create_backend

logger = logging.getLogger("ray")


# TPU v5p hardware constants for memory estimation
# Constants for TPU v5p
HBM_PER_CHIP_GIB = 95
CORES_PER_CHIP = 2
V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]  # TPU slices


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
    output_path: str
    dataset_name: str = "plantcad/Angiosperm_65_genomes_8192bp"
    # Original sequence length for the dataset
    input_seq_len: int = 8192
    # Target sequence length for the dataset after cropping
    output_seq_len: int = 4096
    # Token count in training split (prior to cropping)
    total_token_count: int = 29_670_825_984
    # Per-shard sample limit; e.g. for 60 shards in plantcad2 dataset,
    # multiple this value by 60 to get final sample limit
    sample_limit: int | None = None
    train_split: str = "train"
    validation_split: str = "validation"


@dataclass(frozen=True)
class IsoFlopTokenizeConfig(TokenizeConfig):
    tokenizer: str = "kuleshov-group/PlantCAD2-Small-l24-d0768"
    format: TextLmDatasetFormat = dataclasses.field(default_factory=lambda: TextLmDatasetFormat(text_key="seq"))
    vocab_size: int = 7


@dataclass(frozen=True)
class IsoFlopSweepConfig:
    tokenized_dataset: InputName | str
    vocab_size: int
    seq_len: int
    total_token_count: int
    output_seq_len: int
    input_seq_len: int
    experiment_name: str = "plantcad_isoflop_xbs_01"
    budgets: list[float] = dataclasses.field(default_factory=lambda: [1e17, 3.3e17, 6.6e17, 1e18])
    batch_sizes: list[int] = dataclasses.field(default_factory=lambda: [8, 32, 128, 512])
    num_eval_tokens: int = 268_435_456
    num_evals: int = 3
    min_steps: int = 10
    max_learning_rate: float = 0.01
    flop_tolerance: float = 0.01
    base_hidden_layer_ratio: int = 64
    hidden_head_ratio: int = 128
    lr_constant: float = 0.33
    min_hidden_pow: int = 9
    max_hidden_pow: int = 11
    mlp_ratio: int = 4
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
            resources=TpuPodConfig(tpu_type="v5p-8"),
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
    """Estimate float32 memory usage (in bytes) for one training step."""
    param_bytes = param_count * optim_mult * dtype_size

    act_bytes = (batch * seq_len) * ((hidden_dim * num_layers) + vocab * fudge_factor)

    total_bytes = param_bytes + act_bytes
    return int(total_bytes) * fudge_factor


def pick_v5p_type(
    config: Qwen3Config,
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
    max_eval_batches: int
    eval_tokens: int
    train_tokens: int


def generate_run_configs(cfg: IsoFlopSweepConfig, budget: float) -> Iterator[IsoFlopRunConfig]:
    """Generate ISOFlop run configurations within the FLOP budget."""

    # Hidden size step for grid
    step_size: int = 256

    # Loop over batch size as the primary dimension of the search space
    for batch_size in cfg.batch_sizes:
        # Loop through hidden size on a grid, which will determine the model
        # size and therefore token count for each run config
        for hidden_size in range(2**cfg.min_hidden_pow, (2**cfg.max_hidden_pow) + 1, step_size):
            hs_pow = math.log2(hidden_size)
            intermediate_dim = hidden_size * cfg.mlp_ratio
            num_layers = round(hidden_size / (cfg.base_hidden_layer_ratio + (hs_pow * 4) - cfg.min_hidden_pow))
            n_heads = max(1, hidden_size // cfg.hidden_head_ratio)
            n_kv_heads = n_heads

            # Calculate training steps to meet budget
            flops_per_step = compute_total_flops(
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
            train_steps = round(budget / flops_per_step)
            if train_steps < cfg.min_steps:
                continue

            # Ensure actual flops still within range
            achieved_flops = flops_per_step * train_steps
            if abs(achieved_flops - budget) / budget > cfg.flop_tolerance:
                continue

            # - Scale LR with sqrt(batch) and hidden size
            #   - A particularly useful reference for this is (uTransfer):
            #     https://arxiv.org/pdf/2203.03466 (Section 10 Related Works)
            # - Note that setting LR based on batch size here w/o a constant step count
            #   is controversial; see here for discussion on it:
            #   - https://discordapp.com/channels/1354881461060243556/1442534344622215178/1442602508496273553
            #   - https://github.com/marin-community/marin/pull/1460#issuecomment-3156920910
            # - TODO: Run sensitivity analysis on this vs fixed LR at the end?
            lr = (cfg.lr_constant * math.sqrt(batch_size)) / hidden_size

            # Skip configs with unstable LRs
            if lr > cfg.max_learning_rate:
                continue

            # TODO: Make sure that setting max_eval_batches based on target
            # token count works given that TrainerConfig ultimately sets batch
            # size (eval_batch_size) as `per_device_eval_parallelism * data_axis_size`
            max_eval_batches = cfg.num_eval_tokens // (batch_size * cfg.seq_len)
            eval_tokens = max_eval_batches * batch_size * cfg.seq_len
            train_tokens = train_steps * batch_size * cfg.seq_len
            # Subtract 1 from num_evals to account for the first evaluation
            num_evals = max(1, cfg.num_evals - 1)
            steps_per_eval = max(1, train_steps // num_evals)

            # Set beta2 based on https://arxiv.org/pdf/2507.07101
            b2 = 0.98 ** (batch_size / 128)

            yield IsoFlopRunConfig(
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
                max_eval_batches=max_eval_batches,
                eval_tokens=eval_tokens,
                train_tokens=train_tokens,
            )


def _log_isoflop_run_configs(all_configs: list[IsoFlopRunConfig]):
    """Log summary of generated ISOFlop configurations."""
    if all_configs:
        df = pd.DataFrame([dataclasses.asdict(c) for c in all_configs])

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        logger.info("\n" + "=" * 80)
        logger.info("Configuration Summary Dataframe")
        logger.info("=" * 80)
        logger.info("\n" + str(df))

        logger.info("\n" + "=" * 50)
        logger.info("Configs per Budget")
        logger.info("=" * 50)
        logger.info("\n" + str(df.groupby("budget").size()))

        logger.info("\n" + "=" * 50)
        logger.info("Configs per Batch Size")
        logger.info("=" * 50)
        logger.info("\n" + str(df.groupby("batch_size").size()))

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

    if all_configs:
        # Validate that eval_tokens is consistent to ensure that validation loss is comparable
        logger.info("Validating that eval_tokens is identical across all configs")
        eval_token_counts = {c.eval_tokens for c in all_configs}
        if len(eval_token_counts) > 1:
            raise ValueError(f"eval_tokens must be the same for all run configs, but got: {sorted(eval_token_counts)}")

        # Validate that train_tokens are never too high
        logger.info("Validating that train_tokens doesn't exceed available tokens (no epoching)")
        effective_token_count = config.total_token_count * (config.output_seq_len / config.input_seq_len)
        max_train_tokens = max(c.train_tokens for c in all_configs)
        if max_train_tokens > effective_token_count:
            raise ValueError(
                f"Maximum train_tokens ({max_train_tokens:,}) exceeds available tokens "
                f"after cropping ({effective_token_count:,.0f}). "
                f"Original token count: {config.total_token_count:,}, "
                f"Cropping ratio: {config.output_seq_len}/{config.input_seq_len}"
            )

    _log_isoflop_run_configs(all_configs)

    # Generate executor steps from validated configs
    steps: list[ExecutorStep] = []
    for c in all_configs:
        model_cfg = Qwen3Config(
            seq_len=config.seq_len,
            hidden_dim=c.hidden_size,
            intermediate_dim=c.intermediate_dim,
            num_heads=c.n_heads,
            num_kv_heads=c.n_kv_heads,
            num_layers=c.num_layers,
        )
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
            max_eval_batches=c.max_eval_batches,
            resources=TpuPodConfig(tpu_type=tpu_type),
            optimizer_config=optimizer_cfg,
        )

        param_count = compute_num_parameters(model_cfg, config.vocab_size)
        step = default_train(
            name="-".join(
                [
                    config.experiment_name,
                    f"F{c.budget:.1e}",
                    f"P{format_num(param_count)}",
                    f"T{format_num(c.train_tokens)}",
                    f"S{c.train_steps}",
                    f"B{c.batch_size}",
                ]
            ),
            tokenized=config.tokenized_dataset,
            model_config=model_cfg,
            train_config=train_cfg,
            eval_harness_tasks=[],
            use_default_validation=False,
            tags=(
                f"flops_budget={c.budget:.1e}",
                f"hidden_size={c.hidden_size}",
                f"num_layers={c.num_layers}",
                f"batch_size={c.batch_size}",
                f"steps={c.train_steps}",
                f"tpu={tpu_type}",
                f"params={param_count}",
                f"tokens={c.train_tokens}",
            ),
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


def _verify_jax_cpu_only():
    """Verify that JAX is configured for CPU-only mode.

    See:
    - https://github.com/marin-community/marin/blob/821f9d79d7344960ce023f027ac77952389c8b84/lib/marin/src/marin/processing/tokenize/tokenize.py#L276-L290
    - https://openathena.slack.com/archives/C09AUMZ3QUA/p1764006926655589?thread_ts=1763988866.060449&cid=C09AUMZ3QUA
    """
    # Verify JAX is configured for CPU-only mode
    jax_platforms = os.environ.get("JAX_PLATFORMS", "not set")
    jax_devices = jax.devices()

    # Assert that JAX_PLATFORMS is set to cpu and that all devices are CPU devices
    assert jax_platforms == "cpu", f"JAX_PLATFORMS should be 'cpu' but is '{jax_platforms}'"
    assert all(d.platform == "cpu" for d in jax_devices), f"Expected all CPU devices, got: {jax_devices}"


def _prepare_dataset(config: IsoFlopDataConfig):
    # TODO: Where is the correct place to check this?
    _verify_jax_cpu_only()

    def crop(example):
        seq = example["seq"]
        return {"seq": seq[: config.output_seq_len]}

    # TODO: switch to flow_backend?
    # Keep parallelism very modest for HF reads (else 429s everywhere)
    backend = create_backend(
        backend_type="ray", max_parallelism=8, max_retries=3, memory="8GB", num_cpus=1, chunk_size=1000
    )

    for split_key, split_name in [("train", config.train_split), ("validation", config.validation_split)]:
        output_pattern = f"{config.output_path}/{split_key}/data-{{shard:05d}}.jsonl.gz"
        data_source = WrappedHFDataSource(config.dataset_name, split=split_name)

        ds = (
            Dataset.from_list(data_source.shard_names)
            .flat_map(lambda shard, ds=data_source: ds.open_shard(shard))
            .map(crop)
        )

        if config.sample_limit is not None:
            ds = ds.take_per_shard(config.sample_limit)

        ds = ds.write_jsonl(output_pattern)

        results = list(backend.execute(ds))
        logger.info(
            f"Wrote {len(results)} prepared data files to {config.output_path}/{split_key} (first 5: {results[:5]})"
        )


def prepare_plantcad_dataset() -> ExecutorStep:
    return ExecutorStep(
        name="prepared/plantcad_cropped",
        fn=_prepare_dataset,
        config=IsoFlopDataConfig(output_path=this_output_path()),
    )


def tokenize_plantcad_dataset(
    prepared: ExecutorStep,
) -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/plantcad_cropped",
        fn=tokenize,
        config=IsoFlopTokenizeConfig(
            train_paths=[prepared.cd("train")],
            validation_paths=[prepared.cd("validation")],
            cache_path=this_output_path(),
        ),
    )


def main():
    # Crop to match target sequence length
    plantcad_prepared = prepare_plantcad_dataset()

    # Tokenize
    plantcad_tokenized = tokenize_plantcad_dataset(
        prepared=plantcad_prepared,
    )

    # Generate sweep steps
    plantcad_sweep = generate_isoflop_sweep(
        tokenized=plantcad_tokenized,
        # TODO: Find a better way to chain config dependencies like this
        #       (e.g. this fails if any value is `versioned`)
        vocab_size=plantcad_tokenized.config.vocab_size,
        seq_len=plantcad_prepared.config.output_seq_len,
        total_token_count=plantcad_prepared.config.total_token_count,
        output_seq_len=plantcad_prepared.config.output_seq_len,
        input_seq_len=plantcad_prepared.config.input_seq_len,
    )

    # Execute
    executor_main(steps=[plantcad_prepared, plantcad_tokenized, *plantcad_sweep])


if __name__ == "__main__":
    main()
