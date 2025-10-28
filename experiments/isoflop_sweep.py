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

"""Generate ISOFlop sweep steps for varying model sizes on a target datasett.

This script constructs `ExecutorStep` objects that train models of different
sizes while keeping the total training FLOPs roughly constant.  It is intended
as a lightweight scaffold for ISOFlop scaling law experiments.
"""

import dataclasses
import math
from dataclasses import dataclass, replace

from levanter.data.text import LMMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.defaults import default_train
from experiments.llama import compute_num_parameters
from experiments.metrics.wandb_related import get_vocab_size_for_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from experiments.pretraining_datasets import dclm_baseline
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.processing.tokenize import lm_mixture_data_config
from experiments.common_pile.tokenize_common_pile import comma_main_mixture
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.resources import TpuPodConfig

DEFAULT_BUDGETS = [1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20]
MLP_RATIO = 4

# TPU v5p hardware constants for memory estimation
# Constants for TPU v5p
HBM_PER_CHIP_GIB = 95
CORES_PER_CHIP = 2
V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]  # TPU slices


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
    """
    Estimate float32 memory usage (in bytes) for one training step.
    Note(Will): I had to do more fudging than expected on this,
    but not seems to work ok.

    Parameters:
    - hidden_dim: model hidden size
    - num_layers: number of Transformer layers
    - batch, seq_len: training batch size and sequence length
    - vocab: vocabulary size
    - optim_mult: optimizer memory multiplier (e.g., 100x for Adam + states)
    - dtype_size: bytes per float (4 for float32)
    - fudge_factor: safety margin for extra memory

    Returns:
    - total estimated memory in bytes
    """
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
    """
    Select the smallest TPU v5p slice that fits the model in float32.

    Returns:
    - TPU slice name, e.g., "v5p-8" or "v5p-32"
    """
    param_count = compute_num_parameters(config, vocab)
    need_bytes = estimate_bytes(param_count, hidden, layers, batch, seq_len, vocab)
    chip_bytes = HBM_PER_CHIP_GIB * 1024**3
    chips = math.ceil(need_bytes / chip_bytes)
    cores_req = chips * CORES_PER_CHIP

    valid = [c for c in V5P_CORE_OPTIONS if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available v5p slices (need {cores_req} cores).")

    return f"v5p-{min(valid)}"


@dataclass
class IsoFlopSweepConfig:
    """Configuration for generating ISOFlop sweep steps."""

    tokenized_dataset: InputName | str
    tokenizer: str = "stanford-crfm/marin-tokenizer"
    budgets: list[float] = dataclasses.field(default_factory=lambda: DEFAULT_BUDGETS)
    seq_len: int = 4096
    steps_per_run: int = 2**16
    flop_tolerance: float = 0.01
    base_hidden_layer_ratio: int = 64
    hidden_head_ratio: int = 128
    lr_constant: float = 0.33
    min_hidden_pow: int = 9
    max_hidden_pow: int = 12
    base_optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: CautiousConfig(
            learning_rate=1.0,  # Placeholder
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
            resources=TpuPodConfig(tpu_type="v5p-8"),
            train_batch_size=1,
            num_train_steps=50_000,
            learning_rate=1.0,  # Placeholder
            weight_decay=0.1,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            decay=0.2,
        )
    )


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


def candidate_configs(cfg: IsoFlopSweepConfig, budget: float):
    """Yield candidate model configurations within the FLOP budget."""

    vocab_size = get_vocab_size_for_tokenizer(cfg.tokenizer)

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

        yield (hidden_size, intermediate_dim, num_layers, n_heads, n_kv_heads, batch_size, train_steps, lr, b2)


def generate_isoflop_steps(config: IsoFlopSweepConfig, experiment_name: str) -> list[ExecutorStep]:
    """Generate executor steps for an ISOFlop sweep."""

    steps: list[ExecutorStep] = []
    metadata = []
    vocab_size = get_vocab_size_for_tokenizer(config.tokenizer)

    for budget in config.budgets:
        for (
            hidden_size,
            intermediate_dim,
            num_layers,
            n_heads,
            n_kv_heads,
            batch_size,
            train_steps,
            lr,
            b2,
        ) in candidate_configs(config, budget):
            model_cfg = Qwen3Config(
                seq_len=config.seq_len,
                hidden_dim=hidden_size,
                intermediate_dim=intermediate_dim,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                num_layers=num_layers,
                rope=Llama3RotaryEmbeddingsConfig(),
            )
            tpu_type = pick_v5p_type(
                config=model_cfg,
                hidden=hidden_size,
                layers=num_layers,
                batch=batch_size,
                seq_len=config.seq_len,
                vocab=vocab_size,
            )
            optimizer_cfg = replace(config.base_optimizer_config, learning_rate=lr, beta2=b2)
            train_cfg = replace(
                config.base_train_config,
                train_batch_size=batch_size,
                learning_rate=lr,
                num_train_steps=train_steps,
                resources=TpuPodConfig(tpu_type=tpu_type),
                optimizer_config=optimizer_cfg,
            )

            step = default_train(
                name=f"isoflop-{budget:.0e}-d{hidden_size}-L{num_layers}-B{batch_size}-{experiment_name}",
                tokenized=config.tokenized_dataset,
                model_config=model_cfg,
                train_config=train_cfg,
                eval_harness_tasks=[],
                tags=(
                    f"FLOPs={budget:.1e}",
                    f"d={hidden_size}",
                    f"L={num_layers}",
                    f"B={batch_size}",
                    f"steps={train_steps}",
                    f"tpu={tpu_type}",
                ),
            )
            metadata.append((budget, hidden_size, num_layers, batch_size, train_steps))
            steps.append(step)

    return steps, metadata


def generate_isoflop_sweep(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    experiment_name: str,
    **kwargs,
) -> list[ExecutorStep]:
    sweep_cfg = IsoFlopSweepConfig(tokenized_dataset=tokenized, **kwargs)
    steps, metadata = generate_isoflop_steps(sweep_cfg, experiment_name)

    return steps, metadata


dclm_tokenized = dataclasses.replace(
    default_tokenize(
        name="dclm_baseline",
        dataset=dclm_baseline,
        tokenizer=llama3_tokenizer,
    ).with_output_path("tokenized/dclm_baseline-0206f1/"),
)


dclm_mix = lm_mixture_data_config(
    components={"dclm": dclm_tokenized},
    weights={"dclm": 1.0},
    num_validation_sequences={"dclm": 1024},
)

MARIN_SCALING_SUITES = {
    "nemotron": generate_isoflop_sweep(nemotron_mix, experiment_name="nemo-wider-depth-adapt"),
    "common_pile": generate_isoflop_sweep(comma_main_mixture(permutation_type="linear"), experiment_name="comma-mix"),
    "common_pile_feistel": generate_isoflop_sweep(
        comma_main_mixture(permutation_type="feistel"), experiment_name="comma-mix-feistel"
    ),
    "dclm-default": generate_isoflop_sweep(dclm_mix, experiment_name="dclm-default"),
}

if __name__ == "__main__":
    steps = MARIN_SCALING_SUITES["common_pile"]
    executor_main(steps=steps[0])
