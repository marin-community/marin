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

"""Generate ISOFlop target steps for specific model sizes and budgets.

This script constructs `ExecutorStep` objects that train models of specific
sizes for specific FLOP budgets. Unlike the sweep script, this allows manual
specification of (budget, hidden_size, tpu_type) combinations while keeping
the same hyperparameter derivation logic for compatibility with IsoFLOP plots.
"""

import dataclasses
import math
from dataclasses import dataclass, replace

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.audio.exp1699_marin_audio_all import data_mix_config
from experiments.defaults import default_train
from experiments.llama import compute_num_parameters
from experiments.metrics.wandb_related import get_vocab_size_for_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main

MLP_RATIO = 4


@dataclass
class TargetRunConfig:
    """Configuration for a single target run."""

    budget: float
    hidden_size: int
    tpu_type: str  # e.g., "v5p-8", "v5p-32", etc.


@dataclass
class IsoFlopTargetConfig:
    """Configuration for generating ISOFlop target steps."""

    tokenized_dataset: InputName | str
    tokenizer: str = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"
    seq_len: int = 4096
    steps_per_run: int = 2**16
    base_hidden_layer_ratio: int = 64
    hidden_head_ratio: int = 128
    lr_constant: float = 0.33
    min_hidden_pow: int = 7
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
            resources=ResourceConfig.with_tpu("v5p-8"),
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


def compute_hyperparams_from_hidden_and_budget(
    cfg: IsoFlopTargetConfig,
    budget: float,
    hidden_size: int,
    vocab_size: int,
) -> tuple[int, int, int, int, int, int, float, float]:
    """Compute hyperparameters from hidden size and budget.

    Uses the same logic as candidate_configs in isoflop_audio_sweep.py
    to ensure compatibility with IsoFLOP plots.

    Returns:
        (intermediate_dim, num_layers, n_heads, n_kv_heads, batch_size, train_steps, lr, b2)
    """
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

    return (intermediate_dim, num_layers, n_heads, n_kv_heads, batch_size, train_steps, lr, b2)


def generate_target_step(
    cfg: IsoFlopTargetConfig,
    target: TargetRunConfig,
    experiment_name: str,
    vocab_size: int,
) -> tuple[ExecutorStep, dict]:
    """Generate a single executor step for a target configuration."""
    budget = target.budget
    hidden_size = target.hidden_size
    tpu_type = target.tpu_type

    (
        intermediate_dim,
        num_layers,
        n_heads,
        n_kv_heads,
        batch_size,
        train_steps,
        lr,
        b2,
    ) = compute_hyperparams_from_hidden_and_budget(cfg, budget, hidden_size, vocab_size)

    model_cfg = Qwen3Config(
        max_seq_len=cfg.seq_len,
        hidden_dim=hidden_size,
        intermediate_dim=intermediate_dim,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        num_layers=num_layers,
        rope=Llama3RotaryEmbeddingsConfig(),
    )

    optimizer_cfg = replace(cfg.base_optimizer_config, learning_rate=lr, beta2=b2)
    train_cfg = replace(
        cfg.base_train_config,
        train_batch_size=batch_size,
        learning_rate=lr,
        num_train_steps=train_steps,
        resources=ResourceConfig.with_tpu(tpu_type),
        optimizer_config=optimizer_cfg,
    )

    param_count = compute_num_parameters(model_cfg, vocab_size)
    if param_count > 1e9:
        param_count_str = f"{param_count / 1e9:,.2f}B"
    else:
        param_count_str = f"{param_count / 1e6:,.0f}M"

    num_tokens = batch_size * train_steps * cfg.seq_len
    num_tokens_str = f"{num_tokens / 1e9:,.0f}B"

    step = default_train(
        name=f"{experiment_name}-isoflop-{budget:.0e}-{param_count_str}-d{hidden_size}-L{num_layers}-B{batch_size}",
        tokenized=cfg.tokenized_dataset,
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

    metadata = {
        "budget": budget,
        "hidden_size": hidden_size,
        "intermediate_dim": intermediate_dim,
        "num_layers": num_layers,
        "n_heads": n_heads,
        "batch_size": batch_size,
        "train_steps": train_steps,
        "lr": lr,
        "param_count": param_count,
        "param_count_str": param_count_str,
        "num_tokens": num_tokens,
        "num_tokens_str": num_tokens_str,
        "tpu_type": tpu_type,
    }

    return step, metadata


def generate_target_steps(
    cfg: IsoFlopTargetConfig,
    targets: list[TargetRunConfig],
    experiment_name: str,
) -> tuple[list[ExecutorStep], list[dict]]:
    """Generate executor steps for a list of target configurations."""
    vocab_size = get_vocab_size_for_tokenizer(cfg.tokenizer)
    if vocab_size is None and cfg.tokenizer == "potsawee/marin-mimi-bpe-8cb-16k-tokenizer":
        vocab_size = 144644

    steps = []
    metadata_list = []

    for target in targets:
        step, metadata = generate_target_step(cfg, target, experiment_name, vocab_size)
        steps.append(step)
        metadata_list.append(metadata)

        print(f"Target: budget={target.budget:.1e}, hidden={target.hidden_size}, tpu={target.tpu_type}")
        print(f"  -> L={metadata['num_layers']}, B={metadata['batch_size']}, steps={metadata['train_steps']}")
        print(f"  -> params={metadata['param_count_str']}, tokens={metadata['num_tokens_str']}, lr={metadata['lr']:.6f}")

    return steps, metadata_list


if __name__ == "__main__":
    # Define specific target runs here
    # Each TargetRunConfig specifies: (budget, hidden_size, tpu_type)
    # Other hyperparameters (num_layers, batch_size, etc.) are derived automatically
    # using the same logic as isoflop_audio_sweep.py for compatibility.

    # next hidden sizes: 2048, 2304, 2560, 2816, 3072
    targets = [
        TargetRunConfig(budget=1e20, hidden_size=2304, tpu_type="v5p-64"),
        TargetRunConfig(budget=1e20, hidden_size=2048, tpu_type="v5p-32"),
        TargetRunConfig(budget=6e19, hidden_size=2048, tpu_type="v5p-32"),
        TargetRunConfig(budget=3e19, hidden_size=2048, tpu_type="v5p-16"),
        TargetRunConfig(budget=6e19, hidden_size=2304, tpu_type="v5p-16"),
        TargetRunConfig(budget=1e20, hidden_size=3072, tpu_type="v5p-64"),  # 5.42B
        TargetRunConfig(budget=1e20, hidden_size=2816, tpu_type="v5p-64"),  # 4.24B
    ]

    cfg = IsoFlopTargetConfig(tokenized_dataset=data_mix_config)
    steps, metadata = generate_target_steps(cfg, targets, experiment_name="discrete-audio-target")

    print("-" * 40)
    print("Generated steps summary:")
    for m in metadata:
        print(
            f"  [{m['budget']:.1e}] d={m['hidden_size']} L={m['num_layers']} "
            f"B={m['batch_size']} steps={m['train_steps']} -> {m['param_count_str']}, {m['num_tokens_str']}"
        )

    executor_main(steps=steps, description="Train specific model configurations to improve IsoFLOP curve fitting.")
