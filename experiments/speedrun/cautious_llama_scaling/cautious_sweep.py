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

"""Speedruns using the Cautious optimizer for various Llama model sizes (Chinchilla optimal steps)."""

import dataclasses
import math
import logging

from levanter.models.llama import LlamaConfig
from levanter.optim.cautious import CautiousConfig

from experiments.llama import (
    compute_num_parameters,
    llama_1_4b,
    llama_150m,
    llama_300m,
    llama_600m,
    llama3_tokenizer_vocab_size,
)
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.hparam_regression import predict_lr_from_width, width_lr_fit
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="William Held", affiliation="Georgia Tech", url="https://WilliamHeld.com")

logger = logging.getLogger("ray")

WIDTH_LR_FIT = width_lr_fit()
INTERPOLATED_WIDTHS = {
    "200m": 640,
    "400m": 896,
    "800m": 1408,
}
INTERPOLATED_LRS = {}
for size, width in INTERPOLATED_WIDTHS.items():
    intercept, slope, _, _ = WIDTH_LR_FIT
    INTERPOLATED_LRS[size] = round(predict_lr_from_width(intercept, slope, width), 6)


def get_num_train_steps(param_count, batch_size, max_seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * max_seq_len
    return total_tokens // tokens_per_step


def _build_llama_from_param_count(param_target: int, seq_len: int) -> LlamaConfig:
    """
    Approximate a LlamaConfig for a target parameter count using the heuristics
    from experiments/isoflop_sweep.py (hidden -> layers/heads ratios).
    """
    best_candidate = None
    best_diff = None
    for hidden_dim in range(512, 4097, 128):
        hidden_pow = math.log2(hidden_dim)
        num_layers = max(2, round(hidden_dim / (64 + (hidden_pow * 4) - 9)))
        num_heads = max(1, hidden_dim // 128)
        intermediate_dim = hidden_dim * 4

        llama_candidate = LlamaConfig(
            max_seq_len=seq_len,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_heads,
        )
        param_estimate = compute_num_parameters(llama_candidate, llama3_tokenizer_vocab_size)
        diff = abs(param_estimate - param_target)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_candidate = llama_candidate

    if best_candidate is None:
        raise ValueError(f"Could not build Llama config for {param_target} parameters.")

    return best_candidate


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    # Parameter counts
    param_counts = {
        "130m": 130_000_000,
        "200m": 200_000_000,
        "300m": 300_000_000,
        "400m": 400_000_000,
        "520m": 520_000_000,
        "800m": 800_000_000,
        "1_2b": 1_200_000_000,
    }

    # Model configs
    model_cfgs = {
        "130m": llama_150m,
        "200m": None,
        "300m": llama_300m,
        "400m": None,
        "520m": llama_600m,
        "800m": None,
        "1_2b": llama_1_4b,
    }

    # Training batch sizes
    batch_sizes = {
        "130m": 128,
        "200m": 128,
        "300m": 128,
        "400m": 256,
        "520m": 256,
        "800m": 256,
        "1_2b": 256,
    }

    # Resource configs
    resource_cfgs = {
        "130m": ResourceConfig.with_tpu("v5p-32"),
        "200m": ResourceConfig.with_tpu("v5p-32"),
        "300m": ResourceConfig.with_tpu("v5p-32"),
        "400m": ResourceConfig.with_tpu("v5p-32"),
        "520m": ResourceConfig.with_tpu("v5p-32"),
        "800m": ResourceConfig.with_tpu("v5p-32"),
        "1_2b": ResourceConfig.with_tpu("v5p-32"),
    }

    # Cautious optimizer configs for each size.
    # Base points (from earlier runs): 130m/300m/520m/1_2b.
    # Interpolated points (lr vs sqrt(hidden_dim) fit): 200m/400m/800m.
    # Fit stats on base points: lr = a + b * sqrt(hidden_dim),
    # a=0.010522915936786618, b=-9.476592100146263e-05, R^2=0.8438000577378982, RMSE=0.00034227175854367.
    # See experiments/speedrun/hparam_regression.py for the regression code.
    cautious_configs = {
        "130m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.95,
            beta2=0.98,
            epsilon=1e-15,
            max_grad_norm=1,
        ),
        # lr = a + b * sqrt(hidden_dim) with a=0.010522915936786618, b=-9.476592100146263e-05.
        "200m": CautiousConfig(
            learning_rate=INTERPOLATED_LRS["200m"],
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=2,
        ),
        "300m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=2,
        ),
        "400m": CautiousConfig(
            learning_rate=INTERPOLATED_LRS["400m"],
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=2,
        ),
        "520m": CautiousConfig(
            learning_rate=0.008,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-25,
            max_grad_norm=1,
        ),
        "800m": CautiousConfig(
            learning_rate=INTERPOLATED_LRS["800m"],
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-16,
            max_grad_norm=1,
        ),
        "1_2b": CautiousConfig(
            learning_rate=0.006,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=2000,
            beta1=0.98,
            beta2=0.98,
            epsilon=1e-16,
            max_grad_norm=1,
        ),
    }

    # Descriptions
    descriptions = {
        "130m": "130M parameter model trained with the Cautious optimizer.",
        "200m": "200M parameter model trained with the Cautious optimizer.",
        "300m": "300M parameter model trained with the Cautious optimizer.",
        "400m": "400M parameter model trained with the Cautious optimizer.",
        "520m": "520M parameter model trained with the Cautious optimizer.",
        "800m": "800M parameter model trained with the Cautious optimizer.",
        "1_2b": "1.2B parameter model trained with the Cautious optimizer.",
    }

    # Names for the runs
    run_names = {
        "130m": "llama_130m_cautious_4096",
        "200m": "llama_200m_cautious_4096",
        "300m": "llama_300m_cautious_4096",
        "400m": "llama_400m_cautious_4096",
        "520m": "llama_520m_cautious_4096",
        "800m": "llama_800m_cautious_4096",
        "1_2b": "llama_1_2b_cautious_4096",
    }

    # Gather config for the requested size
    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    batch_size = batch_sizes[size]
    model_cfg = model_cfgs[size]
    if model_cfg is not None:
        model_config = dataclasses.replace(model_cfg, max_seq_len=4096)
    else:
        model_config = _build_llama_from_param_count(param_counts[size], 4096)
    max_seq_len = model_config.max_seq_len
    resource_config = resource_cfgs[size]
    cautious = cautious_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    if model_cfg is not None:
        param_count = param_counts[size]
    else:
        param_count = compute_num_parameters(model_config, llama3_tokenizer_vocab_size)

    num_train_steps = get_num_train_steps(param_count, batch_size, max_seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=cautious.learning_rate,
        optimizer_config=cautious,
    )
    cfg = SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train,
    )
    return run_name, cfg


if __name__ == "__main__":
    runs = [
        build_config("130m"),
        build_config("200m"),
        build_config("300m"),
        build_config("400m"),
        build_config("520m"),
        build_config("800m"),
        build_config("1_2b"),
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Cautious speedruns (Chinchilla optimal)")
