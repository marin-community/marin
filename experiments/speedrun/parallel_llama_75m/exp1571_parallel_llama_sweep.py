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

"""
Speedruns using the Parallel Llama architecture for various model sizes.
Based on the parallel computation pattern where attention and MLP are computed simultaneously.
"""

import logging

from experiments.llama import llama_75m, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.parallel_llama_75m.exp1571_parallel_llama import ParallelLlamaConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig, TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(
    name="Harry Shin",
    affiliation="Independent",
    url="https://www.linkedin.com/in/harry-shin-34743216a/",
)

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def _to_parallel_llama_from_llama(llama_cfg, *, seq_len_override=None) -> ParallelLlamaConfig:
    """
    Build a ParallelLlamaConfig with identical sizes to a given LLaMA config.
    """
    parallel_llama = ParallelLlamaConfig(
        seq_len=seq_len_override if seq_len_override is not None else llama_cfg.seq_len,
        hidden_dim=llama_cfg.hidden_dim,
        intermediate_dim=llama_cfg.intermediate_dim,
        num_layers=llama_cfg.num_layers,
        num_heads=llama_cfg.num_heads,
        num_kv_heads=llama_cfg.num_kv_heads,
        head_dim=getattr(llama_cfg, "head_dim", None),
        use_bias=getattr(llama_cfg, "use_bias", False),
        use_layer_norm_weight=getattr(llama_cfg, "use_layer_norm_weight", True),
        rope=llama_cfg.rope,
        activation_function=llama_cfg.activation_function,
        initializer_range=llama_cfg.initializer_range,
        layer_norm_epsilon=llama_cfg.layer_norm_epsilon,
        tie_word_embeddings=llama_cfg.tie_word_embeddings,
        upcast_attn=llama_cfg.upcast_attn,
        attn_backend=llama_cfg.attn_backend,
        flash_attention_block_size=llama_cfg.flash_attention_block_size,
        scan_layers=getattr(llama_cfg, "scan_layers", True),
        gradient_checkpointing=getattr(llama_cfg, "gradient_checkpointing", True),
        use_parallel_blocks=True,  # Enable parallel computation
    )
    return parallel_llama


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    param_counts = {
        "75m": 75_000_000,
        "150m": 150_000_000,
        "300m": 300_000_000,
        "600m": 600_000_000,
    }

    llama_model_cfgs = {
        "75m": llama_75m,
        "150m": llama_150m,
        "300m": llama_300m,
        "600m": llama_600m,
    }

    batch_sizes = {
        "75m": 64,
        "150m": 64,
        "300m": 64,
        "600m": 64,
    }

    # Resource configurations - using GPUs for smaller models, TPUs for larger
    resource_cfgs = {
        "75m": GpuConfig(gpu_count=1, accelerator_type="H100"),
        "150m": GpuConfig(gpu_count=1, accelerator_type="H100"), 
        "300m": GpuConfig(gpu_count=1, accelerator_type="H100"),
        "600m": GpuConfig(gpu_count=1, accelerator_type="H100"),
    }

    # Learning rates scaled appropriately for each model size
    learning_rates = {
        "75m": 3e-3,
        "150m": 3e-3,
        "300m": 3e-3,
        "600m": 3e-4,  # Lower LR for larger model
    }

    descriptions = {
        "75m": "Parallel Llama ~75M with parallel attention/MLP computation.",
        "150m": "Parallel Llama ~150M with parallel attention/MLP computation.",
        "300m": "Parallel Llama ~300M with parallel attention/MLP computation.",
        "600m": "Parallel Llama ~600M with parallel attention/MLP computation.",
    }

    run_names = {
        "75m": "parallel_llama_75m_1024",
        "150m": "parallel_llama_150m_1024",
        "300m": "parallel_llama_300m_1024",
        "600m": "parallel_llama_600m_1024",
    }

    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    llama_cfg = llama_model_cfgs[size]
    batch_size = batch_sizes[size]
    resource_config = resource_cfgs[size]
    learning_rate = learning_rates[size]
    description = descriptions[size]
    run_name = run_names[size]

    # Convert to ParallelLlamaConfig and keep seq_len from original config
    model_config = _to_parallel_llama_from_llama(llama_cfg)
    seq_len = model_config.seq_len

    num_train_steps = get_num_train_steps(param_counts[size], batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=0.1,
        steps_per_eval=2000,
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
        build_config("75m"),
        build_config("150m"),
        build_config("300m"),
        build_config("600m"),
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Parallel Llama speedruns (Chinchilla-optimal tokens, w/ parallel attention/MLP)")
