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
Speedruns using the MuonH optimizer for various Qwen model sizes (Chinchilla optimal steps)
configs mirroring those in marin/experiments/speedrun/muonh_llama_scaling/muonh_sweep.py
"""

import logging

from levanter.models.qwen import Qwen3Config
from levanter.models.llama import LlamaConfig
from levanter.optim import MuonHConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(
    name="Kaiyue Wen",
    affiliation="Stanford University",
    url="https://whenwen.github.io",
)

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def _to_qwen3_from_llama(llama_cfg: LlamaConfig, *, seq_len_override=None) -> Qwen3Config:
    """
    Build a Qwen3Config with identical sizes to a given LLaMA config.
    """
    qwen = Qwen3Config(
        seq_len=seq_len_override if seq_len_override is not None else llama_cfg.seq_len,
        hidden_dim=llama_cfg.hidden_dim,
        intermediate_dim=llama_cfg.intermediate_dim,
        num_layers=llama_cfg.num_layers,
        num_heads=llama_cfg.num_heads,
        num_kv_heads=llama_cfg.num_kv_heads,
        head_dim=getattr(llama_cfg, "head_dim", None),
        use_bias=getattr(llama_cfg, "use_bias", False),
        rope=llama_cfg.rope,
        activation_function=llama_cfg.activation_function,
        initializer_range=llama_cfg.initializer_range,
        layer_norm_epsilon=llama_cfg.layer_norm_epsilon,
        tie_word_embeddings=llama_cfg.tie_word_embeddings,
        upcast_attn=llama_cfg.upcast_attn,
        attn_backend=llama_cfg.attn_backend,
        flash_attention_block_size=llama_cfg.flash_attention_block_size,
        scan_layers=getattr(llama_cfg, "scan_layers", False),
        gradient_checkpointing=getattr(llama_cfg, "gradient_checkpointing", False),
        hybrid_norm=True,
    )
    return qwen


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    param_counts = {
        "130m": 130_000_000,
        "300m": 300_000_000,
        "520m": 520_000_000,
        "1_2b": 1_200_000_000,
    }

    llama_model_cfgs = {
        "130m": llama_150m,
        "300m": llama_300m,
        "520m": llama_600m,
        "1_2b": llama_1_4b,
    }

    batch_sizes = {
        "130m": 128,
        "300m": 128,
        "520m": 256,
        "1_2b": 256,
    }

    resource_cfgs = {
        "130m": TpuPodConfig(tpu_type="v5litepod-64"),
        "300m": TpuPodConfig(tpu_type="v5litepod-64"),
        "520m": TpuPodConfig(tpu_type="v5litepod-64"),
        "1_2b": TpuPodConfig(tpu_type="v5litepod-64"),
    }

    # Optimizer configs for each size
    muon_configs = {
        "130m": MuonHConfig(
            learning_rate=0.02,
            adam_lr=0.008,
            min_lr_ratio=0,
            momentum=0.95,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            warmup=1000,
        ),
        "300m": MuonHConfig(
            learning_rate=0.01,
            adam_lr=0.002,
            min_lr_ratio=0,
            momentum=0.98,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            warmup=1000,
        ),
        "520m": MuonHConfig(
            learning_rate=0.01,
            adam_lr=0.002,
            min_lr_ratio=0,
            momentum=0.98,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            warmup=1000,
        ),
        "1_2b": MuonHConfig(
            learning_rate=0.01,
            adam_lr=0.0015,
            min_lr_ratio=0,
            momentum=0.98,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=2,
            warmup=1000,
        ),
    }

    descriptions = {
        "130m": "Qwen3 ~130M (LLaMA-geometry-matched) with MuonH.",
        "300m": "Qwen3 ~300M (LLaMA-geometry-matched) with MuonH.",
        "520m": "Qwen3 ~520M (LLaMA-geometry-matched) with MuonH.",
        "1_2b": "Qwen3 ~1.2B (LLaMA-geometry-matched) with MuonH.",
    }

    run_names = {  
        "130m": "qwen3_130m_muonh_4096_lr_0.02_adam_lr_0.008",
        "300m": "qwen3_300m_muonh_4096_lr_0.01",
        "520m": "qwen3_520m_muonh_4096_lr_0.01",
        "1_2b": "qwen3_1_2b_muonh_4096_low_lr",
    }

    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    llama_cfg = llama_model_cfgs[size]
    batch_size = batch_sizes[size]
    resource_config = resource_cfgs[size]
    muon = muon_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    # Convert to Qwen3Config and set seq_len=4096 for the sweep
    model_config = _to_qwen3_from_llama(llama_cfg, seq_len_override=4096)
    seq_len = model_config.seq_len

    num_train_steps = get_num_train_steps(param_counts[size], batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=muon.learning_rate,
        optimizer_config=muon,
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
        build_config("300m"),
        build_config("520m"),
        build_config("1_2b"),
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Qwen3 Muon speedruns (Chinchilla-optimal tokens, w/ QK-Norm)")
