# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Speedruns using the AdaMuon optimizer for various Qwen model sizes (Chinchilla optimal steps)
configs mirroring those in marin/experiments/speedrun/muon_qwen3_scaling/muon_sweep.py
"""

import logging

from levanter.models.qwen import Qwen3Config
from levanter.models.llama import LlamaConfig

from experiments.speedrun.adamuon_qwen3_scaling.adamuon_config import AdaMuonConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


AUTHOR = Author(
    name="Melanie Sclar",
    affiliation="University of Washington",
    url="https://msclar.github.io",
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
        max_seq_len=seq_len_override if seq_len_override is not None else llama_cfg.max_seq_len,
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
    }

    resource_cfgs = {
        "130m": ResourceConfig.with_gpu(),  # ResourceConfig.with_tpu("v5p-32"),
    }

    adaMuon_configs = {
        "130m": AdaMuonConfig(
            learning_rate=0.024,
            adam_lr=0.0048,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.95,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
    }

    descriptions = {
        "130m": "Qwen3 ~130M (LLaMA-geometry-matched) with AdaMuon.",
    }

    run_names = {
        "130m": "test_qwen3_130m_adaMuon_4096",
    }

    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    llama_cfg = llama_model_cfgs[size]
    batch_size = batch_sizes[size]
    resource_config = resource_cfgs[size]
    adaMuon = adaMuon_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    model_config = _to_qwen3_from_llama(llama_cfg, seq_len_override=4096)
    seq_len = model_config.max_seq_len

    num_train_steps = get_num_train_steps(param_counts[size], batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=adaMuon.learning_rate,
        optimizer_config=adaMuon,
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
        build_config("130m")  # including only 130m model, can expand to other sizes
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Qwen3 AdaMuon speedruns (Chinchilla-optimal tokens)")
