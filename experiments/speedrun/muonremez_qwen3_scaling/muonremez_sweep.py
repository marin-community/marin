# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone MuonRemez Qwen3 speedrun submission selected from completed LR sweeps."""

from __future__ import annotations

import inspect
import logging
import os

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from experiments.speedrun.muonremez_qwen3_scaling.muonremez_optimizer import MuonRemezConfig
from marin.execution.executor import executor_main

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.muonremez_qwen3_scaling.selected_runs import AUTHOR_INFO, SELECTED_RUNS
from experiments.speedrun.muonremez_qwen3_scaling.submission_support import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(**AUTHOR_INFO)
SIZES = ("130m", "300m", "520m", "1_2b")
LLAMA_MODEL_CFGS = {
    "130m": llama_150m,
    "300m": llama_300m,
    "520m": llama_600m,
    "1_2b": llama_1_4b,
}

logger = logging.getLogger(__name__)


def _filtered_optimizer_config(optimizer_config: dict[str, object], optimizer_cls: type) -> dict[str, object]:
    accepted_keys = set(inspect.signature(optimizer_cls).parameters)
    return {key: value for key, value in optimizer_config.items() if key in accepted_keys}


def _to_qwen3_from_llama(llama_cfg: LlamaConfig, *, seq_len_override: int = 4096, hybrid_norm: bool = False) -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=seq_len_override,
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
        hybrid_norm=hybrid_norm,
    )


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    selected = SELECTED_RUNS[size]
    optimizer = MuonRemezConfig(**_filtered_optimizer_config(selected["optimizer_config"], MuonRemezConfig))
    model_config = _to_qwen3_from_llama(LLAMA_MODEL_CFGS[size], hybrid_norm=False)
    resources = ResourceConfig.with_tpu(selected["resources"]["device"]["variant"])
    train = SimpleTrainConfig(
        resources,
        train_seq_len=model_config.max_seq_len,
        train_batch_size=selected["train_batch_size"],
        num_train_steps=selected["num_train_steps"],
        learning_rate=optimizer.learning_rate,
        optimizer_config=optimizer,
    )
    config = SpeedrunConfig(
        author=AUTHOR,
        description=selected["description"],
        model_config=model_config,
        train_config=train,
        tokenized_dataset=selected["tokenized_dataset"],
    )
    return selected["source_run_name"], config


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    steps = []
    for size in SIZES:
        name, cfg = build_config(size)
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="MuonRemez Qwen3 speedruns selected from completed LR sweeps.")


if __name__ == "__main__":
    main()
