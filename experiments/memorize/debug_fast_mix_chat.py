"""Exact programmatic replica of submodules/levanter/config/gpt2_small_fast_mix_chat.yaml,
wrapped in an ExecutorStep that launches on a v4-64 via run_levanter_train_lm.

Key points (match YAML exactly):
- Data mixture: owt + wikitext + tulu(chat), weights 0.6/0.3/0.1
- Tokenizer: stanford-crfm/marin-tokenizer
- Cache dir: gs://marin-us-central2/scratch/dlwh/marin_small_fast_mix
- Model: GPT-2 small (12L, 12H, 768d), seq_len=1024, gradient_checkpointing=True,
         scale_attn_by_inverse_layer_idx=True
- Trainer: WandB project="levanter", tags=["openwebtext+wiki","gpt2","itest"],
           mp policy p=f32,c=bfloat16, model_axis_size=1,
           train_batch_size=256, num_train_steps=20000
- Optimizer: Adam lr=1e-3, weight_decay=0.1, warmup=0.01 (cosine schedule by default)

Usage (Executor):
  uv run python experiments/memorize/debug_fast_mix_chat.py
"""

from __future__ import annotations

import jmp

from levanter.main.train_lm import TrainLmConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.optim import AdamConfig
from levanter.data.text import (
    LMMixtureDatasetConfig,
    UrlDatasetSourceConfig,
    HfDatasetSourceConfig,
    ChatLmDatasetFormat,
)
from marin.resources import TpuPodConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def build_config() -> TrainLmConfig:
    # Data mixture configs (exactly as in the YAML)
    owt = UrlDatasetSourceConfig(
        train_urls=[
            "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz",
        ],
        validation_urls=[
            "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz",
        ],
    )

    wikitext = HfDatasetSourceConfig(id="dlwh/wikitext_103_detokenized")

    tulu = HfDatasetSourceConfig(id="allenai/tulu-3-sft-mixture", format=ChatLmDatasetFormat())

    data = LMMixtureDatasetConfig(
        configs={
            "owt": owt,
            "wikitext": wikitext,
            "tulu": tulu,
        },
        train_weights={
            "owt": 0.6,
            "wikitext": 0.3,
            "tulu": 0.1,
        },
        tokenizer="stanford-crfm/marin-tokenizer",
        cache_dir="gs://marin-us-central2/scratch/dlwh/marin_small_fast_mix",
        # YAML leaves shuffle/permutation unspecified; keep defaults (no shuffle).
    )

    # GPT-2 small model, with the same flags as YAML
    model = Gpt2Config(
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        seq_len=1024,
        gradient_checkpointing=True,
        scale_attn_by_inverse_layer_idx=True,
    )

    # Optimizer: Adam, cosine schedule (default), with specified lr/wd/warmup
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.1,
        warmup=0.01,
    )

    trainer = TrainerConfig(
        tracker=WandbConfig(project="levanter", tags=["openwebtext+wiki", "gpt2", "itest"]),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        model_axis_size=1,
        train_batch_size=256,
        num_train_steps=20000,
    )

    return TrainLmConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    cfg = build_config()
    # Wrap in TrainLmOnPodConfig pointing at v4-64, and build an ExecutorStep to run on TPU.
    pod = TpuPodConfig(tpu_type="v4-64", slice_count=1)
    step = ExecutorStep(
        name="checkpoints/debug_fast_mix_chat_v4_64",
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            train_config=cfg,
            resources=pod,
            output_path=this_output_path(),
        ),
        description="GPT2-small OWT+Wikitext+Tulu(chat) fast mix on v4-64 (exact YAML parity)",
        pip_dependency_groups=["tokenize_train"],
    )

    executor_main(steps=[step], description="debug_fast_mix_chat")
