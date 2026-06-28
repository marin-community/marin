# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: Training a tiny model on Wikitext-2 using a GPU (lazy-artifact style).

This script demonstrates how to:
1. Tokenize Wikitext-2 inline and register it as a lazy dataset handle.
2. Define all training decisions inline: model, optimizer, batch size, steps.
3. Run the full pipeline with :class:`~marin.execution.step_runner.StepRunner`.

For CPU training, see train_tiny_model_cpu.py.
"""

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.artifact import Checkpoint
from marin.execution.lazy import Lazy, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm

from experiments.llama import llama_nano
from experiments.marin_tokenizer import marin_tokenizer

# 1. Choose a dataset and tokenize it inline.
# sample_count=1000 caps documents per shard for a quick tutorial run.
tok = tokenized(
    "dlwh/wikitext_2_detokenized",
    tokenizer=marin_tokenizer,
    source="dlwh/wikitext_2_detokenized",
    sample_count=1000,
)


def build(*, version: str = "v1") -> Lazy[Checkpoint]:
    """A tiny Llama model trained on Wikitext-2 (GPU), every decision stated inline."""
    return train_lm(
        name="checkpoints/llama-nano-wikitext",
        version=version,
        model=llama_nano,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        # 2. Single-component mixture: all training tokens from Wikitext-2.
        datasets={tok: 1.0},
        batch_size=256,
        seq_len=llama_nano.max_seq_len,  # 512
        num_train_steps=100,
        z_loss_weight=None,
        evals=None,  # no point running evals on such a tiny model
        resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G"),
        tags=["llama", "nano", "wikitext", "tutorial"],
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: Wikitext-2 tokenizes
    # (cached), then one GPU training job runs.
    StepRunner().run([lower(build())])
