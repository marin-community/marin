# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: Training a ~125M parameter model on FineWeb-Edu using a GPU (lazy-artifact style).

This script demonstrates how to:
1. Reference a pretokenized FineWeb-Edu cache hosted on HuggingFace as a lazy handle.
2. Define all training decisions inline: model, optimizer, batch size, steps.
3. Run the full pipeline with :class:`~marin.execution.step_runner.StepRunner`.

The pretokenized handle downloads the cache on first run and skips re-tokenization on
subsequent runs.
"""

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import Checkpoint, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, pretokenized
from marin.experiment.train import train_lm

from experiments.llama import llama_150m
from experiments.marin_tokenizer import marin_tokenizer

# 1. Reference the FineWeb-Edu 10M-token pretokenized cache from HuggingFace.
# pretokenized() downloads the Levanter cache rather than re-tokenizing from raw text.
tok = pretokenized(
    "fineweb-edu-10M",
    repo_id="marin-community/fineweb-edu-pretokenized-10M",
    tokenizer=marin_tokenizer,
)


def build(*, version: str = "v1") -> Checkpoint:
    """A ~125M Llama model trained on FineWeb-Edu (GPU), every decision stated inline."""
    return train_lm(
        name="checkpoints/llama-150m-fineweb-edu-gpu",
        version=version,
        model=llama_150m,
        optimizer=AdamConfig(learning_rate=3e-4, weight_decay=0.1),
        # 2. Single-component mixture: all training tokens from FineWeb-Edu.
        data=lambda ctx: mixture(ctx, {tok: 1.0}),
        deps=(tok,),
        batch_size=32,
        seq_len=llama_150m.max_seq_len,  # 1024
        num_train_steps=1000,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu("H100", count=1),
        tags=["llama", "150m", "fineweb-edu", "gpu", "tutorial"],
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: the pretokenized cache
    # downloads (cached), then one GPU training job runs.
    StepRunner().run([lower(build())])
