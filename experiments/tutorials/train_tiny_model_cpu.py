# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: Training a tiny model on TinyStories using CPU (lazy-artifact style).

This script demonstrates how to:
1. Tokenize TinyStories inline and register it as a lazy dataset handle.
2. Define all training decisions inline: model, optimizer, batch size, steps.
3. Run the full pipeline with :class:`~marin.execution.step_runner.StepRunner`.

For GPU training, see train_tiny_model_gpu.py.
"""

from fray import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.artifact import Checkpoint
from marin.execution.lazy import Lazy, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm

from experiments.llama import llama_nano
from experiments.marin_tokenizer import marin_tokenizer

# 1. Choose a dataset and tokenize it inline.
# sample_count=1000 caps documents per shard; it bears identity (a sampled cache
# differs from the full one).
tok = tokenized(
    "roneneldan/TinyStories",
    tokenizer=marin_tokenizer,
    source="roneneldan/TinyStories",
    sample_count=1000,
)


def build(*, version: str = "v1") -> Lazy[Checkpoint]:
    """A tiny Llama model trained on TinyStories (CPU), every decision stated inline.

    max_eval_batches=4 from the original SimpleTrainConfig tutorial has no equivalent
    in train_lm; at 100 steps the run is fast enough without it.
    """
    return train_lm(
        name="checkpoints/marin-nano-tinystories",
        version=version,
        # 2-layer, 32-dim Llama — the smallest sensible model for a quick tutorial.
        model=llama_nano,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        # 3. Single-component mixture: all training tokens from TinyStories.
        datasets={tok: 1.0},
        batch_size=4,
        seq_len=llama_nano.max_seq_len,  # 512
        num_train_steps=100,
        z_loss_weight=None,
        evals=None,  # no point running evals on such a tiny model
        resources=ResourceConfig.with_cpu(),
        tags=["llama", "nano", "tinystories", "tutorial"],
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: TinyStories tokenizes
    # (cached), then one CPU training job runs.
    StepRunner().run([lower(build())])
