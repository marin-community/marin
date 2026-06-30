# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: train a tiny model, choosing the accelerator and dataset on the command line.

    python experiments/tutorials/train_tiny_model.py --device cpu --dataset tinystories
    python experiments/tutorials/train_tiny_model.py --device h100x8 --dataset wikitext --cluster marin
    python experiments/tutorials/train_tiny_model.py --device v5litepod-16 --dataset fineweb-edu --cluster marin

Every training decision is stated inline: the model, the data, the optimizer, the token
budget. :func:`~marin.experiment.train.train_lm` handles only the accelerator plumbing (the
mesh, the resumption checkpointer, the Fray dispatch); the same script runs on every device.
Without ``--cluster`` it runs in-process; with ``--cluster`` it ships to a coordinator job.
"""

from dataclasses import dataclass

import draccus
from fray import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import pretokenized, tokenized
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.launch import LaunchConfig, launch, run_steps
from experiments.llama import llama_150m, llama_nano
from experiments.marin_tokenizer import marin_tokenizer

# Each device is one accelerator the same recipe runs on: its resources and a batch size
# that fits. Adding a device is one entry here, not a new file.
DEVICES = {
    "cpu": (ResourceConfig.with_cpu(), 4),
    "h100x8": (ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G"), 256),
    "v5litepod-16": (ResourceConfig.with_tpu("v5litepod-16", slice_count=1, cpu=32, ram="128g", disk="50g"), 128),
}

# Raw HuggingFace text datasets tokenized inline (a small sample for a quick run).
RAW_SOURCES = {
    "tinystories": "roneneldan/TinyStories",
    "wikitext": "dlwh/wikitext_2_detokenized",
}


def dataset(name: str) -> ArtifactStep[TokenizedCache]:
    """The named tutorial dataset as a tokenized handle.

    ``fineweb-edu`` is a prebuilt Levanter cache (downloaded, not re-tokenized); the others
    tokenize a 1000-document sample of a raw HuggingFace text dataset inline.
    """
    if name == "fineweb-edu":
        return pretokenized(
            "fineweb-edu-10M",
            repo_id="marin-community/fineweb-edu-pretokenized-10M",
            tokenizer=marin_tokenizer,
            version="2026.06.28",
        )
    return tokenized(name, tokenizer=marin_tokenizer, source=RAW_SOURCES[name], sample_count=1000, version="2026.06.28")


def build(*, device: str, data: str, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """A tiny Llama trained on ``data`` using ``device``, every decision stated inline.

    The 150M model is used for the prebuilt FineWeb-Edu cache; the nano model keeps the
    raw-text runs fast enough to finish on a laptop CPU. The default ``dev`` version rebuilds
    on every run — what you want while iterating; pin a calendar version for a run to keep.
    """
    resources, batch_size = DEVICES[device]
    model = llama_150m if data == "fineweb-edu" else llama_nano
    return train_lm(
        name=f"checkpoints/tiny-{data}-{device}",
        version=version,
        model=model,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        datasets={dataset(data): 1.0},
        batch_size=batch_size,
        seq_len=model.max_seq_len,
        num_train_steps=100,
        z_loss_weight=None,
        evals=None,  # no point evaluating such a tiny model
        resources=resources,
        tags=["llama", "tutorial", data, device],
    )


@dataclass
class TinyModelLaunch(LaunchConfig):
    """Launcher flags plus this tutorial's device/dataset choice."""

    device: str = "cpu"
    """One of ``DEVICES`` (cpu, h100x8, v5litepod-16)."""

    dataset: str = "tinystories"
    """One of tinystories, wikitext, fineweb-edu."""


def train_tiny(config: TinyModelLaunch) -> None:
    """Build the chosen device/dataset run and execute it, in-process or on ``--cluster``."""
    run_steps(config, build(device=config.device, data=config.dataset))


if __name__ == "__main__":
    launch(draccus.parse(TinyModelLaunch), train_tiny)
