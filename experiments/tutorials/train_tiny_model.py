# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: train a tiny model, choosing the accelerator and dataset on the command line.

    python -m experiments.tutorials.train_tiny_model --device cpu --dataset tinystories --version dev
    python -m experiments.tutorials.train_tiny_model --device h100x8 --dataset wikitext --version dev --run
    python -m experiments.tutorials.train_tiny_model --device v5litepod-16 --dataset fineweb-edu --version dev --run

Every training decision is stated inline: the model, the data, the optimizer, the token
budget. :func:`~marin.experiment.train.train_lm` handles only the accelerator plumbing (the
mesh, the resumption checkpointer, the Fray dispatch); the same script runs on every device.

The checkpoint's version is *deferred*: it is set once for the whole run by the shared
``--version`` (``--run`` builds, the default prints the plan). The datasets pin explicit calendar
versions — they are shared caches that should not rebuild on a mutable version. See
:mod:`marin.experiment.cli`.
"""

import click
from fray.types import ANY_REGION, ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.data import pretokenized, tokenized
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.llama import llama_150m, llama_nano
from experiments.marin_tokenizer import marin_tokenizer

# Each device is one accelerator the same recipe runs on: its resources and a batch size
# that fits. Adding a device is one entry here, not a new file.
#
# The H100 entries pass ``regions=[ANY_REGION]``. A training job inherits the region of the
# worker that submitted it, which keeps it near its data; the GPU fleet lives in a federated
# CoreWeave cluster that advertises no region, so an inherited region would exclude every
# host that has an H100 and the job would be unschedulable.
DEVICES = {
    "cpu": (ResourceConfig.with_cpu(), 4),
    "h100x1": (ResourceConfig.with_gpu("H100", count=1, cpu=8, disk="128G", ram="64G", regions=[ANY_REGION]), 32),
    "h100x8": (
        ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G", regions=[ANY_REGION]),
        256,
    ),
    "v5litepod-16": (ResourceConfig.with_tpu("v5litepod-16", slice_count=1, cpu=32, ram="128g", disk="50g"), 128),
    "v6e-4": (ResourceConfig.with_tpu("v6e-4", slice_count=1, cpu=32, ram="128g", disk="50g"), 32),
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


def build(*, device: str, data: str) -> ArtifactStep[LevanterCheckpoint]:
    """A tiny Llama trained on ``data`` using ``device``, every decision stated inline.

    The 150M model is used for the prebuilt FineWeb-Edu cache; the nano model keeps the
    raw-text runs fast enough to finish on a laptop CPU. The checkpoint's version is deferred to
    the run-wide ``--version`` (``--version dev`` rebuilds every run while iterating; a calendar
    version pins a run to keep).
    """
    resources, batch_size = DEVICES[device]
    model = llama_150m if data == "fineweb-edu" else llama_nano
    return train_lm(
        name=f"checkpoints/tiny-{data}-{device}",
        # A run without an explicit id takes the last segment of its output path, which is the
        # version: every tutorial run would report into one W&B run named "dev".
        run_id=f"tiny-{data}-{device}",
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


@click.command(help=__doc__)
@click.option("--device", type=click.Choice(tuple(DEVICES)), default="cpu", show_default=True)
@click.option(
    "--dataset",
    "data",
    type=click.Choice(("tinystories", "wikitext", "fineweb-edu")),
    default="tinystories",
    show_default=True,
)
@build_options
def main(device: str, data: str) -> ArtifactStep[LevanterCheckpoint]:
    # Return the checkpoint handle; build_options builds it inside a BuildContext, so --version /
    # --override reach the deferred checkpoint version. The dataset tokenizes/downloads (cached),
    # then one training job runs on the chosen device.
    return build(device=device, data=data)


if __name__ == "__main__":
    main()
