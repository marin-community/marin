# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the pinned parent without task generation or training dependencies."""

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint

from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION, parent_public_step


@click.command(help=__doc__)
@click.option("--model-uri", required=True, help="Regional HF export of the pinned September 21 parent.")
@build_options
def main(model_uri: str) -> ArtifactStep:
    model = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        model_uri,
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    return parent_public_step(model, resolve_version("russell-rsi", None))


if __name__ == "__main__":
    main()
