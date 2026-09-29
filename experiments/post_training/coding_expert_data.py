# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the grouped Python coding data for the Snowball expert spike."""

import click
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.post_training.coding_expert import EXPERIMENT_NAME, data_step


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[Artifact]:
    return data_step(resolve_version(f"documents/{EXPERIMENT_NAME}", None))


if __name__ == "__main__":
    main()
