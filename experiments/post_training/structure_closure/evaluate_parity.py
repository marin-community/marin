# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure initializer noise on the frozen native BFCL-Parity cohort."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.post_training.structure_closure.evaluate import build_evaluation, initializer_model


@click.command(help=__doc__)
@click.option("--parity-uri", required=True, help="Immutable native BFCL-Parity dataset directory.")
@click.option("--parity-version", required=True)
@rl_build_options
def main(parity_uri: str, parity_version: str) -> ArtifactStep:
    parity = ArtifactStep.adopt(user_owned_name("datasets/structure-closure-parity"), parity_version, parity_uri)
    return build_evaluation(
        name=user_owned_name("evals/structure-closure-parity-initializer"),
        data=parity,
        model=initializer_model(),
    )


if __name__ == "__main__":
    main()
