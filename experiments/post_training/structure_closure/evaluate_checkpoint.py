# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an exported campaign checkpoint on the full v5 closure cohort."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.datasets.structure_closure import ClosureCohort, structure_closure_dataset
from experiments.post_training.structure_closure.evaluate import build_evaluation, exported_model


@click.command(help=__doc__)
@click.option("--checkpoint-uri", required=True, help="Immutable HF-format checkpoint directory.")
@click.option("--checkpoint-version", required=True)
@click.option("--manifest-identity", required=True, help="Identity recorded in the model manifest.")
@rl_build_options
def main(checkpoint_uri: str, checkpoint_version: str, manifest_identity: str) -> ArtifactStep:
    return build_evaluation(
        name=user_owned_name("evals/structure-closure-checkpoint"),
        data=structure_closure_dataset(ClosureCohort.VALUE_CONTROLS_V5),
        model=exported_model(
            checkpoint_uri=checkpoint_uri,
            checkpoint_version=checkpoint_version,
            manifest_identity=manifest_identity,
        ),
    )


if __name__ == "__main__":
    main()
