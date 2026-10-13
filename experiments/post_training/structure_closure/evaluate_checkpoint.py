# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an exported campaign checkpoint on the full v5 closure cohort."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import LevanterCheckpoint

from experiments.datasets.structure_closure import ClosureCohort, structure_closure_dataset
from experiments.post_training.structure_closure.evaluate import build_evaluation
from experiments.post_training.structure_closure.launch import MODEL_REPOSITORY, MODEL_REVISION


@click.command(help=__doc__)
@click.option("--checkpoint-uri", required=True, help="Immutable HF-format checkpoint directory.")
@click.option("--checkpoint-version", required=True)
@click.option("--manifest-identity", required=True, help="Identity recorded in the model manifest.")
@rl_build_options
def main(checkpoint_uri: str, checkpoint_version: str, manifest_identity: str) -> ArtifactStep:
    model = ArtifactStep.adopt(
        user_owned_name("models/structure-closure-export"),
        checkpoint_version,
        checkpoint_uri,
        kind=LevanterCheckpoint,
        config={"manifest_identity": manifest_identity},
    )
    return build_evaluation(
        name=user_owned_name("evals/structure-closure-checkpoint"),
        data=structure_closure_dataset(ClosureCohort.VALUE_CONTROLS_V5),
        model=ArtifactHfModel(
            step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path=""
        ),
    )


if __name__ == "__main__":
    main()
