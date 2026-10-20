# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an exported campaign checkpoint on frozen native BFCL-Parity."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.post_training.structure_closure.evaluate import build_evaluation, exported_model


@click.command(help=__doc__)
@click.option("--checkpoint-uri", required=True, help="Immutable HF-format checkpoint directory.")
@click.option("--checkpoint-version", required=True)
@click.option("--manifest-identity", required=True, help="Identity recorded in the model manifest.")
@click.option(
    "--parity-uri", required=True, help="Immutable BFCL-Parity directory with current task_rollout_spec Parquet rows."
)
@click.option("--parity-version", required=True)
@rl_build_options
def main(
    checkpoint_uri: str, checkpoint_version: str, manifest_identity: str, parity_uri: str, parity_version: str
) -> ArtifactStep:
    parity = ArtifactStep.adopt(user_owned_name("datasets/structure-closure-parity"), parity_version, parity_uri)
    return build_evaluation(
        name=user_owned_name("evals/structure-closure-parity-checkpoint"),
        data=parity,
        model=exported_model(
            checkpoint_uri=checkpoint_uri,
            checkpoint_version=checkpoint_version,
            manifest_identity=manifest_identity,
        ),
    )


if __name__ == "__main__":
    main()
