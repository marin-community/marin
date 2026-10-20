# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare matched tool-call prompt conditions without allocating a learner."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.datasets.structure_closure import tool_prompt_probe_dataset
from experiments.post_training.structure_closure.evaluate import build_evaluation, exported_model, initializer_model
from experiments.post_training.structure_closure.evaluate_native_tools import CHECKPOINTS


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(["base", "candidate13"]), required=True)
@rl_build_options
def main(model: str) -> ArtifactStep:
    selected = initializer_model()
    if model == "candidate13":
        version, identity = CHECKPOINTS[model]
        selected = exported_model(
            checkpoint_uri=f"s3://marin-us-east-02a/marin/users/benfeuer/models/structure-closure/{version}/exports/global_step_2/policy",
            checkpoint_version=version,
            manifest_identity=identity,
        )
    return build_evaluation(
        name=user_owned_name(f"evals/structure-closure-tool-prompt-{model}"),
        data=tool_prompt_probe_dataset(),
        model=selected,
    )


if __name__ == "__main__":
    main()
