# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare matched tool-call prompt conditions without allocating a learner."""

import click

from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.datasets.structure_closure import tool_prompt_probe_dataset
from experiments.post_training.structure_closure.evaluate import build_evaluation, exported_model, initializer_model


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(["base", "candidate13"]), required=True)
@rl_build_options
def main(model: str) -> ArtifactStep:
    selected = initializer_model()
    if model == "candidate13":
        selected = exported_model(
            checkpoint_uri="s3://marin-us-east-02a/marin/users/benfeuer/models/structure-closure/2026.10.12.13/exports/global_step_2/policy",
            checkpoint_version="2026.10.12.13",
            manifest_identity="sha256:42d23366ce1fb08e818c8d121c0ff455a39e6949efded6e95a35ab417de72fa4",
        )
    return build_evaluation(
        name=user_owned_name(f"evals/structure-closure-tool-prompt-{model}"),
        data=tool_prompt_probe_dataset(),
        model=selected,
    )


if __name__ == "__main__":
    main()
