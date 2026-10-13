# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare frozen checkpoints on native tool prompts and unchanged non-tool cases."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.datasets.structure_closure import ClosureCohort, structure_closure_dataset
from experiments.post_training.structure_closure.evaluate import build_evaluation, exported_model, initializer_model

CHECKPOINTS = {
    "candidate13": ("2026.10.12.13", "sha256:42d23366ce1fb08e818c8d121c0ff455a39e6949efded6e95a35ab417de72fa4"),
    "candidate16": ("2026.10.13.16", "sha256:04ac399a6c01f32adeab4af4632cea7f027d22f468263e3fca8eb15cefa353ea"),
    "candidate18": ("2026.10.13.18", "sha256:5a287f52e4317b1d07caca80d079103e048141796d089211a6684df50e230076"),
}


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(["base", *CHECKPOINTS]), required=True)
@rl_build_options
def main(model: str) -> ArtifactStep:
    selected = initializer_model()
    if model != "base":
        version, identity = CHECKPOINTS[model]
        selected = exported_model(
            checkpoint_uri=f"s3://marin-us-east-02a/marin/users/benfeuer/models/structure-closure/{version}/exports/global_step_2/policy",
            checkpoint_version=version,
            manifest_identity=identity,
        )
    return build_evaluation(
        name=user_owned_name(f"evals/structure-closure-native-tools-{model}"),
        data=structure_closure_dataset(ClosureCohort.NATIVE_TOOLS_V6),
        model=selected,
    )


if __name__ == "__main__":
    main()
