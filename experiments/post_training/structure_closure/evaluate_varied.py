# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate varied string-escape and fence probes on the policy initializer."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.post_training.structure_closure.evaluate import build_evaluation


@click.command(help=__doc__)
@rl_build_options
def main() -> ArtifactStep:
    data = ArtifactStep.adopt(
        user_owned_name("datasets/structure-closure-varied"),
        "2026.10.12.4",
        "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/closure-varied-tasks-v4",
        config={"validation_sha256": "2a3e293887f809b5e21473ae0e400c5d9fd8ed650a3d431dcddb940366426cc8"},
    )
    return build_evaluation(name=user_owned_name("evals/structure-closure-varied"), data=data)


if __name__ == "__main__":
    main()
