# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the separately frozen four-pass teacher study after a nonpromoted continuation."""

import json

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options

from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.teacher_four_pass import (
    PROTOCOL,
    four_pass_post_workflow,
    four_pass_teacher_workflow,
)


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["collect", "sft", "reload", "calibrate", "rl", "evaluate"]), required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if (
        config["protocol"] != PROTOCOL
        or resolve_version(PROTOCOL, None) != config["version"]
        or config["runtime_commit"] != MARIN_SKYRL.commit
    ):
        raise click.UsageError("Four-pass study protocol, version or runtime pin differs")
    if stage in ("collect", "sft", "reload"):
        outputs = four_pass_teacher_workflow(config)
        return [outputs["train" if stage == "sft" else stage]]
    outputs = four_pass_post_workflow(config, "train" if stage == "rl" else stage)
    return [outputs["terminal"]]


if __name__ == "__main__":
    main()
