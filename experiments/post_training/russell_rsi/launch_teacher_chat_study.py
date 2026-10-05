# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the separate four-family chat-SFT feasibility study."""

import json

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options

from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.teacher_chat_study import (
    PROTOCOL,
    chat_study_post_workflow,
    chat_study_workflow,
)


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["collect", "sft", "reload", "calibrate", "rl", "evaluate"]), required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if (
        resolve_version(PROTOCOL, None) != config["version"]
        or config["protocol"] != PROTOCOL
        or config["runtime_commit"] != MARIN_SKYRL.commit
    ):
        raise click.UsageError("Chat study protocol, version or runtime pin differs")
    if stage in ("collect", "sft", "reload"):
        return [chat_study_workflow(config)["train" if stage == "sft" else stage]]
    return [chat_study_post_workflow(config, "train" if stage == "rl" else stage)["terminal"]]


if __name__ == "__main__":
    main()
