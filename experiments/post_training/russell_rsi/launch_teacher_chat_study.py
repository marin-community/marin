# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the separate four-family chat-SFT feasibility study."""

import json

import click
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options

from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.teacher_chat_study import (
    PROTOCOL,
    chat_study_post_workflow,
    chat_study_workflow,
)
from experiments.post_training.russell_rsi.teacher_study_cli import TEACHER_STAGES, teacher_study_stage


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(TEACHER_STAGES), required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    return teacher_study_stage(
        config,
        stage,
        protocol=PROTOCOL,
        workflow=chat_study_workflow,
        post_workflow=chat_study_post_workflow,
        error_message="Chat study protocol, version or runtime pin differs",
    )


if __name__ == "__main__":
    main()
