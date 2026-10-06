# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover saved SFT shards and metadata, then verify serving reload without rerunning training."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.teacher_chat_study import chat_study_workflow
from experiments.post_training.russell_rsi.teacher_sft_export_recovery import export_recovery_workflow


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["recover", "reload"]), required=True)
@build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = PinnedFile(config_uri, config_sha256).read_json()
    producer_config = PinnedFile(**config["producer_config"]).read_json()
    producer = chat_study_workflow(producer_config)["train"]
    return [export_recovery_workflow(config, producer)[stage]]


if __name__ == "__main__":
    main()
