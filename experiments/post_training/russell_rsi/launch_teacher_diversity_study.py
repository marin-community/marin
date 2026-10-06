# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the separately reviewed eight-family teacher SFT study."""

import json

import click
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.teacher_diversity_study import (
    PROTOCOL,
    diversity_post_workflow,
    diversity_workflow,
)
from experiments.post_training.russell_rsi.teacher_study_cli import TEACHER_STAGES, teacher_study_stage


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@click.option("--stage", type=click.Choice(TEACHER_STAGES), required=True)
@foreground_build_options
def main(
    config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str, stage: str
) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    return teacher_study_stage(
        config,
        stage,
        protocol=PROTOCOL,
        workflow=diversity_workflow,
        post_workflow=diversity_post_workflow,
        error_message="Diversity protocol, version or runtime pin differs",
    )


if __name__ == "__main__":
    main()
