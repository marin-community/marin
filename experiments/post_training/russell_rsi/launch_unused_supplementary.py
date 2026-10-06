# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the unused v3 comparison only after an independently recorded promotion."""

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.launch_supplementary import supplementary_workflow
from experiments.post_training.russell_rsi.unused_supplementary import promoted_study_checkpoint


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@foreground_build_options
def main(config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = PinnedFile(config_uri, config_sha256).read_json()
    if resolve_version("russell-rsi-unused-supplementary", None) != config["evaluation_version"]:
        raise click.UsageError("Unused comparison config and artifact version differ")
    return [supplementary_workflow(config, promoted_study_checkpoint(config))]


if __name__ == "__main__":
    main()
