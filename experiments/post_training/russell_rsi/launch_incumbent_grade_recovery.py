# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal one completed CPU grade recovery into a separate calibration artifact."""

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.incumbent_grade_recovery import PROTOCOL, grade_recovery_step
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import require_reviewed_source


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@build_options
def main(config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = PinnedFile(config_uri, config_sha256).read_json()
    if resolve_version(PROTOCOL, None) != config["version"] or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise click.UsageError("Recovery seal config version or installed runtime pin differs")
    return [grade_recovery_step(config["evidence"], config["version"])]


if __name__ == "__main__":
    main()
