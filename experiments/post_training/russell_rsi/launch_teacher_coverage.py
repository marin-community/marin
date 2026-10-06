# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a conditional sixteen-family teacher dose and one separate SFT evaluation."""

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.teacher_coverage_study import (
    PROTOCOL,
    coverage_collection,
    coverage_evaluation,
    coverage_sft_workflow,
)


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@click.option("--stage", type=click.Choice(["collect", "sft", "reload", "evaluate"]), required=True)
@foreground_build_options
def main(
    config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str, stage: str
) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = PinnedFile(config_uri, config_sha256).read_json()
    if resolve_version(PROTOCOL, None) != config["version"] or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise click.UsageError("Coverage config version or installed runtime pin differs")
    if stage == "collect":
        return [coverage_collection(config)]
    if stage == "evaluate":
        return [coverage_evaluation(config)["terminal"]]
    stages = coverage_sft_workflow(config)
    return [stages["train" if stage == "sft" else "reload"]]


if __name__ == "__main__":
    main()
