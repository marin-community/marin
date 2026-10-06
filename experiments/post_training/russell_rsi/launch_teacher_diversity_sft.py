# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the fixed diversity collection with durable optimizer metric events."""

import json
from dataclasses import replace
from typing import cast

import click
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.launch import CLUSTER, evaluation_model
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.teacher_diversity_study import NAMESPACE, UPDATES, diversity_workflow

COLLECTION_VERSION = "2026.10.06.15"
SFT_VERSION = "2026.10.06.17"


def durable_sft_stages(stages: dict[str, ArtifactStep]) -> dict[str, ArtifactStep]:
    """Keep the collection handle and bind telemetry to the new training output."""
    previous = stages["train"]

    def training_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = cast(TrainLmOnPodConfig, previous.build_config(ctx))
        train = cast(TrainLmConfig, pod.train_config)
        trainer = replace(
            train.trainer,
            id=f"russell-rsi-{NAMESPACE}-sft-{SFT_VERSION}",
            tracker=(JsonLoggerConfig(metric_destination=prefix_join(ctx.output_path, "optimizer-telemetry")),),
        )
        return replace(pod, train_config=replace(train, trainer=trainer))

    trained = replace(previous, version=SFT_VERSION, build_config=training_config)
    reload_model = evaluation_model(f"russell-rsi-{NAMESPACE}-sft-reload", "<completed-sft-export>", None)
    reload = eval_step(
        reload_model,
        "mmlu-smoke",
        version=SFT_VERSION,
        deps=(trained,),
        resolve_model=lambda ctx: replace(
            reload_model,
            location=prefix_join(ctx.artifact_path(trained), f"hf/step-{UPDATES - 1}"),
            identity=artifact_identity(trained),
        ),
        limit=1,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"collect": stages["collect"], "train": trained, "reload": reload}


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@click.option("--stage", type=click.Choice(["sft", "reload"]), required=True)
@foreground_build_options
def main(
    config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str, stage: str
) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    if resolve_version("russell-diversity-durable-sft", None) != SFT_VERSION:
        raise click.UsageError("Durable diversity SFT requires its frozen version")
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if config["version"] != COLLECTION_VERSION or config["collection_version"] != COLLECTION_VERSION:
        raise click.UsageError("Durable SFT must consume the fixed v15 collection")
    stages = durable_sft_stages(diversity_workflow(config))
    return [stages["train" if stage == "sft" else stage]]


if __name__ == "__main__":
    main()
