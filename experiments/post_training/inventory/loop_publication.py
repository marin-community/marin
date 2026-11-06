# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a retained loop result independently of expensive training."""

import json
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.inventory.loop_result import LoopRunConfig, run_loop
from experiments.post_training.inventory.publish_loop_result import publish_result


@dataclass(frozen=True)
class PublicationConfig:
    summary_uri: str
    receipt_uri: str


class LoopPublication(Artifact):
    run_id: str
    attempt_id: str


def publish_summary(config: PublicationConfig) -> LoopPublication:
    """Read the retained summary and confirm one immutable dashboard attempt."""
    with StoragePath(config.summary_uri).open("rt") as stream:
        summary = json.load(stream)
    receipt = publish_result(summary)
    destination = StoragePath(config.receipt_uri)
    destination.parent.mkdirs()
    destination.write_text(json.dumps(receipt, allow_nan=False))
    return LoopPublication(path=config.receipt_uri, run_id=summary["run_id"], attempt_id=summary["attempt_id"])


def run_recorded_loop(config: LoopRunConfig) -> SkyRLRun:
    """Publish failed attempts too, while preserving the training failure."""
    try:
        return run_loop(config)
    except Exception as training_error:
        try:
            publish_summary(
                PublicationConfig(
                    summary_uri=prefix_join(config.skyrl.output.export_root, "loop-result.json"),
                    receipt_uri=prefix_join(config.skyrl.output.export_root, "loop-publication.json"),
                )
            )
        except Exception as publication_error:
            raise ExceptionGroup("Training and result publication failed", [training_error, publication_error]) from None
        raise


def publication_step(training: ArtifactStep[SkyRLRun]) -> ArtifactStep[LoopPublication]:
    """Keep publication retries downstream of the cached training artifact."""

    def build_config(ctx: StepContext) -> PublicationConfig:
        return PublicationConfig(
            summary_uri=prefix_join(ctx.artifact_path(training), "exports/loop-result.json"),
            receipt_uri=prefix_join(ctx.output_path, "publication.json"),
        )

    return ArtifactStep(
        name=f"{training.name}-report",
        version=training.version,
        artifact_type=LoopPublication,
        run=publish_summary,
        build_config=build_config,
        deps=(training,),
    )
