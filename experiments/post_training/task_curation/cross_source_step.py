# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the campaign-level cross-source dedup pass to a campaign's source artifacts.

After a FULL campaign's training sources complete, their ``final/`` views can
still hold the same task contract in more than one source. This step runs
``taskcompendium.pipeline.cross_source.deduplicate_across_sources`` over the
sources in campaign order and records one drop manifest per source plus the
campaign evidence, keyed by source and run, so the TaskStore and selection can
exclude cross-source duplicates from the retained population.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

from fray.current_client import current_client, set_current_client
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.cross_source import (
    RetainedSource,
    deduplicate_across_sources,
)
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.pipeline import RlDataArtifact

logger = logging.getLogger(__name__)

CROSS_SOURCE_DEDUP_VERSION = "2026.11.27.1"
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="4g")


@dataclass(frozen=True)
class CrossSourceRequest:
    """The campaign's training sources in catalog order and this step's output."""

    sources: tuple[tuple[str, str, int], ...]
    """(source ID, final/ path, campaign order) per training source."""
    output_path: str


def cross_source_dedup_step(
    curated: dict[str, ArtifactStep[CampaignArtifact]],
    *,
    max_workers: int,
    worker_resources: ResourceConfig,
) -> ArtifactStep[RlDataArtifact] | None:
    """The campaign's cross-source dedup artifact, or ``None`` with fewer than two sources."""
    if len(curated) < 2:
        return None
    deps = list(curated.values())
    digest = hashlib.sha256(",".join(sorted(curated)).encode()).hexdigest()[:20]

    def build_config(ctx: StepContext) -> CrossSourceRequest:
        finals = {source_id: str(StoragePath(ctx.artifact_path(step)) / "final") for source_id, step in curated.items()}
        sources = tuple((source_id, finals[source_id], order) for order, source_id in enumerate(sorted(curated)))
        return CrossSourceRequest(sources=sources, output_path=ctx.output_path)

    return ArtifactStep(
        name=f"task-curation/cross-source-dedup/{digest}",
        version=CROSS_SOURCE_DEDUP_VERSION,
        artifact_type=RlDataArtifact,
        deps=deps,
        run=run_cross_source_dedup,
        build_config=build_config,
    )


def run_cross_source_dedup(
    request: CrossSourceRequest, *, max_workers: int, worker_resources: ResourceConfig
) -> RlDataArtifact:
    """Run the cross-source pass on the campaign's retained training populations."""
    sources = [
        RetainedSource(source_id=source_id, final_path=final_path, order=order)
        for source_id, final_path, order in request.sources
    ]
    client = current_client()
    with set_current_client(client):
        with ZephyrContext(
            client=client, max_workers=max_workers, resources=worker_resources, name="cross-source-dedup"
        ) as context:
            deduplicate_across_sources(sources, request.output_path, context=context)
    manifest = json.loads((StoragePath(request.output_path) / "manifest.json").read_text())
    return RlDataArtifact(path=request.output_path, status="completed", manifest=manifest)
