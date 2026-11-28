# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the campaign-level cross-source dedup pass to a campaign's source artifacts.

After a FULL campaign's training sources complete, their ``final/`` views can
still hold the same task contract in more than one source. This step runs
``taskcompendium.pipeline.cross_source.deduplicate_across_sources`` over the
training sources in the order the driver passes (the campaign's catalog
order) and records one drop manifest per source plus the campaign evidence.

The artifact's identity covers the source set and their artifact versions, so
a rerun with rebuilt sources resolves to a new artifact. The recorded drops
are curation-side evidence for a future TaskStore consumer; the selection
step still deduplicates when it samples and may keep a different
(contract-identical) copy.
"""

import hashlib
import json
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

CROSS_SOURCE_DEDUP_VERSION = "2026.11.27.1"


@dataclass(frozen=True)
class CrossSourceRequest:
    """The campaign's training sources in dedup-priority order and this step's output."""

    sources: tuple[RetainedSource, ...]
    output_path: str
    max_workers: int
    worker_resources: ResourceConfig


def cross_source_dedup_step(
    curated: dict[str, ArtifactStep[CampaignArtifact]],
    *,
    order: tuple[str, ...],
    max_workers: int,
    worker_resources: ResourceConfig,
) -> ArtifactStep[RlDataArtifact] | None:
    """The campaign's cross-source dedup artifact, or ``None`` with fewer than two sources.

    ``order`` lists the training source IDs in the campaign's catalog order and
    must cover exactly the curated sources; the earliest source in that order
    keeps each duplicated contract.
    """
    if len(curated) < 2:
        return None
    if set(order) != set(curated):
        missing = sorted(set(curated) - set(order))
        unknown = sorted(set(order) - set(curated))
        raise ValueError(f"Dedup order must cover exactly the training sources: missing={missing} unknown={unknown}")
    deps = list(curated.values())
    # The digest carries each dep's name@version so a rebuilt source set (new
    # curation attempt, changed recipe) gets a new artifact; dep versions alone
    # would leave the name stable while check_drift only warns on drift.
    digest = hashlib.sha256(
        ",".join(f"{source_id}={curated[source_id].name}@{curated[source_id].version}" for source_id in order).encode()
    ).hexdigest()[:20]

    def build_config(ctx: StepContext) -> CrossSourceRequest:
        finals = {source_id: str(StoragePath(ctx.artifact_path(step)) / "final") for source_id, step in curated.items()}
        sources = tuple(
            RetainedSource(source_id=source_id, final_path=finals[source_id], order=index)
            for index, source_id in enumerate(order)
        )
        return CrossSourceRequest(
            sources=sources,
            output_path=ctx.output_path,
            max_workers=max_workers,
            worker_resources=worker_resources,
        )

    return ArtifactStep(
        name=f"task-curation/cross-source-dedup/{digest}",
        version=CROSS_SOURCE_DEDUP_VERSION,
        artifact_type=RlDataArtifact,
        deps=deps,
        run=run_cross_source_dedup,
        build_config=build_config,
    )


def run_cross_source_dedup(request: CrossSourceRequest) -> RlDataArtifact:
    """Run the cross-source pass on the campaign's retained training populations."""
    client = current_client()
    with set_current_client(client):
        with ZephyrContext(
            client=client,
            max_workers=request.max_workers,
            resources=request.worker_resources,
            name="cross-source-dedup",
        ) as context:
            deduplicate_across_sources(request.sources, request.output_path, context=context)
    manifest = json.loads((StoragePath(request.output_path) / "manifest.json").read_text())
    return RlDataArtifact(path=request.output_path, status="completed", manifest=manifest)
