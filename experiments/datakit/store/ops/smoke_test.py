# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke: 2 small sources -> per-cluster (K=40) Levanter store.

End-to-end exercise of the datakit_store pipeline on the two smallest sources
in :func:`marin.datakit.sources.all_sources` (``cp/peps`` ~0.003B, ``cp/foodista``
~0.02B), reusing all upstream artifacts via fixed paths (no recompute).

Output: ``gs://marin-eu-west4/datakit/store/_smoke_v0/cluster=<K>/`` plus an
``artifact.json`` describing the per-cluster stats.

Submit on iris (eu-west4)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive \\
        -- python experiments/datakit/store/ops/smoke_test.py
"""

import logging

from fray.types import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import LlmQualityOutput
from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    QUALITY_ROOT,
    TOKENIZE_ROOT,
    _build_resolution_index,
)
from experiments.datakit.store.datakit_store import build_clustered_store

logger = logging.getLogger(__name__)


SMOKE_SOURCES = ("cp/peps", "cp/foodista")
OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/_smoke_v0.1_20260518"

CLUSTER_VIEW = 40
SPLIT = "train"

WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="5g")
MAX_WORKERS = 64


def main() -> None:
    configure_logging(logging.INFO)

    dedup = read_artifact(DEDUP_PATH, FuzzyDupsAttrData)

    tokenize_index = _build_resolution_index(TOKENIZE_ROOT)
    decontam_index = _build_resolution_index(DECONTAM_ROOT)
    cluster_assign_index = _build_resolution_index(CLUSTER_ASSIGN_ROOT)
    quality_index = _build_resolution_index(QUALITY_ROOT)

    tokenize: dict[str, TokenizedAttrData] = {}
    decontam: dict[str, DeconAttributes] = {}
    cluster_assign: dict[str, AssignmentAttrData] = {}
    quality: dict[str, LlmQualityOutput] = {}

    for source_name in SMOKE_SOURCES:
        tokenize[source_name] = read_artifact(tokenize_index[source_name], TokenizedAttrData)
        decontam[source_name] = read_artifact(decontam_index[source_name], DeconAttributes)
        cluster_assign[source_name] = read_artifact(cluster_assign_index[source_name], AssignmentAttrData)
        quality[source_name] = read_artifact(quality_index[source_name], LlmQualityOutput)

    artifact = build_clustered_store(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        dedup=dedup,
        output_path=OUTPUT_PATH,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        worker_resources=WORKER_RESOURCES,
        max_workers=MAX_WORKERS,
    )

    logger.info(
        "smoke done: %d buckets, %d total docs, %d total tokens -> %s",
        len(artifact.buckets),
        sum(b.total_elements for b in artifact.buckets),
        sum(b.total_tokens for b in artifact.buckets),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
