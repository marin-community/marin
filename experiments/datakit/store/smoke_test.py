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
        -- python experiments/datakit/store/smoke_test.py
"""

import logging

from fray import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import Artifact
from marin.processing.classification.datakit_store import (
    BuildClusteredStoreConfig,
    ClusterAssignAttrData,
    build_clustered_store,
)
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.log_setup import configure_logging

from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    TOKENIZE_ROOT,
    _resolve_artifact_dir,
)

logger = logging.getLogger(__name__)


SMOKE_SOURCES = ("cp/peps", "cp/foodista")
OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/_smoke_v0"

CLUSTER_VIEW = 40
SPLIT = "train"

WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="5g")
MAX_WORKERS = 64


def main() -> None:
    configure_logging(logging.INFO)

    dedup = Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData)

    tokenize: dict[str, TokenizedAttrData] = {}
    decontam: dict[str, DeconAttributes] = {}
    cluster_assign: dict[str, ClusterAssignAttrData] = {}

    for source_name in SMOKE_SOURCES:
        tokenize[source_name] = Artifact.from_path(_resolve_artifact_dir(TOKENIZE_ROOT, source_name), TokenizedAttrData)
        decontam[source_name] = Artifact.from_path(_resolve_artifact_dir(DECONTAM_ROOT, source_name), DeconAttributes)
        cluster_assign[source_name] = Artifact.from_path(
            _resolve_artifact_dir(CLUSTER_ASSIGN_ROOT, source_name), ClusterAssignAttrData
        )

    config = BuildClusteredStoreConfig(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        dedup=dedup,
        output_path=OUTPUT_PATH,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        worker_resources=WORKER_RESOURCES,
        max_workers=MAX_WORKERS,
    )
    artifact = build_clustered_store(config)

    logger.info(
        "smoke done: %d clusters, %d total docs, %d total tokens -> %s",
        len(artifact.clusters),
        sum(c.total_elements for c in artifact.clusters.values()),
        sum(c.total_tokens for c in artifact.clusters.values()),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
