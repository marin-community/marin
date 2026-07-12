# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Larger smoke: a handful of sources including some big ones.

Stresses the pipeline at meaningful scale before the full fleet run --
~780B tokens (~5-8% of the full registry), with two big nemotron sources
to surface any per-shard memory or GCS write-bandwidth issues.

Source mix:

* ``cp/foodista`` -- tiny baseline.
* ``coderforge`` -- small/medium code corpus.
* ``nemotron_cc_v2_1/high_quality`` -- 25B, medium nemotron.
* ``nemotron_specialized/rqa`` -- 135B, biggish synth.
* ``nemotron_cc_v2/high_quality`` -- 608B, BIG.

Submit on iris (eu-west4) -- bump driver memory (default 1G OOMs while
building shard_specs over ~10K+ nemotron shards)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority production --cpu 2 --memory 4GB --enable-extra-resources \\
        --no-preemptible \\
        -- python experiments/datakit/store/ops/smoke_test_mixed.py
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


SMOKE_SOURCES = (
    "cp/foodista",  # 0.02B -- tiny baseline
    "coderforge",  # 10B -- small/medium code
    "nemotron_cc_v2_1/high_quality",  # 25B -- medium nemotron
    "nemotron_specialized/rqa",  # 135B -- biggish synth
    "nemotron_cc_v2/high_quality",  # 608B -- BIG
)
OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/_smoke_v0.1_20260518_mixed"

CLUSTER_VIEW = 40
SPLIT = "train"

# Streaming refactor caps per-worker peak at ``N_open_buckets * _BATCH_FLUSH *
# avg-doc-size`` (~hundreds of MB worst case at 200 buckets), so 8g is
# comfortable. Smaller bins = more available slots under cluster contention.
# Workers stay preemptible (zephyr auto-retries infra failures) -- under
# production priority their scheduling cost goes way down.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="5g")
MAX_WORKERS = 2048


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

    logger.info("mixed-smoke: %d sources -> %s", len(tokenize), OUTPUT_PATH)

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
        "mixed-smoke done: %d buckets, %d total docs, %d total tokens -> %s",
        len(artifact.buckets),
        sum(b.total_elements for b in artifact.buckets),
        sum(b.total_tokens for b in artifact.buckets),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
