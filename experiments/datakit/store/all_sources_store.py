# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the v0 datakit-clustered Levanter store.

Joins, filters, and routes every Datakit source's tokenized output into one
Levanter cache per cluster (K=40):

    tokenize    gs://marin-eu-west4/datakit/tokenize/<src>_<hash>/
    decontam    gs://marin-eu-west4/datakit/decontam/<src>_<hash>/
    cluster     gs://marin-eu-west4/datakit/cluster/assign/<src>_<hash>/
    dedup       gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/

Excludes ``safety_pt/*`` and ``climblab-ja`` -- both are missing from the
dedup output (see ``experiments/datakit/dedup/all_sources_fuzzy.py``).

Output: ``gs://marin-eu-west4/datakit/store/v0/cluster=<K>/`` for K in 0..39,
plus ``v0/artifact.json`` describing the per-cluster stats.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive --cpu 2 --memory 8GB \\
        -- python experiments/datakit/store/all_sources_store.py
"""

import logging
import os
import re

from fray import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from marin.utils import fsspec_exists, fsspec_glob
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.v0.assign import AssignmentAttrData
from experiments.datakit.store.datakit_store import build_clustered_store

logger = logging.getLogger(__name__)


TOKENIZE_ROOT = "gs://marin-eu-west4/datakit/tokenize"
DECONTAM_ROOT = "gs://marin-eu-west4/datakit/decontam"
CLUSTER_ASSIGN_ROOT = "gs://marin-eu-west4/datakit/cluster/assign"
DEDUP_PATH = "gs://marin-eu-west4/datakit/dedup/dedup_v0_manual"
OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/v0"

# Sources excluded from the v0 store. Both were held out of the dedup run
# (see ``experiments/datakit/dedup/all_sources_fuzzy.py``), so excluding them
# here keeps the invariant that every kept doc has all four attributes.
_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "safety_pt/",
    "climblab-ja",
)

CLUSTER_VIEW = 40
SPLIT = "train"

WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")
MAX_WORKERS = 1024


def _is_excluded(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _EXCLUDE_PREFIXES)


_HASH_LEN = 8
"""Length of the StepSpec content-hash suffix on every output dir, in hex chars."""


def _resolve_artifact_dir(root: str, source_name: str) -> str:
    """Return ``<root>/<source_name>_<hash>`` for the single hashed dir whose
    ``artifact.json`` is present.

    The match is anchored on the leaf component plus an 8-hex hash. A loose
    ``<source_name>_*`` glob would over-match siblings sharing a prefix --
    e.g. ``nemotron_cc_v2_1/high_quality_*`` also matches
    ``high_quality_dqa_<hash>``, ``high_quality_synthetic_<hash>``, etc.
    """
    source_leaf = source_name.rsplit("/", 1)[-1]
    leaf_re = re.compile(rf"^{re.escape(source_leaf)}_[0-9a-f]{{{_HASH_LEN}}}$")
    candidates = [
        p.rstrip("/")
        for p in fsspec_glob(f"{root.rstrip('/')}/{source_name}_*")
        if leaf_re.match(os.path.basename(p.rstrip("/"))) and fsspec_exists(f"{p.rstrip('/')}/artifact.json")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one artifact dir under {root!r} for {source_name!r}, found {candidates}")
    return candidates[0]


def main() -> None:
    configure_logging(logging.INFO)

    dedup = Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData)

    tokenize: dict[str, TokenizedAttrData] = {}
    decontam: dict[str, DeconAttributes] = {}
    cluster_assign: dict[str, AssignmentAttrData] = {}

    for source_name in all_sources():
        if _is_excluded(source_name):
            logger.info("excluding %s", source_name)
            continue
        tokenize[source_name] = Artifact.from_path(_resolve_artifact_dir(TOKENIZE_ROOT, source_name), TokenizedAttrData)
        decontam[source_name] = Artifact.from_path(_resolve_artifact_dir(DECONTAM_ROOT, source_name), DeconAttributes)
        cluster_assign[source_name] = Artifact.from_path(
            _resolve_artifact_dir(CLUSTER_ASSIGN_ROOT, source_name), AssignmentAttrData
        )

    logger.info("resolved %d sources", len(tokenize))

    artifact = build_clustered_store(
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

    logger.info(
        "done: %d clusters, %d total docs, %d total tokens -> %s",
        len(artifact.clusters),
        sum(c.total_elements for c in artifact.clusters.values()),
        sum(c.total_tokens for c in artifact.clusters.values()),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
