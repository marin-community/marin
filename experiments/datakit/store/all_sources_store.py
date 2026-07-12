# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the v0.1 datakit (cluster x quality) Levanter store.

Joins, filters, and routes every Datakit source's tokenized output into one
Levanter cache per (cluster=40, quality=5) bucket -> 200 leaves max:

    tokenize    gs://marin-eu-west4/datakit/tokenize/<src>_<hash>/
    decontam    gs://marin-eu-west4/datakit/decontam/<src>_<hash>/
    cluster     gs://marin-eu-west4/datakit/cluster/assign/<src>_<hash>/
    quality     gs://marin-eu-west4/datakit/llm-quality-classifier/inference/sonnet46-thr05/quality-llm/<src>_<hash>/
                (Sonnet 4.6 rubric distilled to a fasttext classifier; see experiments/datakit/cluster/quality/v0/)
    dedup       gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/

Excludes ``safety_pt/*`` and ``climblab-ja`` -- both are missing from the
dedup output (see ``experiments/datakit/dedup/all_sources_fuzzy.py``).

Output: ``gs://marin-eu-west4/datakit/store/v0.1_20260518/cluster=<K>/quality=<Q>/``
for K in 0..39 and Q in 0..4, plus ``artifact.json`` describing the
per-bucket stats and quality cutoffs.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority production --cpu 2 --memory 8GB --enable-extra-resources \\
        --no-preemptible \\
        -- python experiments/datakit/store/all_sources_store.py
"""

import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from fray.types import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import LlmQualityOutput
from experiments.datakit.store.datakit_store import build_clustered_store

logger = logging.getLogger(__name__)


TOKENIZE_ROOT = "gs://marin-eu-west4/datakit/tokenize"
DECONTAM_ROOT = "gs://marin-eu-west4/datakit/decontam"
CLUSTER_ASSIGN_ROOT = "gs://marin-eu-west4/datakit/cluster/assign"
QUALITY_ROOT = "gs://marin-eu-west4/datakit/llm-quality-classifier/inference/sonnet46-thr05/quality-llm"
DEDUP_PATH = "gs://marin-eu-west4/datakit/dedup/dedup_v0_manual"
OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/v0.1_20260518"

CLUSTER_VIEW = 40
SPLIT = "train"

# Mixed smoke validated 8g workers handle even big nemotron shards
# (numpy-int32 input_ids + bounded iter_batches + _BATCH_FLUSH=256 in
# the join loop). Bigger workers would crowd the cluster's preemptible
# pool without benefit.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="5g")
MAX_WORKERS = 2048


_HASH_LEN = 8
"""Length of the StepSpec content-hash suffix on every output dir, in hex chars."""
_HASH_RE = re.compile(rf"^(.+)_([0-9a-f]{{{_HASH_LEN}}})$")


def _build_resolution_index(root: str) -> dict[str, str]:
    """Walk ``root`` via shallow ``fs.ls`` and return ``{source_name: full_path}``.

    A shallow ``fs.ls`` is used per root rather than a ``<root>/<src>_*`` glob:
    gcsfs implements glob as a recursive ``_find`` that lists every object
    under the prefix and caches the result. With 4 roots x ~100 sources that
    blew up to multiple GB of cached listings, OOM'ing the 8g driver in <2 min.

    Shallow walk:
      - ``fs.ls(root, detail=True)`` yields top-level entries with types.
      - Entries matching ``<name>_<8hex>`` are recorded as flat sources
        (e.g. ``coderforge``); we do NOT recurse into them.
      - Other DIRECTORIES are treated as nested namespaces (e.g. ``cp/``,
        ``finepdfs/``, ``nemotron_cc_v2/``); we recurse one level deeper.
        File entries (e.g. the ``evals/`` parquet tree under ``decontam/``)
        are skipped without recursing.

    Total GCS work: ~1 ls per root + ~1 ls per nested subdirectory
    (~15-20 nested dirs across the datakit), independent of source count.
    """
    fs, _ = url_to_fs(root)
    root_clean = root.removeprefix("gs://").rstrip("/")
    index: dict[str, str] = {}

    def visit(rel_prefix: str) -> None:
        listing_path = f"{root_clean}/{rel_prefix}".rstrip("/")
        for info in fs.ls(listing_path, detail=True):
            entry_path = info["name"]
            bn = os.path.basename(entry_path.rstrip("/"))
            if not bn:
                continue
            m = _HASH_RE.match(bn)
            if m:
                source_name = f"{rel_prefix}{m.group(1)}"
                index[source_name] = f"gs://{entry_path.rstrip('/')}"
            elif info.get("type") == "directory":
                # nested namespace (e.g. "cp", "finepdfs", "nemotron_cc_v2")
                visit(f"{rel_prefix}{bn}/")
            # else: a file (e.g. decontam/evals/.../*.parquet) — skip

    visit("")
    return index


def main() -> None:
    configure_logging(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip the zephyr map-side pass and aggregate from the durable sidecars already on GCS.",
    )
    args = parser.parse_args()

    dedup = read_artifact(DEDUP_PATH, FuzzyDupsAttrData)

    sources_to_resolve = list(all_sources())

    # Build a {source_name: full_dir} index per root via shallow fs.ls walks
    # -- bounded GCS work + bounded memory. See `_build_resolution_index`.
    logger.info("indexing 4 roots via shallow fs.ls")
    tokenize_index = _build_resolution_index(TOKENIZE_ROOT)
    decontam_index = _build_resolution_index(DECONTAM_ROOT)
    cluster_assign_index = _build_resolution_index(CLUSTER_ASSIGN_ROOT)
    quality_index = _build_resolution_index(QUALITY_ROOT)

    def _resolve(name: str) -> tuple[str, TokenizedAttrData, DeconAttributes, AssignmentAttrData, LlmQualityOutput]:
        return (
            name,
            read_artifact(tokenize_index[name], TokenizedAttrData),
            read_artifact(decontam_index[name], DeconAttributes),
            read_artifact(cluster_assign_index[name], AssignmentAttrData),
            read_artifact(quality_index[name], LlmQualityOutput),
        )

    tokenize: dict[str, TokenizedAttrData] = {}
    decontam: dict[str, DeconAttributes] = {}
    cluster_assign: dict[str, AssignmentAttrData] = {}
    quality: dict[str, LlmQualityOutput] = {}

    # Index lookups are O(1); only the 4 ``read_artifact`` reads per
    # source need parallelization. Threadpool of 16 fans those out without
    # the gcsfs cache blow-up we saw at 32+.
    logger.info("loading typed artifacts for %d sources (ThreadPoolExecutor=16)", len(sources_to_resolve))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for name, tok, decon, cluster_a, qual in pool.map(_resolve, sources_to_resolve):
            tokenize[name] = tok
            decontam[name] = decon
            cluster_assign[name] = cluster_a
            quality[name] = qual

    logger.info("resolved %d sources", len(tokenize))

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
        aggregate_only=args.aggregate_only,
    )

    logger.info(
        "done: %d buckets, %d total docs, %d total tokens -> %s",
        len(artifact.buckets),
        sum(b.total_elements for b in artifact.buckets),
        sum(b.total_tokens for b in artifact.buckets),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
