# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fuzzy-dedup the quality-filtered OCR corpus, electing canonicals by quality.

:mod:`~experiments.build_pdf_source.repair_ocr_all` deliberately deferred fuzzy dedup until a
quality signal existed: the library's canonical pick (the member whose tag-prefixed content hash
equals the cluster's component id) is arbitrary with respect to quality, and freezing it would
have discarded better cluster members forever. With the quality step's ``edu_max`` column in
place, this module runs the standard minhash + connected-components stages unchanged and then
**re-elects** each cluster's canonical: the member minimizing ``(-edu_max, id)`` wins, so the
highest-scored member survives and byte-identical scores (the common case inside near-dup
clusters) break deterministically to the smallest content hash. Members without a score are
treated as ``-inf`` — a scored member always beats an unscored one. ``fuzzy_dups`` anticipates
exactly this: its attr rows carry ``is_cluster_canonical`` so the selection policy can live in a
downstream consumer rather than in the library.

The re-election runs as plain pyarrow in the step driver rather than as a Zephyr pipeline: at
311,807 documents the whole non-singleton attr tree plus the ``{id: edu_max}`` map is tens of
megabytes, well inside a small driver, and an in-driver pass keeps the global per-cluster
election trivially correct. It writes a fresh attr tree with the same layout, per-shard
basenames, row sort, and three-column schema as the library's, so ``consolidate`` joins it
exactly as it would the original.

Cluster ids are untouched — only ``is_cluster_canonical`` is recomputed. If connected components
did not converge within its iteration cap, fragments of a true cluster each elect their own
canonical; that matches the library's documented warning-only behavior and keeps the result
deterministic.
"""

import logging
import math
import os
from dataclasses import dataclass
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.fuzzy_dups import (
    FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyDupsAttrData,
    FuzzyDupsPerSource,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs_step
from rigging.filesystem import StoragePath, prefix_join
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

_COUNTER_PREFIX = "focus_crawl_pdf_ocr_fuzzy_reelect"

_CORPUS = "common_crawl_focus_2026_22_pdf_ocr_all"
_MINHASH_NAME = f"data/datakit/minhash/{_CORPUS}"
_FUZZY_DUPS_NAME = f"data/datakit/fuzzy_dups/{_CORPUS}"
_REELECT_NAME = f"data/datakit/fuzzy_elect/{_CORPUS}"
_CLEAN_NAME = f"data/datakit/fuzzy_clean/{_CORPUS}"

# The quality column the election reads; the quality step guarantees it on every document.
_QUALITY_SCORE_COLUMN = "edu_max"
# Identity tag for the election policy: sort key (-edu_max, id), minimum wins.
_ELECTION_POLICY = "edu_max_desc_id_asc"

_ATTR_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("dup_cluster_id", pa.string()),
        pa.field("is_cluster_canonical", pa.bool_()),
    ]
)

# MinHash parameters stay the compute_minhash_attrs_step defaults (num_perms 286, bands 26,
# ngram 5, seed 42), matching the reference pipeline; the factory restates them in hash_attrs.
_FUZZY_CC_MAX_ITERATIONS = 10
_FUZZY_MAX_PARALLELISM = 64

_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# The corpus has 23 shards, so the streaming stages have ~23 tasks; asking for more workers than
# that queues for capacity the stage cannot use (see repair_ocr_all's decon sizing note).
_MAX_WORKERS = 12
# Never Zephyr's 1 GB coordinator default (exit-137 at run end; see repair_ocr_all). The CC
# group_by shuffle gets the larger allowance.
_FUZZY_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="16g", preemptible=False)
_CONSOLIDATE_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
_REELECT_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="8g")


@dataclass(frozen=True)
class _AttrShard:
    """One fuzzy attr shard, decoded: parallel per-member columns, rows sorted by id."""

    basename: str
    ids: list[str]
    cluster_ids: list[str]
    was_canonical: list[bool]


def _load_attr_shards(attr_dir: str) -> list[_AttrShard]:
    """Read every attr shard under *attr_dir*, asserting the sorted-by-id join invariant."""
    paths = sorted(StoragePath(prefix_join(attr_dir, "*.parquet")).glob(), key=str)
    if not paths:
        raise RuntimeError(f"No fuzzy attr shards under {attr_dir}")

    shards: list[_AttrShard] = []
    for path in paths:
        with path.open("rb") as stream:
            table = pq.read_table(stream)
        if table.num_rows == 0:
            # A shard with no cluster members is written schema-less by the library; carry it
            # through so the tree stays 1:1 with the corpus shards.
            shards.append(_AttrShard(basename=path.name, ids=[], cluster_ids=[], was_canonical=[]))
            continue
        ids = table["id"].to_pylist()
        assert ids == sorted(ids), f"attr shard {path} is not sorted by id"
        shards.append(
            _AttrShard(
                basename=path.name,
                ids=ids,
                cluster_ids=table["dup_cluster_id"].to_pylist(),
                was_canonical=table["is_cluster_canonical"].to_pylist(),
            )
        )
    return shards


def _quality_scores(main_output_dir: str) -> dict[str, float]:
    """Map content-hash ``id`` to ``edu_max`` across every quality corpus shard.

    Null scores are omitted so the election treats them as unscored.
    """
    paths = sorted(StoragePath(prefix_join(main_output_dir, "*.parquet")).glob(), key=str)
    if not paths:
        raise RuntimeError(f"No quality corpus shards under {main_output_dir}")

    scores: dict[str, float] = {}
    for path in paths:
        with path.open("rb") as stream:
            table = pq.read_table(stream, columns=["id", _QUALITY_SCORE_COLUMN])
        for doc_id, score in zip(table["id"].to_pylist(), table[_QUALITY_SCORE_COLUMN].to_pylist(), strict=True):
            if score is not None:
                scores[doc_id] = score
    return scores


def reelect_cluster_canonicals(
    *, fuzzy: FuzzyDupsAttrData, quality: NormalizedData, output_path: str
) -> FuzzyDupsAttrData:
    """Rewrite the fuzzy attr tree with canonicals re-elected by quality.

    Per ``dup_cluster_id``, the member minimizing ``(-edu_max, id)`` becomes the canonical:
    highest score wins, ties break to the smallest id, unscored members rank below every scored
    one. Cluster ids and the tree layout (per-shard basenames, row order) are preserved so
    ``consolidate`` joins the result exactly like the library's own output.
    """
    if len(fuzzy.sources) != 1:
        # A global election over this source's scores alone would leave other sources'
        # canonicals stale; this pipeline only ever has the quality corpus.
        raise ValueError(f"re-election expects exactly one source, got {sorted(fuzzy.sources)}")

    source_key = datakit_source_key(quality.main_output_dir)
    attr_dir = fuzzy.attr_dir_for_source(quality.main_output_dir)
    shards = _load_attr_shards(attr_dir)
    scores = _quality_scores(quality.main_output_dir)

    winners: dict[str, tuple[float, str]] = {}
    members_unscored = 0
    for shard in shards:
        for doc_id, cluster_id in zip(shard.ids, shard.cluster_ids, strict=True):
            score = scores.get(doc_id)
            # NaN would poison the tuple comparison below (never wins, never loses), so it
            # ranks as unscored alongside missing rows.
            if score is None or math.isnan(score):
                members_unscored += 1
                score = float("-inf")
            key = (-score, doc_id)
            if cluster_id not in winners or key < winners[cluster_id]:
                winners[cluster_id] = key
    winner_ids = {cluster_id: doc_id for cluster_id, (_, doc_id) in winners.items()}

    source_tag = os.path.basename(attr_dir.rstrip("/"))
    new_attr_dir = prefix_join(output_path, f"outputs/{source_tag}")
    canonicals_per_cluster = dict.fromkeys(winner_ids, 0)
    cluster_members = 0
    canonicals_changed = 0
    for shard in shards:
        rows = []
        for doc_id, cluster_id, was_canonical in zip(shard.ids, shard.cluster_ids, shard.was_canonical, strict=True):
            is_canonical = winner_ids[cluster_id] == doc_id
            canonicals_per_cluster[cluster_id] += is_canonical
            canonicals_changed += is_canonical and not was_canonical
            rows.append({"id": doc_id, "dup_cluster_id": cluster_id, "is_cluster_canonical": is_canonical})
        cluster_members += len(rows)
        write_parquet_file(rows, prefix_join(new_attr_dir, shard.basename), schema=_ATTR_SCHEMA)

    broken = {cluster_id: n for cluster_id, n in canonicals_per_cluster.items() if n != 1}
    assert not broken, f"clusters without exactly one canonical after re-election: {broken}"

    write_copartitioned_source_manifest(output_path=output_path, attr_dirs={source_key: new_attr_dir})
    logger.info(
        "Re-elected %d clusters (%d members, %d canonicals moved, %d members unscored)",
        len(winner_ids),
        cluster_members,
        canonicals_changed,
        members_unscored,
    )
    return FuzzyDupsAttrData(
        params=fuzzy.params,
        sources={source_key: FuzzyDupsPerSource(attr_dir=new_attr_dir)},
        counters={
            f"{_COUNTER_PREFIX}/clusters": len(winner_ids),
            f"{_COUNTER_PREFIX}/cluster_members": cluster_members,
            f"{_COUNTER_PREFIX}/canonicals_changed": canonicals_changed,
            f"{_COUNTER_PREFIX}/members_unscored": members_unscored,
        },
    )


def reelect_fuzzy_canonicals(
    output_path: str, fuzzy_dups_output_path: str, quality_output_path: str
) -> FuzzyDupsAttrData:
    """Step fn: load the fuzzy and quality artifacts, then re-elect canonicals."""
    fuzzy = read_artifact(fuzzy_dups_output_path, FuzzyDupsAttrData)
    quality = read_artifact(quality_output_path, NormalizedData)
    return reelect_cluster_canonicals(fuzzy=fuzzy, quality=quality, output_path=output_path)


def consolidate_fuzzy_clean(output_path: str, quality_output_path: str, reelected_output_path: str) -> NormalizedData:
    """Keep the re-elected canonical of every cluster, plus every singleton.

    Fuzzy attrs are sparse — singletons have no row — so ``keep_if_missing=True`` is load-bearing:
    without it every unique document would be dropped.
    """
    quality = read_artifact(quality_output_path, NormalizedData)
    reelected = read_artifact(reelected_output_path, FuzzyDupsAttrData)
    outcome = consolidate(
        input_path=quality.main_output_dir,
        output_path=prefix_join(output_path, "outputs/main"),
        filetype="parquet",
        filters=[
            FilterConfig(
                type=FilterType.KEEP_DOC,
                attribute_path=reelected.attr_dir_for_source(quality.main_output_dir),
                name="is_cluster_canonical",
                attribute_filetype="parquet",
                keep_if_missing=True,
            )
        ],
        worker_resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        coordinator_resources=_CONSOLIDATE_COORDINATOR_RESOURCES,
    )
    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def fuzzy_steps(quality: StepSpec, schema: pa.Schema) -> list[StepSpec]:
    """Build the fuzzy-dedup DAG over the quality-filtered corpus.

    Returns ``[minhash, fuzzy_dups, reelect, fuzzy_clean]``; the last step produces the
    :class:`~marin.datakit.normalize.NormalizedData` at
    ``data/datakit/fuzzy_clean/common_crawl_focus_2026_22_pdf_ocr_all``. ``quality`` must be a
    step whose artifact is a ``NormalizedData`` with *schema* — the election needs its
    ``edu_max`` column.
    """
    if _QUALITY_SCORE_COLUMN not in schema.names:
        raise ValueError(f"quality schema must carry {_QUALITY_SCORE_COLUMN!r}; got {schema.names}")

    minhash = compute_minhash_attrs_step(
        name=_MINHASH_NAME,
        normalize=quality,
        worker_resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
    )
    fuzzy_dups = compute_fuzzy_dups_attrs_step(
        name=_FUZZY_DUPS_NAME,
        minhash_steps=[minhash],
        cc_max_iterations=_FUZZY_CC_MAX_ITERATIONS,
        max_parallelism=_FUZZY_MAX_PARALLELISM,
        worker_resources=_WORKER_RESOURCES,
        coordinator_resources=_FUZZY_COORDINATOR_RESOURCES,
    )
    reelect = StepSpec(
        name=_REELECT_NAME,
        deps=[fuzzy_dups, quality],
        hash_attrs={
            "election": _ELECTION_POLICY,
            "artifact_version": FUZZY_DUPS_ATTR_DATA_VERSION,
            "v": 1,
        },
        fn=remote(
            partial(
                reelect_fuzzy_canonicals,
                fuzzy_dups_output_path=fuzzy_dups.output_path,
                quality_output_path=quality.output_path,
            ),
            resources=_REELECT_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    fuzzy_clean = StepSpec(
        name=_CLEAN_NAME,
        deps=[quality, reelect],
        hash_attrs={"filters": ("keep_cluster_canonical",), "v": 1},
        fn=partial(
            consolidate_fuzzy_clean,
            quality_output_path=quality.output_path,
            reelected_output_path=reelect.output_path,
        ),
    )
    return [minhash, fuzzy_dups, reelect, fuzzy_clean]
