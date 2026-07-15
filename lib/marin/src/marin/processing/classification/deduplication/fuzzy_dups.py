# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute fuzzy duplicate markers from one or more ``MinHashAttrData`` inputs.

Loads MinHash bucket attrs from each input, runs LSH-graph connected
components globally across all inputs, and writes per-source attribute trees
annotating every non-singleton cluster member. Each source's attr tree is
co-partitioned with its underlying ``NormalizedData``, so
:mod:`marin.processing.classification.consolidate` can join them directly.

Per-document attr rows have schema::

    {
      id: str,
      attributes: {
        dup_cluster_id: str,         # global CC component id — shared by all cluster members
        is_cluster_canonical: bool,  # True for the canonical member(s); see canonical_scope
      }
    }

Rows are emitted for every member of a non-singleton cluster (canonical +
non-canonicals). Singletons get no row, preserving the
``consolidate(..., keep_if_missing=True)`` pattern. This shape lets the
canonical-selection policy live in consolidate (e.g. the default
``keep is_cluster_canonical=True``, or any custom per-cluster reducer) rather
than being baked in here. ``canonical_scope`` decides how many members are
flagged canonical: one per cluster (``GLOBAL``) or one per ``(source, cluster)``
(``PER_SOURCE``, which prevents cross-source dedup from wiping out whole
sources) — ``dup_cluster_id`` stays the global component id either way.

Combining multiple ``MinHashAttrData`` inputs is the foundation for iterative
global dedup: re-running this job over the union of all per-dataset MinHash
artifacts produces fresh markers without re-reading any source text.
"""

import logging
import os
from collections.abc import Iterator
from enum import StrEnum
from typing import Any

from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_WORKERS_PER_JOB, ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.connected_components import connected_components
from marin.processing.classification.deduplication.dedup_commons import _load_batches
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams

logger = logging.getLogger(__name__)


class CanonicalScope(StrEnum):
    """Which members of a duplicate cluster are marked ``is_cluster_canonical``.

    Connected components always run globally across every input, so
    ``dup_cluster_id`` identifies a cross-source cluster regardless of scope.
    This only controls *which* members of that cluster survive the default
    keep-canonical policy:

    - ``GLOBAL``: exactly one member per cluster (the min-``id_norm`` node),
      regardless of source. Cross-source near-duplicates collapse to a single
      surviving copy — appropriate when source boundaries are irrelevant (e.g.
      deduping one logical corpus).
    - ``PER_SOURCE``: one member per ``(source, cluster)`` — the min-``id_norm``
      node *within each source*. Every source that participates in a cluster
      keeps a representative, so no source can be wiped out just because its
      content near-duplicates content in another source. Genuine *intra*-source
      near-duplicates are still collapsed to one.
    """

    GLOBAL = "global"
    PER_SOURCE = "per_source"


class FuzzyDupsPerSource(BaseModel):
    """Per-source output entry inside :class:`FuzzyDupsAttrData`.

    Attributes:
        attr_dir: Directory containing per-shard duplicate marker Parquet
            files. Filenames mirror the source's MinHash attr (and thus its
            normalized) shards.
    """

    attr_dir: str


class FuzzyDupsAttrData(BaseModel):
    """Co-partitioned fuzzy-duplicate marker attrs for one or more sources.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, FuzzyDupsAttrData)``.

    Attributes:
        version: Schema version of this artifact.
        params: MinHash params; equal to every input's params.
        sources: Mapping from each input's ``MinHashAttrData.source_main_dir``
            to its per-source attr output entry.
        counters: Aggregated zephyr counters across all sources.
    """

    version: str = "v1"
    params: MinHashParams
    sources: dict[str, FuzzyDupsPerSource]
    counters: dict[str, int | float]


def _validate_inputs(inputs: list[MinHashAttrData]) -> MinHashParams:
    """Ensure every input shares the same MinHash params and a unique source; raise otherwise."""
    if not inputs:
        raise ValueError("compute_fuzzy_dups_attrs requires at least one input")

    head = inputs[0].params
    mismatched = [(i, m.params) for i, m in enumerate(inputs) if m.params != head]
    if mismatched:
        details = "; ".join(f"inputs[{i}]={p}" for i, p in mismatched)
        raise ValueError(
            f"All MinHashAttrData inputs must share identical MinHash params. "
            f"inputs[0]={head} but mismatches: {details}"
        )

    seen: dict[str, int] = {}
    for i, m in enumerate(inputs):
        if m.source_main_dir in seen:
            raise ValueError(
                f"Duplicate source_main_dir in inputs: inputs[{seen[m.source_main_dir]}] and "
                f"inputs[{i}] both point to {m.source_main_dir!r}. Each source must be "
                "represented at most once so its output attr tree is unambiguous."
            )
        seen[m.source_main_dir] = i

    return head


def _build_shard_index(inputs: list[MinHashAttrData]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Enumerate every source shard across *inputs* and assign a global file_idx.

    Returns:
        (entries, source_tag_for_input) where ``entries[file_idx]`` holds
        ``{attr_path, source_main_dir, source_tag, basename}`` and
        ``source_tag_for_input[source_main_dir] = "source_NNN"``.
    """
    # Sort inputs by source_main_dir so source_tags are deterministic regardless
    # of the order callers happen to pass them in.
    ordered = sorted(enumerate(inputs), key=lambda iv: iv[1].source_main_dir)
    source_tag: dict[str, str] = {}
    for new_idx, (_, m) in enumerate(ordered):
        source_tag[m.source_main_dir] = f"source_{new_idx:03d}"

    entries: list[dict[str, Any]] = []
    for m in inputs:
        attr_shards = sorted(str(shard) for shard in StoragePath(f"{m.attr_dir.rstrip('/')}/*.parquet").glob())
        if not attr_shards:
            raise FileNotFoundError(f"No attr parquet shards under {m.attr_dir}")
        for attr_path in attr_shards:
            entries.append(
                {
                    "file_idx": len(entries),
                    "attr_path": attr_path,
                    "source_main_dir": m.source_main_dir,
                    "source_tag": source_tag[m.source_main_dir],
                    "basename": os.path.basename(attr_path),
                }
            )
    return entries, source_tag


# Separator between the per-source CC tag and the original content-hash id.
# "|" can't appear in the hex-digit content hashes produced by normalize's
# generate_id, so splitting on the first "|" is unambiguous.
_CC_ID_SEP = "|"


def _cc_id(source_tag: str, doc_id: str) -> str:
    """Prefix *doc_id* with *source_tag* so CC treats cross-source collisions as distinct nodes.

    ``connected_components`` keys nodes by a hash of the record id. Two
    inputs can carry byte-identical normalized ids (e.g. exact text overlap
    across datasets), and without this prefix they collapse to a single
    node — under-reporting dups and potentially clobbering co-partitioned
    attr files. The prefix is split back off in :func:`_split_cc_id` before
    the final attr parquet is written.
    """
    return f"{source_tag}{_CC_ID_SEP}{doc_id}"


def _split_cc_id(record_id: str) -> tuple[str, str]:
    """Reverse :func:`_cc_id`, returning ``(source_tag, doc_id)``."""
    source_tag, doc_id = record_id.split(_CC_ID_SEP, 1)
    return source_tag, doc_id


def _emit_bucket_records(entries: list[dict[str, Any]]) -> Iterator[dict]:
    """For each (bucket, id) pair across all attr shards in *entries*, emit a routing record."""
    for entry in entries:
        for batch in _load_batches(entry["attr_path"], columns=["id", "buckets"]):
            ids = batch["id"]
            buckets_col = batch["buckets"]
            for doc_id, doc_buckets in zip(ids, buckets_col, strict=True):
                if not doc_buckets.is_valid:
                    continue
                cc_id = _cc_id(entry["source_tag"], doc_id.as_py())
                for b in doc_buckets.as_py():
                    yield {"bucket": str(b), "id": cc_id, "file_idx": entry["file_idx"]}


# Key under which ``entries`` is staged via ``ZephyrContext.put`` for the
# stage1 reducer. Holding the list in the closure would serialize it into
# every ``pull_task`` RPC pickle (one per dispatched reduce task) — at
# ~100k entries this OOMs the coord under the high-fan-out dispatch that
# kicks in when ``max_parallelism`` is large. Pulling it from shared data
# instead means each worker fetches and caches the list once.
_SHARED_ENTRIES_KEY = "fuzzy_dups_entries"


def _make_per_shard_writer(output_path: str, counter_prefix: str):
    """Return a group_by reducer that writes per-shard cluster-annotation parquet files.

    Skips singletons entirely. For every non-singleton cluster member, writes
    ``{id, attributes: {dup_cluster_id, is_cluster_canonical}}``. Rows are
    already sorted by ``id`` thanks to the upstream ``group_by(sort_by=id)``.

    The ``entries`` list is loaded via ``zephyr_worker_ctx().get_shared`` so it
    is shipped to workers once (via Zephyr shared-data) rather than captured
    in this closure and re-pickled per ``pull_task`` RPC.
    """

    def aggregate(file_idx: int, records: Iterator[dict]) -> dict:
        entries = zephyr_worker_ctx().get_shared(_SHARED_ENTRIES_KEY)
        entry = entries[file_idx]
        out_path = f"{output_path}/outputs/{entry['source_tag']}/{entry['basename']}"

        cluster_members = 0
        canonicals = 0

        def cluster_member_rows():
            nonlocal cluster_members, canonicals
            for record in records:
                if record["is_singleton"]:
                    counters.pipeline.update_counter(f"{counter_prefix}/singletons_skipped", 1)
                    continue
                cluster_members += 1
                counters.pipeline.update_counter(f"{counter_prefix}/cluster_members", 1)
                if record["is_canonical"]:
                    canonicals += 1
                    counters.pipeline.update_counter(f"{counter_prefix}/canonicals", 1)
                yield {
                    "id": record["id"],
                    "attributes": {
                        "dup_cluster_id": record["component_id"],
                        "is_cluster_canonical": record["is_canonical"],
                    },
                }

        result = write_parquet_file(cluster_member_rows(), out_path)
        return {
            **result,
            "file_idx": file_idx,
            "source_tag": entry["source_tag"],
            "cluster_members": cluster_members,
            "canonicals": canonicals,
        }

    return aggregate


def _to_cc_member(record: dict) -> dict:
    """Project one raw ``CCNode`` parquet row into a cluster-member record.

    Returns ``{id, id_norm, source_tag, component_id, file_idx, is_singleton}``:
    the source-stripped doc ``id``, the orderable ``id_norm``, its
    ``source_tag``, the global ``component_id`` (the ``dup_cluster_id``), the
    ``file_idx`` write shard, and whether the node is a singleton.
    """
    source_tag, doc_id = _split_cc_id(record["record_id"])
    adjacency = record["adjacency_list"]
    return {
        "id": doc_id,
        "id_norm": record["id_norm"],
        "source_tag": source_tag,
        "component_id": record["component_id"],
        "file_idx": record["file_idx"],
        # preserve_singletons wires singletons as a self-link, so a node is a
        # singleton iff its adjacency is exactly [id_norm] — no cluster peers.
        "is_singleton": len(adjacency) == 1 and adjacency[0] == record["id_norm"],
    }


def _mark_source_canonicals(key: tuple[str, str], members: Iterator[dict]) -> Iterator[dict]:
    """Flag the min-``id_norm`` member of one ``(source_tag, component_id)`` group as canonical.

    Relies on the upstream ``group_by(sort_by=id_norm)`` delivering members in
    ascending ``id_norm`` order, so the first member is the per-source minimum
    and the rest are non-canonical. Streaming (O(1) memory) — a single cluster
    within one over-merged source can hold millions of members. ``id_norm`` is
    unique per node within a source, so the minimum is unambiguous.
    """
    is_first = True
    for member in members:
        yield {**member, "is_canonical": is_first}
        is_first = False


def compute_fuzzy_dups_attrs(
    *,
    inputs: list[MinHashAttrData],
    output_path: str,
    canonical_scope: CanonicalScope,
    cc_max_iterations: int = 10,
    cc_resume: bool = False,
    max_parallelism: int = MAX_WORKERS_PER_JOB,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
) -> FuzzyDupsAttrData:
    """Mark fuzzy-duplicate cluster membership across one or more ``MinHashAttrData`` inputs.

    All inputs must share identical :class:`MinHashParams`. The job builds a
    global LSH bucket graph across every input shard, runs connected
    components, and emits a per-source attribute tree under
    ``<output_path>/outputs/source_NNN/`` with one parquet file per source
    shard (filenames preserved from the source). Each row annotates one
    cluster member with ``{id: str, attributes: {dup_cluster_id: str,
    is_cluster_canonical: bool}}``; singletons are omitted.

    ``dup_cluster_id`` is always the global connected-component id, so it
    identifies a cross-source cluster regardless of ``canonical_scope``.
    ``canonical_scope`` controls how many members are flagged
    ``is_cluster_canonical=True`` (the members the default keep-canonical
    policy retains):

    - :attr:`CanonicalScope.GLOBAL`: exactly one per cluster (the min-``id_norm``
      node). Cross-source near-duplicates collapse to a single surviving copy.
    - :attr:`CanonicalScope.PER_SOURCE`: one per ``(source, cluster)`` — the
      min-``id_norm`` node within each source. Prevents whole-source wipeouts
      when a source's content near-duplicates content in other sources, while
      still collapsing genuine intra-source near-duplicates.

    Consolidate may honor the flag (default policy) or ignore it and apply a
    custom per-``dup_cluster_id`` policy.

    Args:
        inputs: ``MinHashAttrData`` artifacts to fuzzy-dedup together.
        output_path: Output root. Per-source attr trees land under
            ``<output_path>/outputs/source_NNN/``.
        canonical_scope: Whether ``is_cluster_canonical`` marks one member per
            cluster (:attr:`CanonicalScope.GLOBAL`) or one per ``(source,
            cluster)`` (:attr:`CanonicalScope.PER_SOURCE`).
        cc_max_iterations: Max iterations for connected components.
        max_parallelism: Worker count for the ZephyrContext.
        worker_resources: Per-worker resource request. Required when
            ``map_task_resources`` is set.
        coordinator_resources: Coordinator resource request.
        map_task_resources: ResourceConfig for map-stage tasks.
        reduce_task_resources: ResourceConfig for reduce-stage tasks (e.g.
            the per-shard ``group_by`` writer).

    Returns:
        :class:`FuzzyDupsAttrData` describing per-source attr directories,
        the shared MinHash params, and aggregated counters.

    Canonical selection is deterministic (the min content-hash per component) and
    reproducible across executor counts: ``connected_components`` sorts each LSH
    bucket by ``id_norm`` so the graph topology does not depend on shuffle order.
    If CC does not converge within ``cc_max_iterations`` the result is still
    deterministic but *incomplete* (some near-dup clusters stay split); a warning
    is logged and the caller can raise ``cc_max_iterations`` for complete dedup.

    Raises:
        ValueError: If inputs is empty or input params disagree.
        FileNotFoundError: If any input ``attr_dir`` is missing parquet shards.
    """
    params = _validate_inputs(inputs)
    entries, source_tag = _build_shard_index(inputs)

    logger.info(
        "Computing fuzzy dups for %d inputs (%d total shards) → %s, params=%s",
        len(inputs),
        len(entries),
        output_path,
        params,
    )

    ctx_kwargs: dict = {
        "name": "fuzzy-dups",
        "max_workers": max_parallelism,
        "resources": worker_resources or ResourceConfig(cpu=1, ram="32g", disk="5g"),
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    if map_task_resources is not None:
        ctx_kwargs["map_task_resources"] = map_task_resources
    if reduce_task_resources is not None:
        ctx_kwargs["reduce_task_resources"] = reduce_task_resources
    ctx = ZephyrContext(**ctx_kwargs)

    # Cap shard count at max_parallelism. Each group reads its attr files
    # sequentially and emits bucket records; file_idx is preserved on the entry
    # itself, not by enumeration order, so grouping is safe.
    n_groups = min(max_parallelism, len(entries))
    entry_groups: list[list[dict[str, Any]]] = [[] for _ in range(n_groups)]
    for i, entry in enumerate(entries):
        entry_groups[i % n_groups].append(entry)

    bucket_ds = Dataset.from_list(entry_groups).flat_map(_emit_bucket_records)
    converged, cc_files = connected_components(
        bucket_ds,
        ctx,
        output_dir=f"{output_path}/metadata/cc",
        max_iterations=cc_max_iterations,
        resume=cc_resume,
    )
    if not converged:
        # A non-converged CC is still deterministic and reproducible across
        # runs/executor counts (the bucket group_by sorts by id_norm, pinning the
        # graph topology -- see connected_components), but it is *incomplete*: some
        # true duplicate-clusters remain split across several component_ids, each
        # keeping its own local-min canonical, so a few extra near-dups survive.
        # Warn rather than fail -- callers that cap iterations get a stable,
        # under-deduped result; raise cc_max_iterations for complete dedup (see
        # marin#6798).
        logger.warning(
            "Connected components did not converge within cc_max_iterations=%d; dedup is deterministic but "
            "incomplete (some near-dup clusters remain split). Raise cc_max_iterations for complete dedup.",
            cc_max_iterations,
        )

    ctx.put(_SHARED_ENTRIES_KEY, entries)
    aggregator = _make_per_shard_writer(output_path, counter_prefix="dedup/fuzzy/document")

    members = Dataset.from_list(cc_files).load_parquet().map(_to_cc_member)
    if canonical_scope is CanonicalScope.GLOBAL:
        # CC's Hash-to-Min guarantees component_id == min(id_norm) across a
        # cluster, so `component_id == id_norm` cheaply identifies the single
        # global canonical — no extra shuffle.
        labeled = members.map(lambda m: {**m, "is_canonical": m["component_id"] == m["id_norm"]})
    else:
        # Per-source canonical: regroup members by (source_tag, component_id)
        # sorted by id_norm and flag the per-source minimum. One extra shuffle
        # over the cluster members (marginal vs CC's iterations), gated behind
        # PER_SOURCE so global callers keep the single-shuffle fast path.
        labeled = members.group_by(
            lambda m: (m["source_tag"], m["component_id"]),
            sort_by=lambda m: m["id_norm"],
            reducer=_mark_source_canonicals,
        )

    # Co-partition annotations with their source shard for the writer. Rows are
    # sorted by id so the per-shard attr parquet mirrors its NormalizedData shard.
    shard_pipeline = labeled.group_by(
        lambda r: r["file_idx"],
        sort_by=lambda r: r["id"],
        reducer=aggregator,
    )

    outcome = ctx.execute(shard_pipeline, verbose=True)
    shard_results = outcome.results

    # Aggregate per-source counters across shards for the final artifact.
    sources: dict[str, FuzzyDupsPerSource] = {
        src_dir: FuzzyDupsPerSource(attr_dir=f"{output_path}/outputs/{tag}") for src_dir, tag in source_tag.items()
    }

    cluster_members = sum(r["cluster_members"] for r in shard_results)
    # One canonical per cluster (GLOBAL) or per (source, cluster) (PER_SOURCE).
    canonicals = sum(r["canonicals"] for r in shard_results)
    logger.info(
        "Fuzzy dups (%s): %d cluster members, %d canonical survivors (non-canonicals to drop by default: %d)",
        canonical_scope.value,
        cluster_members,
        canonicals,
        cluster_members - canonicals,
    )

    return FuzzyDupsAttrData(
        params=params,
        sources=sources,
        counters=dict(outcome.counters),
    )


def compute_fuzzy_dups_attrs_step(
    *,
    name: str,
    minhash_steps: list[StepSpec],
    canonical_scope: CanonicalScope,
    cc_max_iterations: int = 10,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that computes fuzzy duplicate attrs from ``MinHashAttrData`` step outputs."""
    return StepSpec(
        name=name,
        deps=list(minhash_steps),
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(s.output_path, MinHashAttrData) for s in minhash_steps],
            output_path=output_path,
            canonical_scope=canonical_scope,
            cc_max_iterations=cc_max_iterations,
            max_parallelism=max_parallelism,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
        ),
        hash_attrs={"cc_max_iterations": cc_max_iterations, "canonical_scope": canonical_scope.value},
        override_output_path=override_output_path,
    )
