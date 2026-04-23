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
        dup_cluster_id: str,         # CC component id — shared by all cluster members
        is_cluster_canonical: bool,  # True for exactly one member per cluster
      }
    }

Rows are emitted for every member of a non-singleton cluster (canonical +
non-canonicals). Singletons get no row, preserving the
``consolidate(..., keep_if_missing=True)`` pattern. This shape lets the
canonical-selection policy live in consolidate (e.g. the default
``keep is_cluster_canonical=True``, or any custom per-cluster reducer) rather
than being baked in here.

Combining multiple ``MinHashAttrData`` inputs is the foundation for iterative
global dedup: re-running this job over the union of all per-dataset MinHash
artifacts produces fresh markers without re-reading any source text.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from fray.v2 import ResourceConfig
from pydantic import BaseModel
from zephyr import Dataset, ZephyrContext, counters, write_parquet_file

from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.connected_components import connected_components
from marin.processing.classification.deduplication.dedup_commons import _load_batches
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


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
    ``Artifact.load(step, FuzzyDupsAttrData)``.

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
    counters: dict[str, int]


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
        attr_shards = sorted(fsspec_glob(f"{m.attr_dir.rstrip('/')}/*.parquet"))
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
    attr files. The prefix is stripped in :func:`_strip_cc_prefix` before
    the final attr parquet is written.
    """
    return f"{source_tag}{_CC_ID_SEP}{doc_id}"


def _strip_cc_prefix(record_id: str) -> str:
    """Reverse :func:`_cc_id`, returning the original ``doc_id``."""
    return record_id.split(_CC_ID_SEP, 1)[1]


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


def _make_per_shard_writer(output_path: str, entries: list[dict[str, Any]], counter_prefix: str):
    """Return a group_by reducer that writes per-shard cluster-annotation parquet files.

    Skips singletons entirely. For every non-singleton cluster member, writes
    ``{id, attributes: {dup_cluster_id, is_cluster_canonical}}``. Rows are
    already sorted by ``id`` thanks to the upstream ``group_by(sort_by=id)``.
    """

    def aggregate(file_idx: int, records: Iterator[dict]) -> dict:
        entry = entries[file_idx]
        out_path = f"{output_path}/outputs/{entry['source_tag']}/{entry['basename']}"

        cluster_members = 0
        canonicals = 0

        def cluster_member_rows():
            nonlocal cluster_members, canonicals
            for record in records:
                if record["is_singleton"]:
                    counters.increment(f"{counter_prefix}/singletons_skipped")
                    continue
                cluster_members += 1
                counters.increment(f"{counter_prefix}/cluster_members")
                if record["is_canonical"]:
                    canonicals += 1
                    counters.increment(f"{counter_prefix}/canonicals")
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


def compute_fuzzy_dups_attrs(
    *,
    inputs: list[MinHashAttrData],
    output_path: str,
    cc_max_iterations: int = 10,
    cc_resume: bool = False,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
) -> FuzzyDupsAttrData:
    """Mark fuzzy-duplicate cluster membership across one or more ``MinHashAttrData`` inputs.

    All inputs must share identical :class:`MinHashParams`. The job builds a
    global LSH bucket graph across every input shard, runs connected
    components, and emits a per-source attribute tree under
    ``<output_path>/outputs/source_NNN/`` with one parquet file per source
    shard (filenames preserved from the source). Each row annotates one
    cluster member with ``{id: str, attributes: {dup_cluster_id: str,
    is_cluster_canonical: bool}}``; singletons are omitted.

    Exactly one member per cluster has ``is_cluster_canonical=True`` — the
    one CC's Hash-to-Min picked as the natural canonical (min ``id_norm``).
    Consolidate may honor that flag (default policy) or ignore it and apply
    a custom per-``dup_cluster_id`` policy.

    Args:
        inputs: ``MinHashAttrData`` artifacts to fuzzy-dedup together.
        output_path: Output root. Per-source attr trees land under
            ``<output_path>/outputs/source_NNN/``.
        cc_max_iterations: Max iterations for connected components.
        max_parallelism: Worker count for the ZephyrContext.
        worker_resources: Per-worker resource request.
        coordinator_resources: Coordinator resource request.

    Returns:
        :class:`FuzzyDupsAttrData` describing per-source attr directories,
        the shared MinHash params, and aggregated counters.

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
        # TODO (rav): log the number of changed nodes?
        logger.warning("Connected components did not converge")

    aggregator = _make_per_shard_writer(output_path, entries, counter_prefix="dedup/fuzzy/document")

    # CC's Hash-to-Min guarantees component_id == min(id_norm) across a cluster,
    # so `component_id == id_norm` cheaply identifies the natural canonical.
    # `preserve_singletons=True` wires singletons as self-links, so a node is a
    # singleton iff its adjacency_list is exactly [id_norm] — no cluster peers.
    shard_pipeline = (
        Dataset.from_list(cc_files)
        .load_parquet()
        .map(
            lambda r: {
                "id": _strip_cc_prefix(r["record_id"]),
                "component_id": r["component_id"],
                "is_canonical": r["component_id"] == r["id_norm"],
                "is_singleton": len(r["adjacency_list"]) == 1 and r["adjacency_list"][0] == r["id_norm"],
                "file_idx": r["file_idx"],
            }
        )
        .group_by(
            lambda r: r["file_idx"],
            sort_by=lambda r: r["id"],
            reducer=aggregator,
        )
    )

    outcome = ctx.execute(shard_pipeline, verbose=True)
    shard_results = outcome.results

    # Aggregate per-source counters across shards for the final artifact.
    sources: dict[str, FuzzyDupsPerSource] = {
        src_dir: FuzzyDupsPerSource(attr_dir=f"{output_path}/outputs/{tag}") for src_dir, tag in source_tag.items()
    }

    cluster_members = sum(r["cluster_members"] for r in shard_results)
    clusters = sum(r["canonicals"] for r in shard_results)  # one canonical per cluster
    logger.info(
        "Fuzzy dups: %d cluster members across %d clusters (non-canonicals to drop by default: %d)",
        cluster_members,
        clusters,
        cluster_members - clusters,
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
            inputs=[Artifact.load(s, MinHashAttrData) for s in minhash_steps],
            output_path=output_path,
            cc_max_iterations=cc_max_iterations,
            max_parallelism=max_parallelism,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
        ),
        hash_attrs={"cc_max_iterations": cc_max_iterations},
        override_output_path=override_output_path,
    )
