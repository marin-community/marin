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
from collections.abc import Iterator

from fray import ResourceConfig
from pydantic import BaseModel
from zephyr import Dataset, ZephyrContext, counters, write_parquet_file

from marin.datakit import partition_filename
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


def _assign_source_tags(inputs: list[MinHashAttrData]) -> dict[str, str]:
    """Assign a deterministic ``source_NNN`` tag per input ``source_main_dir``.

    Sorting by ``source_main_dir`` keeps tags stable regardless of the order
    callers happen to pass inputs in. Returns ``{source_main_dir: source_tag}``.
    """
    source_tag: dict[str, str] = {}
    for new_idx, m in enumerate(sorted(inputs, key=lambda m: m.source_main_dir)):
        source_tag[m.source_main_dir] = f"source_{new_idx:03d}"
    return source_tag


def _list_attr_files(inputs: list[MinHashAttrData], source_tag: dict[str, str]) -> list[dict[str, str]]:
    """Return ``[{attr_path, source_tag}, ...]`` for every attr shard across *inputs*."""
    files: list[dict[str, str]] = []
    for m in inputs:
        attr_shards = sorted(fsspec_glob(f"{m.attr_dir.rstrip('/')}/*.parquet"))
        if not attr_shards:
            raise FileNotFoundError(f"No attr parquet shards under {m.attr_dir}")
        for attr_path in attr_shards:
            files.append({"attr_path": attr_path, "source_tag": source_tag[m.source_main_dir]})
    return files


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
    attr files. The prefix is recovered in :func:`_split_cc_id` before the
    final attr parquet is written.
    """
    return f"{source_tag}{_CC_ID_SEP}{doc_id}"


def _split_cc_id(record_id: str) -> tuple[str, str]:
    """Reverse :func:`_cc_id`, returning ``(source_tag, doc_id)``."""
    source_tag, doc_id = record_id.split(_CC_ID_SEP, 1)
    return source_tag, doc_id


def _emit_bucket_records(group: list[dict[str, str]]) -> Iterator[dict]:
    """For each (bucket, id) pair across all attr shards in *group*, emit a routing record.

    ``CCInput.file_idx`` is an opaque routing payload: we stuff each row's
    source-stamped ``partition_id`` (read directly from the attr parquet
    column) into it so the post-CC writer can reconstruct co-partitioned
    output filenames without re-globbing the source tree.
    """
    for entry in group:
        for batch in _load_batches(entry["attr_path"], columns=["id", "partition_id", "buckets"]):
            ids = batch["id"]
            partition_ids = batch["partition_id"]
            buckets_col = batch["buckets"]
            for doc_id, partition_id, doc_buckets in zip(ids, partition_ids, buckets_col, strict=True):
                if not doc_buckets.is_valid:
                    continue
                cc_id = _cc_id(entry["source_tag"], doc_id.as_py())
                p_id = partition_id.as_py()
                for b in doc_buckets.as_py():
                    yield {"bucket": str(b), "id": cc_id, "file_idx": p_id}


def _make_per_shard_writer(
    output_path: str,
    source_tag_to_num_partitions: dict[str, int],
    counter_prefix: str,
):
    """Return a group_by reducer that writes per-shard cluster-annotation parquet files.

    Group key is ``(source_tag, partition_id)``: the writer reconstructs the
    output filename via ``partition_filename`` so attr files stay in lockstep
    with the source's ``part-NNNNN-of-MMMMM.parquet`` naming and consolidate's
    filename-rebase join keeps finding them. Skips singletons entirely. For
    every non-singleton cluster member, writes
    ``{id, attributes: {dup_cluster_id, is_cluster_canonical}}``. Rows are
    already sorted by ``id`` thanks to the upstream ``group_by(sort_by=id)``.
    """

    def aggregate(group_key: tuple[str, int], records: Iterator[dict]) -> dict:
        source_tag, partition_id = group_key
        num_partitions = source_tag_to_num_partitions[source_tag]
        out_path = f"{output_path}/outputs/{source_tag}/{partition_filename(partition_id, num_partitions)}"

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
            "source_tag": source_tag,
            "partition_id": partition_id,
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
    source_tag = _assign_source_tags(inputs)
    attr_files = _list_attr_files(inputs, source_tag)
    source_tag_to_num_partitions: dict[str, int] = {source_tag[m.source_main_dir]: m.num_partitions for m in inputs}

    logger.info(
        "Computing fuzzy dups for %d inputs (%d total shards) → %s, params=%s",
        len(inputs),
        len(attr_files),
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

    # Cap fan-out at max_parallelism. Each group reads its attr files
    # sequentially and emits bucket records; partition_id rides on each row
    # in the parquet itself, so chunking by file order is safe.
    n_groups = min(max_parallelism, len(attr_files))
    file_groups: list[list[dict[str, str]]] = [[] for _ in range(n_groups)]
    for i, af in enumerate(attr_files):
        file_groups[i % n_groups].append(af)

    bucket_ds = Dataset.from_list(file_groups).flat_map(_emit_bucket_records)
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

    aggregator = _make_per_shard_writer(
        output_path,
        source_tag_to_num_partitions,
        counter_prefix="dedup/fuzzy/document",
    )

    # CC's Hash-to-Min guarantees component_id == min(id_norm) across a cluster,
    # so `component_id == id_norm` cheaply identifies the natural canonical.
    # `preserve_singletons=True` wires singletons as self-links, so a node is a
    # singleton iff its adjacency_list is exactly [id_norm] — no cluster peers.
    def _enrich(r: dict) -> dict:
        source_tag_str, doc_id = _split_cc_id(r["record_id"])
        return {
            "id": doc_id,
            "source_tag": source_tag_str,
            # CC's file_idx slot carried partition_id through the shuffle.
            "partition_id": r["file_idx"],
            "component_id": r["component_id"],
            "is_canonical": r["component_id"] == r["id_norm"],
            "is_singleton": len(r["adjacency_list"]) == 1 and r["adjacency_list"][0] == r["id_norm"],
        }

    shard_pipeline = (
        Dataset.from_list(cc_files)
        .load_parquet()
        .map(_enrich)
        .group_by(
            lambda r: (r["source_tag"], r["partition_id"]),
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
