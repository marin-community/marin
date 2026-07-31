# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute fuzzy duplicate markers from one or more ``MinHashAttrData`` inputs.

Loads MinHash bucket attrs from each input, runs LSH-graph connected
components globally across all inputs, and writes per-source attribute trees
that identify every non-singleton cluster member. The full-text verification
job consumes this candidate artifact.

Per-document attr rows have schema::

    {
      id: str,
      dup_cluster_id: str,         # CC component id — shared by all cluster members
      is_cluster_canonical: bool,  # True for exactly one member per cluster
    }

Rows are emitted for every member of a non-singleton cluster. Singletons get
no row. The ``is_cluster_canonical`` field records the connected-components
canonical for cluster diagnostics. It is not a verified duplicate decision.

Combining multiple ``MinHashAttrData`` inputs is the foundation for iterative
global dedup: re-running this job over the union of all per-dataset MinHash
artifacts produces fresh markers without re-reading any source text.
"""

import logging
from collections.abc import Iterator

from fray.types import ResourceConfig
from pydantic import BaseModel
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_IRIS_WORKER_REPLICAS, ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.datakit.copartitioned import (
    CopartitionedShard,
    CopartitionedSource,
    build_copartitioned_shards,
    write_copartitioned_source_manifest,
)
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.connected_components import connected_components
from marin.processing.classification.deduplication.dedup_commons import _load_batches
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams

logger = logging.getLogger(__name__)
FUZZY_DUPS_ATTR_DATA_VERSION = 4
DEFAULT_CC_MAX_ITERATIONS = 10


class FuzzyDupsPerSource(BaseModel):
    """Per-source output entry inside :class:`FuzzyDupsAttrData`.

    Attributes:
        attr_dir: Directory containing per-shard duplicate marker Parquet
            files. Filenames mirror the source's MinHash attr (and thus its
            normalized) shards.
    """

    attr_dir: DatakitArtifactPath


class FuzzyDupsAttrData(BaseModel):
    """Co-partitioned fuzzy-duplicate marker attrs for one or more sources.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, FuzzyDupsAttrData)``.

    Attributes:
        version: Schema version of this artifact.
        params: MinHash params; equal to every input's params.
        sources: Mapping from each input's ``MinHashAttrData.source_key``
            to its per-source attr output entry.
        counters: Aggregated zephyr counters across all sources.
    """

    version: str = f"v{FUZZY_DUPS_ATTR_DATA_VERSION}"
    params: MinHashParams
    sources: dict[str, FuzzyDupsPerSource]
    counters: dict[str, int | float]

    def attr_dir_for_source(self, source_path: str) -> str:
        """Return the attribute directory for a materialized source path."""
        source_key = datakit_source_key(source_path)
        entry = self.sources.get(source_key)
        if entry is None:
            raise KeyError(f"Fuzzy duplicate attributes have no entry for source_key={source_key!r}")
        return entry.attr_dir


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
        if m.source_key in seen:
            raise ValueError(
                f"Duplicate source_key in inputs: inputs[{seen[m.source_key]}] and "
                f"inputs[{i}] both point to {m.source_key!r}. Each source must be "
                "represented at most once so its output attr tree is unambiguous."
            )
        seen[m.source_key] = i

    return head


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


def _emit_bucket_records(entries: list[CopartitionedShard]) -> Iterator[dict]:
    """For each (bucket, id) pair across all attr shards in *entries*, emit a routing record."""
    for entry in entries:
        for batch in _load_batches(entry.input_path, columns=["id", "buckets"]):
            ids = batch["id"]
            buckets_col = batch["buckets"]
            for doc_id, doc_buckets in zip(ids, buckets_col, strict=True):
                if not doc_buckets.is_valid:
                    continue
                cc_id = _cc_id(entry.source_tag, doc_id.as_py())
                for b in doc_buckets.as_py():
                    yield {"bucket": str(b), "id": cc_id, "file_idx": entry.file_idx}


# Key under which ``entries`` is staged via ``ZephyrContext.put`` for the
# stage1 reducer. Holding the list in the closure would serialize it into
# every ``pull_task`` RPC pickle (one per dispatched reduce task) — at
# ~100k entries this OOMs the coord under the high-fan-out dispatch that
# kicks in when ``max_parallelism`` is large. Pulling it from shared data
# instead means each worker fetches and caches the list once.
_SHARED_ENTRIES_KEY = "fuzzy_dups_entries"


def _make_per_shard_writer(counter_prefix: str):
    """Return a group_by reducer that writes per-shard cluster-annotation parquet files.

    Skips singletons entirely. For every non-singleton cluster member, writes
    ``{id, dup_cluster_id, is_cluster_canonical}``. Rows are
    already sorted by ``id`` thanks to the upstream ``group_by(sort_by=id)``.

    The ``entries`` list is loaded via ``zephyr_worker_ctx().get_shared`` so it
    is shipped to workers once (via Zephyr shared-data) rather than captured
    in this closure and re-pickled per ``pull_task`` RPC.
    """

    def aggregate(file_idx: int, records: Iterator[dict]) -> dict:
        entries: list[CopartitionedShard] = zephyr_worker_ctx().get_shared(_SHARED_ENTRIES_KEY)
        entry = entries[file_idx]

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
                    "dup_cluster_id": record["component_id"],
                    "is_cluster_canonical": record["is_canonical"],
                }

        result = write_parquet_file(cluster_member_rows(), entry.output_path)
        return {
            **result,
            "file_idx": file_idx,
            "source_tag": entry.source_tag,
            "cluster_members": cluster_members,
            "canonicals": canonicals,
        }

    return aggregate


def compute_fuzzy_dups_attrs(
    *,
    inputs: list[MinHashAttrData],
    output_path: str,
    cc_max_iterations: int = DEFAULT_CC_MAX_ITERATIONS,
    cc_resume: bool = False,
    max_parallelism: int = MAX_IRIS_WORKER_REPLICAS,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    zephyr_context: ZephyrContext | None = None,
) -> FuzzyDupsAttrData:
    """Mark fuzzy-duplicate cluster membership across one or more ``MinHashAttrData`` inputs.

    All inputs must share identical :class:`MinHashParams`. The job builds a
    global LSH bucket graph across every input shard, runs connected
    components, and emits a per-source attribute tree under
    ``<output_path>/outputs/source_NNN/`` with one parquet file per source
    shard (filenames preserved from the source). Each row annotates one
    cluster member with ``{id: str, dup_cluster_id: str,
    is_cluster_canonical: bool}``; singletons are omitted.

    Exactly one member per cluster has ``is_cluster_canonical=True`` — the
    one CC's Hash-to-Min picked as the natural canonical (min ``id_norm``).
    The full-text verifier selects its own deterministic representative. It
    does not use this candidate-only canonical flag as a duplicate decision.

    Args:
        inputs: ``MinHashAttrData`` artifacts to fuzzy-dedup together.
        output_path: Output root. Per-source attr trees land under
            ``<output_path>/outputs/source_NNN/``.
        cc_max_iterations: Max iterations for connected components.
        max_parallelism: Worker count for the ZephyrContext.
        worker_resources: Per-worker resource request. Required when
            ``map_task_resources`` is set.
        coordinator_resources: Coordinator resource request.
        map_task_resources: ResourceConfig for map-stage tasks.
        reduce_task_resources: ResourceConfig for reduce-stage tasks (e.g.
            the per-shard ``group_by`` writer).
        zephyr_context: Optional shared Zephyr context.

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
    sources = [CopartitionedSource(source_key=minhash.source_key, input_dir=minhash.attr_dir) for minhash in inputs]
    entries, attr_dirs = build_copartitioned_shards(
        sources=sources,
        output_path=output_path,
    )

    logger.info(
        "Computing fuzzy dups for %d inputs (%d total shards) → %s, params=%s",
        len(inputs),
        len(entries),
        output_path,
        params,
    )

    resources = worker_resources or ResourceConfig(cpu=1, ram="32g", disk="5g")
    ctx_kwargs: dict = {
        "name": "fuzzy-dups",
        "max_workers": max_parallelism,
        "resources": resources,
    }
    if coordinator_resources is not None:
        ctx_kwargs["coordinator_resources"] = coordinator_resources
    ctx = zephyr_context or ZephyrContext(**ctx_kwargs)
    map_resources = map_task_resources or resources

    # Cap shard count at max_parallelism. Each group reads its attr files
    # sequentially and emits bucket records; file_idx is preserved on the entry
    # itself, not by enumeration order, so grouping is safe.
    n_groups = min(max_parallelism, len(entries))
    entry_groups: list[list[CopartitionedShard]] = [[] for _ in range(n_groups)]
    for i, entry in enumerate(entries):
        entry_groups[i % n_groups].append(entry)

    bucket_ds = Dataset.from_list(entry_groups).flat_map(_emit_bucket_records)
    converged, cc_files = connected_components(
        bucket_ds,
        ctx,
        output_dir=f"{output_path}/metadata/cc",
        max_iterations=cc_max_iterations,
        resume=cc_resume,
        num_reduce_shards=max_parallelism,
        map_task_resources=map_resources,
        reduce_task_resources=reduce_task_resources,
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
    aggregator = _make_per_shard_writer(counter_prefix="dedup/fuzzy/document")

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

    outcome = ctx.execute(
        shard_pipeline,
        verbose=True,
        map_task_resources=map_resources,
        reduce_task_resources=reduce_task_resources,
    )
    shard_results = outcome.results
    write_copartitioned_source_manifest(output_path=output_path, attr_dirs=attr_dirs)

    # Aggregate per-source counters across shards for the final artifact.
    sources = {source_key: FuzzyDupsPerSource(attr_dir=attr_dir) for source_key, attr_dir in attr_dirs.items()}

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
    cc_max_iterations: int = DEFAULT_CC_MAX_ITERATIONS,
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
            cc_max_iterations=cc_max_iterations,
            max_parallelism=max_parallelism,
            worker_resources=worker_resources,
            coordinator_resources=coordinator_resources,
        ),
        # Match the identity the Datakit DAG builds, so a step created here
        # resolves to the artifacts that graph already produced. The MinHash
        # content parameters reach this hash through the dependency IDs.
        hash_attrs={
            "v": FUZZY_DUPS_ATTR_DATA_VERSION,
            **({"cc_max_iterations": cc_max_iterations} if cc_max_iterations != DEFAULT_CC_MAX_ITERATIONS else {}),
        },
        override_output_path=override_output_path,
    )
