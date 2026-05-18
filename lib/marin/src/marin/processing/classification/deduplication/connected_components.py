# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator, Sequence
from typing import Any, TypedDict

import dupekit
from zephyr import Dataset, ShardInfo, ZephyrContext, counters, write_parquet_file

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


def _find_last_complete_iteration(
    output_dir: str, max_iterations: int, expected_parquets: int
) -> tuple[int, list[str]] | None:
    """Return (last_iteration, parquet_paths) from prior run outputs, or None if nothing reusable.

    A CC iteration ``it_N/`` is considered complete iff its parquet file count equals
    ``expected_parquets`` (= ``ctx.max_workers`` at write time). Iteration 0 uses the
    ``part-{shard:05d}.parquet`` naming; iterations 1+ use ``part-{shard:05d}-of-{total:05d}.parquet``.
    Both are detected by globbing ``it_N/*.parquet``.
    """
    last_complete = -1
    last_paths: list[str] = []
    for i in range(max_iterations + 1):
        paths = fsspec_glob(f"{output_dir}/it_{i}/*.parquet")
        if len(paths) != expected_parquets:
            break
        last_complete = i
        last_paths = paths
    if last_complete < 0:
        return None
    return last_complete, last_paths


# TODO (rav): can we have just a single id that's expected to be clean on the inputs?
class RecordId(TypedDict):
    record_id: Any
    id_norm: str
    file_idx: int


class CCNode(TypedDict):
    record_id: Any
    id_norm: str
    adjacency_list: list[str]
    component_id: str
    changed: bool
    file_idx: int


class CCInput(TypedDict):
    bucket: str
    id: Any
    file_idx: int


def _internal_orderable_id(record_id: Any) -> str:
    """
    We need an id that has a total ordering for connected components. If the id is an int,
    we can use it as is. If it's a string, we hash it and convert it to string. We convert
    to string to make internal zephyr/pyarrow serde happy.
    """
    if isinstance(record_id, int):
        id_norm = str(record_id)
    elif isinstance(record_id, str):
        id_norm = str(dupekit.hash_xxh3_128(record_id.encode()))
    else:
        raise ValueError(f"Unsupported id type: {type(record_id)}")
    return id_norm


def connected_components(
    ds: Dataset[CCInput],
    ctx: ZephyrContext,
    *,
    output_dir: str,
    max_iterations: int = 10,
    preserve_singletons: bool = True,
    resume: bool = False,
) -> tuple[bool, Sequence[str]]:
    """
    Connected Components implementation using Zephyr Dataset API and Hash-to-Min algorithm (https://arxiv.org/abs/1203.5387)

    Args:
        ds: Input dataset containing 'bucket' and 'ids' fields, most likely from MinHash LSH output
        ctx: ZephyrContext to use for execution.
        output_dir: Directory to write intermediate and final output files
        max_iterations: Maximum number of iterations to run the connected components algorithm
        preserve_singletons: Whether to preserve single-node buckets in the output
        resume: If True, skip iterations whose ``it_N/`` already contains a complete set of
            parquet files (count == ``ctx.max_workers``). Starts from the first incomplete
            iteration. If no complete prior state exists, runs from scratch.
    """

    def _reduce_bucket_to_links(bucket: str, items: Iterator[CCInput]) -> Iterator[dict]:
        """Generator reducer: dedup items by id_norm, yield star-topology links.

        Streams through items tracking only the current minimum hub RecordId (O(1))
        and a set of seen id_norms for dedup (O(n) strings). This is ~4x cheaper
        than the previous dict[str, RecordId] approach which stored a full RecordId
        per unique item. When a new minimum hub is found mid-stream, the old hub is
        linked to the new hub so all prior nodes remain transitively connected.
        """
        seen: set[str] = set()
        hub: RecordId | None = None
        num_unique = 0

        for item in items:
            norm = _internal_orderable_id(item["id"])
            if norm in seen:
                continue
            seen.add(norm)
            num_unique += 1

            record = RecordId(record_id=item["id"], id_norm=norm, file_idx=item["file_idx"])

            if hub is None:
                hub = record
            elif norm < hub["id_norm"]:
                yield _make_link(record, hub)
                yield _make_link(hub, record)
                counters.increment("cc/links", 2)
                hub = record
            else:
                yield _make_link(hub, record)
                yield _make_link(record, hub)
                counters.increment("cc/links", 2)

        if hub is None:
            return

        counters.increment("cc/buckets")
        counters.increment("cc/bucket_nodes", num_unique)

        if preserve_singletons and num_unique == 1:
            yield _make_link(hub, hub)

    def _dedup_combiner(bucket: str, items: Iterator[CCInput]) -> Iterator[CCInput]:
        """Local pre-aggregation: deduplicate items by record_id within each scatter buffer."""
        # TODO (rav): replace this with bloom filter? Reduce mem overhead.
        seen: set[str] = set()
        for item in items:
            norm = _internal_orderable_id(item["id"])
            if norm not in seen:
                seen.add(norm)
                yield item

    # Determine reduce shard count. Default to ctx max_workers to avoid
    # I/O amplification.
    num_reduce_shards = ctx.max_workers

    start_iteration = 1
    curr_it: Sequence[str]
    resumed = _find_last_complete_iteration(output_dir, max_iterations, num_reduce_shards) if resume else None
    if resumed is not None:
        last_it, last_paths = resumed
        logger.info("CC resume: skipping through it_%d (%d parquets present)", last_it, len(last_paths))
        curr_it = last_paths
        start_iteration = last_it + 1
    else:
        curr_it = ctx.execute(
            ds
            # Group nodes in buckets, deduplicate, and emit pairwise links
            .group_by(
                lambda x: x["bucket"],
                reducer=_reduce_bucket_to_links,
                combiner=_dedup_combiner,
                num_output_shards=num_reduce_shards,
            )
            # Construct Node state, init with:
            #  * each node is its own component
            #  * adjacency list from links
            .group_by(
                lambda x: x["source_id_norm"],
                reducer=_build_adjacency,
                num_output_shards=num_reduce_shards,
            ).write_parquet(f"{output_dir}/it_0/part-{{shard:05d}}.parquet"),
            verbose=True,
        ).results

    def _get_write_shard_and_count_fn(iteration: int):
        # NOTE: this function exists to make the iteration number closure capture explicit
        def _write_shard_and_count(nodes: Iterator[CCNode], shard_info: ShardInfo) -> Iterator[dict]:
            num_changes = 0

            def counting_iter():
                nonlocal num_changes
                for node in nodes:
                    counters.increment("cc/iteration_nodes")
                    if node["changed"]:
                        num_changes += 1
                        counters.increment("cc/changes")
                    yield node

            path = (
                f"{output_dir}/it_{iteration}/part-{shard_info.shard_idx:05d}-of-{shard_info.total_shards:05d}.parquet"
            )
            result = write_parquet_file(counting_iter(), path)
            yield {**result, "num_changes": num_changes}

        return _write_shard_and_count

    converged = False
    for i in range(start_iteration, max_iterations + 1):  # type: ignore[bad-assignment]
        logger.info(f"Connected components iteration {i}...")

        shard_results = ctx.execute(
            Dataset.from_list(curr_it)
            .load_parquet()
            .map(lambda record: CCNode(**record))
            .flat_map(_emit_messages)
            .group_by(key=lambda x: x["key"], reducer=_reduce_node_step, num_output_shards=num_reduce_shards)
            .map_shard(_get_write_shard_and_count_fn(i)),
            verbose=True,
        ).results

        curr_it = [r["path"] for r in shard_results]
        num_changes = sum(r["num_changes"] for r in shard_results)

        if num_changes == 0:
            converged = True
            logger.info(f"Connected components converged after {i} iterations.")
            break
        else:
            logger.info(f"Connected components iteration {i} found {num_changes:,} changes.")

    return converged, curr_it


def _make_link(source: RecordId, dest: RecordId) -> dict:
    return {
        "source_record_id": source["record_id"],
        "source_id_norm": source["id_norm"],
        "source_file_idx": source["file_idx"],
        "dest_id_norm": dest["id_norm"],
    }


def _build_adjacency(node_id: str, links: Iterator[dict]) -> CCNode:
    first = next(links)
    adj: set[str] = {first["dest_id_norm"]}
    for link in links:
        adj.add(link["dest_id_norm"])
    counters.increment("cc/nodes")
    return CCNode(
        record_id=first["source_record_id"],
        id_norm=first["source_id_norm"],
        adjacency_list=list(adj),
        component_id=node_id,
        changed=True,
        file_idx=first["source_file_idx"],
    )


def _emit_messages(node: CCNode) -> Iterator[dict]:
    """
    1. Emit the node structure to itself (to preserve graph topology).
    2. Emit the current component ID to all neighbors.
    """
    # 1. Preserve structure (self-message carries all node fields)
    yield {
        "key": node["id_norm"],
        "is_self": True,
        "record_id": node["record_id"],
        "id_norm": node["id_norm"],
        "adjacency_list": node["adjacency_list"],
        "component_id": node["component_id"],
        "changed": node["changed"],
        "file_idx": node["file_idx"],
    }

    # 2. Propagate component ID to neighbors
    # Use [""] instead of [] so Arrow infers list<string> consistently
    # with the self-message's adjacency_list, avoiding schema evolution.
    for neighbor_id in node["adjacency_list"]:
        yield {
            "key": neighbor_id,
            "is_self": False,
            "record_id": node["record_id"],
            "id_norm": "",
            "adjacency_list": [""],
            "component_id": node["component_id"],
            "changed": False,
            "file_idx": 0,
        }


def _reduce_node_step(key: str, incoming: Iterator[dict]) -> CCNode:
    """
    1. Recover NodeState from self-message.
    2. Find minimum component ID from all messages.
    3. Update state if a smaller ID is found.
    """
    # NOTE: init the minimum component ID with the current key
    min_comp = key
    node_structure: CCNode | None = None

    for msg in incoming:
        if msg["is_self"]:
            node_structure = CCNode(
                record_id=msg["record_id"],
                id_norm=msg["id_norm"],
                adjacency_list=msg["adjacency_list"],
                component_id=msg["component_id"],
                changed=msg["changed"],
                file_idx=msg["file_idx"],
            )
            if node_structure["component_id"] < min_comp:
                min_comp = node_structure["component_id"]
        else:
            if msg["component_id"] < min_comp:
                min_comp = msg["component_id"]

    if node_structure is None:
        raise ValueError(f"Lost/corrupted structure for node {key}")

    if min_comp < node_structure["component_id"]:
        node_structure["component_id"] = min_comp
        node_structure["changed"] = True
    else:
        node_structure["changed"] = False

    return node_structure
