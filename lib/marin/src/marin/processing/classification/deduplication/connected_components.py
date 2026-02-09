# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, TypedDict

import dupekit
from zephyr import Dataset, ZephyrContext
from zephyr.expr import col

logger = logging.getLogger(__name__)


# TODO (rav): can we have just a single id that's expected to be clean on the inputs?
class RecordId(TypedDict):
    record_id: Any
    record_id_norm: str


# TODO: dataclasses are not writable to parquet ATM, is there something better than TypedDict that works out of the box?
class CCNode(TypedDict):
    node_id: RecordId
    adjacency_list: list[str]
    component_id: str
    changed: bool


@dataclass
class CCMessage: ...


@dataclass
class SelfMessage(CCMessage):
    node: CCNode


@dataclass
class CompMessage(CCMessage):
    component_id: str


class CCInput(TypedDict):
    bucket: str
    id: Any


def _internal_orderable_id(record_id: Any) -> str:
    """
    We need an id that has a total ordering for connected components. If the id is an int,
    we can use it as is. If it's a string, we hash it and convert it to string. We convert
    to string to make internal zephyr/ray/pyarrow serde happy.
    """
    if isinstance(record_id, int):
        record_id_norm = str(record_id)
    elif isinstance(record_id, str):
        record_id_norm = str(dupekit.hash_xxh3_128(record_id.encode()))
    else:
        raise ValueError(f"Unsupported id type: {type(record_id)}")
    return record_id_norm


class BucketWithIds(TypedDict):
    bucket: str
    ids: list[RecordId]


def connected_components(
    ds: Dataset[CCInput],
    ctx: ZephyrContext,
    *,
    output_dir: str,
    max_iterations: int = 10,
    preserve_singletons: bool = True,
) -> tuple[bool, Sequence[str]]:
    """
    Connected Components implementation using Zephyr Dataset API and Hash-to-Min algorithm (https://arxiv.org/abs/1203.5387)

    Args:
        ds: Input dataset containing 'bucket' and 'ids' fields, most likely from MinHash LSH output
        ctx: ZephyrContext to use for execution.
        output_dir: Directory to write intermediate and final output files
        max_iterations: Maximum number of iterations to run the connected components algorithm
        preserve_singletons: Whether to preserve single-node buckets in the output
    """
    curr_it = ctx.execute(
        ds
        # Group nodes in buckets
        .group_by(
            lambda x: x["bucket"],
            _reduce_buckets,
        )
        # Go from bucket -> links
        .flat_map(partial(_gen_links_within_buckets, preserve_singletons=preserve_singletons))
        # Construct Node state, init with:
        #  * each node is its own component
        #  * adjacency list from links
        .group_by(
            lambda x: x[0]["record_id_norm"],
            _build_adjacency,
        ).write_parquet(f"{output_dir}/it_0/part-{{shard:05d}}.parquet"),
        verbose=True,
    )

    converged = False
    for i in range(1, max_iterations + 1):  # type: ignore[bad-assignment]
        logger.info(f"Connected components iteration {i}...")
        curr_it = ctx.execute(
            Dataset.from_list(curr_it)
            .load_parquet()
            .map(lambda record: CCNode(**record))
            .flat_map(_emit_messages)
            .group_by(key=lambda x: x[0], reducer=_reduce_node_step)
            # NOTE: parquet built-in does not support list of int :/
            .write_parquet(f"{output_dir}/it_{i}/part-{{shard:05d}}.parquet"),
            verbose=True,
        )

        # Check for convergence
        changes = ctx.execute(
            Dataset.from_list(curr_it).load_parquet(columns=["changed"]).filter(col("changed")).count(),
        )

        num_changes = changes[0]

        if num_changes == 0:
            converged = True
            logger.info(f"Connected components converged after {i} iterations.")
            break
        else:
            logger.info(f"Connected components iteration {i} found {num_changes} changes.")

    return converged, curr_it


def _reduce_buckets(bucket: str, items: Iterator[CCInput]) -> BucketWithIds:
    # TODO: do we want/need this optimization?
    # if len(all_items) <= 1:
    #    return None  # No duplicates in this bucket
    return {
        "bucket": bucket,
        "ids": [RecordId(record_id=item["id"], record_id_norm=_internal_orderable_id(item["id"])) for item in items],
    }


def _gen_links_within_buckets(record: CCInput, *, preserve_singletons: bool) -> Iterator[tuple[RecordId, RecordId]]:
    ids = record.get("ids", [])

    norm_ids = [i["record_id_norm"] for i in ids]
    # TODO: this will materialize ids!
    if len(norm_ids) != len(set(norm_ids)):
        duplicate_ids = [x for x in norm_ids if norm_ids.count(x) > 1]
        raise ValueError(f"Duplicate found in bucket during link_reduce: {duplicate_ids}")

    if preserve_singletons and len(ids) == 1:
        yield (ids[0], ids[0])
        return

    # Emit all pairwise links
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] == ids[j]:
                continue

            yield (ids[i], ids[j])
            yield (ids[j], ids[i])


def _build_adjacency(node_id: RecordId, links: Iterator[tuple[RecordId, RecordId]]) -> CCNode:
    all_links = list(links)
    return CCNode(
        node_id=all_links[0][0],
        adjacency_list=list(set([link[1]["record_id_norm"] for link in all_links])),
        # init with own id as component
        component_id=node_id,
        changed=True,
    )


def _emit_messages(node: CCNode) -> Iterator[tuple[str, CCMessage]]:
    """
    1. Emit the node structure to itself (to preserve graph topology).
    2. Emit the current component ID to all neighbors.
    """
    # 1. Preserve structure
    yield (node["node_id"]["record_id_norm"], SelfMessage(node=node))

    # 2. Propagate component ID to neighbors
    # (Optimization: Only send if we changed recently, but strictly Hash-to-Min sends always)
    msg = CompMessage(component_id=node["component_id"])
    for neighbor_id in node["adjacency_list"]:
        yield (neighbor_id, msg)


def _reduce_node_step(key: str, incoming: Iterator[tuple[str, CCMessage]]) -> CCNode:
    """
    1. Recover NodeState.
    2. Find minimum component ID from messages.
    3. Update state if a smaller ID is found.
    """
    # NOTE: init the minimum component ID with the current key
    min_comp = key
    node_structure: CCNode | None = None

    # Iterate through mixed stream of structure and messages
    for _, msg in incoming:
        if isinstance(msg, SelfMessage):
            node_structure = msg.node
            if node_structure["component_id"] < min_comp:
                min_comp = node_structure["component_id"]
        else:
            assert isinstance(msg, CompMessage)
            remote_comp = msg.component_id
            if remote_comp < min_comp:
                min_comp = remote_comp

    if node_structure is None:
        # Should technically not happen if graph is well-formed
        raise ValueError(f"Lost/corrupted structure for node {key}")

    if min_comp < node_structure["component_id"]:
        node_structure["component_id"] = min_comp
        node_structure["changed"] = True
    else:
        node_structure["changed"] = False

    return node_structure
