# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator, Sequence
from typing import Any, TypedDict

import dupekit
from zephyr import Dataset, ZephyrContext
from zephyr.expr import col

logger = logging.getLogger(__name__)


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
    to string to make internal zephyr/ray/pyarrow serde happy.
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

    def _reduce_bucket_to_links(bucket: str, items: Iterator[CCInput]) -> Iterator[dict]:
        """Generator reducer: dedup items by id_norm, then yield pairwise links."""
        seen: dict[str, RecordId] = {}
        for item in items:
            norm = _internal_orderable_id(item["id"])
            if norm in seen:
                existing = seen[norm]
                if existing["file_idx"] != item["file_idx"]:
                    logger.warning(
                        "Document %s appears in multiple files (file_idx %d and %d) within bucket %s",
                        item["id"],
                        existing["file_idx"],
                        item["file_idx"],
                        bucket,
                    )
            else:
                seen[norm] = RecordId(
                    record_id=item["id"],
                    id_norm=norm,
                    file_idx=item["file_idx"],
                )

        ids = list(seen.values())

        if preserve_singletons and len(ids) == 1:
            yield _make_link(ids[0], ids[0])
            return

        # Use a star topology: pick the node with the smallest id_norm
        # as the hub and link all others to it. This is O(n) instead of O(n²)
        # and produces the same connected components via hash-to-min propagation.
        hub = min(ids, key=lambda x: x["id_norm"])
        for node in ids:
            if node["id_norm"] == hub["id_norm"]:
                continue
            yield _make_link(hub, node)
            yield _make_link(node, hub)

    def _dedup_combiner(bucket: str, items: Iterator[CCInput]) -> Iterator[CCInput]:
        """Local pre-aggregation: deduplicate items by record_id within each scatter buffer."""
        seen: set[str] = set()
        for item in items:
            norm = _internal_orderable_id(item["id"])
            if norm not in seen:
                seen.add(norm)
                yield item

    curr_it = ctx.execute(
        ds
        # Group nodes in buckets, deduplicate, and emit pairwise links
        .group_by(
            lambda x: x["bucket"],
            reducer=_reduce_bucket_to_links,
            combiner=_dedup_combiner,
        )
        # Construct Node state, init with:
        #  * each node is its own component
        #  * adjacency list from links
        .group_by(
            lambda x: x["source_id_norm"],
            reducer=_build_adjacency,
        ).write_vortex(f"{output_dir}/it_0/part-{{shard:05d}}.vortex"),
        verbose=True,
    )

    converged = False
    for i in range(1, max_iterations + 1):  # type: ignore[bad-assignment]
        logger.info(f"Connected components iteration {i}...")
        curr_it = ctx.execute(
            Dataset.from_list(curr_it)
            .load_vortex()
            .map(lambda record: CCNode(**record))
            .flat_map(_emit_messages)
            .group_by(key=lambda x: x["key"], reducer=_reduce_node_step)
            .write_vortex(f"{output_dir}/it_{i}/part-{{shard:05d}}.vortex"),
            verbose=True,
        )

        # Check for convergence
        changes = ctx.execute(
            Dataset.from_list(curr_it).load_vortex(columns=["changed"]).filter(col("changed")).count(),
        )

        num_changes = changes[0]

        if num_changes == 0:
            converged = True
            logger.info(f"Connected components converged after {i} iterations.")
            break
        else:
            logger.info(f"Connected components iteration {i} found {num_changes} changes.")

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
