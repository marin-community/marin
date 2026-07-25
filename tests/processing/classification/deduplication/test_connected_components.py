# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re
from collections import defaultdict

from marin.processing.classification.deduplication.connected_components import CCInput, connected_components
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext


def test_connected_components_happy_path(tmp_path):
    input_data: list[CCInput] = [
        {"bucket": "bucket_1", "id": "doc_1", "file_idx": 0},
        {"bucket": "bucket_1", "id": "doc_2", "file_idx": 0},
        {"bucket": "bucket_2", "id": "doc_2", "file_idx": 0},
        {"bucket": "bucket_2", "id": "doc_3", "file_idx": 1},
        {"bucket": "bucket_3", "id": "doc_4", "file_idx": 1},
    ]

    ds = Dataset.from_list(input_data)

    ctx = ZephyrContext(name="test-cc")
    converged, output_path = connected_components(ds, ctx, output_dir=tmp_path.as_posix(), max_iterations=5)
    assert converged
    results = ctx.execute(Dataset.from_list(output_path).load_parquet()).results
    assert len(results) == len(set(r["id"] for r in input_data))

    components = defaultdict(list)
    for r in results:
        components[r["component_id"]].append(r["record_id"])

    sorted_components = sorted(sorted(group) for group in components.values())
    assert sorted_components == [["doc_1", "doc_2", "doc_3"], ["doc_4"]]


def test_connected_components_already_converged(tmp_path):
    """Single bucket with a single doc converges in one iteration (no changes)."""
    input_data: list[CCInput] = [
        {"bucket": "bucket_1", "id": "doc_1", "file_idx": 0},
    ]

    ds = Dataset.from_list(input_data)
    ctx = ZephyrContext(name="test-cc-single")
    converged, output_path = connected_components(ds, ctx, output_dir=tmp_path.as_posix(), max_iterations=5)
    assert converged

    results = ctx.execute(Dataset.from_list(output_path).load_parquet()).results
    assert len(results) == 1
    assert results[0]["record_id"] == "doc_1"


def test_connected_components_resume_at_cap_reports_converged(tmp_path):
    """Resuming a run that converged exactly at the iteration cap must still report converged.

    Regression for marin#6798: ``connected_components`` must not report a false
    non-convergence for a ``resume`` that finds all ``it_0..it_cap`` shards
    present (the non-convergence warning keys off the returned flag). A 4-node
    path graph propagates the min one hop per iteration, so it converges after a
    few iterations; resuming with the cap set exactly at that convergence
    iteration would otherwise skip the loop entirely.
    """
    # doc_0 - doc_1 - doc_2 - doc_3 path: consecutive docs share one bucket.
    input_data: list[CCInput] = [
        {"bucket": "b01", "id": "doc_0", "file_idx": 0},
        {"bucket": "b01", "id": "doc_1", "file_idx": 0},
        {"bucket": "b12", "id": "doc_1", "file_idx": 0},
        {"bucket": "b12", "id": "doc_2", "file_idx": 0},
        {"bucket": "b23", "id": "doc_2", "file_idx": 0},
        {"bucket": "b23", "id": "doc_3", "file_idx": 0},
    ]
    out_dir = tmp_path.as_posix()

    ctx = ZephyrContext(name="test-cc-resume-seed")
    converged, _ = connected_components(Dataset.from_list(input_data), ctx, output_dir=out_dir, max_iterations=10)
    assert converged
    # Highest completed iteration = the convergence iteration; resuming with
    # the cap set there makes the resume find every it_ shard already present.
    cap = max(int(re.search(r"it_(\d+)$", p).group(1)) for p in glob.glob(os.path.join(out_dir, "it_*")))
    assert cap >= 1, "path graph should take at least one propagation iteration"

    ctx_resume = ZephyrContext(name="test-cc-resume-run")
    converged_resume, output_path = connected_components(
        Dataset.from_list(input_data), ctx_resume, output_dir=out_dir, max_iterations=cap, resume=True
    )
    assert converged_resume, "resume of a run that converged at the cap must report converged=True"

    results = ctx_resume.execute(Dataset.from_list(output_path).load_parquet()).results
    components = defaultdict(list)
    for r in results:
        components[r["component_id"]].append(r["record_id"])
    assert len(components) == 1, "the path graph is a single connected component"


def _labels_at(input_data: list[CCInput], out_dir: str, *, max_iterations: int, max_workers: int) -> dict[str, str]:
    ctx = ZephyrContext(name=f"cc-exec-{max_workers}", max_workers=max_workers)
    _, paths = connected_components(
        Dataset.from_list(input_data), ctx, output_dir=out_dir, max_iterations=max_iterations
    )
    res = ctx.execute(Dataset.from_list(paths).load_parquet()).results
    return {r["record_id"]: r["component_id"] for r in res}


def test_connected_components_capped_run_is_executor_independent(tmp_path):
    """A capped (unconverged) run yields identical labels regardless of executor count (marin#6798).

    ``num_reduce_shards = ctx.max_workers``, so without a stable bucket order the
    star-vs-chain link topology -- and thus the K-hop labeling of an unconverged
    run -- depends on how many executors ran. The bucket ``group_by`` sorts by
    ``id_norm`` to pin the topology; this checks the labels match across executor
    counts even though the graph is deliberately capped below convergence.

    The graph is a long bridging chain (buckets of 4 nodes, consecutive buckets
    sharing a node) so convergence needs many iterations; ``max_iterations=2``
    leaves it unconverged.
    """
    data: list[CCInput] = []
    for b in range(24):
        members = [f"n_{b}_{k}" for k in range(4)]
        if b > 0:
            members[0] = f"n_{b - 1}_3"  # bridge consecutive buckets into one chain
        data.extend({"bucket": f"bk_{b}", "id": m, "file_idx": 0} for m in members)

    labels_2 = _labels_at(data, (tmp_path / "w2").as_posix(), max_iterations=2, max_workers=2)
    labels_8 = _labels_at(data, (tmp_path / "w8").as_posix(), max_iterations=2, max_workers=8)

    assert labels_2 == labels_8, "capped CC labels differ across executor counts (bucket order not pinned)"
