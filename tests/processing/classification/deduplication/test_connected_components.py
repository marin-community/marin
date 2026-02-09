# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from marin.processing.classification.deduplication.connected_components import CCInput, connected_components
from zephyr import Dataset, ZephyrContext


def test_connected_components_happy_path(tmp_path):
    input_data: list[CCInput] = [
        {"bucket": "bucket_1", "id": "doc_1"},
        {"bucket": "bucket_1", "id": "doc_2"},
        {"bucket": "bucket_2", "id": "doc_2"},
        {"bucket": "bucket_2", "id": "doc_3"},
        {"bucket": "bucket_3", "id": "doc_4"},
    ]

    ds = Dataset.from_list(input_data)

    with ZephyrContext(name="test-cc") as ctx:
        converged, output_path = connected_components(ds, ctx, output_dir=tmp_path.as_posix(), max_iterations=5)
        assert converged
        results = ctx.execute(Dataset.from_list(output_path).load_parquet())
    assert len(results) == len(set(r["id"] for r in input_data))

    components = defaultdict(list)
    for r in results:
        components[r["component_id"]].append(r["node_id"]["record_id"])

    sorted_components = sorted(sorted(group) for group in components.values())
    assert sorted_components == [["doc_1", "doc_2", "doc_3"], ["doc_4"]]
