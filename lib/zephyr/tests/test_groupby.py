# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for deduplicate and group_by operations."""

import hashlib

import pytest
from zephyr.context import create_backend_context
from zephyr import Backend, Dataset


@pytest.fixture(
    params=[
        pytest.param(create_backend_context("sync"), id="sync"),
        pytest.param(create_backend_context("threadpool", max_workers=2), id="thread"),
        pytest.param(create_backend_context("ray"), id="ray"),
    ]
)
def backend(request):
    """Parametrized fixture providing all backend types."""
    return request.param


@pytest.fixture
def large_document_dataset():
    """Generate 500 documents: 100 unique content values, each appears 5 times."""
    docs = []
    for content_id in range(100):
        for copy_id in range(5):
            docs.append(
                {
                    "id": content_id * 5 + copy_id,
                    "content": f"document_{content_id}",
                    "value": content_id * 1000 + copy_id,
                }
            )
    return docs


def test_deduplicate_basic(backend):
    """Test basic deduplication by id."""
    data = [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
        {"id": 1, "val": "c"},
        {"id": 3, "val": "d"},
        {"id": 2, "val": "e"},
    ]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = list(Backend.execute(ds, context=backend))

    # Should have exactly 3 unique items (ids 1, 2, 3)
    assert len(results) == 3

    # Check that we have one of each id
    ids = sorted([r["id"] for r in results])
    assert ids == [1, 2, 3]


def test_deduplicate_empty(backend):
    """Test deduplication on empty dataset."""
    ds = Dataset.from_list([]).deduplicate(key=lambda x: x["id"])
    results = list(Backend.execute(ds, context=backend))
    assert results == []


def test_deduplicate_all_unique(backend):
    """Test deduplication when all items are unique."""
    data = [{"id": i, "val": f"item_{i}"} for i in range(10)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 10


def test_deduplicate_all_duplicates(backend):
    """Test deduplication when all items have same key."""
    data = [{"id": 1, "val": f"item_{i}"} for i in range(10)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = list(Backend.execute(ds, context=backend))
    assert len(results) == 1
    assert results[0]["id"] == 1


def test_group_by_count(backend):
    """Test groupby with count reducer."""
    data = [
        {"cat": "A", "val": 1},
        {"cat": "B", "val": 2},
        {"cat": "A", "val": 3},
        {"cat": "C", "val": 4},
        {"cat": "B", "val": 5},
        {"cat": "A", "val": 6},
    ]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)}
    )

    results = list(Backend.execute(ds, context=backend))

    # Should have 3 groups
    assert len(results) == 3

    # Sort by category for consistent testing
    results = sorted(results, key=lambda x: x["cat"])

    assert results[0] == {"cat": "A", "count": 3}
    assert results[1] == {"cat": "B", "count": 2}
    assert results[2] == {"cat": "C", "count": 1}


def test_group_by_sum(backend):
    """Test groupby with sum reducer."""
    data = [
        {"cat": "A", "val": 1},
        {"cat": "B", "val": 2},
        {"cat": "A", "val": 3},
        {"cat": "C", "val": 4},
        {"cat": "B", "val": 5},
    ]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "sum": sum(item["val"] for item in items)}
    )

    results = list(Backend.execute(ds, context=backend))
    results = sorted(results, key=lambda x: x["cat"])

    assert results[0] == {"cat": "A", "sum": 4}  # 1 + 3
    assert results[1] == {"cat": "B", "sum": 7}  # 2 + 5
    assert results[2] == {"cat": "C", "sum": 4}


def test_group_by_list(backend):
    """Test groupby with list aggregation."""
    data = [
        {"cat": "A", "val": 1},
        {"cat": "B", "val": 2},
        {"cat": "A", "val": 3},
    ]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "vals": [item["val"] for item in items]}
    )

    results = list(Backend.execute(ds, context=backend))
    results = sorted(results, key=lambda x: x["cat"])

    assert results[0]["cat"] == "A"
    assert sorted(results[0]["vals"]) == [1, 3]

    assert results[1]["cat"] == "B"
    assert results[1]["vals"] == [2]


def test_group_by_empty(backend):
    """Test groupby on empty dataset."""
    ds = Dataset.from_list([]).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)}
    )

    results = list(Backend.execute(ds, context=backend))
    assert results == []


def test_deduplicate_with_num_output_shards(backend):
    """Test deduplication with explicit num_output_shards."""
    data = [{"id": i % 3, "val": f"item_{i}"} for i in range(20)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"], num_output_shards=5)

    results = list(Backend.execute(ds, context=backend))

    # Should have exactly 3 unique items (ids 0, 1, 2)
    assert len(results) == 3
    ids = sorted([r["id"] for r in results])
    assert ids == [0, 1, 2]


def test_group_by_with_hash_key_large(backend, large_document_dataset):
    """Test group_by with MD5 hash on larger dataset, counting duplicates."""

    def compute_hash(doc):
        content = doc["content"]
        return hashlib.md5(content.encode()).hexdigest()

    # Add hash field, group by hash, and count
    def count_and_extract(hash_key, items):
        """Reducer that counts items and extracts content from first item."""
        items_list = list(items)
        return {
            "hash": hash_key,
            "count": len(items_list),
            "content": items_list[0]["content"],
        }

    ds = (
        Dataset.from_list(large_document_dataset)
        .reshard(10)
        .map(lambda doc: {**doc, "hash": compute_hash(doc)})
        .group_by(key=lambda doc: doc["hash"], reducer=count_and_extract)
    )

    results = list(Backend.execute(ds, context=backend))

    # Should have exactly 100 groups (one per unique content)
    assert len(results) == 100

    # Each group should have exactly count of 5 (5 copies per unique content)
    for result in results:
        assert result["count"] == 5, f"Expected count 5 for {result['content']}, got {result['count']}"

    # Verify all hashes are unique
    hashes = [r["hash"] for r in results]
    assert len(set(hashes)) == 100


def test_group_by_with_none_and_filter(backend):
    """Test group_by with None results followed by filter operations."""
    data = [
        {"cat": "a"},
        {"cat": "a"},
        {"cat": "b"},
        {"cat": "foo"},
        {"cat": "foo"},
        {"cat": "bar"},
    ]

    # Reducer returns None for non-duplicates, key for duplicates
    def reducer(key, items):
        items_list = list(items)
        if len(items_list) > 1:
            return key
        return None

    ds = (
        Dataset.from_list(data)
        .group_by(key=lambda x: x["cat"], reducer=reducer)
        .filter(lambda x: x is not None)  # Filter out None values
    )

    results = list(Backend.execute(ds, context=backend))

    # Should only have duplicate keys: "a" and "foo"
    assert len(results) == 2
    assert sorted(results) == ["a", "foo"]
