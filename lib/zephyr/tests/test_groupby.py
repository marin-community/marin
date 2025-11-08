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

import pytest

from zephyr import Dataset, create_backend


@pytest.fixture(autouse=True)
def ensure_ray(ray_cluster):
    """Ensure Ray is initialized for all tests."""
    pass


@pytest.fixture(
    params=[
        pytest.param(create_backend("sync"), id="sync"),
        pytest.param(create_backend("threadpool", max_parallelism=2), id="thread"),
        pytest.param(create_backend("ray", max_parallelism=2), id="ray"),
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

    results = list(backend.execute(ds))

    # Should have exactly 3 unique items (ids 1, 2, 3)
    assert len(results) == 3

    # Check that we have one of each id
    ids = sorted([r["id"] for r in results])
    assert ids == [1, 2, 3]


def test_deduplicate_empty(backend):
    """Test deduplication on empty dataset."""
    ds = Dataset.from_list([]).deduplicate(key=lambda x: x["id"])
    results = list(backend.execute(ds))
    assert results == []


def test_deduplicate_all_unique(backend):
    """Test deduplication when all items are unique."""
    data = [{"id": i, "val": f"item_{i}"} for i in range(10)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = list(backend.execute(ds))
    assert len(results) == 10


def test_deduplicate_all_duplicates(backend):
    """Test deduplication when all items have same key."""
    data = [{"id": 1, "val": f"item_{i}"} for i in range(10)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = list(backend.execute(ds))
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

    results = list(backend.execute(ds))

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

    results = list(backend.execute(ds))
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

    results = list(backend.execute(ds))
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

    results = list(backend.execute(ds))
    assert results == []


def test_group_by_single_group(backend):
    """Test groupby when all items belong to same group."""
    data = [{"cat": "A", "val": i} for i in range(5)]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)}
    )

    results = list(backend.execute(ds))
    assert len(results) == 1
    assert results[0] == {"cat": "A", "count": 5}


def test_deduplicate_with_num_output_shards(backend):
    """Test deduplication with explicit num_output_shards."""
    data = [{"id": i % 3, "val": f"item_{i}"} for i in range(20)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"], num_output_shards=5)

    results = list(backend.execute(ds))

    # Should have exactly 3 unique items (ids 0, 1, 2)
    assert len(results) == 3
    ids = sorted([r["id"] for r in results])
    assert ids == [0, 1, 2]


def test_group_by_with_num_output_shards(backend):
    """Test groupby with explicit num_output_shards."""
    data = [{"cat": chr(65 + i % 3), "val": i} for i in range(20)]  # A, B, C

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"],
        reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)},
        num_output_shards=5,
    )

    results = list(backend.execute(ds))
    results = sorted(results, key=lambda x: x["cat"])

    assert len(results) == 3
    # Each category should have items (A: 0,3,6,9,12,15,18 = 7, B: 1,4,7,10,13,16,19 = 7, C: 2,5,8,11,14,17 = 6)
    assert results[0] == {"cat": "A", "count": 7}
    assert results[1] == {"cat": "B", "count": 7}
    assert results[2] == {"cat": "C", "count": 6}


def test_deduplicate_chained_with_map(backend):
    """Test deduplication chained with other operations."""
    data = [1, 2, 1, 3, 2, 4, 1]

    ds = (
        Dataset.from_list(data)
        .map(lambda x: {"id": x, "val": x * 2})
        .deduplicate(key=lambda x: x["id"])
        .map(lambda x: x["val"])
    )

    results = sorted(list(backend.execute(ds)))
    assert results == [2, 4, 6, 8]  # 1*2, 2*2, 3*2, 4*2


def test_group_by_chained_with_filter(backend):
    """Test groupby chained with other operations."""
    data = [
        {"cat": "A", "val": 1},
        {"cat": "B", "val": 2},
        {"cat": "A", "val": 3},
        {"cat": "B", "val": 4},
    ]

    ds = (
        Dataset.from_list(data)
        .filter(lambda x: x["val"] > 1)
        .group_by(key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)})
    )

    results = sorted(list(backend.execute(ds)), key=lambda x: x["cat"])
    assert results == [{"cat": "A", "count": 1}, {"cat": "B", "count": 2}]


def test_deduplicate_with_hash_key_large(backend, large_document_dataset):
    """Test deduplication with MD5 hash on larger dataset with duplicates."""
    import hashlib

    def compute_hash(doc):
        """Compute MD5 hash of content field."""
        content = doc["content"]
        return hashlib.md5(content.encode()).hexdigest()

    # Add hash field and deduplicate
    ds = (
        Dataset.from_list(large_document_dataset)
        .map(lambda doc: {**doc, "hash": compute_hash(doc)})
        .deduplicate(key=lambda doc: doc["hash"])
    )

    results = list(backend.execute(ds))

    # Should have exactly 100 unique documents (one per unique content)
    assert len(results) == 100

    # Verify all hashes are unique
    hashes = [r["hash"] for r in results]
    assert len(set(hashes)) == 100

    # Verify we have the expected content patterns
    contents = sorted([r["content"] for r in results])
    expected = sorted([f"document_{i}" for i in range(100)])
    assert contents == expected


def test_group_by_with_hash_key_large(backend, large_document_dataset):
    """Test group_by with MD5 hash on larger dataset, counting duplicates."""
    import hashlib

    def compute_hash(doc):
        """Compute MD5 hash of content field."""
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
        .map(lambda doc: {**doc, "hash": compute_hash(doc)})
        .group_by(key=lambda doc: doc["hash"], reducer=count_and_extract)
    )

    results = list(backend.execute(ds))

    # Should have exactly 100 groups (one per unique content)
    assert len(results) == 100

    # Each group should have exactly count of 5 (5 copies per unique content)
    for result in results:
        assert result["count"] == 5, f"Expected count 5 for {result['content']}, got {result['count']}"

    # Verify all hashes are unique
    hashes = [r["hash"] for r in results]
    assert len(set(hashes)) == 100
