# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for deduplicate and group_by operations."""
import pyarrow as pa

import hashlib

import pytest

from zephyr import Dataset


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


def test_deduplicate_basic(zephyr_ctx):
    """Test basic deduplication by id."""
    data = [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
        {"id": 1, "val": "c"},
        {"id": 3, "val": "d"},
        {"id": 2, "val": "e"},
    ]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = zephyr_ctx.execute(ds).results

    # Should have exactly 3 unique items (ids 1, 2, 3)
    assert len(results) == 3

    # Check that we have one of each id
    ids = sorted([r["id"] for r in results])
    assert ids == [1, 2, 3]


def test_deduplicate_empty(zephyr_ctx):
    """Test deduplication on empty dataset."""
    ds = Dataset.from_list([]).deduplicate(key=lambda x: x["id"])
    results = zephyr_ctx.execute(ds).results
    assert results == []


def test_deduplicate_all_unique(zephyr_ctx):
    """Test deduplication when all items are unique."""
    data = [{"id": i, "val": f"item_{i}"} for i in range(10)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = zephyr_ctx.execute(ds).results
    assert len(results) == 10


def test_deduplicate_all_duplicates(zephyr_ctx):
    """Test deduplication when all items have same key."""
    data = [{"id": 1, "val": f"item_{i}"} for i in range(10)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = zephyr_ctx.execute(ds).results
    assert len(results) == 1
    assert results[0]["id"] == 1


def test_group_by_count(zephyr_ctx):
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

    results = zephyr_ctx.execute(ds).results

    # Should have 3 groups
    assert len(results) == 3

    # Sort by category for consistent testing
    results = sorted(results, key=lambda x: x["cat"])

    assert results[0] == {"cat": "A", "count": 3}
    assert results[1] == {"cat": "B", "count": 2}
    assert results[2] == {"cat": "C", "count": 1}


def test_group_by_sum(zephyr_ctx):
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

    results = zephyr_ctx.execute(ds).results
    results = sorted(results, key=lambda x: x["cat"])

    assert results[0] == {"cat": "A", "sum": 4}  # 1 + 3
    assert results[1] == {"cat": "B", "sum": 7}  # 2 + 5
    assert results[2] == {"cat": "C", "sum": 4}


def test_group_by_list(zephyr_ctx):
    """Test groupby with list aggregation."""
    data = [
        {"cat": "A", "val": 1},
        {"cat": "B", "val": 2},
        {"cat": "A", "val": 3},
    ]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "vals": [item["val"] for item in items]}
    )

    results = zephyr_ctx.execute(ds).results
    results = sorted(results, key=lambda x: x["cat"])

    assert results[0]["cat"] == "A"
    assert sorted(results[0]["vals"]) == [1, 3]

    assert results[1]["cat"] == "B"
    assert results[1]["vals"] == [2]


def test_group_by_empty(zephyr_ctx):
    """Test groupby on empty dataset."""
    ds = Dataset.from_list([]).group_by(
        key=lambda x: x["cat"], reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)}
    )

    results = zephyr_ctx.execute(ds).results
    assert results == []


def test_deduplicate_with_num_output_shards(zephyr_ctx):
    """Test deduplication with explicit num_output_shards."""
    data = [{"id": i % 3, "val": f"item_{i}"} for i in range(20)]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"], num_output_shards=5)

    results = zephyr_ctx.execute(ds).results

    # Should have exactly 3 unique items (ids 0, 1, 2)
    assert len(results) == 3
    ids = sorted([r["id"] for r in results])
    assert ids == [0, 1, 2]


def test_group_by_with_hash_key_large(zephyr_ctx, large_document_dataset):
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

    results = zephyr_ctx.execute(ds).results

    # Should have exactly 100 groups (one per unique content)
    assert len(results) == 100

    # Each group should have exactly count of 5 (5 copies per unique content)
    for result in results:
        assert result["count"] == 5, f"Expected count 5 for {result['content']}, got {result['count']}"

    # Verify all hashes are unique
    hashes = [r["hash"] for r in results]
    assert len(set(hashes)) == 100


def test_group_by_generator_reducer(zephyr_ctx):
    """Test group_by with a generator reducer that yields multiple items per group."""
    data = [
        {"cat": "A", "val": 1},
        {"cat": "B", "val": 2},
        {"cat": "A", "val": 3},
        {"cat": "B", "val": 5},
    ]

    def explode_reducer(key, items):
        """Generator reducer: yield each item individually with group metadata."""
        for item in items:
            yield {"cat": key, "val": item["val"], "from_group": True}

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"],
        reducer=explode_reducer,
    )

    results = zephyr_ctx.execute(ds).results

    # Generator reducer should flatten: 4 items total (2 from A, 2 from B)
    assert len(results) == 4

    results = sorted(results, key=lambda x: (x["cat"], x["val"]))
    assert results[0] == {"cat": "A", "val": 1, "from_group": True}
    assert results[1] == {"cat": "A", "val": 3, "from_group": True}
    assert results[2] == {"cat": "B", "val": 2, "from_group": True}
    assert results[3] == {"cat": "B", "val": 5, "from_group": True}


def test_group_by_secondary_sort(zephyr_ctx):
    """Test group_by with sort_by delivers items in sorted order within each group."""
    data = [
        {"user": "A", "ts": 3, "event": "c"},
        {"user": "B", "ts": 2, "event": "b"},
        {"user": "A", "ts": 1, "event": "a"},
        {"user": "B", "ts": 5, "event": "e"},
        {"user": "A", "ts": 2, "event": "b"},
        {"user": "B", "ts": 1, "event": "a"},
    ]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["user"],
        reducer=lambda key, items: {"user": key, "events": [i["event"] for i in items]},
        sort_by=lambda x: x["ts"],
    )

    results = zephyr_ctx.execute(ds).results
    results = sorted(results, key=lambda x: x["user"])

    # Items within each group should be sorted by timestamp
    assert results[0] == {"user": "A", "events": ["a", "b", "c"]}
    assert results[1] == {"user": "B", "events": ["a", "b", "e"]}


def test_group_by_secondary_sort_with_generator_reducer(zephyr_ctx):
    """Test sort_by combined with a generator reducer."""
    data = [
        {"cat": "X", "rank": 2, "val": "second"},
        {"cat": "X", "rank": 1, "val": "first"},
        {"cat": "Y", "rank": 3, "val": "third"},
        {"cat": "Y", "rank": 1, "val": "first"},
    ]

    def ranked_explode(key, items):
        for item in items:
            yield {"cat": key, "val": item["val"], "rank": item["rank"]}

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"],
        reducer=ranked_explode,
        sort_by=lambda x: x["rank"],
    )

    results = zephyr_ctx.execute(ds).results
    results = sorted(results, key=lambda x: (x["cat"], x["rank"]))

    # Items should arrive at reducer sorted by rank
    assert results == [
        {"cat": "X", "val": "first", "rank": 1},
        {"cat": "X", "val": "second", "rank": 2},
        {"cat": "Y", "val": "first", "rank": 1},
        {"cat": "Y", "val": "third", "rank": 3},
    ]


def test_group_by_with_none_and_filter(zephyr_ctx):
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

    results = zephyr_ctx.execute(ds).results

    # Should only have duplicate keys: "a" and "foo"
    assert len(results) == 2
    assert sorted(results) == ["a", "foo"]


def test_group_by_non_vortex_serializable(zephyr_ctx):
    """Shuffle with items that Vortex/Arrow cannot serialize uses pickle-in-parquet.

    Uses frozenset (not Arrow-serializable) so the pickle envelope path is
    exercised. Items are serialized via cloudpickle into a binary ``__pickle__``
    column inside Parquet, avoiding the N*M pickle file blowup.
    """

    from zephyr.writers import infer_arrow_schema

    # NOTE: confirm frozenset is not arrow-serializable type to trigger the pickle envelope path
    with pytest.raises(pa.lib.ArrowInvalid, match="Could not convert frozenset"):
        infer_arrow_schema([{"foo": frozenset([1, 2, 3])}])

    data = [
        {"key": "a", "values": frozenset([1, 2, 3])},
        {"key": "b", "values": frozenset([2])},
        {"key": "a", "values": frozenset([3, 4])},
    ]

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["key"],
        reducer=lambda key, items: {"key": key, "value": frozenset().union(*(item["values"] for item in items))},
    )

    results = zephyr_ctx.execute(ds).results
    results = sorted(results, key=lambda x: x["key"])
    assert len(results) == 2
    assert results[0] == {"key": "a", "value": frozenset([1, 2, 3, 4])}
    assert results[1] == {"key": "b", "value": frozenset([2])}


def test_scatter_file_iterator_pickle_roundtrip(tmp_path):
    """ScatterFileIterator round-trips non-Arrow-serializable items (e.g. frozenset)."""
    from zephyr.shuffle import ScatterFileIterator, _write_chunk_frame

    items = [frozenset([1, 2]), frozenset([3, 4, 5])]
    frame = _write_chunk_frame(items)

    path = str(tmp_path / "test.shuffle")
    with open(path, "wb") as f:
        f.write(frame)

    it = ScatterFileIterator(path=path, chunks=((0, len(frame)),))
    chunks = [list(chunk_iter) for chunk_iter in it.get_chunk_iterators()]
    assert len(chunks) == 1
    assert chunks[0] == items


def test_group_by_schema_evolution(zephyr_ctx):
    """Schema evolution: a field that is null in some chunks gains a type in others."""
    data = []
    # First batch of items: score is None (Arrow infers Null type)
    for i in range(20):
        data.append({"id": i, "cat": f"g{i % 5}", "score": None})
    # Second batch: score is int (Arrow infers int64)
    for i in range(20, 40):
        data.append({"id": i, "cat": f"g{i % 5}", "score": i})

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"],
        reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)},
    )

    results = zephyr_ctx.execute(ds).results
    results = sorted(results, key=lambda x: x["cat"])

    assert len(results) == 5
    for r in results:
        assert r["count"] == 8  # 4 with None score + 4 with int score


def test_group_by_combiner(zephyr_ctx):
    """Combiner deduplicates locally during scatter; reducer still deduplicates globally."""
    data = [
        {"key": "a", "id": 1},
        {"key": "a", "id": 1},  # duplicate
        {"key": "a", "id": 2},
        {"key": "b", "id": 3},
        {"key": "b", "id": 3},  # duplicate
        {"key": "b", "id": 3},  # duplicate
        {"key": "b", "id": 4},
    ]

    def dedup_combiner(key, items):
        seen = set()
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                yield item

    def dedup_reducer(key, items):
        seen = set()
        ids = []
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                ids.append(item["id"])
        return {"key": key, "ids": sorted(ids)}

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["key"],
        reducer=dedup_reducer,
        combiner=dedup_combiner,
    )

    results = sorted(zephyr_ctx.execute(ds).results, key=lambda x: x["key"])
    assert results == [
        {"key": "a", "ids": [1, 2]},
        {"key": "b", "ids": [3, 4]},
    ]


def test_group_by_combiner_sum(zephyr_ctx):
    """Combiner pre-aggregates partial sums, reducer produces final sum."""
    data = [
        {"cat": "x", "val": 1},
        {"cat": "x", "val": 2},
        {"cat": "x", "val": 3},
        {"cat": "y", "val": 10},
        {"cat": "y", "val": 20},
    ]

    def sum_combiner(key, items):
        total = sum(item["val"] for item in items)
        yield {"cat": key, "val": total}

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["cat"],
        reducer=lambda key, items: {"cat": key, "total": sum(item["val"] for item in items)},
        combiner=sum_combiner,
    )

    results = sorted(zephyr_ctx.execute(ds).results, key=lambda x: x["cat"])
    assert results == [
        {"cat": "x", "total": 6},
        {"cat": "y", "total": 30},
    ]


# --- Integration tests (all backends) ---


def test_deduplicate_basic_integration(integration_ctx):
    data = [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
        {"id": 1, "val": "c"},
        {"id": 3, "val": "d"},
        {"id": 2, "val": "e"},
    ]

    ds = Dataset.from_list(data).deduplicate(key=lambda x: x["id"])

    results = integration_ctx.execute(ds).results
    assert len(results) == 3
    ids = sorted([r["id"] for r in results])
    assert ids == [1, 2, 3]


def test_group_by_combiner_integration(integration_ctx):
    data = [
        {"key": "a", "id": 1},
        {"key": "a", "id": 1},
        {"key": "a", "id": 2},
        {"key": "b", "id": 3},
        {"key": "b", "id": 3},
        {"key": "b", "id": 3},
        {"key": "b", "id": 4},
    ]

    def dedup_combiner(key, items):
        seen = set()
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                yield item

    def dedup_reducer(key, items):
        seen = set()
        ids = []
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                ids.append(item["id"])
        return {"key": key, "ids": sorted(ids)}

    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["key"],
        reducer=dedup_reducer,
        combiner=dedup_combiner,
    )

    results = sorted(integration_ctx.execute(ds).results, key=lambda x: x["key"])
    assert results == [
        {"key": "a", "ids": [1, 2]},
        {"key": "b", "ids": [3, 4]},
    ]


def test_group_by_with_multiple_combine_tasks(zephyr_ctx, monkeypatch):
    """Force K>1 via a tiny memory budget so the combine stage splits into
    multiple parallel tasks; output must match the K=1 path."""
    from zephyr import execution

    # 1-byte budget forces _auto_combine_tasks to cap K at R (== num_shards).
    monkeypatch.setattr(execution, "_COMBINE_TASK_MEM_BUDGET", 1)

    data = [{"k": i % 4, "v": i} for i in range(40)]
    ds = Dataset.from_list(data).group_by(
        key=lambda x: x["k"],
        reducer=lambda key, items: {"k": key, "sum": sum(x["v"] for x in items)},
        num_output_shards=4,
    )
    results = sorted(zephyr_ctx.execute(ds).results, key=lambda x: x["k"])
    expected = [{"k": k, "sum": sum(v for v in range(40) if v % 4 == k)} for k in range(4)]
    assert results == expected
