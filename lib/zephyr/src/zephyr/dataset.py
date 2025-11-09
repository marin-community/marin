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

"""Core Dataset API with lazy evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import braceexpand
import fsspec


@dataclass
class MapOp:
    """Map operation - applies function to each element."""

    fn: Callable


@dataclass
class FilterOp:
    """Filter operation - keeps elements matching predicate."""

    predicate: Callable


@dataclass
class BatchOp:
    """Batch operation - groups elements into fixed-size batches."""

    batch_size: int


@dataclass
class WriteJsonlOp:
    """Write operation for JSONL files.

    Writes records to JSONL files. Compression is inferred from the file extension
    (.gz suffix enables gzip compression).
    Supports path patterns with {shard}, {total}, {basename} substitutions.
    """

    output_pattern: str


@dataclass
class WriteParquetOp:
    """Write operation for Parquet files.

    Writes records to Parquet files with optional schema.
    Batches records before writing for efficiency.
    Supports path patterns with {shard}, {total}, {basename} substitutions.
    """

    output_pattern: str
    schema: object | None = None  # pyarrow.Schema, but avoid import here
    batch_size: int = 1000


@dataclass
class FlatMapOp:
    """FlatMap operation - applies function that yields multiple results per input.

    Use when a function processes one input and yields/returns many outputs.
    For example, reading a file that contains multiple records.
    """

    fn: Callable


@dataclass
class MapShardOp:
    """MapShard operation - applies function to entire shard iterator.

    The converse of flat_map: function receives an iterator of all items
    in the shard and returns an iterator of results. Enables stateful
    shard processing without requiring callable classes.

    Use when you need to maintain state across all items in a shard, such as
    deduplication, reservoir sampling, or loading expensive resources once.
    """

    fn: Callable


@dataclass
class ReshardOp:
    """Reshard operation - redistributes data across target number of shards.

    Best-effort operation that changes parallelism for subsequent operations.
    For RayBackend, redistributes object refs without materializing data.
    For other backends, this is a no-op.
    """

    num_shards: int


@dataclass
class FusedMapOp:
    """Fused operation - chains multiple operations into a single pipeline.

    Executes a sequence of operations on each item without materializing
    intermediate results. Operations are executed in order, avoiding object
    store overhead for intermediate values.
    """

    operations: list  # List of operations to fuse (MapOp, FlatMapOp, FilterOp, BatchOp, WriteJsonlOp, WriteParquetOp)


@dataclass
class GroupByLocalOp:
    """Phase 1 of GroupBy: Local grouping per shard by hash(key) % num_output_shards.

    This operation is fusible into a set of preceding operations.
    """

    key_fn: Callable  # Function from item -> hashable key
    num_output_shards: int  # Number of output shards


@dataclass
class GroupByShuffleReduceOp:
    """Phase 2 of GroupBy: Shuffle and reduce.

    This operation is not-fusible - it requires a shuffle boundary and must
    see all items with the same key together to apply the reducer.
    """

    key_fn: Callable  # Function from item -> hashable key
    reducer_fn: Callable  # Function from (key, Iterator[items]) -> result


@dataclass
class ReduceLocalOp:
    """Phase 1 of Reduce: Local reduction per shard."""

    local_reducer: Callable


@dataclass
class ReduceGlobalOp:
    """Phase 2 of Reduce: Pull to controller and apply final reduction."""

    global_reducer: Callable


@dataclass
class SortedMergeJoinOp:
    """Streaming merge join for pre-sorted, co-partitioned datasets.

    Single-phase join that pairs up corresponding shards and streams through them.
    Much faster than hash join but requires strict preconditions:
    - Both datasets have the same number of shards
    - Corresponding shards (left[i], right[i]) contain the same key ranges
    - Items within each shard are sorted by join key

    Only supports inner and left joins.
    """

    left_key_fn: Callable
    right_key_fn: Callable
    right_dataset: Dataset
    combiner_fn: Callable
    join_type: str  # "inner" or "left"


# Type alias for operations
Operation = (
    MapOp
    | FilterOp
    | BatchOp
    | WriteJsonlOp
    | WriteParquetOp
    | FlatMapOp
    | MapShardOp
    | ReshardOp
    | FusedMapOp
    | GroupByLocalOp
    | GroupByShuffleReduceOp
    | ReduceLocalOp
    | ReduceGlobalOp
    | SortedMergeJoinOp
)

T = TypeVar("T")
R = TypeVar("R")


class Dataset(Generic[T]):
    """Lazy dataset with method chaining for data processing pipelines.

    Dataset represents a data processing pipeline as a source and a chain of
    operations. Operations are stored as dataclasses, making the pipeline
    inspectable and treating transformations as data.

    Execution is handled by Backend classes via backend.execute(dataset).

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_list([1, 2, 3, 4, 5])
        ...     .filter(lambda x: x % 2 == 0)
        ...     .map(lambda x: x * 2)
        ... )
        >>> results = list(backend.execute(ds))
        [4, 8]
    """

    def __init__(self, source: Iterable[T], operations: list[Operation] | None = None):
        """Create a dataset from a source and optional operations.

        Args:
            source: Source data iterable
            operations: List of operations to apply
        """
        self.source = source
        self.operations = operations or []

    @staticmethod
    def from_list(items: list[T]) -> Dataset[T]:
        """Create a dataset from a list.

        Args:
            items: List of items

        Returns:
            Dataset wrapping the list
        """
        return Dataset(items)

    @staticmethod
    def from_iterable(iterable: Iterable[T]) -> Dataset[T]:
        """Create a dataset from any iterable.

        Args:
            iterable: Source iterable

        Returns:
            Dataset wrapping the iterable
        """
        return Dataset(iterable)

    @staticmethod
    def from_files(
        pattern: str,
        empty_glob_ok: bool = False,
    ) -> Dataset[str]:
        """Create dataset from file glob pattern.

        This method finds all files matching the glob pattern and returns a
        dataset of file paths. Use .write_jsonl() or .write_parquet() to write
        output files with explicit path patterns.

        Args:
            pattern: Glob pattern (e.g., "/input/**/*.jsonl.gz", "gs://bucket/data/*.parquet")
            empty_glob_ok: If True, empty glob won't raise an error (default: False)

        Returns:
            Dataset of input file paths

        Raises:
            FileNotFoundError: If no files match the pattern and empty_glob_ok is False

        Example:
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> ds = (Dataset
            ...     .from_files("/input/*.txt")
            ...     .map(lambda path: process_file(path))
            ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        import re

        # Normalize double slashes while preserving protocol (e.g., gs://, s3://, http://)
        pattern = re.sub(r"(?<!:)//+", "/", pattern)

        fs, _ = fsspec.core.url_to_fs(pattern)
        protocol = fsspec.core.split_protocol(pattern)[0]

        files = []
        for expanded in braceexpand.braceexpand(pattern):
            for f in fs.glob(expanded):
                if protocol:
                    files.append(f"{protocol}://{f}")
                else:
                    files.append(f)
        files = sorted(files)

        if len(files) == 0 and not empty_glob_ok:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        return Dataset.from_list(files)

    def map(self, fn: Callable[[T], R]) -> Dataset[R]:
        """Map a function over the dataset.

        The execution strategy (parallel vs sequential) is determined by the
        backend used to execute the dataset.

        Args:
            fn: Function to apply to each element

        Returns:
            New dataset with map operation appended

        Example:
            >>> backend = create_backend("sync")
            >>> ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
            >>> list(backend.execute(ds))
            [2, 4, 6]
        """
        return Dataset(self.source, [*self.operations, MapOp(fn)])

    def filter(self, predicate: Callable[[T], bool]) -> Dataset[T]:
        """Filter dataset elements by a predicate.

        Filter always executes synchronously as it's lightweight.

        Args:
            predicate: Function returning True to keep element, False to drop

        Returns:
            New dataset with filter operation appended

        Example:
            >>> backend = create_backend("sync")
            >>> ds = Dataset.from_list([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
            >>> list(backend.execute(ds))
            [2, 4]
        """
        return Dataset(self.source, [*self.operations, FilterOp(predicate)])

    def batch(self, batch_size: int) -> Dataset[list[T]]:
        """Batch dataset elements into fixed-size lists.

        Args:
            batch_size: Number of elements per batch

        Returns:
            New dataset with batch operation appended

        Example:
            >>> backend = create_backend("sync")
            >>> ds = Dataset.from_list([1, 2, 3, 4, 5]).batch(2)
            >>> list(backend.execute(ds))
            [[1, 2], [3, 4], [5]]
        """
        return Dataset(self.source, [*self.operations, BatchOp(batch_size)])

    def flat_map(self, fn: Callable[[T], Iterable[R]]) -> Dataset[R]:
        """Apply function that returns an iterable, flattening results.

        Args:
            fn: Function that takes an item and returns an iterable of results

        Returns:
            New dataset with flat_map operation appended

        Example:
            >>> from zephyr import load_jsonl
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> ds = (Dataset
            ...     .from_files("/input", "*.jsonl.gz")
            ...     .flat_map(load_jsonl)  # Each file yields many records
            ...     .filter(lambda r: r["score"] > 0.5)
            ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        return Dataset(self.source, [*self.operations, FlatMapOp(fn)])

    def map_shard(self, fn: Callable[[Iterator[T]], Iterator[R]]) -> Dataset[R]:
        """Apply function to entire shard iterator.

        The function receives an iterator of all items in the shard and returns
        an iterator of results. This is the recommended pattern for stateful
        processing across a shard (deduplication, sampling, windowing, etc.)
        without requiring callable classes.

        Args:
            fn: Function from Iterator[items] -> Iterator[results]

        Returns:
            New dataset with map_shard operation appended

        Example:
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> # Deduplicate items within each shard
            >>> def deduplicate_shard(items: Iterator):
            ...     seen = set()
            ...     for item in items:
            ...         key = item["id"]
            ...         if key not in seen:
            ...             seen.add(key)
            ...             yield item
            >>>
            >>> ds = (Dataset
            ...     .from_files("data/*.jsonl")
            ...     .flat_map(load_jsonl)
            ...     .map_shard(deduplicate_shard)
            ...     .write_jsonl("output/deduped-{shard:05d}.jsonl.gz")
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        return Dataset(self.source, [*self.operations, MapShardOp(fn)])

    def reshard(self, num_shards: int) -> Dataset[T]:
        """Redistribute data across target number of shards (best-effort).

        Changes parallelism for subsequent operations.

        Useful after operations that reduce parallelism (like filtering) or when
        starting with a small number of input files.

        Args:
            num_shards: Target number of shards

        Returns:
            New dataset with reshard operation appended

        Example:
            >>> backend = create_backend("ray", max_parallelism=20)
            >>> ds = (Dataset
            ...     .from_files("/input", "*.jsonl.gz")  # 3 files = 3 shards
            ...     .flat_map(load_jsonl)                 # Still 3 shards
            ...     .filter(lambda r: r["score"] > 0.9)  # Still 3 shards
            ...     .reshard(num_shards=20)              # Redistribute to 20 shards
            ...     .map(expensive_transform)            # Now uses up to 20 workers
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        return Dataset(self.source, [*self.operations, ReshardOp(num_shards)])

    def write_jsonl(self, output_pattern: str) -> Dataset[str]:
        """Write records as JSONL files.

        Writes each input stream to a separate JSONL file. The output pattern
        supports substitutions: {shard:05d}, {total:05d}, {basename}.

        Compression is automatically inferred from the file extension.
        Args:
            output_pattern: Output path pattern (e.g., "dir/data-{shard:05d}-of-{total:05d}.jsonl.gz")

        Returns:
            Dataset of output file paths written

        Example:
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> ds = (Dataset
            ...     .from_files("/input", "*.jsonl.gz")
            ...     .map(lambda path: process_file(path))
            ...     .write_jsonl("/output/processed-{shard:05d}-of-{total:05d}.jsonl.gz")
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        return Dataset(self.source, [*self.operations, WriteJsonlOp(output_pattern)])

    def write_parquet(
        self,
        output_pattern: str,
        schema: object | None = None,
        batch_size: int = 1000,
    ) -> Dataset[str]:
        """Write records as Parquet files.

        Writes records to Parquet files, batching for efficiency. Schema can be
        provided or inferred from the first record or dataclass type.
        The output pattern supports substitutions: {shard:05d}, {total:05d}, {basename}.

        Args:
            output_pattern: Output path pattern (e.g., "dir/data-{shard:05d}-of-{total:05d}.parquet")
            schema: PyArrow schema (optional, will be inferred if not provided)
            batch_size: Number of records to batch before writing (default: 1000)

        Returns:
            Dataset of output file paths written

        Example:
            >>> import pyarrow as pa
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> schema = pa.schema([("name", pa.string()), ("age", pa.int64())])
            >>> ds = (Dataset
            ...     .from_files("/input", "*.csv")
            ...     .map(lambda path: read_csv_records(path))
            ...     .write_parquet("/output/data-{shard:05d}.parquet", schema=schema)
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        return Dataset(self.source, [*self.operations, WriteParquetOp(output_pattern, schema, batch_size)])

    def group_by(
        self,
        key: Callable[[T], object],
        reducer: Callable[[object, Iterator[T]], R],
        num_output_shards: int | None = None,
    ) -> Dataset[R]:
        """Group items by key and apply reducer function.

        The reducer receives (key, iterator_of_items) and returns a single result.

        Args:
            key: Function extracting grouping key from item (must be hashable)
            reducer: Function from (key, Iterator[items]) -> result
            num_output_shards: Number of output shards (None = auto-detect, uses current shard count)

        Returns:
            New dataset with group_by operations appended

        Example:
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> # Count items by category
            >>> ds = (Dataset
            ...     .from_list([{"cat": "A", "val": 1}, {"cat": "A", "val": 2}, {"cat": "B", "val": 3}])
            ...     .group_by(
            ...         key=lambda x: x["cat"],
            ...         reducer=lambda key, items: {"cat": key, "count": sum(1 for _ in items)}
            ...     )
            ... )
            >>> list(backend.execute(ds))
            [{"cat": "A", "count": 2}, {"cat": "B", "count": 1}]
        """
        # Split into two explicit operations: local grouping + shuffle/reduce
        # If num_output_shards is None, backend will detect from current shard count
        if num_output_shards is None:
            # Use a sentinel value (-1) to indicate auto-detect at execution time
            # Backend will replace this with len(shards)
            num_output_shards = -1

        return Dataset(
            self.source, [*self.operations, GroupByLocalOp(key, num_output_shards), GroupByShuffleReduceOp(key, reducer)]
        )

    def deduplicate(self, key: Callable[[T], object], num_output_shards: int | None = None) -> Dataset[T]:
        """Deduplicate items by key.

        Example:
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> ds = (Dataset
            ...     .from_list([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}, {"id": 1, "val": "c"}])
            ...     .deduplicate(key=lambda x: x["id"])
            ... )
            >>> list(backend.execute(ds))
            [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]  # Or {"id": 1, "val": "c"}
        """

        def streaming_dedup(items: Iterator[T]) -> Iterator[T]:
            """Deduplicate items within a shard."""
            seen = set()
            for item in items:
                k = key(item)
                if k not in seen:
                    seen.add(k)
                    yield item

        def keep_first(k, items: Iterator[T]) -> T:
            """Reducer that keeps the first item."""
            return next(iter(items))

        return self.map_shard(streaming_dedup).group_by(key=key, reducer=keep_first, num_output_shards=num_output_shards)

    def reduce(
        self,
        local_reducer: Callable[[Iterator[T]], R],
        global_reducer: Callable[[Iterator[R]], R] | None = None,
    ) -> Dataset[R]:
        """Reduce dataset to a single value via two-phase reduction.

        Phase 1: Apply local_reducer to each shard independently
        Phase 2: Pull shard results to controller and apply global_reducer

        Args:
            local_reducer: Reduces iterator of items to single value per shard
            global_reducer: Reduces shard results to final value (defaults to local_reducer)

        Returns:
            Dataset containing a single reduced value

        Example:
            >>> ds = Dataset.from_list(range(100)).reduce(sum)
            >>> result = list(backend.execute(ds))[0]
            4950
        """
        if global_reducer is None:
            global_reducer = local_reducer

        return Dataset(self.source, [*self.operations, ReduceLocalOp(local_reducer), ReduceGlobalOp(global_reducer)])

    def sorted_merge_join(
        self,
        right: Dataset[R],
        left_key: Callable[[T], object],
        right_key: Callable[[R], object],
        combiner: Callable[[T | None, R | None], object] | None = None,
        how: str = "inner",
    ) -> Dataset:
        """Streaming merge join for already-sorted, co-partitioned datasets.

        **PRECONDITIONS (undefined behavior if violated):**
        - Both datasets have the same number of shards
        - Corresponding shards (left[i], right[i]) contain the same key ranges
        - Items within each shard are sorted by their join key

        These preconditions are typically met when both datasets come from
        group_by() with the same key and num_output_shards.

        **Use this when:**
        - Both datasets already partitioned by join key (e.g., from group_by)
        - Want to avoid shuffle overhead
        - Know datasets are compatible

        **Only supports inner and left joins** (no right or outer joins).

        Args:
            right: Right dataset to join with
            left_key: Function to extract join key from left items
            right_key: Function to extract join key from right items
            combiner: Function to combine (left_item, right_item) or (left_item, None).
                      Defaults to merging dicts: {**left, **right}
            how: Join type - "inner" or "left" (default: "inner")

        Returns:
            New dataset with joined results

        Raises:
            ValueError: If join type is not "inner" or "left"

        Example:
            >>> # Both come from group_by - safe to use sorted_merge_join
            >>> docs = Dataset.from_files(...).group_by(
            ...     key=lambda x: x["id"],
            ...     reducer=keep_first,
            ...     num_output_shards=100
            ... )
            >>> attrs = Dataset.from_files(...).group_by(
            ...     key=lambda x: x["id"],
            ...     reducer=keep_first,
            ...     num_output_shards=100
            ... )
            >>> joined = docs.sorted_merge_join(
            ...     attrs,
            ...     left_key=lambda x: x["id"],
            ...     right_key=lambda x: x["id"]
            ... )
        """
        if how not in ("inner", "left"):
            raise ValueError(f"sorted_merge_join only supports 'inner' and 'left' joins, got: {how}")

        # Default combiner merges dicts
        if combiner is None:

            def default_combiner(left, right):
                if left is None or right is None:
                    raise ValueError(
                        "Default combiner requires both left and right items (use custom combiner for outer joins)"
                    )
                return {**left, **right}

            combiner = default_combiner

        return Dataset(
            self.source,
            [*self.operations, SortedMergeJoinOp(left_key, right_key, right, combiner, how)],
        )
