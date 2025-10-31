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

import fnmatch
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import fsspec

T = TypeVar("T")
R = TypeVar("R")


class BackendProtocol(Protocol):
    """Protocol for execution backends.

    Avoids circular import by defining interface without importing Backend class.
    """

    def execute_operations(self, source: Iterable, operations: list) -> Iterator:
        """Execute a chain of operations on a data source.

        Args:
            source: Source data iterable
            operations: List of operation dataclasses (MapOp, FilterOp, etc.)

        Returns:
            Iterator over results
        """
        ...


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


# Type alias for operations
Operation = MapOp | FilterOp | BatchOp | WriteJsonlOp | WriteParquetOp | FlatMapOp | ReshardOp | FusedMapOp


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
        input_path: str,
        pattern: str,
        empty_glob_ok: bool = False,
    ) -> Dataset[str]:
        """Create dataset from file glob.

        This method finds all files matching the pattern in input_path and
        returns a dataset of file paths. Use .write_jsonl() or .write_parquet()
        to write output files with explicit path patterns.

        Args:
            input_path: Directory to search for input files
            pattern: Glob pattern (e.g., "*.jsonl.gz", "**/*.parquet")
            empty_glob_ok: If True, empty glob won't raise an error (default: False)

        Returns:
            Dataset of input file paths

        Raises:
            FileNotFoundError: If no files match the pattern and empty_glob_ok is False

        Example:
            >>> backend = create_backend("ray", max_parallelism=10)
            >>> ds = (Dataset
            ...     .from_files("/input", "*.txt")
            ...     .map(lambda path: process_file(path))
            ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
            ... )
            >>> output_files = list(backend.execute(ds))
        """
        import os

        from marin.utils import fsspec_glob

        files = fsspec_glob(os.path.join(input_path, pattern))

        if len(files) == 0 and not empty_glob_ok:
            raise FileNotFoundError(f"No files found in {input_path} with pattern {pattern}")

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

        Batch always executes synchronously as it's lightweight.

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

        Use when your function processes one input and yields/returns multiple outputs.
        Common use case: reading a file that contains multiple records.

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

    def reshard(self, num_shards: int) -> Dataset[T]:
        """Redistribute data across target number of shards (best-effort).

        Changes parallelism for subsequent operations. For RayBackend, redistributes
        object refs without materializing data. For SyncBackend/ThreadPoolBackend,
        this is a no-op.

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

        Compression is automatically inferred from the file extension:
        - Files ending in .gz use gzip compression
        - Other files are written uncompressed

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


# Predefined transform functions for common file loading patterns


def load_jsonl(file_path: str) -> Iterator[dict]:
    """Load JSONL file and yield records.

    Handles gzip compression automatically via fsspec.
    Use with .flat_map() to read files containing multiple records.

    Args:
        file_path: Path to JSONL file (local or remote, .gz supported)

    Yields:
        Parsed JSON records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*.jsonl.gz")
        ...     .flat_map(load_jsonl)  # Each file yields many records
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    import json

    with fsspec.open(file_path, "rt", compression="infer", block_size=64_000_000, cache_type="background") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_parquet(file_path: str, **kwargs) -> Iterator[dict]:
    """Load Parquet file and yield records as dicts.

    Use with .flat_map() to read Parquet files containing multiple records.

    Args:
        file_path: Path to Parquet file
        **kwargs: Additional arguments for pd.read_parquet (e.g., columns, engine)

    Yields:
        Records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*.parquet")
        ...     .flat_map(load_parquet)
        ...     .map(lambda r: transform_record(r))
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    import pandas as pd

    df = pd.read_parquet(file_path, **kwargs)
    for _, row in df.iterrows():
        yield row.to_dict()


def load_file(file_path: str, **parquet_kwargs) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL or Parquet format.

    Detects format based on file extension and delegates to load_jsonl or load_parquet.
    Supports .jsonl, .jsonl.gz, .jsonl.zstd, jsonl.xz, and .parquet files.

    Args:
        file_path: Path to file (local or remote)
        **parquet_kwargs: Additional arguments for load_parquet (ignored for JSONL)

    Yields:
        Parsed records as dictionaries

    Raises:
        ValueError: If file extension is not supported

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> # Works with mixed JSONL and Parquet files
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*")
        ...     .flat_map(load_file)
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    if (
        file_path.endswith(".jsonl")
        or file_path.endswith(".jsonl.gz")
        or file_path.endswith(".jsonl.zstd")
        or file_path.endswith(".jsonl.xz")
    ):
        yield from load_jsonl(file_path)
    elif file_path.endswith(".parquet"):
        yield from load_parquet(file_path, **parquet_kwargs)
    else:
        raise ValueError(
            f"Unsupported extension: {file_path}. Supported formats: .jsonl, .jsonl.gz, .jsonl.zstd, .jsonl.xz, .parquet"
        )


def load_zip_members(zip_path: str, pattern: str = "*") -> Iterator[dict]:
    """Load zip members matching pattern, yielding filename and content.

    Opens zip file (supports fsspec paths like gs://), finds members matching
    the pattern, and yields dicts with 'filename' and 'content' (bytes).

    Args:
        zip_path: Path to zip file (local or remote via fsspec)
        pattern: Glob pattern to match member names (default: "*")

    Yields:
        Dicts with 'filename' (str) and 'content' (bytes)

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_list(["gs://bucket/data.zip"])
        ...     .flat_map(lambda p: load_zip_members(p, pattern="test.jsonl"))
        ...     .map(lambda m: process_file(m["filename"], m["content"]))
        ... )
        >>> output_files = list(backend.execute(ds))
    """

    import fsspec

    with fsspec.open(zip_path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            for member_name in zf.namelist():
                if not member_name.endswith("/") and fnmatch.fnmatch(member_name, pattern):
                    with zf.open(member_name, "r") as member_file:
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
