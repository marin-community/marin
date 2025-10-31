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

"""Core Dataset API wrapping Apache Beam PCollection."""

import logging
from typing import TYPE_CHECKING, TypeVar
from collections.abc import Callable, Iterator

import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.options.pipeline_options import PipelineOptions

from ml_flow.io import ReadJsonLines, ReadTextLines, WriteJsonLines

if TYPE_CHECKING:
    from apache_beam.runners.runner import PipelineResult

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class Dataset:
    """
    Dataset abstraction wrapping Apache Beam PCollection.

    Provides a fluent, composable API for data transformations that compiles
    down to efficient Beam pipelines. Designed to replace Ray Data patterns
    in Marin with Dataflow's superior scaling and cost optimization.

    Example:
        # Read JSONL files
        ds = Dataset.from_jsonl_files(
            "gs://bucket/**/*.jsonl.gz",
            pipeline_options=options
        )

        # Transform
        ds = ds.filter(lambda x: x["score"] > 0.5)
        ds = ds.map(lambda x: {"text": x["text"].upper(), "id": x["id"]})

        # Write
        ds.write_jsonl_gz("gs://output/data")

        # Execute
        result = ds.run_and_wait()
    """

    def __init__(
        self,
        pcollection: beam.PCollection,
        pipeline: beam.Pipeline,
    ):
        self._pcollection = pcollection
        self._pipeline = pipeline

    @classmethod
    def from_jsonl_files(
        cls,
        pattern: str,
        pipeline_options: PipelineOptions,
    ) -> "Dataset":
        """
        Read JSONL/JSONL.gz files matching glob pattern.

        Automatically detects compression from file extension.
        Each line is parsed as JSON.

        Args:
            pattern: File glob pattern (e.g., "gs://bucket/**/*.jsonl.gz")
            pipeline_options: Beam pipeline options

        Returns:
            Dataset with parsed JSON records

        Example:
            options = DataflowOptions(project="my-project").to_pipeline_options()
            ds = Dataset.from_jsonl_files("gs://input/*.jsonl.gz", options)
        """
        pipeline = beam.Pipeline(options=pipeline_options)
        pcoll = pipeline | "ReadJsonLines" >> ReadJsonLines(pattern)
        return cls(pcoll, pipeline)

    @classmethod
    def from_text_files(
        cls,
        pattern: str,
        pipeline_options: PipelineOptions,
    ) -> "Dataset":
        """
        Read text files matching glob pattern.

        Automatically detects compression from file extension.
        Yields each line as a string.

        Args:
            pattern: File glob pattern (e.g., "gs://bucket/**/*.txt")
            pipeline_options: Beam pipeline options

        Returns:
            Dataset with text lines

        Example:
            options = DataflowOptions(project="my-project").to_pipeline_options()
            ds = Dataset.from_text_files("gs://input/*.txt", options)
        """
        pipeline = beam.Pipeline(options=pipeline_options)
        pcoll = pipeline | "ReadTextLines" >> ReadTextLines(pattern)
        return cls(pcoll, pipeline)

    def map(self, fn: Callable[[T], U], name: str | None = None) -> "Dataset":
        """
        Apply function to each element.

        Args:
            fn: Function taking element and returning transformed element
            name: Optional name for this transform (for monitoring)

        Returns:
            New Dataset with transformed elements

        Example:
            ds = ds.map(lambda x: x.upper())
            ds = ds.map(lambda doc: {"id": doc["id"], "len": len(doc["text"])})
        """
        label = name or "Map"
        pcoll = self._pcollection | label >> beam.Map(fn)
        return Dataset(pcoll, self._pipeline)

    def flat_map(self, fn: Callable[[T], Iterator[U]], name: str | None = None) -> "Dataset":
        """
        Apply function that yields multiple elements per input.

        Useful for parsing files where each input produces multiple records.

        Args:
            fn: Function taking element and yielding zero or more elements
            name: Optional name for this transform

        Returns:
            New Dataset with flattened elements

        Example:
            # Parse file into multiple documents
            def parse_file(line):
                for doc in extract_docs(line):
                    yield doc

            ds = ds.flat_map(parse_file)
        """
        label = name or "FlatMap"
        pcoll = self._pcollection | label >> beam.FlatMap(fn)
        return Dataset(pcoll, self._pipeline)

    def filter(self, predicate: Callable[[T], bool], name: str | None = None) -> "Dataset":
        """
        Filter elements based on predicate.

        Args:
            predicate: Function returning True to keep element, False to drop
            name: Optional name for this transform

        Returns:
            New Dataset with filtered elements

        Example:
            ds = ds.filter(lambda x: len(x["text"]) > 100)
            ds = ds.filter(lambda x: x["score"] > 0.5)
        """
        label = name or "Filter"
        pcoll = self._pcollection | label >> beam.Filter(predicate)
        return Dataset(pcoll, self._pipeline)

    def reshuffle(self, name: str | None = None) -> "Dataset":
        """
        Force a reshuffle/reshard for load balancing.

        Useful after filtering or when worker utilization is uneven.

        Args:
            name: Optional name for this transform

        Returns:
            New Dataset with reshuffled elements

        Example:
            ds = ds.filter(lambda x: x["important"])
            ds = ds.reshuffle()  # Rebalance after heavy filtering
        """
        label = name or "Reshuffle"
        pcoll = self._pcollection | label >> beam.Reshuffle()
        return Dataset(pcoll, self._pipeline)

    def write_jsonl_gz(
        self,
        output_path: str,
        num_shards: int | None = None,
        file_name_suffix: str = ".jsonl.gz",
    ) -> "Dataset":
        """
        Write dataset as gzip-compressed JSONL files.

        Args:
            output_path: Output file prefix (e.g., "gs://bucket/output/data")
            num_shards: Number of output shards (None = auto)
            file_name_suffix: Suffix for output files

        Returns:
            Self (for chaining run_and_wait)

        Example:
            ds.write_jsonl_gz("gs://output/data")
            ds.write_jsonl_gz("gs://output/data", num_shards=10)
        """
        # Remove extension if present (Beam adds it)
        if output_path.endswith(file_name_suffix):
            output_path = output_path[: -len(file_name_suffix)]

        self._pcollection | "WriteJsonLines" >> WriteJsonLines(
            output_path,
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
            compression_type="gzip",
        )
        return self

    def write_text_files(
        self,
        output_path: str,
        file_name_suffix: str = ".txt",
        num_shards: int | None = None,
        compression_type: str = "none",
    ) -> "Dataset":
        """
        Write dataset as text files (one element per line).

        Args:
            output_path: Output file prefix
            file_name_suffix: Suffix for output files
            num_shards: Number of output shards (None = auto, 0 = runner decides)
            compression_type: "gzip", "bz2", or "none"

        Returns:
            Self (for chaining run_and_wait)

        Example:
            ds.write_text_files("gs://output/lines.txt")
        """
        # Convert string compression type to Beam enum
        compression_map = {
            "none": CompressionTypes.UNCOMPRESSED,
            "gzip": CompressionTypes.GZIP,
            "bz2": CompressionTypes.BZIP2,
        }
        beam_compression = compression_map.get(compression_type, CompressionTypes.AUTO)

        # Work around beam 2.69 bug - use 0 instead of None for auto-sharding
        shards = 0 if num_shards is None else num_shards

        self._pcollection | "WriteToText" >> beam.io.WriteToText(
            output_path,
            file_name_suffix=file_name_suffix,
            num_shards=shards,
            compression_type=beam_compression,
        )
        return self

    def run(self) -> "PipelineResult":
        """
        Execute pipeline asynchronously.

        Returns immediately with a result object for monitoring.

        Returns:
            PipelineResult for monitoring job status
        """
        return self._pipeline.run()

    def run_and_wait(self) -> "PipelineResult":
        """
        Execute pipeline and wait for completion.

        Blocks until pipeline finishes or fails.

        Returns:
            PipelineResult with final status

        Example:
            result = ds.run_and_wait()
            print(f"Job completed: {result.state}")
        """
        result = self._pipeline.run()
        result.wait_until_finish()
        return result
