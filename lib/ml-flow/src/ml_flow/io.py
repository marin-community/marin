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

"""File I/O utilities and custom sources for Beam pipelines."""

import json
import logging
from collections.abc import Iterator

import apache_beam as beam
from apache_beam.io import fileio

import fsspec

logger = logging.getLogger(__name__)


class ReadJsonLines(beam.PTransform):
    """
    Read JSONL files (compressed or uncompressed) using fsspec.

    Supports glob patterns and automatic compression detection.
    Each line is parsed as JSON and yielded as a dictionary.

    Example:
        pcoll = pipeline | ReadJsonLines("gs://bucket/**/*.jsonl.gz")
    """

    def __init__(self, file_pattern: str):
        self.file_pattern = file_pattern

    def expand(self, pcoll):
        return (
            pcoll
            | "MatchFiles" >> fileio.MatchFiles(self.file_pattern)
            | "ReadMatches" >> fileio.ReadMatches()
            | "ParseJsonLines" >> beam.FlatMap(self._parse_jsonl_file)
        )

    def _parse_jsonl_file(self, readable_file: fileio.ReadableFile) -> Iterator[dict]:
        """Parse JSONL file and yield records."""
        # Use fsspec for consistent handling with Marin's existing code
        file_path = readable_file.metadata.path

        try:
            with fsspec.open(file_path, "rt", compression="infer") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON at {file_path}:{line_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise


class ReadTextLines(beam.PTransform):
    """
    Read text files (compressed or uncompressed) using fsspec.

    Supports glob patterns and automatic compression detection.
    Yields each line as a string.

    Example:
        pcoll = pipeline | ReadTextLines("gs://bucket/**/*.txt.gz")
    """

    def __init__(self, file_pattern: str):
        self.file_pattern = file_pattern

    def expand(self, pcoll):
        return (
            pcoll
            | "MatchFiles" >> fileio.MatchFiles(self.file_pattern)
            | "ReadMatches" >> fileio.ReadMatches()
            | "ParseLines" >> beam.FlatMap(self._read_lines)
        )

    def _read_lines(self, readable_file: fileio.ReadableFile) -> Iterator[str]:
        """Read text file and yield lines."""
        file_path = readable_file.metadata.path

        try:
            with fsspec.open(file_path, "rt", compression="infer") as f:
                for line in f:
                    yield line.rstrip("\n\r")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise


class WriteJsonLines(beam.PTransform):
    """
    Write records as JSONL with gzip compression.

    Automatically creates output directory and handles sharding.

    Example:
        pcoll | WriteJsonLines("gs://bucket/output/data", num_shards=10)
    """

    def __init__(
        self,
        file_path_prefix: str,
        file_name_suffix: str = ".jsonl.gz",
        num_shards: int | None = None,
        compression_type: str = "gzip",
    ):
        self.file_path_prefix = file_path_prefix
        self.file_name_suffix = file_name_suffix
        self.num_shards = num_shards
        self.compression_type = compression_type

    def expand(self, pcoll):
        # Work around beam 2.69 bug - use 0 instead of None for auto-sharding
        shards = 0 if self.num_shards is None else self.num_shards

        return (
            pcoll
            | "SerializeJson" >> beam.Map(self._serialize_json)
            | "WriteToText"
            >> beam.io.WriteToText(
                self.file_path_prefix,
                file_name_suffix=self.file_name_suffix,
                num_shards=shards,
                compression_type=self.compression_type,
            )
        )

    def _serialize_json(self, record: dict) -> str:
        """Serialize record to JSON string."""
        return json.dumps(record, ensure_ascii=False)


def format_jsonl_output_path(base_path: str, extension: str = ".jsonl.gz") -> str:
    """
    Format output path for JSONL files.

    Ensures path doesn't end with extension (Beam adds it).

    Example:
        format_jsonl_output_path("gs://bucket/output/data.jsonl.gz")
        -> "gs://bucket/output/data"
    """
    if base_path.endswith(extension):
        return base_path[: -len(extension)]
    return base_path
