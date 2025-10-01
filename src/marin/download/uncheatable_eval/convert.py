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

"""Convert downloaded Uncheatable Eval data dumps to the correct format."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, simple_backpressure
from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, VersionedValue, ensure_versioned, this_output_path
from marin.utils import fsspec_exists, fsspec_mkdirs

from src.marin.download.huggingface.download import DownloadConfig
from src.marin.download.huggingface.download_hf import download_hf

logger = logging.getLogger("ray")

FILENAME_PATTERN = re.compile(r"^(?P<benchmark>.+)_(?P<start>\d{8})to(?P<end>\d{8})(?P<suffix>(?:\.[^.]+)*)$")

TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "body",
    "content",
    "article",
    "document",
    "raw_text",
    "code",
    "message",
    "description",
    "story",
)

LIST_FIELD_CANDIDATES: tuple[str, ...] = (
    "paragraphs",
    "sentences",
    "lines",
    "messages",
)

ID_FIELD_CANDIDATES: tuple[str, ...] = (
    "id",
    "uuid",
    "guid",
    "doc_id",
    "document_id",
    "article_id",
    "hash",
    "sha",
    "uid",
)


@dataclass(frozen=True)
class ConversionDataset:
    """Information about a single data file to be converted."""

    benchmark: str
    start_date: str
    end_date: str
    name: str
    input_path: str
    output_path: str

    @property
    def date_range(self) -> str:
        return f"{self.start_date}to{self.end_date}"

    @property
    def source_label(self) -> str:
        return f"{self.benchmark}:{self.date_range}"


@dataclass
class UncheatableEvalConvertConfig:
    """Configuration for converting Uncheatable Eval dumps."""

    input_path: str | VersionedValue[str]
    output_path: str | VersionedValue[str] = THIS_OUTPUT_PATH
    max_concurrent_conversions: int = 8
    skip_existing: bool = True
    metadata_filename: str = "conversion_metadata.json"


def _discover_input_files(input_path: str) -> list[str]:
    """Discover all JSON/JSONL files in the input directory."""

    if not fsspec_exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    files = []
    fs = fsspec.filesystem(input_path.split("://")[0] if "://" in input_path else "file")

    try:
        items = fs.listdir(input_path, detail=True)
        for item in items:
            if item["type"] == "file":
                name = item["name"]
                basename = os.path.basename(name)
                if basename.endswith(".json") or basename.endswith(".jsonl"):
                    files.append(name)
    except Exception as e:
        logger.error("Failed to list files in %s: %s", input_path, e)
        raise

    return sorted(files)


def _parse_filename(file_path: str) -> ConversionDataset | None:
    """Parse a file path to extract dataset information."""

    filename = os.path.basename(file_path)
    base_name = filename.split(".")[0]  # Remove extensions

    match = FILENAME_PATTERN.match(base_name)
    if not match:
        return None

    benchmark = match.group("benchmark")
    start = match.group("start")
    end = match.group("end")

    return ConversionDataset(
        benchmark=benchmark,
        start_date=start,
        end_date=end,
        name=filename,
        input_path=file_path,
        output_path=""  # Will be set later
    )


def _extract_id(raw: Any, dataset: ConversionDataset, index: int) -> str:
    """Extract or generate an ID for a record."""

    if isinstance(raw, dict):
        for key in ID_FIELD_CANDIDATES:
            value = raw.get(key)
            if value:
                return str(value)
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            for key in ID_FIELD_CANDIDATES:
                value = metadata.get(key)
                if value:
                    return str(value)
    return f"{dataset.benchmark}_{dataset.date_range}_{index:06d}"


def _join_list_field(value: Any) -> str | None:
    """Join list fields into a single string."""

    if isinstance(value, list):
        text_items = [str(item) for item in value if item is not None]
        if text_items:
            return "\n".join(text_items)
    return None


def _extract_text(raw: Any) -> str | None:
    """Extract text content from a raw record."""

    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # Try standard text fields first
        for key in TEXT_FIELD_CANDIDATES:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # Try list fields that can be joined
        for key in TEXT_FIELD_CANDIDATES:
            value = raw.get(key)
            joined = _join_list_field(value)
            if joined:
                return joined

        for key in LIST_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined

        # Try title + body combination
        title = raw.get("title")
        body = raw.get("body")
        if isinstance(title, str) and isinstance(body, str):
            combined = f"{title.strip()}\n\n{body.strip()}"
            if combined.strip():
                return combined

        if isinstance(title, str) and title.strip():
            return title

        # Fall back to JSON representation
        return json.dumps(raw, ensure_ascii=False)

    return str(raw)


def _normalize_record(raw: Any, dataset: ConversionDataset, index: int) -> dict[str, str]:
    """Normalize a raw record to the standard format."""

    text = _extract_text(raw)
    if text is None or not str(text).strip():
        raise ValueError(f"Record {index} in {dataset.name} does not contain text")

    record_id = _extract_id(raw, dataset, index)
    return {"id": record_id, "text": text, "source": dataset.source_label}


@ray.remote(max_retries=3)
@cached_or_construct_output(success_suffix="SUCCESS")
def _convert_single_file(
    input_file_path: str,
    output_file_path: str,
    dataset: ConversionDataset,
) -> dict[str, Any]:
    """Convert a single input file to the normalized format."""

    logger.info("Converting %s to %s", input_file_path, output_file_path)

    # Read input file
    with fsspec.open(input_file_path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)

    if not isinstance(payload, list):
        raise ValueError(f"Expected list in dataset {dataset.name}, found {type(payload).__name__}")

    # Ensure output directory exists
    fsspec_mkdirs(os.path.dirname(output_file_path), exist_ok=True)

    # Convert and write output
    record_count = 0
    with fsspec.open(output_file_path, "wt", encoding="utf-8", compression="gzip") as outfile:
        for index, raw in enumerate(payload):
            normalized = _normalize_record(raw, dataset, index)
            json.dump(normalized, outfile, ensure_ascii=False)
            outfile.write("\n")
            record_count += 1

    logger.info("Converted %s records from %s to %s", record_count, input_file_path, output_file_path)
    return {"records": record_count, "input_file": input_file_path, "output_file": output_file_path}


def _generate_conversion_tasks(
    input_files: Iterable[str],
    output_path: str,
    skip_existing: bool,
) -> tuple[list[tuple[str, str, ConversionDataset]], list[ConversionDataset]]:
    """Generate conversion tasks for the input files."""

    tasks: list[tuple[str, str, ConversionDataset]] = []
    datasets: list[ConversionDataset] = []

    for input_file in input_files:
        dataset = _parse_filename(input_file)
        if not dataset:
            logger.warning("Skipping file with unrecognized name pattern: %s", input_file)
            continue

        output_filename = f"{dataset.benchmark}_{dataset.date_range}.jsonl.gz"
        output_file = posixpath.join(output_path, output_filename)
        success_file = f"{output_file}.SUCCESS"

        if skip_existing and fsspec_exists(success_file):
            logger.info("Skipping %s because output already exists", dataset.name)
            continue

        # Update dataset with output path
        dataset = ConversionDataset(
            benchmark=dataset.benchmark,
            start_date=dataset.start_date,
            end_date=dataset.end_date,
            name=dataset.name,
            input_path=dataset.input_path,
            output_path=output_file
        )

        tasks.append((input_file, output_file, dataset))
        datasets.append(dataset)

    return tasks, datasets


def _write_conversion_metadata(cfg: UncheatableEvalConvertConfig, records: list[dict[str, Any]]) -> None:
    """Write conversion metadata to a JSON file."""

    if not records:
        return

    metadata_path = posixpath.join(str(cfg.output_path), cfg.metadata_filename)
    with fsspec.open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(records, meta_file, indent=2, ensure_ascii=False)

    logger.info("Wrote conversion metadata to %s", metadata_path)


# def convert_uncheatable_eval_files(cfg: UncheatableEvalConvertConfig) -> dict[str, Any]:
#     """Convert downloaded Uncheatable Eval files to the normalized format."""

#     input_path = str(cfg.input_path)
#     output_path = str(cfg.output_path)

#     logger.info("Starting conversion process")
#     logger.info("Input path: %s", input_path)
#     logger.info("Output path: %s", output_path)

#     # Discover input files
#     try:
#         input_files = _discover_input_files(input_path)
#         logger.info("Found %d input files", len(input_files))
#         for file in input_files[:5]:  # Log first 5 files
#             logger.info("Input file: %s", file)
#         if len(input_files) > 5:
#             logger.info("... and %d more files", len(input_files) - 5)
#     except Exception as e:
#         logger.error("Failed to discover input files: %s", e)
#         return {"success": False, "reason": "discovery_failed", "error": str(e)}

#     if not input_files:
#         logger.warning("No input files found in %s", input_path)
#         return {"success": False, "reason": "no_input_files"}

#     # Create output directory
#     fsspec_mkdirs(output_path, exist_ok=True)

#     # Generate conversion tasks
#     tasks, datasets = _generate_conversion_tasks(input_files, output_path, cfg.skip_existing)

#     if not tasks:
#         logger.info("No files to convert")
#         return {"success": True, "reason": "already_converted", "skipped": True}

#     metadata_records: list[dict[str, Any]] = []

#     # Execute conversion tasks
#     refs = simple_backpressure(
#         _convert_single_file,
#         iter(tasks),
#         max_in_flight=cfg.max_concurrent_conversions,
#         fetch_local=True,
#     )

#     for dataset, ref in zip(datasets, refs, strict=False):

#         result = ray.get(ref)


#         metadata_records.append(
#             {
#                 "benchmark": dataset.benchmark,
#                 "start_date": dataset.start_date,
#                 "end_date": dataset.end_date,
#                 "source": dataset.source_label,
#                 "input_file": dataset.input_path,
#                 "output_file": dataset.output_path,
#                 "records": result.get("records"),
#             }
#         )

#     # Write metadata
#     _write_conversion_metadata(cfg, metadata_records)

#     return {"success": True, "converted": metadata_records}

def convert_uncheatable_eval_files(cfg: UncheatableEvalConvertConfig) -> dict[str, Any]:
    """Convert downloaded Uncheatable Eval files to the normalized format (sequential, no Ray)."""

    input_path = str(cfg.input_path)
    output_path = str(cfg.output_path)

    logger.info("Starting conversion process")
    logger.info("Input path: %s", input_path)
    logger.info("Output path: %s", output_path)

    # Discover input files
    try:
        input_files = _discover_input_files(input_path)
        logger.info("Found %d input files", len(input_files))
        for file in input_files[:5]:
            logger.info("Input file: %s", file)
        if len(input_files) > 5:
            logger.info("... and %d more files", len(input_files) - 5)
    except Exception as e:
        logger.error("Failed to discover input files: %s", e)
        return {"success": False, "reason": "discovery_failed", "error": str(e)}

    if not input_files:
        logger.warning("No input files found in %s", input_path)
        return {"success": False, "reason": "no_input_files"}

    # Create output directory
    fsspec_mkdirs(output_path, exist_ok=True)

    # Generate conversion tasks
    tasks, datasets = _generate_conversion_tasks(input_files, output_path, cfg.skip_existing)

    if not tasks:
        logger.info("No files to convert")
        return {"success": True, "reason": "already_converted", "skipped": True}

    metadata_records: list[dict[str, Any]] = []

    # Sequentially run conversions
    for input_file, output_file, dataset in tasks:
        logger.info("Converting %s -> %s", input_file, output_file)

        with fsspec.open(input_file, "r", encoding="utf-8") as infile:
            payload = json.load(infile)

        if not isinstance(payload, list):
            raise ValueError(f"Expected list in dataset {dataset.name}, found {type(payload).__name__}")

        fsspec_mkdirs(os.path.dirname(output_file), exist_ok=True)

        record_count = 0
        logger.info("Using fsspec version: %s", fsspec.__version__)
        with fsspec.open(output_file, "wt", encoding="utf-8", compression="gzip") as outfile:
            for index, raw in enumerate(payload):
                normalized = _normalize_record(raw, dataset, index)
                json.dump(normalized, outfile, ensure_ascii=False)
                outfile.write("\n")
                record_count += 1

        metadata_records.append(
            {
                "benchmark": dataset.benchmark,
                "start_date": dataset.start_date,
                "end_date": dataset.end_date,
                "source": dataset.source_label,
                "input_file": dataset.input_path,
                "output_file": dataset.output_path,
                "records": record_count,
            }
        )
        logger.info("Finished %s: %d records", dataset.name, record_count)

    # Write metadata
    _write_conversion_metadata(cfg, metadata_records)

    return {"success": True, "converted": metadata_records}


def make_hf_download_step(
    *,
    name: str = "raw/uncheatable-eval/download",
    hf_dataset_id: str = "ziqinghuang/uncheatable",
    revision: str = "main",
) -> ExecutorStep[DownloadConfig]:
    """Create an :class:`ExecutorStep` that downloads Uncheatable Eval datasets from Hugging Face."""

    config = DownloadConfig(
        hf_dataset_id=hf_dataset_id,
        revision=revision,
    )

    return ExecutorStep(
        name=name,
        fn=download_hf,
        config=config,
    )

def make_uncheatable_eval_convert_step(
    *,
    name: str = "raw/uncheatable-eval/convert",
    input_path: str,
    max_concurrent_conversions: int = 8,
    skip_existing: bool = True,
) -> ExecutorStep[UncheatableEvalConvertConfig]:
    """Create an :class:`ExecutorStep` that converts Uncheatable Eval dumps."""

    config = UncheatableEvalConvertConfig(
        input_path=input_path,
        output_path=this_output_path(),
        max_concurrent_conversions=max_concurrent_conversions,
        skip_existing=skip_existing,
    )

    return ExecutorStep(
        name=name,
        fn=convert_uncheatable_eval_files,
        config=config,
        pip_dependency_groups=["quality_dedup_consolidate"],
    )


__all__ = [
    "ConversionDataset",
    "UncheatableEvalConvertConfig",
    "convert_uncheatable_eval_files",
    "make_hf_download_step"
    "make_uncheatable_eval_convert_step",
]