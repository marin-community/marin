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

"""Utilities for filtering Common Pile datasets by metadata extensions."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field

import draccus
import fsspec
import ray

from marin.core.runtime import TaskConfig, cached_or_construct_output, map_files_in_directory
from marin.utils import fsspec_rm

logger = logging.getLogger("ray")


def _normalize_extension(value: object, *, casefold: bool) -> str | None:
    """Normalise an extension value to a lower-cased string that starts with a dot."""

    if value is None:
        return None

    if isinstance(value, list) or isinstance(value, tuple):
        # Some datasets store extension metadata as a singleton list. Prefer the first entry.
        value = value[0] if value else None

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.casefold() if casefold else text
    if not text.startswith("."):
        text = f".{text}"
    return text


def _normalise_allowed_extensions(extensions: Sequence[str], *, casefold: bool) -> set[str]:
    normalised = set()
    for item in extensions:
        normalised_extension = _normalize_extension(item, casefold=casefold)
        if normalised_extension is None:
            raise ValueError("Allowed extensions must be non-empty strings.")
        normalised.add(normalised_extension)
    if not normalised:
        raise ValueError("allowed_extensions must contain at least one extension.")
    return normalised


@dataclass(frozen=True)
class FilterByMetadataExtensionConfig:
    """Configuration for filtering JSONL datasets by metadata extension."""

    input_path: str
    output_path: str
    allowed_extensions: Sequence[str]
    metadata_column: str = "metadata"
    extension_key: str = "extension"
    input_glob: str = "*.json*"
    casefold: bool = True
    drop_missing_extensions: bool = True
    task_config: TaskConfig = field(default_factory=TaskConfig)

    def normalised_allowed_extensions(self) -> set[str]:
        return _normalise_allowed_extensions(self.allowed_extensions, casefold=self.casefold)


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _filter_file_by_metadata_extension(
    input_file_path: str, output_file_path: str, config: FilterByMetadataExtensionConfig
) -> dict[str, int]:
    """Filter a JSONL(.gz) file to include only rows with allowed extensions."""

    allowed_extensions = config.normalised_allowed_extensions()
    # Remove any stale outputs before we start processing.
    fsspec_rm(output_file_path)

    total_records = 0
    kept_records = 0

    with fsspec.open(input_file_path, "rt", compression="infer") as input_file, ExitStack() as stack:
        output_handle = None
        for line in input_file:
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON record from %s", input_file_path)
                continue

            metadata = record.get(config.metadata_column)
            if not isinstance(metadata, dict):
                if config.drop_missing_extensions:
                    continue
                metadata = {}

            extension_value = metadata.get(config.extension_key)
            normalised_extension = _normalize_extension(extension_value, casefold=config.casefold)
            if normalised_extension is None:
                if config.drop_missing_extensions:
                    continue
            if normalised_extension not in allowed_extensions:
                continue

            if output_handle is None:
                output_handle = stack.enter_context(fsspec.open(output_file_path, "wt", compression="infer"))
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_records += 1

    if output_handle is None:
        # Remove the output if we didn't emit any rows.
        fsspec_rm(output_file_path)

    logger.info(
        "Filtered %s -> %s | kept %d of %d records", input_file_path, output_file_path, kept_records, total_records
    )
    return {"total": total_records, "kept": kept_records}


def filter_dataset_by_metadata_extension(config: FilterByMetadataExtensionConfig) -> None:
    """Filter every file in ``config.input_path`` by extension metadata."""

    allowed_extensions = config.normalised_allowed_extensions()
    logger.info(
        "Filtering %s for %d allowed extensions: %s",
        config.input_path,
        len(allowed_extensions),
        sorted(allowed_extensions),
    )

    if config.task_config.task_options:
        raise ValueError("task_options are not supported for metadata extension filtering steps.")

    futures = list(
        map_files_in_directory(
            _filter_file_by_metadata_extension,
            config.input_path,
            config.input_glob,
            config.output_path,
            task_config=config.task_config,
            config=config,
        )
    )
    ray.get(futures)


@draccus.wrap()
def main(config: FilterByMetadataExtensionConfig) -> None:
    filter_dataset_by_metadata_extension(config)


if __name__ == "__main__":
    main()
