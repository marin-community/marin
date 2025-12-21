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

"""Utilities for filtering Common Pile datasets by metadata extensions.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --cluster=us-central2 \
    lib/marin/src/marin/transform/common_pile/filter_by_extension.py \
    --input_path gs://marin-data/raw/common-pile/ \
    --output_path gs://marin-data/processed/common-pile/filtered/ \
    --allowed_extensions .txt .md
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from zephyr import Dataset, execute, load_jsonl

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

    @cached_property
    def normalised_allowed_extensions(self) -> set[str]:
        return _normalise_allowed_extensions(self.allowed_extensions, casefold=self.casefold)


def _filter_record_by_metadata_extension(record: dict, config: FilterByMetadataExtensionConfig):
    """Filter a single record and return it if it has an allowed extension.

    Args:
        record: Record from JSONL file
        config: Filter configuration

    Returns:
        The record if it matches allowed extensions, None otherwise
    """
    allowed_extensions = config.normalised_allowed_extensions

    metadata = record.get(config.metadata_column)
    if not isinstance(metadata, dict):
        if config.drop_missing_extensions:
            return None
        metadata = {}

    extension_value = metadata.get(config.extension_key)
    normalised_extension = _normalize_extension(extension_value, casefold=config.casefold)
    if normalised_extension is None:
        if config.drop_missing_extensions:
            return None
    if normalised_extension not in allowed_extensions:
        return None

    return record


def filter_dataset_by_metadata_extension(config: FilterByMetadataExtensionConfig) -> None:
    """Filter every file in ``config.input_path`` by extension metadata."""

    allowed_extensions = config.normalised_allowed_extensions
    logger.info(
        "Filtering %s for %d allowed extensions: %s",
        config.input_path,
        len(allowed_extensions),
        sorted(allowed_extensions),
    )

    pipeline = (
        Dataset.from_files(f"{config.input_path}/{config.input_glob}")
        .flat_map(load_jsonl)
        .map(lambda record: _filter_record_by_metadata_extension(record, config))
        .filter(lambda record: record is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    execute(pipeline)
