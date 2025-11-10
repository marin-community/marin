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

"""
Tokenize datasets using zephyr pipeline and write to Levanter cache format.

Supports both regular file paths and HuggingFace datasets. For HF datasets, downloads
them first then tokenizes the downloaded files.

Usage:
    # Tokenize a single file for testing
    uv run zephyr --entry-point=main \
        --backend=ray --cluster=us-central2 --max-parallelism=10 --memory=2GB \
        lib/marin/src/marin/processing/tokenize/tokenize.py \
        --train_paths '["gs://marin-us-central2/raw/ar5iv/ar5iv-04-2024-no-problem-d1522a/0001.jsonl.gz"]' \
        --cache_path gs://marin-us-central2/cache/test-tokenize \
        --tokenizer gpt2 \
        --validation_paths '[]'

    # Tokenize entire directory
    uv run zephyr --entry-point=main \
        --backend=ray --cluster=us-central2 --max-parallelism=500 --memory=4GB \
        lib/marin/src/marin/processing/tokenize/tokenize.py \
        --train_paths '["gs://marin-us-central2/raw/ar5iv/ar5iv-04-2024-no-problem-d1522a/"]' \
        --cache_path gs://marin-us-central2/cache/ar5iv-gpt2 \
        --tokenizer gpt2 \
        --validation_paths '[]'

Args:
    train_paths: List of input paths (files or directories) for training, or empty list
    validation_paths: List of input paths for validation, or empty list
    cache_path: Base directory to save tokenized files (train/ and validation/ subdirs)
    tokenizer: Tokenizer name (must match tokenizer used in training run)
    format: LmDatasetFormatBase (default: TextLmDatasetFormat())

The tokenized data will be written to:
    {cache_path}/train/     - Training cache
    {cache_path}/validation/ - Validation cache (if provided)
"""

import abc
import dataclasses
import logging
import os
import re
from collections.abc import Sequence

import draccus
import transformers
from levanter.data.text import (
    HfDatasetSourceConfig,
    LmDatasetFormatBase,
    LMDatasetSourceConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from zephyr import Dataset, create_backend, flow_backend
from zephyr.readers import load_file

from marin.execution.executor import ExecutorStep, InputName, VersionedValue
from marin.utils import fsspec_glob, fsspec_isdir, fsspec_size

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class HfDatasetSpec:
    """Specification for a HuggingFace dataset and optional subset name."""

    id: str
    name: str | None = None


class TokenizeConfigBase(abc.ABC):
    """Base class for tokenize configs."""

    @abc.abstractmethod
    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
    ) -> LMDatasetSourceConfig:
        """
        Create a Levanter dataset source config from this config and the actual output path.
        """
        pass


@dataclasses.dataclass(frozen=True)
class TokenizeConfig(TokenizeConfigBase):
    train_paths: list[str]  # path to training data
    validation_paths: list[str]  # path to validation data
    cache_path: str  # base path to save the tokenized files
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa
    window_size_bytes: int = 10_000_000_000
    allow_test_in_train: bool = False
    """
    If True, allows 'test' or 'validation' in the train_paths. This is useful for datasets that have
    'test' or 'validation' in the file names, but are not actually test or validation sets.
    """

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
    ) -> LMDatasetSourceConfig:
        """
        For use in Levanter training runs with mixtures of datasets.

        Args:
            actual_output_path: The actual output path to use for the cache. Since we often pass in an InputName,
                we need to resolve it to a string.

            include_raw_paths: if false, don't include paths to raw data in Levanter's config. This means we'll be able
                to run training without the original training data, but hte provenance won't be recorded in wandb.

        """
        return UrlDatasetSourceConfig(
            tags=self.tags,
            train_urls=self.train_paths if include_raw_paths else [],
            validation_urls=self.validation_paths if include_raw_paths else [],
            cache_dir=actual_output_path,
            format=self.format,
        )

    def __post_init__(self):
        if not self.train_paths and not self.validation_paths:
            raise ValueError("At least one of train_paths or validation_paths must be specified")

        assert not isinstance(self.train_paths, str | InputName)
        assert not isinstance(self.validation_paths, str | InputName)

        if isinstance(self.train_paths, Sequence):
            assert "/" not in self.train_paths, "don't use the entire fs for train paths!"

        if isinstance(self.validation_paths, Sequence):
            assert "/" not in self.validation_paths, "don't use the entire fs for validation paths!"

        _validate_train_urls(self.train_paths, self.allow_test_in_train)


@dataclasses.dataclass(frozen=True)
class HfTokenizeConfig(TokenizeConfigBase):
    """
    Tokenize a HuggingFace dataset directly without having to download it first.
    """

    id: str  # HF dataset id
    cache_path: str  # base path to save the tokenized files
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run
    name: str | None = None  # HF dataset name
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa: RUF009
    window_size_bytes: int = 10_000_000_000

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
    ) -> LMDatasetSourceConfig:
        return HfDatasetSourceConfig(
            id=self.id,
            name=self.name,
            tags=self.tags,
            cache_dir=actual_output_path,
            format=self.format,
        )


def _validate_train_urls(train_paths: list[str | InputName], warn):
    """
    Validates the training data URLs or InputName attributes to ensure they do not contain forbidden patterns.
    Raises a ValueError if a forbidden pattern is found.
    """
    for item in train_paths:
        url_or_name_to_check: str = ""
        if isinstance(item, str):
            url_or_name_to_check = item
        elif isinstance(item, InputName):
            url_or_name_to_check = item.name or ""

        # \b doesn't work because of underscores
        if re.search(r"[^a-zA-Z]test[^a-zA-Z]", url_or_name_to_check) or re.search(r"validation", url_or_name_to_check):
            if warn:
                logger.warning(
                    f"Warning: Training data URL or InputName '{url_or_name_to_check}' contains a forbidden pattern "
                )
            else:
                raise ValueError(
                    f"Error: Training data URL or InputName '{url_or_name_to_check}' contains a forbidden pattern "
                    "('test' or 'validation'). "
                    "Please ensure training data does not include test or validation sets."
                )


def _get_files_by_extensions(input_paths: list[str], extensions: list[str]) -> list[str]:
    """
    Get a list of all filepaths with the specified extension from the input paths.
    """
    output_paths = []
    for path in input_paths:
        assert path != "/"
        if path.endswith("/") or fsspec_isdir(path):
            logger.info(f"Getting all {extensions} files in {path}")
            for ex in extensions:
                output_paths.extend(fsspec_glob(os.path.join(path, f"**/*.{ex}")))
        else:
            output_paths.extend(fsspec_glob(path))

    return output_paths


def _get_filepaths_to_tokenize(input_paths: list[str]) -> list[str]:
    """
    Get all file paths to tokenize from the input paths.
    Handles json/jsonl.{gz,zst,zstd}, and parquet.
    """
    if isinstance(input_paths, VersionedValue):
        input_paths = input_paths.value

    if len(input_paths) == 0:
        return []
    elif any(isinstance(x, InputName | ExecutorStep) for x in input_paths):
        return input_paths

    # we're only going to have one or the other, but might as well return both
    out = _get_files_by_extensions(input_paths, ["json.{gz,zst,zstd}", "jsonl.{gz,zst,zstd}", "parquet", "json"])

    # NOTE(chris): When downloading the datasets from HF using default_download, we create a
    # provenance.json file but we seek to exclude this since we shouldn't train on this.
    out = [x for x in out if "provenance.json" not in x]

    if not len(out):
        raise ValueError(
            f"No valid jsonl or parquet files found in {input_paths}. "
            "Please provide a path to a directory containing jsonl or parquet files."
        )

    return out


def _group_files_by_size(file_infos, max_bytes: int):
    """Group files into lists by total size threshold."""
    current_group = []
    current_size = 0

    for info in file_infos:
        if current_size + info["size"] >= max_bytes and current_group:
            yield current_group
            current_group = []
            current_size = 0
        current_group.append(info["filename"])
        current_size += info["size"]

    if current_group:
        yield current_group


def _tokenize_batches(config: TokenizeConfig | HfTokenizeConfig, batches):
    """Load tokenizer once per shard and process all batches."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer)
    batch_processor = preprocessor_for_format(config.format, tokenizer)

    for batch in batches:
        yield from batch_processor(batch)


def tokenize(config: TokenizeConfigBase):
    """Tokenize datasets using zephyr pipeline.

    Processes train and validation splits separately, writing to Levanter cache format.
    For HuggingFace datasets, downloads them first then tokenizes the downloaded files.
    """

    if isinstance(config, TokenizeConfig):
        train_paths = _get_filepaths_to_tokenize(config.train_paths) if config.train_paths else []
        validation_paths = _get_filepaths_to_tokenize(config.validation_paths) if config.validation_paths else []
    elif isinstance(config, HfTokenizeConfig):
        # Download HF dataset first, then tokenize the downloaded files
        from marin.download.huggingface.download import DownloadConfig
        from marin.download.huggingface.download_hf import download_hf

        download_dir = os.path.join(config.cache_path, "raw")
        download_config = DownloadConfig(
            hf_dataset_id=config.id,
            gcs_output_path=download_dir,
            hf_urls_glob=[],  # Download entire dataset
        )
        logger.info(f"Downloading HuggingFace dataset {config.id} to {download_dir}")
        download_hf(download_config)

        # Tokenize the downloaded files
        train_paths = _get_filepaths_to_tokenize([os.path.join(download_dir, "train")])
        validation_paths = _get_filepaths_to_tokenize([os.path.join(download_dir, "validation")])
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    if not train_paths and not validation_paths:
        raise ValueError("No input files specified. Nothing to do.")

    backend = flow_backend()

    def create_pipeline(paths: list[str], split_name: str) -> Dataset:
        """Create a tokenization pipeline for a set of file paths."""
        logger.info(f"Statting {len(paths)} {split_name} files to compute groups")

        file_stats = list(
            create_backend("threadpool").execute(
                Dataset.from_list(paths).map(lambda path: {"filename": path, "size": fsspec_size(path)})
            )
        )
        file_groups = list(_group_files_by_size(file_stats, config.window_size_bytes))
        logger.info(f"Grouped {len(paths)} files into {len(file_groups)} groups by size")

        # Main pipeline processes file groups
        return (
            Dataset.from_list(file_groups)
            .flat_map(lambda file_list: file_list)  # Unpack groups to individual files
            .flat_map(load_file)
            .batch(100)
            .map_shard(lambda batches: _tokenize_batches(config, batches))
            .write_levanter_cache(os.path.join(config.cache_path, split_name), config.tokenizer, config.format)
        )

    if train_paths:
        list(backend.execute(create_pipeline(train_paths, "train-{shard:05d}")))

    if validation_paths:
        list(backend.execute(create_pipeline(validation_paths, "validation-{shard:05d}")))


@draccus.wrap()
def main(config: TokenizeConfig):
    """Main entry point for tokenizing datasets with TokenizeConfig."""
    tokenize(config)
