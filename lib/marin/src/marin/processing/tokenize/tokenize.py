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
"""

import abc
import dataclasses
import json
import logging
import os
import re
from collections.abc import Iterator, Sequence
from typing import Any

import fsspec

import draccus
from fray.job.context import JobContext
import humanfriendly
import transformers
from datasets import load_dataset_builder
from fray.job import create_job_ctx
from levanter.data.text import (
    HfDatasetSourceConfig,
    LmDatasetFormatBase,
    LmDatasetSourceConfigBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from levanter.store.cache import consolidate_shard_caches
from zephyr import Backend, Dataset
from zephyr.readers import load_file

from marin.execution.executor import ExecutorStep, InputName, VersionedValue
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_size

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
    ) -> LmDatasetSourceConfigBase:
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

    sample_count: int | None = None
    """Number of samples to tokenize. If None, tokenize all samples."""

    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa
    """
    The format of the dataset. This is used to determine how to tokenize the data.
    See Levanter's documentation for more details.
    """
    window_size_bytes: int = 10_000_000_000
    allow_test_in_train: bool = False
    """
    If True, allows 'test' or 'validation' in the train_paths. This is useful for datasets that have
    'test' or 'validation' in the file names, but are not actually test or validation sets.
    """
    # TODO (rav): remove this once there's better way to capture this in datakit
    zephyr_num_cpus: int = 2
    zephyr_memory: int = humanfriendly.parse_size("32GB", binary=True)

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
    ) -> LmDatasetSourceConfigBase:
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
    revision: str | None = None  # HF dataset revision (commit hash, branch, or tag). Defaults to "main"
    name: str | None = None  # HF dataset name
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa: RUF009
    window_size_bytes: int = 10_000_000_000

    sample_count: int | None = None
    """Number of samples to tokenize. If None, tokenize all samples."""

    # TODO (rav): remove this once there's better way to capture this in datakit
    zephyr_num_cpus: int = 2
    zephyr_memory: int = humanfriendly.parse_size("32GB", binary=True)

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
    ) -> LmDatasetSourceConfigBase:
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

    out = _get_files_by_extensions(input_paths, ["json.{gz,zst,zstd}", "jsonl.{gz,zst,zstd}", "parquet", "json"])
    out = [x for x in out if "provenance.json" not in x]

    if not len(out):
        raise ValueError(
            f"No valid jsonl or parquet files found in {input_paths}. "
            "Please provide a path to a directory containing jsonl or parquet files."
        )

    return out


def _bundle_files_by_size(file_infos, max_bytes: int):
    """Bundle files into groups, with each group having a total size less than max_bytes."""
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


def _tokenize_batches(
    *, ctx: JobContext, tokenizer_ref: Any, config: TokenizeConfig | HfTokenizeConfig, batches: Iterator[dict]
) -> Iterator[dict]:
    """Tokenize a list of batches using the specified tokenizer and format."""
    tokenizer: transformers.PreTrainedTokenizer = ctx.get(tokenizer_ref)
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
        # Validate expanded paths to catch validation/test files that were inside directories
        _validate_train_urls(train_paths, warn=config.allow_test_in_train)
    elif isinstance(config, HfTokenizeConfig):
        logger.info(f"Loading dataset metadata for {config.id}" + (f" (config: {config.name})" if config.name else ""))

        builder = load_dataset_builder(config.id, name=config.name, revision=config.revision)
        data_files = builder.config.data_files

        if data_files is None:
            raise ValueError(
                f"Dataset {config.id} does not have data_files metadata. "
                "This might be a dataset that requires custom loading logic."
            )

        train_paths = data_files.get("train", [])
        validation_paths = data_files.get("validation", data_files.get("test", []))

        if train_paths:
            logger.info(f"Found {len(train_paths)} training files in {config.id}")
        if validation_paths:
            logger.info(f"Found {len(validation_paths)} validation files in {config.id}")
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    if not train_paths and not validation_paths:
        raise ValueError("No input files specified. Nothing to do.")

    def run_pipeline(paths: list[str], split_name: str) -> None:
        prefix = os.path.join(config.cache_path, split_name)
        ledger_path = os.path.join(prefix, "shard_ledger.json")

        if fsspec_exists(ledger_path):
            logger.info(
                "Shard ledger already exists for %s at %s; skipping tokenization step",
                split_name,
                ledger_path,
            )
            return

        thread_ctx = create_job_ctx("threadpool")
        file_stats = list(
            Backend.execute(
                Dataset.from_list(paths).map(lambda path: {"filename": path, "size": fsspec_size(path)}),
                context=thread_ctx,
                verbose=False,
            )
        )
        file_groups = list(_bundle_files_by_size(file_stats, config.window_size_bytes))
        logger.info(f"Grouped {len(paths)} files into {len(file_groups)} groups by size.")

        ds = Dataset.from_list(file_groups).flat_map(lambda file_list: file_list).flat_map(load_file)

        if config.sample_count is not None:
            logger.info(f"Sampling {config.sample_count} examples from {split_name} set for tokenization")
            ds = ds.take_per_shard(config.sample_count)

        cluster_ctx = create_job_ctx(context_type="auto", num_cpus=config.zephyr_num_cpus, memory=config.zephyr_memory)
        # NOTE: "broadcast" the tokenizer on the cluster context
        tokenizer_ref = cluster_ctx.put(transformers.AutoTokenizer.from_pretrained(config.tokenizer))

        temp_shards = (
            ds.window(64)
            .map_shard(
                lambda batches: _tokenize_batches(
                    ctx=cluster_ctx, tokenizer_ref=tokenizer_ref, config=config, batches=batches
                )
            )
            .write_levanter_cache(f"{prefix}/part-{{shard:05d}}", metadata={}, skip_existing=True)
        )

        shard_paths = Backend.execute(temp_shards, context=cluster_ctx)

        logger.info("Computing exemplar for cache consolidation")
        exemplar = Backend.execute(
            Dataset.from_list(paths[0:1])
            .flat_map(load_file)
            .take_per_shard(1)
            .map_shard(
                lambda example: _tokenize_batches(
                    ctx=cluster_ctx, tokenizer_ref=tokenizer_ref, config=config, batches=[example]
                )
            ),
            context=cluster_ctx,
            verbose=False,
        )[0]

        logger.info(f"Tokenization complete, consolidating {len(shard_paths)} shards into {prefix}")
        consolidate_shard_caches(
            shard_cache_paths=shard_paths, output_path=prefix, exemplar=exemplar, context=cluster_ctx
        )

        # Aggregate token counts from shard stats
        total_tokens = 0
        total_elements = 0
        for shard_path in shard_paths:
            stats_path = f"{shard_path}/.stats.json"
            with fsspec.open(stats_path) as f:
                stats = json.load(f)
                total_tokens += stats.get("token_count", 0)
                total_elements += stats.get("num_rows", 0)

        stats_path = os.path.join(prefix, ".stats.json")
        logger.info(
            f"Writing total token count ({total_tokens:,}) and element count ({total_elements:,}) to {stats_path}"
        )
        with fsspec.open(stats_path, "w") as f:
            json.dump({"total_tokens": total_tokens, "total_elements": total_elements}, f)

    if train_paths:
        run_pipeline(train_paths, "train")

    if validation_paths:
        run_pipeline(validation_paths, "validation")


@draccus.wrap()
def main(config: TokenizeConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    tokenize(config)
