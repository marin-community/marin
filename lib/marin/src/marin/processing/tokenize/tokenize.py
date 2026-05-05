# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize datasets directly into a Levanter cache (legacy end-to-end path).

Supports both regular file paths and HuggingFace datasets. For HF datasets, downloads
them first then tokenizes the downloaded files. The output is a single consolidated
Levanter ``TreeStore`` per split (train/validation), as it was before the datakit
split. No intermediate parquet round-trip is performed on this path.

For datakit-style separation (per-doc attribute parquet → store assembly), see
:mod:`marin.processing.tokenize.attributes` and :mod:`marin.processing.tokenize.store_builder`.
The shared tokenization core lives in :mod:`marin.processing.tokenize._core`.
"""
import abc
import dataclasses
import logging
import os
import re
import time
from collections.abc import Sequence

import draccus
from datasets import load_dataset_builder
from fray import ResourceConfig
from levanter.data.text import (
    HfDatasetSourceConfig,
    LmDatasetFormatBase,
    LmDatasetSourceConfigBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.tokenizers import TokenizerBackend
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.dataset import FileEntry
from zephyr.readers import load_file

from marin.execution.executor import InputName, VersionedValue
from marin.processing.tokenize._core import (
    MIN_GROUP_BYTES,
    bundle_files_by_size,
    compute_target_group_bytes,
    drop_sidecars,
    expand_tokenize_paths,
    glob_with_sizes,
    parquet_window_hint,
    tokenize_pipeline,
)
from marin.processing.tokenize.store_builder import build_from_datasets, write_stats_json
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)


# Re-exports for callers / tests that import these from tokenize.py.
__all__ = [
    "MIN_GROUP_BYTES",
    "HfDatasetSpec",
    "HfTokenizeConfig",
    "TokenizeConfig",
    "TokenizeConfigBase",
    "_bundle_files_by_size",
    "_compute_target_group_bytes",
    "main",
    "tokenize",
]

# Backward-compatible aliases — kept because tests import these names.
_bundle_files_by_size = bundle_files_by_size
_compute_target_group_bytes = compute_target_group_bytes


@dataclasses.dataclass(frozen=True)
class HfDatasetSpec:
    """Specification for a HuggingFace dataset and optional subset name."""

    id: str
    name: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenizeConfigBase(abc.ABC):
    """Base class for tokenize configs."""

    max_workers: int = 4096
    worker_resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig(ram="10g", disk="5g"))

    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF
    """Backend to use for tokenization. HF uses the HuggingFace tokenizers library directly.
    KITOKEN uses the kitoken library."""

    num_shards: int | None = None
    """Override the number tokenize shards. When set, files are grouped to produce approximately
    this many shards instead of deriving the count from max_workers. This can be useful if you want
    more shards than max_workers, for example to mitigate the cost of retrying a single shard."""

    levanter_batch_size: int | None = None
    """Number of tokenized records to accumulate before flushing to disk. Defaults to 16384.
    Lower values reduce peak memory for datasets with large documents."""

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
    allow_test_in_train: bool = False
    """
    If True, allows 'test' or 'validation' in the train_paths. This is useful for datasets that have
    'test' or 'validation' in the file names, but are not actually test or validation sets.
    """

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

    sample_count: int | None = None
    """Number of samples to tokenize. If None, tokenize all samples."""

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


def _resolve_input_paths(input_paths: list[str] | VersionedValue) -> list[str]:
    """Unwrap a ``VersionedValue`` if needed and return the concrete path list."""
    if isinstance(input_paths, VersionedValue):
        return list(input_paths.value)
    return list(input_paths)


def _local_preprocess_paths(files: list[FileEntry], config: TokenizeConfigBase) -> list[list[str]]:
    """Bundle files into size-balanced groups for distributed processing."""
    files = sorted(files, key=lambda f: f.path)
    total_input_bytes = sum(f.size for f in files)
    if config.num_shards is not None:
        target_group_bytes = compute_target_group_bytes(total_input_bytes, config.num_shards)
    else:
        target_group_bytes = compute_target_group_bytes(total_input_bytes, config.max_workers)
    file_groups = list(bundle_files_by_size(files, target_group_bytes))
    logger.info(
        f"Grouped {len(files):,} files ({total_input_bytes / 1e9:.2f} GB) into {len(file_groups):,} groups "
        f"(target {target_group_bytes / 1e9:.2f} GB/group)."
    )
    return file_groups


def _split_already_done(cache_path: str, split_name: str) -> bool:
    ledger_path = os.path.join(cache_path, split_name, "shard_ledger.json")
    if fsspec_exists(ledger_path):
        logger.info("Shard ledger already exists for %s at %s; skipping", split_name, ledger_path)
        return True
    return False


def _exemplar_for(
    file_groups: list[list[str]],
    *,
    config: TokenizeConfigBase,
) -> dict:
    """Compute a post-tokenize exemplar by running one record through the pipeline.

    The exemplar still carries ``id`` (because the shared tokenize pipeline emits
    ``{id, input_ids, ...}``); :func:`build_from_datasets` strips it before opening
    the TreeStore.
    """
    sample_path = parquet_window_hint(file_groups)
    ds = Dataset.from_list(file_groups[0][0:1]).flat_map(load_file)
    pipeline, _ = tokenize_pipeline(
        ds,
        data_format=config.format,
        sample_count=1,
        sample_parquet_path=sample_path,
        levanter_batch_size=None,
    )
    ctx = ZephyrContext(
        resources=config.worker_resources,
        max_workers=1,
        name="tokenize-exemplar",
    )
    ctx.put("tokenizer_name", config.tokenizer)
    ctx.put("tokenizer_backend", config.tokenizer_backend)
    return ctx.execute(pipeline, verbose=False).results[0]


def _strip_id(record: dict) -> dict:
    if "id" not in record:
        return record
    return {k: v for k, v in record.items() if k != "id"}


def _run_split(
    *,
    config: TokenizeConfigBase,
    file_groups: list[list[str]],
    split_name: str,
) -> None:
    """End-to-end pipeline for one split: glob → tokenize → consolidate → stats."""
    prefix = os.path.join(config.cache_path, split_name)
    pipeline_start = time.monotonic()

    sample_path = parquet_window_hint(file_groups)
    ds = Dataset.from_list(file_groups).flat_map(lambda file_list: file_list).flat_map(load_file)
    tokenized_ds, batch_size = tokenize_pipeline(
        ds,
        data_format=config.format,
        sample_count=config.sample_count,
        sample_parquet_path=sample_path,
        levanter_batch_size=config.levanter_batch_size,
    )

    ctx = ZephyrContext(
        resources=config.worker_resources,
        max_workers=min(config.max_workers, len(file_groups)),
        name=f"tokenize-{split_name}",
    )
    # Broadcast tokenizer config to workers. We send name + backend rather than
    # the tokenizer object because not all backends support pickling.
    ctx.put("tokenizer_name", config.tokenizer)
    ctx.put("tokenizer_backend", config.tokenizer_backend)

    logger.info("Computing exemplar for cache consolidation")
    exemplar = _strip_id(_exemplar_for(file_groups, config=config))

    ledger = build_from_datasets(
        ctx=ctx,
        dataset=tokenized_ds,
        output_path=prefix,
        exemplar=exemplar,
        batch_size=batch_size,
    )

    stats_path, _ = write_stats_json(prefix, exemplar, ledger)
    pipeline_elapsed = time.monotonic() - pipeline_start
    logger.info(
        "%s pipeline complete: %d docs in %.1fs. Wrote stats to %s",
        split_name,
        ledger.total_num_rows,
        pipeline_elapsed,
        stats_path,
    )


def tokenize(config: TokenizeConfigBase) -> None:
    """Tokenize datasets using zephyr pipeline.

    Processes train and validation splits separately, writing to Levanter cache format.
    For HuggingFace datasets, downloads them first then tokenizes the downloaded files.
    """
    if isinstance(config, TokenizeConfig):
        train_patterns = expand_tokenize_paths(_resolve_input_paths(config.train_paths)) if config.train_paths else []
        validation_patterns = (
            expand_tokenize_paths(_resolve_input_paths(config.validation_paths)) if config.validation_paths else []
        )
    elif isinstance(config, HfTokenizeConfig):
        logger.info(f"Loading dataset metadata for {config.id}" + (f" (config: {config.name})" if config.name else ""))

        builder = load_dataset_builder(config.id, name=config.name, revision=config.revision)
        data_files = builder.config.data_files

        if data_files is None:
            raise ValueError(
                f"Dataset {config.id} does not have data_files metadata. "
                "This might be a dataset that requires custom loading logic."
            )

        train_patterns = list(data_files.get("train", []))
        validation_patterns = list(data_files.get("validation", data_files.get("test", [])))
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    # Resolve patterns → concrete files with sizes (single list-objects call per pattern)
    train_files = drop_sidecars(glob_with_sizes(train_patterns))
    validation_files = drop_sidecars(glob_with_sizes(validation_patterns))

    if isinstance(config, TokenizeConfig):
        _validate_train_urls([f.path for f in train_files], warn=config.allow_test_in_train)

    if train_files:
        logger.info(f"Found {len(train_files)} training files")
    if validation_files:
        logger.info(f"Found {len(validation_files)} validation files")

    if train_patterns and not train_files:
        raise ValueError(f"No training files matched configured patterns: {train_patterns}")
    if validation_patterns and not validation_files:
        raise ValueError(f"No validation files matched configured patterns: {validation_patterns}")
    if not train_files and not validation_files:
        raise ValueError("No input files specified. Nothing to do.")

    # TODO (rav): both train and val could run at the same time
    if train_files and not _split_already_done(config.cache_path, "train"):
        train_groups = _local_preprocess_paths(train_files, config)
        _run_split(config=config, file_groups=train_groups, split_name="train")

    if validation_files and not _split_already_done(config.cache_path, "validation"):
        validation_groups = _local_preprocess_paths(validation_files, config)
        _run_split(config=config, file_groups=validation_groups, split_name="validation")


@draccus.wrap()
def main(config: TokenizeConfig):

    configure_logging(level=logging.INFO)
    tokenize(config)
