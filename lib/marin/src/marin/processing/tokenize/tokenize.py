# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
import time
from collections.abc import Iterator, Sequence

import draccus
import fsspec
import transformers
from datasets import load_dataset_builder
from fray.v2 import ResourceConfig
from fray.v2.local_backend import LocalClient
from levanter.data.text import (
    HfDatasetSourceConfig,
    LmDatasetFormatBase,
    LmDatasetSourceConfigBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from levanter.store.cache import consolidate_shard_caches
from levanter.store.tree_store import TreeStore
from zephyr import Dataset, ZephyrContext, zephyr_worker_ctx
from zephyr.readers import load_file

from marin.execution.executor import ExecutorStep, InputName, VersionedValue
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_size

logger = logging.getLogger(__name__)

MIN_GROUP_BYTES = 100_000_000  # 100 MB floor to avoid degenerate tiny shards


def _compute_target_group_bytes(total_input_bytes: int, max_workers: int) -> int:
    """Compute target group size to produce approximately max_workers groups.

    Applies a floor of MIN_GROUP_BYTES to avoid degenerate tiny shards.
    """
    return max(total_input_bytes // max_workers, MIN_GROUP_BYTES)


@dataclasses.dataclass(frozen=True)
class HfDatasetSpec:
    """Specification for a HuggingFace dataset and optional subset name."""

    id: str
    name: str | None = None


class TokenizeConfigBase(abc.ABC):
    """Base class for tokenize configs."""

    max_workers: int = 4096
    worker_resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig(ram="5g", disk="5g"))
    writer_batch_size: int = 65536
    """Larger values mean fewer, bigger writes to the Levanter cache, which reduces per-op
    overhead. Too large a value increases memory usage and delays progress checkpointing."""

    num_shards: int | None = None
    """Override the number tokenize shards. When set, files are grouped to produce approximately
    this many shards instead of deriving the count from max_workers. This can be useful if you want
    more shards than max_workers, for example to mitigate the cost of retrying a single shard."""

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


def _tokenize_batches(*, config: TokenizeConfig | HfTokenizeConfig, batches: Iterator[Sequence[dict]]) -> Iterator[dict]:
    """Tokenize a list of batches using the specified tokenizer and format."""
    tokenizer: transformers.PreTrainedTokenizer = zephyr_worker_ctx().get_shared("tokenizer")
    batch_processor = preprocessor_for_format(config.format, tokenizer)

    batch_count = 0
    record_count = 0
    token_count = 0
    start_time = time.monotonic()

    for batch in batches:
        batch_count += 1
        for record in batch_processor(batch):
            record_count += 1
            token_count += len(record.get("input_ids", []))
            yield record
        if batch_count % 10 == 0:
            elapsed = time.monotonic() - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0
            doc_per_sec = record_count / elapsed if elapsed > 0 else 0
            avg_tok_per_doc = token_count / record_count if record_count > 0 else 0
            logger.info(
                f"Tokenized {batch_count:,} batches, {record_count:,} docs, {token_count:,} tokens in {elapsed:.1f}s "
                f"({tok_per_sec:,.0f} tokens/s, {doc_per_sec:,.1f} docs/s, {avg_tok_per_doc:,.0f} avg tokens/doc)"
            )

    elapsed = time.monotonic() - start_time
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0
    doc_per_sec = record_count / elapsed if elapsed > 0 else 0
    avg_tok_per_doc = token_count / record_count if record_count > 0 else 0
    logger.info(
        f"Tokenization done: {batch_count:,} batches, {record_count:,} docs, {token_count:,} tokens in {elapsed:.1f}s "
        f"({tok_per_sec:,.0f} tokens/s, {doc_per_sec:,.1f} docs/s, {avg_tok_per_doc:,.0f} avg tokens/doc)"
    )


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

    def local_preprocess_paths(paths: list[str]) -> list[list[str]]:
        """Scan file sizes locally and bundle into groups for distributed processing."""
        filescan_start = time.monotonic()
        local_ctx = ZephyrContext(client=LocalClient(), max_workers=8, name="tokenize-filescan")
        file_stats = list(
            local_ctx.execute(
                Dataset.from_list(paths).map(lambda path: {"filename": path, "size": fsspec_size(path)}),
                verbose=False,
            )
        )
        total_input_bytes = sum(f["size"] for f in file_stats)
        if config.num_shards is not None:
            target_group_bytes = _compute_target_group_bytes(total_input_bytes, config.num_shards)
        else:
            target_group_bytes = _compute_target_group_bytes(total_input_bytes, config.max_workers)
        file_groups = list(_bundle_files_by_size(file_stats, target_group_bytes))
        logger.info(
            f"Grouped {len(paths):,} files ({total_input_bytes / 1e9:.2f} GB) into {len(file_groups):,} groups "
            f"(target {target_group_bytes / 1e9:.2f} GB/group) in {time.monotonic() - filescan_start:.1f}s."
        )
        return file_groups

    def split_already_done(split_name: str) -> bool:
        ledger_path = os.path.join(config.cache_path, split_name, "shard_ledger.json")
        if fsspec_exists(ledger_path):
            logger.info(
                "Shard ledger already exists for %s at %s; skipping",
                split_name,
                ledger_path,
            )
            return True
        return False

    def run_pipeline(ctx: ZephyrContext, file_groups: list[list[str]], split_name: str) -> None:
        prefix = os.path.join(config.cache_path, split_name)
        pipeline_start = time.monotonic()

        ds = Dataset.from_list(file_groups).flat_map(lambda file_list: file_list).flat_map(load_file)

        if config.sample_count is not None:
            logger.info(f"Sampling {config.sample_count} examples from {split_name} set for tokenization")
            ds = ds.take_per_shard(config.sample_count)

        temp_shards = (
            ds.window(config.writer_batch_size)
            .map_shard(lambda batches: _tokenize_batches(config=config, batches=batches))
            .write_levanter_cache(
                f"{prefix}/part-{{shard:05d}}-of-{{total:05d}}",
                metadata={},
                batch_size=config.writer_batch_size,
                skip_existing=True,
            )
        )

        # Broadcast the tokenizer to all workers via ZephyrContext
        ctx.put("tokenizer", transformers.AutoTokenizer.from_pretrained(config.tokenizer))

        tokenize_start = time.monotonic()
        shard_paths = ctx.execute(temp_shards)
        tokenize_elapsed = time.monotonic() - tokenize_start

        logger.info("Computing exemplar for cache consolidation")
        exemplar = ctx.execute(
            Dataset.from_list(file_groups[0][0:1])
            .flat_map(load_file)
            .take_per_shard(1)
            .map_shard(lambda example: _tokenize_batches(config=config, batches=[example])),
            verbose=False,
        )[0]

        consolidate_start = time.monotonic()
        logger.info(f"Consolidating {len(shard_paths)} shards into {prefix}")
        ledger = consolidate_shard_caches(shard_cache_paths=shard_paths, output_path=prefix, exemplar=exemplar)
        consolidate_elapsed = time.monotonic() - consolidate_start

        total_elements = ledger.total_num_rows
        store = TreeStore.open(exemplar, prefix, mode="r", cache_metadata=True)
        total_tokens = store.tree["input_ids"].data_size if "input_ids" in store.tree else 0

        stats_path = os.path.join(prefix, ".stats.json")
        with fsspec.open(stats_path, "w") as f:
            json.dump({"total_tokens": total_tokens, "total_elements": total_elements}, f)

        pipeline_elapsed = time.monotonic() - pipeline_start
        overall_tok_per_sec = total_tokens / tokenize_elapsed if tokenize_elapsed > 0 else 0
        overall_doc_per_sec = total_elements / tokenize_elapsed if tokenize_elapsed > 0 else 0
        logger.info(
            f"{split_name} pipeline complete: {total_elements:,} docs, {total_tokens:,} tokens "
            f"in {pipeline_elapsed:.1f}s (tokenize: {tokenize_elapsed:.1f}s at {overall_tok_per_sec:,.0f} tokens/s "
            f"{overall_doc_per_sec:,.1f} docs/s, consolidate: {consolidate_elapsed:.1f}s). "
            f"Wrote stats to {stats_path}"
        )

    # TODO (rav): both train and val could run at the same time
    if train_paths and not split_already_done("train"):
        train_groups = local_preprocess_paths(train_paths)
        ctx = ZephyrContext(
            resources=config.worker_resources,
            max_workers=min(config.max_workers, len(train_groups)),
            name="tokenize-train",
        )
        run_pipeline(ctx, train_groups, "train")

    if validation_paths and not split_already_done("validation"):
        validation_groups = local_preprocess_paths(validation_paths)
        ctx = ZephyrContext(
            resources=config.worker_resources,
            max_workers=min(config.max_workers, len(validation_groups)),
            name="tokenize-validation",
        )
        run_pipeline(ctx, validation_groups, "validation")


@draccus.wrap()
def main(config: TokenizeConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    tokenize(config)
