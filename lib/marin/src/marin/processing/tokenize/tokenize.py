# Copyright The Marin Authors
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

import numpy as np

import braceexpand
import draccus
import fsspec
from rigging.filesystem import open_url, url_to_fs
from datasets import load_dataset_builder
from fray.v2 import ResourceConfig
from levanter.data.text import (
    HfDatasetSourceConfig,
    LmDatasetFormatBase,
    LmDatasetSourceConfigBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from levanter.tokenizers import MarinTokenizer, TokenizerBackend, load_tokenizer
from levanter.store.cache import CacheLedger, CacheMetadata
from levanter.store.tree_store import TreeStore
from zephyr import Dataset, ZephyrContext, zephyr_worker_ctx
from zephyr.dataset import FileEntry
from zephyr.readers import InputFileSpec, load_file

from marin.execution.executor import InputName, VersionedValue
from marin.utils import fsspec_exists, fsspec_isdir
from rigging.log_setup import configure_logging

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenizeConfigBase(abc.ABC):
    """Base class for tokenize configs."""

    max_workers: int = 4096
    cache_copy_max_workers: int = 128
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


_TOKENIZE_EXTENSIONS = ["json.{gz,zst,zstd}", "jsonl.{gz,zst,zstd}", "parquet"]


# NOTE(chris): Marin's `default_download` writes a `provenance.json` sidecar next to
# downloaded HF data. Downstream TokenizeConfig jobs glob those directories and must
# exclude sidecars so we don't train on provenance records. Applied uniformly to both
# splits and both config types — HF hub datasets don't ship sidecars named this way,
# so the filter is a no-op on the HfTokenizeConfig path.
_MARIN_SIDECAR_NAMES = frozenset({"provenance.json"})


def _drop_sidecars(files: list[FileEntry]) -> list[FileEntry]:
    return [f for f in files if os.path.basename(f.path) not in _MARIN_SIDECAR_NAMES]


def _glob_with_sizes(patterns: list[str]) -> list[FileEntry]:
    """Glob patterns and return FileEntry objects (spec + size).

    Uses fsspec glob(detail=True) which returns file metadata from the same
    list-objects API call — no per-file stat RPCs needed. Works for gs://, hf://, s3://, local.
    """
    results: list[FileEntry] = []
    for pattern in patterns:
        pattern = re.sub(r"(?<!:)//+", "/", pattern)
        fs, _ = url_to_fs(pattern)
        protocol = fsspec.core.split_protocol(pattern)[0]
        for expanded in braceexpand.braceexpand(pattern):
            detail = fs.glob(expanded, detail=True)
            for path, info in detail.items():
                full = f"{protocol}://{path}" if protocol else path
                results.append(FileEntry(spec=InputFileSpec(path=full), size=info.get("size", 0)))
    return results


def _expand_tokenize_paths(input_paths: list[str]) -> list[str]:
    """Expand input paths into glob patterns for tokenizable file types.

    Directories get expanded to recursive globs for each supported extension.
    Concrete paths/patterns pass through unchanged.
    """
    if isinstance(input_paths, VersionedValue):
        input_paths = input_paths.value

    patterns: list[str] = []
    for path in input_paths:
        assert path != "/"
        if path.endswith("/") or fsspec_isdir(path):
            logger.info(f"Getting all {_TOKENIZE_EXTENSIONS} files in {path}")
            for ex in _TOKENIZE_EXTENSIONS:
                patterns.append(os.path.join(path, f"**/*.{ex}"))
        else:
            patterns.append(path)
    return patterns


def _bundle_files_by_size(files: list[FileEntry], max_bytes: int):
    """Bundle files into groups, with each group having a total size less than max_bytes."""
    current_group: list[str] = []
    current_size = 0

    for f in files:
        if current_size + f.size >= max_bytes and current_group:
            yield current_group
            current_group = []
            current_size = 0
        current_group.append(f.path)
        current_size += f.size

    if current_group:
        yield current_group


def _tokenize_batches(*, config: TokenizeConfig | HfTokenizeConfig, batches: Iterator[Sequence[dict]]) -> Iterator[dict]:
    """Tokenize a list of batches using the specified tokenizer and format."""
    ctx = zephyr_worker_ctx()
    name = ctx.get_shared("tokenizer_name")
    backend = ctx.get_shared("tokenizer_backend")
    # load_tokenizer is @lru_cache, so this only loads once per worker process.
    tokenizer: MarinTokenizer = load_tokenizer(name, backend=backend)
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
        train_patterns = _expand_tokenize_paths(config.train_paths) if config.train_paths else []
        validation_patterns = _expand_tokenize_paths(config.validation_paths) if config.validation_paths else []
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
    train_files = _drop_sidecars(_glob_with_sizes(train_patterns))
    validation_files = _drop_sidecars(_glob_with_sizes(validation_patterns))

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

    def local_preprocess_paths(files: list[FileEntry]) -> list[list[str]]:
        """Bundle files into size-balanced groups for distributed processing."""
        files = sorted(files, key=lambda f: f.path)
        total_input_bytes = sum(f.size for f in files)
        if config.num_shards is not None:
            target_group_bytes = _compute_target_group_bytes(total_input_bytes, config.num_shards)
        else:
            target_group_bytes = _compute_target_group_bytes(total_input_bytes, config.max_workers)
        file_groups = list(_bundle_files_by_size(files, target_group_bytes))
        logger.info(
            f"Grouped {len(files):,} files ({total_input_bytes / 1e9:.2f} GB) into {len(file_groups):,} groups "
            f"(target {target_group_bytes / 1e9:.2f} GB/group)."
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
            # NOTE: https://github.com/marin-community/marin/issues/2829#issuecomment-3963661943
            # Window set to 64 ^
            ds.window(64)
            .map_shard(lambda batches, _: _tokenize_batches(config=config, batches=batches))
            .write_levanter_cache(
                f"{prefix}/part-{{shard:05d}}-of-{{total:05d}}",
                metadata={},
                skip_existing=True,
                batch_size=config.levanter_batch_size,
            )
        )

        # Broadcast tokenizer config to workers. We send name + backend rather than
        # the tokenizer object because not all backends support pickling.
        ctx.put("tokenizer_name", config.tokenizer)
        ctx.put("tokenizer_backend", config.tokenizer_backend)

        tokenize_start = time.monotonic()
        shard_paths = ctx.execute(temp_shards).results
        tokenize_elapsed = time.monotonic() - tokenize_start

        # Build sharded ledger — each shard is directly readable, no consolidation needed.
        shard_ledgers = [CacheLedger.load(p) for p in shard_paths]
        shard_rows = {}
        total_elements = 0
        field_counts: dict[str, int] = {}
        for path, sl in zip(shard_paths, shard_ledgers, strict=True):
            shard_name = os.path.basename(path)
            shard_rows[shard_name] = sl.total_num_rows
            total_elements += sl.total_num_rows
            for field, count in sl.field_counts.items():
                field_counts[field] = field_counts.get(field, 0) + count

        ledger = CacheLedger(
            total_num_rows=total_elements,
            shard_rows=shard_rows,
            is_finished=True,
            finished_shards=list(shard_rows.keys()),
            field_counts=field_counts,
            metadata=CacheMetadata.empty(),
            shard_paths=shard_paths,
        )
        ledger._serialize_and_commit(prefix)

        # Sum token counts across shards
        exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
        total_tokens = 0
        for path in shard_paths:
            store = TreeStore.open(exemplar, path, mode="r", cache_metadata=True)
            if "input_ids" in store.tree:
                total_tokens += store.tree["input_ids"].data_size

        stats_path = os.path.join(prefix, ".stats.json")
        with open_url(stats_path, "w") as f:
            json.dump({"total_tokens": total_tokens, "total_elements": total_elements}, f)

        pipeline_elapsed = time.monotonic() - pipeline_start
        overall_tok_per_sec = total_tokens / tokenize_elapsed if tokenize_elapsed > 0 else 0
        overall_doc_per_sec = total_elements / tokenize_elapsed if tokenize_elapsed > 0 else 0
        logger.info(
            f"{split_name} pipeline complete: {total_elements:,} docs, {total_tokens:,} tokens "
            f"in {pipeline_elapsed:.1f}s (tokenize: {tokenize_elapsed:.1f}s at {overall_tok_per_sec:,.0f} tokens/s "
            f"{overall_doc_per_sec:,.1f} docs/s). "
            f"Wrote stats to {stats_path}"
        )

    # TODO (rav): both train and val could run at the same time
    if train_files and not split_already_done("train"):
        train_groups = local_preprocess_paths(train_files)
        ctx = ZephyrContext(
            resources=config.worker_resources,
            max_workers=min(config.max_workers, len(train_groups)),
            name="tokenize-train",
        )
        run_pipeline(ctx, train_groups, "train")

    if validation_files and not split_already_done("validation"):
        validation_groups = local_preprocess_paths(validation_files)
        ctx = ZephyrContext(
            resources=config.worker_resources,
            max_workers=min(config.max_workers, len(validation_groups)),
            name="tokenize-validation",
        )
        run_pipeline(ctx, validation_groups, "validation")


@draccus.wrap()
def main(config: TokenizeConfig):

    configure_logging(level=logging.INFO)
    tokenize(config)
