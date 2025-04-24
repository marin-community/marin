"""
Main for running Levanter's tokenizer infrastructure on a dataset using an existing Ray cluster.

Usage:
    ray job submit --working-dir . --no-wait -- python -m marin.processing.tokenize \
        --train_paths '[<input-dir>]' --cache_path <cache-path> --tokenizer <tokenizer_name>
        --validation_paths null

    train_paths: The input directory containing the jsonl files for training, or null/None
    validation_paths: The input directory containing jsonl files for validation, or null/None
    cache_path: The base directory to save the tokenized files
    tokenizer: The name of the tokenizer to use. This must be the same as the tokenizer used in the Levanter
               training run

    The data will be tokenized to $cache_path
"""

import dataclasses
import logging
import os
from collections.abc import Sequence

import draccus
import fsspec
import humanfriendly
import levanter
import ray
import transformers
from levanter.data.sharded_datasource import ShardedDataSource, UrlDataSource
from levanter.data.text import (
    LmDatasetFormatBase,
    LMDatasetSourceConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from levanter.store.cache import CacheOptions
from ray.runtime_env import RuntimeEnv

from marin.execution.executor import ExecutorStep, InputName, VersionedValue
from marin.utils import fsspec_glob, fsspec_isdir, fsspec_size

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TokenizeConfig:
    train_paths: list[str]  # path to training data
    validation_paths: list[str]  # path to validation data
    cache_path: str  # base path to save the tokenized files
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config
    cache_options: CacheOptions | None = None
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa
    """
    The format of the dataset. This is used to determine how to tokenize the data.
    See Levanter's documentation for more details.
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


def _expand_directories(config: UrlDatasetSourceConfig) -> UrlDatasetSourceConfig:
    """
    Expand directories in the config to globs.
    """

    train_paths = _get_filepaths_to_tokenize(config.train_urls)
    validation_paths = _get_filepaths_to_tokenize(config.validation_urls)

    return dataclasses.replace(config, train_urls=train_paths, validation_urls=validation_paths)


def tokenize(config: TokenizeConfig):
    source_config = config.as_lm_dataset_source_config(config.cache_path)

    # TODO: Levanter doesn't automatically expand directories to globs, but by convention we do in Marin
    # we should backport this to Levanter

    source_config = _expand_directories(source_config)

    train_source = source_config.get_shard_source("train")
    validation_source = source_config.get_shard_source("validation")

    if train_source is None and validation_source is None:
        raise ValueError("No input files specified. Nothing to do.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer)
    batch_tokenizer = preprocessor_for_format(config.format, tokenizer)

    if train_source is not None:
        options = config.cache_options
        if options is None:
            options = _heuristic_cache_options(config.train_paths)

        train_ledger = (
            ray.remote(_levanter_build_cache)
            .options(
                name=f"tokenize::{config.cache_path}",
                runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}),
                max_retries=50,
            )
            .remote(
                train_source,
                batch_tokenizer,
                os.path.join(config.cache_path, "train"),
                options,
            )
        )
    else:
        train_ledger = None

    if validation_source is not None:
        options = config.cache_options
        if options is None:
            options = _heuristic_cache_options(config.validation_paths)

        validation_ledger = (
            ray.remote(_levanter_build_cache)
            .options(
                name=f"tokenize::{config.cache_path}",
                runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}),
                max_retries=50,
            )
            .remote(
                validation_source,
                batch_tokenizer,
                os.path.join(config.cache_path, "validation"),
                options,
            )
        )
    else:
        validation_ledger = None

    if train_ledger is not None:
        ray.get(train_ledger)
    if validation_ledger is not None:
        ray.get(validation_ledger)


def _heuristic_cache_options(paths: list[str]):
    # attempt to sniff out a good default. We don't want to use tons of processors if there are a lot of small
    # data files. Rule of thumb: 1 processor per 10GB (capping at 1024, per normal)
    # This reduces contention when writing to gcs and should hopefully mitigate some cost spikes

    paths = _get_filepaths_to_tokenize(paths)
    if paths:
        total_size = sum(fsspec_size(path) for path in paths)
        num_files = len(paths)
        num_processors = min(1024, max(1, total_size // 10_000_000_000, num_files))
        human_size = humanfriendly.format_size(total_size)
        logger.info(f"Using {num_processors} processors for caching {num_files} files of total size {human_size}")
        options = CacheOptions(num_shard_groups=num_processors)
    else:
        options = CacheOptions()
    return options


def _levanter_build_cache(source, batch_tokenizer, output_path, options: CacheOptions):
    from levanter.data.metrics_monitor import LoggerMetricsMonitor
    from levanter.store.cache import build_or_load_cache

    cache = build_or_load_cache(
        cache_dir=output_path,
        source=source,
        processor=batch_tokenizer,
        await_finished=False,
        monitors=[LoggerMetricsMonitor("ray")],
        options=options,
    )
    cache.await_finished()


def _create_source(input_paths: str | list[str]) -> ShardedDataSource:
    if isinstance(input_paths, str) and not _is_probably_path(input_paths):
        source = levanter.data.datasource_from_hf(input_paths, split="train")
    else:
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        filepaths_to_tokenize = _get_filepaths_to_tokenize(input_paths)

        if len(filepaths_to_tokenize) == 0:
            raise ValueError(f"No valid jsonl/parquet files found to tokenize in {input_paths}")

        logger.info(f"Found {len(filepaths_to_tokenize)} files to tokenize.")
        source = UrlDataSource(filepaths_to_tokenize)

    return source


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
    Handles jsonl.{gz,zst,zstd}, and parquet.
    """
    if isinstance(input_paths, VersionedValue):
        input_paths = input_paths.value

    if len(input_paths) == 0:
        return []
    elif any(isinstance(x, InputName | ExecutorStep) for x in input_paths):
        return input_paths

    # we're only going to have one or the other, but might as well return both
    return _get_files_by_extensions(input_paths, ["jsonl.{gz,zst,zstd}", "parquet"])


def _is_probably_path(path: str) -> bool:
    """see if it looks like a real path or not, in which case it might be an hf dataset"""

    protocol, _ = fsspec.core.split_protocol(path)

    if protocol is not None:
        return True

    if fsspec_isdir(path):
        return True

    return False


@draccus.wrap()
def main(config: TokenizeConfig):
    tokenize(config)


if __name__ == "__main__":
    main()
