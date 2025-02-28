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
from levanter.data.sharded_datasource import ShardedDataSource, TextUrlDataSource
from levanter.data.text import (
    BatchTokenizer,
    ChatUrlDataSourceConfig,
    LMDatasetSourceConfig,
    LMSupervisedDatasetConfig,
    mk_chat_sft_dataset,
    mk_supervised_dataset,
)
from levanter.store.cache import CacheOptions
from ray.runtime_env import RuntimeEnv

from marin.execution.executor import InputName
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
    input_field: str = ""
    output_field: str = ""
    text_key: str = "text"

    def train_source(self) -> ShardedDataSource | None:
        if len(self.train_paths) == 0:
            return None
        return _create_source(self.train_paths, self.text_key)

    def validation_source(self) -> ShardedDataSource | None:
        if len(self.validation_paths) == 0:
            return None
        return _create_source(self.validation_paths, self.text_key)

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName, *, include_raw_paths=True
    ) -> LMDatasetSourceConfig:
        """
        For use in Levanter training runs with mixtures of datasets.

        Args:
            include_raw_paths: if false, don't include paths to raw data in Levanter's config. This means we'll be able
                to run training without the original training data, but hte provenance won't be recorded in wandb.

        """
        return LMDatasetSourceConfig(
            tags=self.tags,
            train_urls=self.train_paths if include_raw_paths else [],
            validation_urls=self.validation_paths if include_raw_paths else [],
            cache_dir=actual_output_path,
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


def tokenize(config: TokenizeConfig):
    train_source = config.train_source()
    validation_source = config.validation_source()

    if train_source is None and validation_source is None:
        raise ValueError("No input files specified. Nothing to do.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer)
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    if train_source is not None:
        options = config.cache_options
        if options is None:
            options = _heuristic_cache_options(config.train_paths)

        train_ledger = (
            ray.remote(_levanter_build_cache)
            .options(name=f"tokenize::{config.cache_path}", runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))
            .remote(
                train_source,
                batch_tokenizer,
                os.path.join(config.cache_path, "train"),
                options,
            )
        )
    else:
        train_ledger = None

    validation_source = config.validation_source()

    if validation_source is not None:
        options = config.cache_options
        if options is None:
            options = _heuristic_cache_options(config.validation_paths)

        validation_ledger = (
            ray.remote(_levanter_build_cache)
            .options(name=f"tokenize::{config.cache_path}", runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))
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


@ray.remote(runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))
def levanter_tokenize_sft(config: TokenizeConfig):
    """
    Tokenize chat SFT data using the mk_chat_sft_dataset function.
    """

    def add_special_tokens(tokenizer, use_unk_instead_of_adding=False):
        special_tokens_dict = dict()
        if use_unk_instead_of_adding:
            if tokenizer.unk_token is None:
                raise ValueError("use_unk_instead_of_add is True but tokenizer doesn't have an unk token")

        unk = tokenizer.unk_token if use_unk_instead_of_adding else None

        if tokenizer.pad_token is None:
            logger.info(f"Adding pad token to {tokenizer}")
            special_tokens_dict["pad_token"] = "[PAD]" if not use_unk_instead_of_adding else unk
        if tokenizer.eos_token is None:
            logger.info(f"Adding eos token to {tokenizer}")
            special_tokens_dict["eos_token"] = "</s>" if not use_unk_instead_of_adding else unk
        if tokenizer.bos_token is None:
            logger.info(f"Adding bos token to {tokenizer}")
            special_tokens_dict["bos_token"] = "<s>" if not use_unk_instead_of_adding else unk
        if tokenizer.unk_token is None:
            logger.info(f"Adding unk token to {tokenizer}")
            special_tokens_dict["unk_token"] = "<unk>"

        return tokenizer.add_special_tokens(special_tokens_dict)

    logging.basicConfig(level=logging.INFO)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer, padding_side="right", trust_remote_code=True
    )
    num_new_tokens = add_special_tokens(tokenizer)
    logger.info(f"Added {num_new_tokens} special tokens to tokenizer")

    sft_config = ChatUrlDataSourceConfig(
        train_urls=config.train_paths,
        cache_dir=config.cache_path,
        messages_field="messages",  # Adjust these fields based on your data format
        input_role="user",
        output_role="assistant",
    )

    logger.info(f"Caching SFT data to {config.cache_path}")
    import haliax

    # Use the existing mk_chat_sft_dataset function, position axis is arbitrary
    # it shouldn't matter what the value is during cache creation
    mk_chat_sft_dataset(sft_config, tokenizer, haliax.Axis("position", 2048))

    logger.info(f"Finished caching SFT dataset to {config.cache_path}")


@ray.remote(runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))
def levanter_tokenize_supervised(config: TokenizeConfig):
    supervised_config = LMSupervisedDatasetConfig(
        validation_urls=config.validation_paths,
        cache_dir=config.cache_path,
        input_field=config.input_field,
        output_field=config.output_field,
    )
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Caching {config.validation_paths} to {config.cache_path}.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer)

    import haliax

    # this axis doesn't actually matter for just building the cache
    mk_supervised_dataset(supervised_config, "validation", tokenizer, haliax.Axis("position", 2048))
    logger.info(f"Finished caching supervised dataset to {config.cache_path}.")


def _levanter_build_cache(source, batch_tokenizer, output_path, options: CacheOptions):
    from levanter.data.metrics_monitor import LoggerMetricsMonitor
    from levanter.store.cache import build_or_load_cache

    cache = build_or_load_cache(
        cache_dir=output_path,
        input_shards=source,
        processor=batch_tokenizer,
        await_finished=False,
        monitors=[LoggerMetricsMonitor("ray")],
        options=options,
    )
    cache.await_finished()


def _create_source(input_paths: str | list[str], text_key) -> ShardedDataSource:
    if isinstance(input_paths, str) and not _is_probably_path(input_paths):
        source = levanter.data.datasource_from_hf(input_paths, split="train")
        source = source.map(lambda d: d["text"])
    else:
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        filepaths_to_tokenize = _get_filepaths_to_tokenize(input_paths)

        if len(filepaths_to_tokenize) == 0:
            raise ValueError(f"No valid jsonl/parquet files found to tokenize in {input_paths}")

        logger.info(f"Found {len(filepaths_to_tokenize)} files to tokenize.")
        source = TextUrlDataSource(filepaths_to_tokenize, text_key=text_key)

    return source


def _get_files_by_extensions(input_paths: list[str], extensions: list[str]) -> list[str]:
    """
    Get a list of all filepaths with the specified extension from the input paths.
    """
    print(input_paths)
    output_paths = []
    for path in input_paths:
        assert path != "/"
        if fsspec_isdir(path) or path.endswith("/"):
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
    if len(input_paths) == 0:
        return []

    # we're only going to have one or the other, but might as well return both
    return _get_files_by_extensions(input_paths, ["jsonl.{gz,zst,zstd}", "parquet"])


def _is_probably_path(path: str) -> bool:
    """see if looks like a real path or not, in which case it might be an hf dataset"""

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
