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
    uv run zephyr --entry-point=main \
        --backend=ray --cluster=us-central2 --max-parallelism=10 --memory=2GB \
        lib/marin/src/marin/processing/tokenize/tokenize.py \
        --train_paths '["gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_04_of_10/local-shard_0_of_10/*.zst"]' \
        --cache_path gs://marin-us-central2/cache/test-tokenize \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --validation_paths '[]'

The tokenized data will be written to:
    {cache_path}/train/     - Training cache
    {cache_path}/validation/ - Validation cache (if provided)
"""  # noqa: E501

import abc
import asyncio
import copy
import dataclasses
import logging
import operator
import os
import re
from collections.abc import Iterator, Sequence

import draccus
import fsspec
import jax
import transformers
from datasets import load_dataset_builder
from levanter.data.text import (
    HfDatasetSourceConfig,
    LmDatasetFormatBase,
    LMDatasetSourceConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from levanter.store.cache import (
    CacheLedger,
    expose_cache_rows,
    extend_cache_metadata_with_other,
    extend_cache_with_other_cache,
    merge_ledgers,
)
from levanter.store.tree_store import TreeStore
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
    revision: str | None = None  # HF dataset revision (commit hash, branch, or tag). Defaults to "main"
    name: str | None = None  # HF dataset name
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa: RUF009
    window_size_bytes: int = 10_000_000_000

    sample_count: int | None = None
    """Number of samples to tokenize. If None, tokenize all samples."""

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


def _tokenize_batches(config: TokenizeConfig | HfTokenizeConfig, batches: Iterator[dict]) -> Iterator[dict]:
    """Tokenize a list of batches using the specified tokenizer and format."""
    # Verify JAX is configured for CPU-only mode
    jax_platforms = os.environ.get("JAX_PLATFORMS", "not set")
    jax_devices = jax.devices()

    # Assert that JAX_PLATFORMS is set to cpu and that all devices are CPU devices
    assert jax_platforms == "cpu", f"JAX_PLATFORMS should be 'cpu' but is '{jax_platforms}'"
    assert all(d.platform == "cpu" for d in jax_devices), f"Expected all CPU devices, got: {jax_devices}"

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer)
    batch_processor = preprocessor_for_format(config.format, tokenizer)

    for batch in batches:
        yield from batch_processor(batch)


def consolidate_shard_caches(
    shard_cache_paths: list[str],
    output_path: str,
    exemplar,
) -> CacheLedger:
    """
    Consolidate multiple shard caches into a single cache.

    Args:
        shard_cache_paths: List of paths to individual shard cache directories
        output_path: Path to the output unified cache directory
        exemplar: Example data structure matching the cache format
        processor_metadata: Metadata dictionary from the batch processor

    Returns:
        The final consolidated CacheLedger
    """
    logger.info(f"Consolidating {len(shard_cache_paths)} shard caches into {output_path}")

    # Initialize data_offset_tree to zero
    first_cache = TreeStore.open(exemplar, shard_cache_paths[0], mode="r", cache_metadata=True)
    data_offset_tree = jax.tree.map(lambda x: 0, first_cache.tree)

    # Compute data offsets for each shard
    shard_info = []
    total_rows = 0

    logger.info("Computing data offsets for each shard")
    for shard_path in shard_cache_paths:
        logger.info(f"Processing shard: {shard_path}")
        ledger = CacheLedger.load(shard_path)

        # Store this shard's info with current offsets
        shard_info.append(
            {
                "path": shard_path,
                "row_offset": total_rows,
                "data_offset_tree": copy.deepcopy(data_offset_tree),
            }
        )

        total_rows += ledger.total_num_rows

        # Update offsets for next shard
        this_cache = TreeStore.open(exemplar, shard_path, mode="r", cache_metadata=True)
        this_offsets = jax.tree.map(lambda x: x.data_size, this_cache.tree)
        data_offset_tree = jax.tree.map(operator.add, data_offset_tree, this_offsets)

    logger.info(f"Computed offsets for {len(shard_info)} shards, total rows: {total_rows}")

    def copy_shard(info: dict):
        asyncio.run(
            extend_cache_with_other_cache(
                output_path, info["path"], exemplar, info["data_offset_tree"], info["row_offset"]
            )
        )
        asyncio.run(
            extend_cache_metadata_with_other(
                output_path, info["path"], exemplar, info["data_offset_tree"], info["row_offset"]
            )
        )

    logger.info("Copying shard data and metadata to final cache.")
    list(
        cpu_only_backend().execute(
            Dataset.from_list(shard_info)
            .map(copy_shard)
            .write_jsonl(f"{output_path}/.copy/copy-shard-{{shard:05d}}.jsonl", skip_existing=True)
        )
    )

    logger.info("Merging ledgers into final ledger.")
    ledgers = []
    for shard_path in shard_cache_paths:
        ledgers.append(CacheLedger.load(shard_path))

    final_ledger = ledgers[0]
    final_ledger.is_finished = False
    for ledger in ledgers[1:]:
        merge_ledgers(final_ledger, ledger)

    final_ledger.is_finished = True
    final_ledger._serialize_and_commit(output_path)

    logger.info(f"Exposing {total_rows} rows in final cache")
    expose_cache_rows(output_path, exemplar, total_rows)

    logger.info(f"Consolidation complete: {total_rows} total rows across {len(shard_info)} shards")

    # I should probably stick this somewhere at the end, it's computed by levanter somewhere,
    # but the shard writers don't give me a place to put it.
    # metadata = {
    #     "append_bos": False,
    #     "append_eos": True,
    #     "max_length": 131072,
    #     "padding": False,
    #     "return_attention_mask": False,
    #     "tokenizer": "marin-community/marin-tokenizer",
    #     "vocab_size": 128256,
    # }

    # now we can delete our temporary shards
    _ = list(
        create_backend("threadpool").execute(
            Dataset.from_list(shard_cache_paths).map(lambda path: fsspec.url_to_fs(path)[0].rm(path, recursive=True))
        )
    )
    return final_ledger


def cpu_only_backend():
    """Return a Zephyr flow backend that uses only CPU devices."""
    return flow_backend(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "CPU"}})


def tokenize(config: TokenizeConfigBase):
    """Tokenize datasets using zephyr pipeline.

    Processes train and validation splits separately, writing to Levanter cache format.
    For HuggingFace datasets, downloads them first then tokenizes the downloaded files.
    """

    if isinstance(config, TokenizeConfig):
        train_paths = _get_filepaths_to_tokenize(config.train_paths) if config.train_paths else []
        validation_paths = _get_filepaths_to_tokenize(config.validation_paths) if config.validation_paths else []
    elif isinstance(config, HfTokenizeConfig):
        # Use datasets library to get the actual file list for each split
        logger.info(f"Loading dataset metadata for {config.id}" + (f" (config: {config.name})" if config.name else ""))

        builder = load_dataset_builder(config.id, name=config.name, revision=config.revision)

        # Get the data files for each split from the builder config
        # This returns a dict like {'train': ['hf://...'], 'validation': ['hf://...']}
        data_files = builder.config.data_files

        if data_files is None:
            raise ValueError(
                f"Dataset {config.id} does not have data_files metadata. "
                "This might be a dataset that requires custom loading logic."
            )

        # Map common split names to our train/validation convention
        # Some datasets use 'test' instead of 'validation'
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
        # Compute the file groups we should process per shard.
        cluster_backend = cpu_only_backend()

        thread_backend = create_backend("threadpool")
        file_stats = list(
            thread_backend.execute(
                Dataset.from_list(paths).map(lambda path: {"filename": path, "size": fsspec_size(path)}),
                verbose=False,
            )
        )
        file_groups = list(_bundle_files_by_size(file_stats, config.window_size_bytes))
        logger.info(f"Grouped {len(paths)} files into {len(file_groups)} groups by size.")

        prefix = os.path.join(config.cache_path, split_name)
        ds = Dataset.from_list(file_groups).flat_map(lambda file_list: file_list).flat_map(load_file)

        if config.sample_count is not None:
            logger.info(f"Sampling {config.sample_count} examples from {split_name} set for tokenization")
            ds = ds.take_per_shard(config.sample_count)

        # tokenization runs a bit faster with batching
        temp_shards = (
            ds.batch(64)
            .map_shard(lambda batches: _tokenize_batches(config, batches))
            .write_levanter_cache(f"{prefix}/part-{{shard:05d}}", metadata={}, skip_existing=True)
        )

        shard_paths = list(cluster_backend.execute(temp_shards))

        logger.info("Computing exemplar for cache consolidation")
        exemplar = cluster_backend.execute(
                Dataset.from_list(paths[0:1])
                .flat_map(load_file)
                .take_per_shard(1)
                .map_shard(lambda example: _tokenize_batches(config, [example]))
            )[0]

        logger.info(f"Tokenization complete, consolidating {len(shard_paths)} shards into {prefix}")
        consolidate_shard_caches(
            shard_cache_paths=shard_paths,
            output_path=prefix,
            exemplar=exemplar,
        )

    if train_paths:
        run_pipeline(train_paths, "train")

    if validation_paths:
        run_pipeline(validation_paths, "validation")


@draccus.wrap()
def main(config: TokenizeConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    tokenize(config)
