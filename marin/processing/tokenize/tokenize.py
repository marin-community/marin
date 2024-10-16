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
import levanter
import ray
import transformers
from levanter.data.sharded_datasource import ShardedDataSource, TextUrlDataSource
from levanter.data.text import BatchTokenizer, LMDatasetConfig, LMDatasetSourceConfig, LMMixtureDatasetConfig
from levanter.store.cache import CacheOptions
from ray.runtime_env import RuntimeEnv

from marin.execution.executor import ExecutorStep, output_path_of
from marin.processing.tokenize.independent_tokenize import tokenize_and_concatenate_shards
from marin.utils import fsspec_glob, fsspec_isdir

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TokenizeConfig:
    train_paths: list[str]  # path to training data in jsonl format
    validation_paths: list[str]  # path to validation data in jsonl format
    cache_path: str  # base path to save the tokenized files
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config
    cache_options: CacheOptions = CacheOptions()  # noqa: RUF009

    def train_source(self) -> ShardedDataSource | None:
        if len(self.train_paths) == 0:
            return None
        return _create_source(self.train_paths)

    def validation_source(self) -> ShardedDataSource | None:
        if len(self.validation_paths) == 0:
            return None
        return _create_source(self.validation_paths)

    def as_lm_dataset_source_config(self, actual_output_path: str) -> LMDatasetSourceConfig:
        """
        For use in Levanter training runs with mixtures of datasets.
        """
        return LMDatasetSourceConfig(
            tags=self.tags,
            train_urls=self.train_paths,
            validation_urls=self.validation_paths,
            cache_dir=actual_output_path,
        )

    def as_lm_dataset_task_config(self, actual_output_path: str) -> LMDatasetConfig:
        """
        For use in Levanter training runs with a single dataset.
        """
        return LMDatasetConfig(
            cache_dir=actual_output_path,
            train_urls=self.train_paths,
            validation_urls=self.validation_paths,
            tags=self.tags,
            tokenizer=self.tokenizer,
        )


def tokenize(config: TokenizeConfig):
    if len(config.train_paths) == 0 and len(config.validation_paths) == 0:
        raise ValueError("No input files specified. Nothing to do.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer)
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    train_source = config.train_source()
    if train_source is not None:
        ledger = (
            ray.remote(tokenize_and_concatenate_shards)
            .options(
                name=f"tokenize::{config.cache_path}", runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORM_NAME": "cpu"})
            )
            .remote(
                train_source,
                batch_tokenizer,
                os.path.join(config.cache_path, "train"),
                config.cache_options,
            )
        )
    else:
        ledger = None

    validation_source = config.validation_source()

    if validation_source is not None:
        validation_ledger = (
            ray.remote(tokenize_and_concatenate_shards)
            .options(
                name=f"tokenize::{config.cache_path}", runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORM_NAME": "cpu"})
            )
            .remote(
                validation_source,
                batch_tokenizer,
                os.path.join(config.cache_path, "validation"),
                config.cache_options,
            )
        )
    else:
        validation_ledger = None

    if ledger is not None:
        ray.get(ledger)
    if validation_ledger is not None:
        ray.get(validation_ledger)


@ray.remote(runtime_env=RuntimeEnv(env_vars={"JAX_PLATFORM_NAME": "cpu"}))
def levanter_tokenize(input_paths: list[str] | str, tokenizer_name: str, output_path: str):
    from levanter.data.metrics_monitor import LoggerMetricsMonitor
    from levanter.data.text import BatchTokenizer
    from levanter.store.cache import build_or_load_cache

    if len(input_paths) == 0:
        logger.warning("No input files found. Nothing to do.")
        return

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Caching {input_paths} to {output_path}.")

    source = _create_source(input_paths)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    cache = build_or_load_cache(
        cache_dir=output_path,
        input_shards=source,
        processor=batch_tokenizer,
        await_finished=False,
        monitors=[LoggerMetricsMonitor("ray")],
    )

    cache.await_finished()
    logger.info(f"Finished caching {input_paths} to {output_path}.")


def _create_source(input_paths) -> ShardedDataSource:
    if isinstance(input_paths, str) and not _is_probably_path(input_paths):
        source = levanter.data.datasource_from_hf(input_paths, split="train")
        source = source.map(lambda d: d["text"])
    else:
        if isinstance(input_paths, str):
            input_paths = [input_paths]
        jsonls = _get_jsonls(input_paths)
        if len(jsonls) == 0:
            raise ValueError(f"No jsonl files found in {input_paths}")
        logger.info(f"Found {len(jsonls)} jsonl files.")
        source = TextUrlDataSource(jsonls)
    return source


TokenizerStep = ExecutorStep[TokenizeConfig]


def step_to_lm_mixture_component(step: TokenizerStep) -> LMDatasetSourceConfig:
    """
    Converts a tokenizer step to a Levanter dataset source config. This is useful for creating
    data mixture configs.
    """
    return step.config.as_lm_dataset_source_config(output_path_of(step))


def step_to_lm_training_config(step: TokenizerStep) -> LMDatasetConfig:
    """
    Converts a tokenizer step to a Levanter dataset config. This is useful for creating
    data mixture configs.
    """
    return step.config.as_lm_dataset_task_config(output_path_of(step))


def lm_training_config(
    training_set: TokenizerStep,
    validation_sets: Sequence[TokenizerStep] = (),
    shuffle: bool | int = True,
) -> LMDatasetConfig | LMMixtureDatasetConfig:
    """
    Creates a dataset config suitable for Levanter's TrainLMConfig from a single training set

    Args:
        training_set: The training set to use
        validation_sets: A sequence of validation sets to use
        shuffle: Whether to shuffle the data. If int, uses era shuffling.
    """
    tokenizer = training_set.config.tokenizer

    if len(validation_sets) == 0:
        return dataclasses.replace(step_to_lm_training_config(training_set), shuffle=shuffle)

    for step in validation_sets:
        if step.config.tokenizer != tokenizer:
            raise ValueError(
                f"Validation set {step.name} must have same tokenizer as training set's,"
                f" but got: {step.config.tokenizer} vs {tokenizer}"
            )

    prefix = os.path.commonprefix([training_set.name, *(dset.name for dset in validation_sets)])

    def _strip_prefix(name):
        return name[len(prefix) :]

    weights = {_strip_prefix(training_set.name): 1.0, **{_strip_prefix(step.name): 0.0 for step in validation_sets}}

    components = {
        _strip_prefix(training_set.name): step_to_lm_mixture_component(training_set),
        **{_strip_prefix(step.name): step_to_lm_mixture_component(step) for step in validation_sets},
    }

    return LMMixtureDatasetConfig(
        configs=components, train_weights=weights, tokenizer=tokenizer, cache_dir=None, shuffle=shuffle
    )


def lm_mixture_training_config(
    components: dict[str, TokenizerStep],
    weights: dict[str, float],
    *,
    shuffle: bool | int = True,
    missing_weights_are_validation: bool = True,
):
    """
    Creates a training config from a mixture of datasources.

    Args:
        components: dict from names of datasets to the steps that produced them.
        weights: dict from names of datasets to their weights.
        shuffle: shuffling policy. int means era shuffling (~shuffle buffer).
        missing_weights_are_validation: whether to pad out missing weights with 0's, indicating validation-only sets
    """
    configs = {name: step_to_lm_mixture_component(step) for name, step in components.items()}

    if missing_weights_are_validation:
        missing_keys = {k: 0.0 for k in components if k not in weights}
        weights = {**weights, **missing_keys}

    first_name, first_step = next(iter(configs.items()))
    tokenizer = first_step.config.tokenizer
    for name, step in components.items():
        if step.config.tokenizer != tokenizer:
            raise ValueError(
                "All components must have the same tokenizer, but got:"
                f" {step.config.tokenizer} ({name}) vs {tokenizer} ({name})"
            )

    return LMMixtureDatasetConfig(
        configs=configs, train_weights=weights, tokenizer=tokenizer, cache_dir=None, shuffle=shuffle
    )


def _get_jsonls(input_path: list[str]):
    output_paths = []
    for path in input_path:
        if fsspec_isdir(path) or path.endswith("/"):
            logger.info(f"Getting all jsonl files in {path}")
            output_paths.extend(fsspec_glob(os.path.join(path, "**/*.jsonl.{gz,zst,zstd}")))
        else:
            output_paths.extend(fsspec_glob(path))

    return output_paths


def _is_probably_path(path: str) -> bool:
    """see if looks like a real path or not, in which case it might be an hf dataset"""

    protocol, _ = fsspec.core.split_protocol(path)

    if protocol is not None:
        return False

    if fsspec_isdir(path):
        return False

    return True


@draccus.wrap()
def main(config: TokenizeConfig):
    tokenize(config)


if __name__ == "__main__":
    main()
