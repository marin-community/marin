"""
Main for running Levanter's tokenizer infrastructure on a dataset using an existing Ray cluster.

Usage:
    ray job submit --working-dir . --no-wait -- python -m marin.processing.tokenize \
        --train_urls '[<input-dir>]' --cache_path <cache-path> --tokenizer <tokenizer_name>
        --validation_urls null

    train_urls: The input directory containing the jsonl files for training, or null/None
    validation_urls: The input directory containing jsonl files for validation, or null/None
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
import ray
import transformers

from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.data.text import LMDatasetConfig, LMDatasetSourceConfig, LMMixtureDatasetConfig
from marin.execution.executor import ExecutorStep, output_path_of
from marin.utils import fsspec_glob, fsspec_isdir

logger = logging.getLogger(__name__)


@ray.remote
def levanter_tokenize(input_urls: list[str] | str, tokenizer_name: str, output_path: str):
    import levanter
    from levanter.data.metrics_monitor import LoggerMetricsMonitor
    from levanter.data.text import BatchTokenizer
    from levanter.store.cache import build_or_load_cache

    if len(input_urls) == 0:
        logger.warning("No input files found. Nothing to do.")
        return

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Caching {input_urls} to {output_path}.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    if isinstance(input_urls, str) and is_hf_dataset(input_urls):
        source = levanter.data.datasource_from_hf(input_urls, split="train")
        source = source.map(lambda d: d["text"])
    else:
        if isinstance(input_urls, str):
            input_urls = [input_urls]
        jsonls = _get_jsonls(input_urls)
        if len(jsonls) == 0:
            raise ValueError(f"No jsonl files found in {input_urls}")
        logger.info(f"Found {len(jsonls)} jsonl files.")
        source = TextUrlDataSource(jsonls)

    cache = build_or_load_cache(
        cache_dir=output_path,
        input_shards=source,
        processor=batch_tokenizer,
        await_finished=False,
        monitors=[LoggerMetricsMonitor("ray")],
    )

    cache.await_finished()
    logger.info(f"Finished caching {input_urls} to {output_path}.")


@dataclasses.dataclass(frozen=True)
class TokenizeConfig:
    train_urls: list[str]  # path to training data in jsonl format
    validation_urls: list[str]  # path to validation data in jsonl format
    cache_path: str  # base path to save the tokenized files
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run
    tags: list[str] = dataclasses.field(default_factory=list)  # tags to be added to config

    def as_lm_dataset_source_config(self, actual_output_dir) -> LMDatasetSourceConfig:
        """
        For use in Levanter training runs with mixtures of datasets.
        """
        return LMDatasetSourceConfig(
            tags=self.tags,
            train_urls=(self.train_urls),
            validation_urls=(self.validation_urls),
            cache_dir=actual_output_dir,
        )

    def as_lm_dataset_task_config(self, actual_output_dir) -> LMDatasetConfig:
        """
        For use in Levanter training runs with a single dataset.
        """
        train_urls = self.train_urls
        validation_urls = self.validation_urls
        return LMDatasetConfig(
            cache_dir=actual_output_dir,
            train_urls=train_urls,
            validation_urls=validation_urls,
            tags=self.tags,
            tokenizer=self.tokenizer,
        )


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
    else:

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


def tokenize(config: TokenizeConfig):
    if len(config.train_urls) == 0 and len(config.validation_urls) == 0:
        raise ValueError("No input files specified. Nothing to do.")
    output_path = os.path.join(config.cache_path, "train")
    response_train = levanter_tokenize.remote(config.train_urls, config.tokenizer, output_path)
    validation_out = os.path.join(config.cache_path, "validation")
    response_validation = levanter_tokenize.remote(config.validation_urls, config.tokenizer, validation_out)
    return ray.get([response_train, response_validation])[0]


def _get_jsonls(input_path: list[str]):
    output_paths = []
    for path in input_path:
        if fsspec_isdir(path) or path.endswith("/"):
            logger.info(f"Getting all jsonl files in {path}")
            output_paths.extend(fsspec_glob(os.path.join(path, "**/*.jsonl.{gz,zst,zstd}")))
        else:
            output_paths.extend(fsspec_glob(path))

    return output_paths


def is_hf_dataset(path):
    # see if looks like an hf dataset or not.
    import fsspec

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
