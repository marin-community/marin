import dataclasses
import logging
import os
from functools import lru_cache

import transformers
from levanter.data.text import LMDatasetSourceConfig, LMMixtureDatasetConfig

from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.processing.tokenize.tokenize import TokenizeConfig

TokenizerStep = ExecutorStep[TokenizeConfig]

logger = logging.getLogger(__name__)


def step_to_lm_mixture_component(step: TokenizerStep | TokenizeConfig, include_raw_paths: bool) -> LMDatasetSourceConfig:
    """
    Converts a tokenizer step to a Levanter dataset source config. This is useful for creating
    data mixture configs.
    """

    if isinstance(step, TokenizeConfig):
        return step.as_lm_dataset_source_config(step.cache_path, include_raw_paths=include_raw_paths)
    else:
        return step.config.as_lm_dataset_source_config(output_path_of(step), include_raw_paths=include_raw_paths)


def as_single_lm_data_source(
    input: TokenizerStep | InputName,  # noqa: A002
) -> LMDatasetSourceConfig:
    """
    Creates a dataset config suitable for Levanter's TrainLMConfig from a single training set

    Args:
        input: The training set to use
    """
    return step_to_lm_mixture_component(input, include_raw_paths=True)


def lm_data_config(
    training_set: TokenizerStep | InputName,
    validation_sets: dict[str, TokenizerStep] | None = None,
    shuffle: bool | int = True,
    max_train_batches: dict[str, int] | None = None,
    num_validation_sequences: dict[str, int] | None = None,
) -> LMMixtureDatasetConfig:
    """
    Creates a dataset config suitable for Levanter's TrainLMConfig from a single training set

    Args:
        training_set: The training set to use
        validation_sets: A sequence of validation sets to use
        shuffle: Whether to shuffle the data. If int, uses era shuffling.
        max_train_batches: Maximum number of batches to use for the training set per dataset.
        num_validation_sequences: Number of validation sequences to take from the training set per dataset.
    """
    tokenizer = training_set.config.tokenizer

    if validation_sets is not None:
        for name, step in validation_sets.items():
            if step.config.tokenizer != tokenizer:
                raise ValueError(
                    f"Validation set {name} ({step.name}) must have same tokenizer as training set's,"
                    f" but got: {step.config.tokenizer} vs {tokenizer}"
                )

    train_set_name = os.path.basename(training_set.name)

    return lm_mixture_data_config(
        {train_set_name: training_set, **(validation_sets or {})},
        {train_set_name: 1.0},
        shuffle=shuffle,
        missing_weights_are_validation=True,
        max_train_batches=max_train_batches,
        num_validation_sequences=num_validation_sequences,
    )


def lm_mixture_data_config(
    components: dict[str, TokenizerStep | TokenizeConfig],
    weights: dict[str, float],
    *,
    shuffle: bool | int = True,
    missing_weights_are_validation: bool = True,
    include_raw_paths: bool = True,
    max_train_batches: dict[str, int] | None = None,
    num_validation_sequences: dict[str, int] | None = None,
) -> LMMixtureDatasetConfig:
    """
    Creates a training config from a mixture of datasources.

    Args:
        components: dict from names of datasets to the steps that produced them.
        weights: dict from names of datasets to their weights.
        shuffle: shuffling policy. int means era shuffling (~shuffle buffer).
        missing_weights_are_validation: whether to pad out missing weights with 0's, indicating validation-only sets
        include_raw_paths: whether to include raw paths in the dataset config. This is mostly for logging purposes.
        max_train_batches: Maximum number of batches to use for the training set per dataset.
        num_validation_sequences: Number of validation sequences to take from the training set per dataset.
    """
    configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=include_raw_paths)
        for name, step in components.items()
    }

    if missing_weights_are_validation:
        missing_keys = {k: 0.0 for k in components if k not in weights}
        weights = {**weights, **missing_keys}

    tokenizer = _verify_tokenizers_same(components)

    return LMMixtureDatasetConfig(
        configs=configs,
        train_weights=weights,
        tokenizer=tokenizer,
        cache_dir=None,
        shuffle=shuffle,
        max_train_batches=max_train_batches,
        num_validation_sequences=num_validation_sequences,
    )


def lm_varying_mixture_data_config(
    components: dict[str, TokenizerStep],
    weights_list: list[tuple[int, dict[str, float]]],
    *,
    shuffle: bool | int = True,
    missing_weights_are_validation: bool = True,
    include_raw_paths: bool = True,
    mixture_block_size: int | None = None,
    max_train_batches: dict[str, int] | None = None,
    num_validation_sequences: dict[str, int] | None = None,
) -> LMMixtureDatasetConfig:
    """
    Creates a training config from a mixture of datasources with varying weights.

    Args:
        components: dict from names of datasets to the steps that produced them.
        weights_list: list of tuples of (start_seq_index, weights_dict)
            weights_dict maps dataset names to their weights.
            The weights will change at each start_seq_index. start_seq_index's must be sorted in ascending order.
            Note that start_seq_index should be the index of the sequence (not batch) where the transition should occur.
        shuffle: shuffling policy. int means era shuffling (~shuffle buffer).
        missing_weights_are_validation: whether to pad out missing weights with 0's, indicating validation-only sets
        include_raw_paths: whether to include raw paths in the dataset config. This is mostly for logging purposes.
        mixture_block_size: The block size to use for the mixture.
        max_train_batches: Maximum number of batches to use for the training set per dataset.
        num_validation_sequences: Number of validation sequences to take from the training set per dataset.

    Returns:
        LMMixtureDatasetConfig configured with the varying weights
    """
    configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=include_raw_paths)
        for name, step in components.items()
    }

    # Validate and normalize weights
    if not weights_list:
        raise ValueError("weights_list cannot be empty")

    if weights_list[0][0] != 0:
        raise ValueError("First weight stage must start at index 0")

    # If missing_weights_are_validation, pad out weights with zeros
    if missing_weights_are_validation:
        padded_weights_list = []
        for step_idx, weights in weights_list:
            missing_keys = {k: 0.0 for k in components if k not in weights}
            padded_weights_list.append((step_idx, {**weights, **missing_keys}))
        weights_list = padded_weights_list

    tokenizer = _verify_tokenizers_same(components)

    return LMMixtureDatasetConfig(
        configs=configs,
        train_weights=weights_list,
        tokenizer=tokenizer,
        cache_dir=None,
        shuffle=shuffle,
        mixture_block_size=mixture_block_size or 2048,
        max_train_batches=max_train_batches,
        num_validation_sequences=num_validation_sequences,
    )


def add_validation_sets_to_mixture(
    config: LMMixtureDatasetConfig, validation_sets: dict[str, TokenizerStep]
) -> LMMixtureDatasetConfig:
    """
    Adds validation sets to a mixture config. Works with both fixed and varying mixture weights.
    """
    valid_configs = {
        name: step.config.as_lm_dataset_source_config(output_path_of(step)) for name, step in validation_sets.items()
    }
    new_configs = {
        **config.configs,
        **{name: source for name, source in valid_configs.items() if name not in config.configs},
    }

    if isinstance(config.train_weights, dict):
        # Handle fixed weights case
        if any(name in config.train_weights for name in validation_sets):
            overlap = set(config.train_weights) & set(validation_sets)
            logger.warning(f"Validation sets {overlap} already present in mixture. Skipping.")

        new_weights = {
            **config.train_weights,
            **{name: 0.0 for name in validation_sets if name not in config.train_weights},
        }
    elif isinstance(config.train_weights, list):
        for step_idx, weights_dict in config.train_weights:
            assert isinstance(step_idx, int)
            assert isinstance(weights_dict, dict)

        # Handle varying weights case
        overlap_sets = set()
        for _, weights_dict in config.train_weights:
            overlap_sets.update(set(weights_dict) & set(validation_sets))

        if overlap_sets:
            logger.warning(f"Validation sets {overlap_sets} already present in mixture. Skipping.")

        new_weights = []
        for step_idx, weights_dict in config.train_weights:
            new_weights_dict = {**weights_dict, **{name: 0.0 for name in validation_sets if name not in weights_dict}}
            new_weights.append((step_idx, new_weights_dict))
    else:
        raise ValueError(f"Invalid train_weights type: {type(config.train_weights)}")

    return dataclasses.replace(config, configs=new_configs, train_weights=new_weights)


def mixture_for_evaluation(inputs: dict[str, ExecutorStep]) -> LMMixtureDatasetConfig:
    """
    Creates a mixture of datasets purely for evaluation purposes. Used mostly for visualizing log probabilities.

    Args:
        inputs (dict[str, ExecutorStep]): The inputs to the mixture.

    Returns:
        LMMixtureDatasetConfig: The mixture of datasets.
    """
    return lm_mixture_data_config(
        {name: step for name, step in inputs.items()},
        {name: 0.0 for name in inputs},
        shuffle=False,
        missing_weights_are_validation=True,
    )


@lru_cache(maxsize=32)
def _load_tokenizer(tokenizer_name: str) -> transformers.PreTrainedTokenizer:
    """Load and cache a tokenizer by name"""
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name)


def _are_tokenizers_equivalent(tokenizer1: str, tokenizer2: str) -> bool:
    """Compare two tokenizers by loading them and comparing their vocabularies and token IDs"""
    t1 = _load_tokenizer(tokenizer1)
    t2 = _load_tokenizer(tokenizer2)

    # Compare vocab sizes
    if len(t1.get_vocab()) != len(t2.get_vocab()):
        return False

    # Compare vocab contents and IDs
    vocab1 = t1.get_vocab()
    vocab2 = t2.get_vocab()

    # Check that all tokens exist in both vocabs with the same IDs
    for token, id1 in vocab1.items():
        if token not in vocab2:
            return False
        if vocab2[token] != id1:
            return False

    if getattr(t1, "chat_template", None) is not None and getattr(t2, "chat_template", None) is not None:
        if t1.chat_template != t2.chat_template:
            return False

    return True


def _verify_tokenizers_same(components: dict[str, TokenizerStep]):
    first_name, first_step = next(iter(components.items()))
    tokenizer = first_step.config.tokenizer
    for name, step in components.items():
        if step.config.tokenizer != tokenizer:
            # If string comparison fails, try comparing loaded tokenizers
            if not _are_tokenizers_equivalent(step.config.tokenizer, tokenizer):
                raise ValueError(
                    "All components must have the same tokenizer, but got:"
                    f" {step.config.tokenizer} ({name}) vs {tokenizer} ({name})"
                )
            else:
                logger.warning(
                    f"Tokenizers ({name}) and {tokenizer} ({name}) have equivalent vocabularies but are not the same"
                    f"tokenizer. This may cause issues with training."
                )
    return tokenizer
