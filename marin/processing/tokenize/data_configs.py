import dataclasses
import logging
import os

from levanter.data.text import LMDatasetSourceConfig, LMMixtureDatasetConfig

from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.processing.tokenize.tokenize import TokenizeConfig

TokenizerStep = ExecutorStep[TokenizeConfig]

logger = logging.getLogger(__name__)


def step_to_lm_mixture_component(step: TokenizerStep, include_raw_paths: bool) -> LMDatasetSourceConfig:
    """
    Converts a tokenizer step to a Levanter dataset source config. This is useful for creating
    data mixture configs.
    """
    return step.config.as_lm_dataset_source_config(output_path_of(step), include_raw_paths=include_raw_paths)


def lm_data_config(
    training_set: TokenizerStep | InputName,
    validation_sets: dict[str, TokenizerStep] | None = None,
    shuffle: bool | int = True,
) -> LMMixtureDatasetConfig:
    """
    Creates a dataset config suitable for Levanter's TrainLMConfig from a single training set

    Args:
        training_set: The training set to use
        validation_sets: A sequence of validation sets to use
        shuffle: Whether to shuffle the data. If int, uses era shuffling.
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
    )


def lm_mixture_data_config(
    components: dict[str, TokenizerStep],
    weights: dict[str, float],
    *,
    shuffle: bool | int = True,
    missing_weights_are_validation: bool = True,
    include_raw_paths: bool = True,
) -> LMMixtureDatasetConfig:
    """
    Creates a training config from a mixture of datasources.

    Args:
        components: dict from names of datasets to the steps that produced them.
        weights: dict from names of datasets to their weights.
        shuffle: shuffling policy. int means era shuffling (~shuffle buffer).
        missing_weights_are_validation: whether to pad out missing weights with 0's, indicating validation-only sets
        include_raw_paths: whether to include raw paths in the dataset config. This is mostly for logging purposes.
    """
    configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=include_raw_paths)
        for name, step in components.items()
    }

    if missing_weights_are_validation:
        missing_keys = {k: 0.0 for k in components if k not in weights}
        weights = {**weights, **missing_keys}

    first_name, first_step = next(iter(components.items()))
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


def lm_varying_mixture_data_config(
    components: dict[str, TokenizerStep],
    weights_list: list[tuple[int, dict[str, float]]],
    *,
    shuffle: bool | int = True,
    missing_weights_are_validation: bool = True,
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

    Returns:
        LMMixtureDatasetConfig configured with the varying weights
    """
    configs = {name: step_to_lm_mixture_component(step) for name, step in components.items()}

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

    # Validate tokenizer consistency
    first_name, first_step = next(iter(components.items()))
    tokenizer = first_step.config.tokenizer
    for name, step in components.items():
        if step.config.tokenizer != tokenizer:
            raise ValueError(
                "All components must have the same tokenizer, but got:"
                f" {step.config.tokenizer} ({name}) vs {tokenizer} ({first_name})"
            )

    return LMMixtureDatasetConfig(
        configs=configs,
        train_weights=weights_list,
        tokenizer=tokenizer,
        cache_dir=None,
        shuffle=shuffle,
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
