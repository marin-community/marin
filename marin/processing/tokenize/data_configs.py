import dataclasses
import logging
import os

from levanter.data.text import LMDatasetSourceConfig, LMMixtureDatasetConfig

from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.processing.tokenize.tokenize import TokenizeConfig

TokenizerStep = ExecutorStep[TokenizeConfig]

logger = logging.getLogger(__name__)


def step_to_lm_mixture_component(step: TokenizerStep) -> LMDatasetSourceConfig:
    """
    Converts a tokenizer step to a Levanter dataset source config. This is useful for creating
    data mixture configs.
    """
    return step.config.as_lm_dataset_source_config(output_path_of(step))


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
) -> LMMixtureDatasetConfig:
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


def add_validation_sets_to_mixture(
    config: LMMixtureDatasetConfig, validation_sets: dict[str, TokenizerStep]
) -> LMMixtureDatasetConfig:
    """
    Adds validation sets to a mixture config.
    """
    valid_configs = {
        name: step.config.as_lm_dataset_source_config(output_path_of(step)) for name, step in validation_sets.items()
    }
    new_configs = {
        **config.configs,
        **{name: source for name, source in valid_configs.items() if name not in config.configs},
    }

    if any(name in config.train_weights for name in validation_sets):
        overlap = set(config.train_weights) & set(validation_sets)
        logger.warning(f"Validation sets {overlap} already present in mixture. Skipping.")

    new_weights = {**config.train_weights, **{name: 0.0 for name in validation_sets if name not in config.train_weights}}
    return dataclasses.replace(config, configs=new_configs, train_weights=new_weights)
