# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import logging
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from fray.v2 import ResourceConfig
from levanter.data.text import LmDatasetFormatBase, LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.utils import fsspec_utils
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    VersionedValue,
    ensure_versioned,
    this_output_path,
    unwrap_versioned_value,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfigBase
from marin.training import default_train as library_default_train

from experiments.evals.task_configs import (
    CORE_TASKS,
)
from experiments.paloma import paloma_tokenized
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.processing.tokenize import (
    HfDatasetSpec,
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    get_vocab_size_for_tokenizer,
    lm_data_config,
    tokenize,
)

logger = logging.getLogger("ray")


def default_download(
    name: str,
    hf_dataset_id: str,
    revision: str,
    override_output_path: str | None = None,
    **kwargs: Any,
) -> InputName:
    """
    Download a HuggingFace dataset and upload it to a specified path with default configuration.

    Args:
        name: The name of the Download step. It forms the basis of the output path
            unless override_output_path is explicitly specified.
        hf_dataset_id: The HuggingFace dataset ID to download. As `$ORG/$DATASET` on HF Hub
        revision: The revision of the dataset to download.
            Short Commit Hash from HF Dataset Repo (7 characters)
        override_output_path: Optional. The output path for the dataset.
        **kwargs: Additional keyword arguments that are passed to the download config.

    The final output data will reside in '{output_path}/{revision}'.
    """

    step = ExecutorStep(
        name=name,
        description=f"Download {hf_dataset_id} revision {revision}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=hf_dataset_id,
            revision=revision,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            **kwargs,
        ),
        override_output_path=override_output_path,
    )

    return step.as_input_name()


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str | HfDatasetSpec,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa
    *,
    sample_count: int | VersionedValue[int] | None = None,
    is_validation: bool = False,
) -> ExecutorStep:
    """
    Tokenizes a dataset using the specified tokenizer and Levanter's tokenization infrastructure.

    Args:
        name: The name of the tokenized dataset. This is used to form the output path for the executor step.
            `tokenized/` will be prepended to the name.
        dataset:  The dataset to tokenize. This can be an InputName, ExecutorStep, a string as a
            path to the dataset or a HuggingFace dataset ID, or ``HfDatasetSpec`` to specify a
            dataset with a particular subset name.
        tokenizer: string HuggingFace tokenizer name. Should be the same as you intend to use in the tokenizer
            spec for the training run.
        format: The format of the dataset. This is used to determine how to tokenize the data.

            See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/)
            for more details.
        sample_count: Optional limit on the number of samples to tokenize per shard. If ``None``, tokenize everything.
        is_validation: Whether the dataset is a validation set. Doesn't do anything for HF datasets.
    Returns:
        An ExecutorStep that represents the tokenized dataset.
    """

    # sniff out if it's a HuggingFace dataset
    if isinstance(dataset, HfDatasetSpec):
        config = HfTokenizeConfig(
            id=dataset.id,
            name=dataset.name,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
        )
    elif isinstance(dataset, str) and dataset.count("/") == 1 and not fsspec_utils.exists(dataset):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
        )

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=tokenize,
        config=config,
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
        pip_dependency_groups=["cpu"],
        env_vars={
            "TRANSFORMERS_NO_TORCH": "1",
            "TRANSFORMERS_NO_TORCHVISION": "1",
            "USE_TORCH": "0",
            "TORCH_DISABLE_GLOBAL_DEPS": "1",
        },
    )


@lru_cache  # LRU to make the executor happier
def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, TokenizerStep]:
    # Avoid circular dependencies
    # TODO: Will - break apart defaults a bit
    from experiments.evals.exp1600_uncheatable_evals import uncheatable_eval_tokenized

    validation_sets = dict(paloma_tokenized(base_path=base_path, tokenizer=tokenizer))
    validation_sets.update(uncheatable_eval_tokenized(base_path=base_path, tokenizer=tokenizer))
    return validation_sets


def simulated_epoching_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    target_budget: int,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
) -> ExecutorStep:
    """
    Simulates the number of epochs seen in a full training run by sub-sampling individual datasets.
    Otherwise, operates the same as default_train.

    Args:
        name:  The name of the training run. Will form the basis of the output path for the executor step.
        tokenized:  The tokenized data to train on. This can be an InputName, ExecutorStep, or LMMixtureDatasetConfig.
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        target_budget: Target token budget to simulate.
        tags: Any additional tags to add to the Wandb tracker.
        use_default_validation: Whether to use the default validation sets (currently Paloma).
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable
    """
    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    # Use explicit training length rather than inferring from the model
    actual_model_config = unwrap_versioned_value(model_config)
    train_length = train_config.train_seq_len or actual_model_config.max_seq_len
    if train_length > actual_model_config.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {actual_model_config.max_seq_len}.")

    # Calculate the experiment token budget
    experiment_budget = train_config.train_batch_size * train_config.num_train_steps * train_length

    simulated_pretraining_data = dataclasses.replace(
        pretraining_data, target_budget=target_budget, experiment_budget=experiment_budget
    )

    logger.info(
        f"Simulating Epoching Behavior, Experiment Tokens {experiment_budget}, "
        + "Simulated Target Tokens {target_budget}"
    )

    return library_default_train(
        name=name,
        tokenized=simulated_pretraining_data,
        model_config=model_config,
        train_config=train_config,
        tags=tags,
        use_default_validation=use_default_validation,
        eval_harness_tasks=eval_harness_tasks,
    )


def default_sft(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    sft_config: SimpleSFTConfig,
    tags: Sequence[str] = (),
) -> ExecutorStep:
    """
    Creates an ExecutorStep for supervised fine-tuning of a language model.

    This function provides a unified interface for both single-dataset SFT and mixture-based
    SFT with a simplified configuration approach.

    Args:
        name: The name of the training run, forms the basis of the output path.
        tokenized: The tokenized data to train on:
                  - For single dataset: an InputName or ExecutorStep for a tokenized dataset.
                  - For mixture: a LMMixtureDatasetConfig with multiple datasets.
        model_config: Levanter LlamaConfig for the model architecture to train.
        sft_config: Configuration for the SFT training process.
        tags: Additional tags for WandB logging. Default: ().

    Returns:
        An ExecutorStep configured for supervised fine-tuning.
    """
    # Set up common configurations
    if "sft" not in tags:
        tags = [*tags, "sft"]

    if sft_config.initialize_from_hf is not None and sft_config.initialize_from_checkpoint_path is not None:
        raise ValueError("Cannot specify both initialize_from_hf and initialize_from_checkpoint_path!")

    # now we just shell out to default_train
    normal_train_config = SimpleTrainConfig(
        resources=sft_config.resources,
        train_batch_size=sft_config.train_batch_size,
        num_train_steps=sft_config.num_train_steps,
        learning_rate=sft_config.learning_rate,
        lr_schedule=sft_config.lr_schedule,
        decay=sft_config.decay,
        weight_decay=sft_config.weight_decay,
        min_lr_ratio=sft_config.min_lr_ratio,
        max_grad_norm=sft_config.max_grad_norm,
        warmup=sft_config.warmup,
        steps_per_eval=sft_config.steps_per_eval,
        steps_per_export=sft_config.steps_per_checkpoint,
        int8=sft_config.int8,
        steps_per_hf_export=sft_config.steps_per_hf_export,
        initialize_from_hf=sft_config.initialize_from_hf,
        initialize_from_checkpoint_path=sft_config.initialize_from_checkpoint_path,
        train_seq_len=sft_config.max_seq_len,
        data_seed=sft_config.seed,
        z_loss_weight=sft_config.z_loss_weight,
        beta1=sft_config.beta1,
        beta2=sft_config.beta2,
        pad_tokenizer_to_match_model=sft_config.pad_tokenizer_to_match_model,
        per_device_parallelism=sft_config.per_device_parallelism,
    )

    if sft_config.reinit_tokens:
        raise NotImplementedError("reinit_tokens is not supported by default_train")

    # Create and return the ExecutorStep
    return library_default_train(
        name=name,
        tokenized=tokenized,
        model_config=model_config,
        train_config=normal_train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def _get_vocab_size(pretraining_data):
    tokenizer = unwrap_versioned_value(pretraining_data.tokenizer)
    return get_vocab_size_for_tokenizer(tokenizer)


def _prepare_data_config(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    use_default_validation: bool,
) -> LMMixtureDatasetConfig:
    """
    Prepare a tokenized dataset for training. This is mostly just combining the tokenized data with the validation sets.

    Returns:
        The data config to use for training with any validation sets added.
        The evaluation data config for internal evaluation.

    """
    tokenizer = _get_tokenizer_for_train(tokenized)
    if use_default_validation:
        validation_sets = default_validation_sets(tokenizer=tokenizer)
    else:
        validation_sets = {}

    if isinstance(tokenized, InputName | ExecutorStep):
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
        )
    else:
        # TODO: would be better to expose hooks in levanter instead of relying on mixtures
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    return pretraining_data


def _get_tokenizer_for_train(tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig) -> str:
    match tokenized:
        case LMMixtureDatasetConfig(tokenizer=tokenizer):
            pass
        case ExecutorStep(config=config) if isinstance(config, TokenizeConfigBase):
            tokenizer = config.tokenizer
        case ExecutorStep(config=HfTokenizeConfig(tokenizer=tokenizer)):
            pass
        case InputName(step=ExecutorStep(config)) if isinstance(config, TokenizeConfigBase):
            tokenizer = config.tokenizer
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")

    return tokenizer
