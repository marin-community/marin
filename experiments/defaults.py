"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import os
from collections.abc import Sequence
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.llama import compute_num_parameters
from experiments.paloma import paloma_tokenized
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    lm_mixture_data_config,
    tokenize,
)
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep,
    tokenizer: str,
    options: CacheOptions | None = None,
    text_key: str = "text",
) -> ExecutorStep:
    config = TokenizeConfig(
        train_paths=[dataset],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(tokenizer),
        text_key=text_key,
    )
    if options is not None:
        config = dataclasses.replace(config, cache_options=options)

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=tokenize,
        config=config,
    )


def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, TokenizerStep]:
    return paloma_tokenized(base_path=base_path, tokenizer=tokenizer)


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | dict[str, ExecutorStep] | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    weights: dict[str, float] | None = None,
    use_default_validation: bool = True,
) -> ExecutorStep:
    data = _prepare_data_config(tokenized, weights, use_default_validation)

    # TODO: right now, assume architecture is a LlamaConfig, generalize this
    assert isinstance(model_config, LlamaConfig)
    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=f"Train a {compute_num_parameters(model_config):,} parameter model for "
        f"{train_config.num_train_steps} (steps) * "
        f"{train_config.train_batch_size} (batch_size) * "
        f"{model_config.seq_len} (seq_len) "
        f"= {train_config.num_train_steps * train_config.train_batch_size * model_config.seq_len:,} tokens.",
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            tpu_type=train_config.tpu_type,
            data=data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    project="marin",
                    tags=[name, *tags],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=train_config.train_batch_size,
                num_train_steps=train_config.num_train_steps,
                steps_per_eval=1000,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=[dict(every=25000)],
                ),
            ),
            z_loss_weight=train_config.z_loss_weight,
            model=model_config,
            optimizer=AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=(
                    train_config.weight_decay if train_config.weight_decay is not None else AdamConfig().weight_decay
                ),
                warmup=train_config.warmup if train_config.warmup is not None else AdamConfig().warmup,
                cooldown=train_config.cooldown if train_config.cooldown is not None else AdamConfig().cooldown,
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
            ),
            hf_save_steps=25000,
        ),
    )


def _prepare_data_config(
    tokenized: InputName | ExecutorStep | dict[str, ExecutorStep] | LMMixtureDatasetConfig,
    weights: dict[str, float],
    use_default_validation: bool,
) -> LMMixtureDatasetConfig:
    """
    Prepare a tokenized dataset for training. This is mostly just combining the tokenized data with the validation sets.

    Returns:
        The data config to use for training with any validation sets added.

    """
    tokenizer = _get_tokenizer_for_train(tokenized)
    if use_default_validation:
        validation_sets = default_validation_sets(tokenizer=tokenizer)
    else:
        validation_sets = []
    if isinstance(tokenized, InputName | ExecutorStep):
        data = lm_data_config(training_set=tokenized, validation_sets=validation_sets)
    elif isinstance(tokenized, dict):
        tokenized.update(validation_sets)
        data = lm_mixture_data_config(components=tokenized, weights=weights)
    else:
        # TODO: would be better to expose hooks in levanter instead of relying on mixtures
        data = tokenized
        if validation_sets:
            data = add_validation_sets_to_mixture(data, validation_sets)
    return data


def _get_tokenizer_for_train(
    tokenized: InputName | ExecutorStep | dict[str, ExecutorStep] | LMMixtureDatasetConfig,
) -> str:
    match tokenized:
        case LMMixtureDatasetConfig(tokenizer=tokenizer):
            pass
        case ExecutorStep(config=TokenizeConfig(tokenizer=tokenizer)):
            pass
        case InputName(step=ExecutorStep(config=TokenizeConfig(tokenizer=tokenizer))):
            pass
        case dict():
            first_executor_step = next(iter(tokenized.values()))
            tokenizer = _get_tokenizer_for_train(first_executor_step)
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")

    return tokenizer
