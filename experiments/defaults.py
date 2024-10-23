"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import os
from collections.abc import Sequence
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMDatasetConfig, LMMixtureDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

import marin.processing.tokenize as tokenize
from experiments.dolma.paloma import tokenize_paloma_steps
from experiments.llama import compute_num_parameters
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config
from marin.processing.tokenize import TokenizerStep
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def default_tokenize(
    name: str, dataset: InputName | ExecutorStep, tokenizer: str, options: CacheOptions | None = None
) -> ExecutorStep:
    config = TokenizeConfig(
        train_paths=[dataset], validation_paths=[], cache_path=this_output_path(), tokenizer=versioned(tokenizer)
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
    return tokenize_paloma_steps(base_path=base_path, tokenizer=tokenizer)


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMDatasetConfig | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
) -> ExecutorStep:

    tokenizer = _get_tokenizer_for_train(tokenized)

    if use_default_validation:
        validation_sets = default_validation_sets(tokenizer=tokenizer)
    else:
        validation_sets = []

    if isinstance(tokenized, InputName | ExecutorStep):
        data = lm_data_config(training_set=tokenized, validation_sets=validation_sets)
    else:
        # TODO: would be better to expose hooks in levanter instead of relying on mixtures
        data = tokenized
        if validation_sets:
            if isinstance(data, LMDatasetConfig):
                # replace with a mixture
                data = tokenize.convert_to_mixture_config(data, "train")

            data = tokenize.add_validation_sets_to_mixture(data, validation_sets)

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
            model=model_config,
            optimizer=AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
            ),
            hf_save_steps=25000,
        ),
    )


def _get_tokenizer_for_train(tokenized):
    match tokenized:
        case LMDatasetConfig(tokenizer=tokenizer):
            tokenizer = tokenizer
        case LMMixtureDatasetConfig(tokenizer=tokenizer):
            tokenizer = tokenizer
        case ExecutorStep(config=TokenizeConfig(tokenizer=tokenizer)):
            tokenizer = tokenizer
        case InputName(step=ExecutorStep(config=TokenizeConfig(tokenizer=tokenizer))):
            tokenizer = tokenizer
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")

    return tokenizer
