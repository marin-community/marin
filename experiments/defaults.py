"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMDatasetConfig, LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.dolma.paloma import tokenize_paloma_steps
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.processing.tokenize.tokenize import TokenizerStep
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
        fn=tokenize,
        config=config,
    )


def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> list[TokenizerStep]:
    return list(tokenize_paloma_steps(base_path=base_path, tokenizer=tokenizer).values())


@dataclass(frozen=True)
class SimpleTrainConfig:
    """Simplified configuration for training (the things that matter)."""

    tpu_type: str
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    weight_decay: float


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMDatasetConfig | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
) -> ExecutorStep:

    tokenizer = _get_tokenizer_for_train(tokenized)

    if isinstance(tokenized, InputName | ExecutorStep):
        if use_default_validation:
            validation_sets = default_validation_sets(tokenizer=tokenizer)
        else:
            validation_sets = []
        data = lm_data_config(training_set=tokenized, validation_sets=validation_sets)
    else:
        data = tokenized

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
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
