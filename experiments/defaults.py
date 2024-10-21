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

from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
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


@dataclass(frozen=True)
class SimpleTrainConfig:
    tpu_type: str | None = None
    train_batch_size: int | None = None
    num_train_steps: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    min_lr_ratio: float | None = None
    warmup: int | None = None
    cooldown: float | None = None


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMDatasetConfig | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
) -> ExecutorStep:

    if isinstance(tokenized, InputName | ExecutorStep):
        data = lm_data_config(training_set=tokenized)
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
