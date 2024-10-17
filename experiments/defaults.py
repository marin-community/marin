"""
This file represents the best practices for each stage of the pipeline.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_training_config, tokenize
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def default_tokenize(name: str, dataset: InputName | ExecutorStep, tokenizer: str) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join("tokenized", name),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[dataset],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )


@dataclass(frozen=True)
class SimpleTrainConfig:
    """Simplified configuration for training (the things that matter)."""

    tpu_type: str
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    weight_decay: float


def default_train(
    name: str, tokenized: InputName | ExecutorStep, model_config: LmConfig, train_config: SimpleTrainConfig
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            tpu_type=train_config.tpu_type,
            data=lm_training_config(training_set=tokenized),
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    project="marin",
                    tags=[name],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=train_config.train_batch_size,
                num_train_steps=train_config.num_train_steps,
                steps_per_eval=10000,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=[dict(every=10000)],
                ),
            ),
            model=model_config,
            optimizer=AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
            ),
            hf_save_steps=1,
        ),
    )


def default_eval(
    name: str,
    evalution_config: EvaluationConfig,
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join("evaluation", name),
        fn=evaluate,
        config=EvaluationConfig(
            evaluator=evalution_config.evaluator,
            model_name=evalution_config.model_name,
            model_path=evalution_config.model_path,
            evaluation_path=this_output_path(),
            evals=evalution_config.evals,
            launch_with_ray=evalution_config.launch_with_ray,
        ),
    )
