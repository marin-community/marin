"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

import os
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.eval_harness import LmEvalHarnessConfig

from experiments.defaults import default_tokenize, default_train, _prepare_data_config
from experiments.llama import llama3_tokenizer, llama_150m, llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.pretraining_datasets import fineweb_edu, slimpajama_6b
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config

from marin.execution.executor import executor_main
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned, output_path_of
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    levanter_tokenize_supervised,
    lm_data_config,
    tokenize,
)

from experiments.curriculum.curriculum_stages import tokenize_train_validation, train_executor_step

def construct_steps():
    slimpajama_6b_tokenized = default_tokenize(name="SlimPajama-6B", dataset=slimpajama_6b, tokenizer=llama3_tokenizer)
    pretraining_data, evaluation_data = _prepare_data_config(slimpajama_6b_tokenized, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = llama_150m
    train_batch_size=1024
    num_train_steps=500 # 1024 * 1024 * 500 = 500M tokens
    learning_rate=3e-3
    weight_decay=0.1
    steps_per_eval=num_train_steps // 2
    steps_per_export=num_train_steps // 2
    name_prefix = "debug-resume-optimizer-v9"

    optimizer_config = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        cycles=[num_train_steps],
        warmup=0.01 * num_train_steps,
        decay=0.99 * num_train_steps,
    )

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-stage1",
        pretraining_data=pretraining_data,
        evaluation_data=evaluation_data,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
    )

    stage1_output_path = output_path_of(train_step_stage1).step.override_output_path

    train_step_stage2 = train_executor_step(
        name=f"{name_prefix}-stage2",
        pretraining_data=pretraining_data,
        evaluation_data=evaluation_data,
        model=model,
        model_checkpoint=f"gs://marin-us-central2/{stage1_output_path}/checkpoints/step-{num_train_steps // 2}",
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
    )

    return train_step_stage1, train_step_stage2

############################################################

if __name__ == "__main__":
    train_step_stage1, train_step_stage2 = construct_steps()

    # executor_main(
    #     steps=[train_step_stage1],
    #     description=f"Test training with varying mixtures",
    # )

    executor_main(
        steps=[train_step_stage2],
        description=f"Test training with varying mixtures",
    )
