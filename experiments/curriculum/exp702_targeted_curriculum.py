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
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
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

# Stage 1

dolma_starcoder_stage1_tokenized = tokenize_train_validation(
    train_files=["starcoder-{0000..0023}.json.gz"],
    validation_files=["starcoder-{0048..0048}.json.gz"],
    name="dolma_starcoder_stage1",
)

dolma_c4_stage1_tokenized = tokenize_train_validation(
    train_files=["c4-{0000..0084}.json.gz"],
    validation_files=["c4-{0170..0170}.json.gz"],
    name="dolma_c4_stage1",
)

data_config_stage1 = lm_mixture_data_config(
    components={"starcoder": dolma_starcoder_stage1_tokenized, "c4": dolma_c4_stage1_tokenized},
    weights={"starcoder": 0.5, "c4": 0.5},
)

pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

# Stage 2

dolma_starcoder_stage2_tokenized = tokenize_train_validation(
    train_files=["starcoder-{0024..0047}.json.gz"],
    validation_files=["starcoder-{0048..0048}.json.gz"],
    name="dolma_starcoder_stage2",
)

dolma_c4_stage2_tokenized = tokenize_train_validation(
    train_files=["c4-{0085..0169}.json.gz"],
    validation_files=["c4-{0170..0170}.json.gz"],
    name="dolma_c4_stage2",
)

data_config_stage2 = lm_mixture_data_config(
    components={"starcoder": dolma_starcoder_stage2_tokenized, "c4": dolma_c4_stage2_tokenized},
    weights={"starcoder": 0.5, "c4": 0.5},
)

pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

# Construct executor steps for training

tpu_type="v4-128"
model = llama_150m
train_batch_size=1024
num_train_steps=600 # 1024 * 1024 * 600 = 600M tokens
learning_rate=3e-3
weight_decay=0.1
steps_per_eval=num_train_steps // 10
steps_per_export=num_train_steps // 2
name_prefix = "starcoder-c4-curriculum-150m-0.5-0.5-debug"

train_step_stage1 = train_executor_step(
    name=f"{name_prefix}-stage1",
    pretraining_data=pretraining_data_stage1,
    evaluation_data=evaluation_data_stage1,
    model=model,
    model_checkpoint=None,
    train_batch_size=train_batch_size,
    num_train_steps=num_train_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    steps_per_eval=steps_per_eval,
    steps_per_export=steps_per_export,
    tpu_type=tpu_type,
)

train_step_stage2 = train_executor_step(
    name=f"{name_prefix}-stage2",
    pretraining_data=pretraining_data_stage2,
    evaluation_data=evaluation_data_stage2,
    model=model,
    model_checkpoint=f"gs://marin-us-central2/checkpoints/{name_prefix}-stage1/checkpoints/step-{num_train_steps // 2}",
    train_batch_size=train_batch_size,
    num_train_steps=num_train_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    steps_per_eval=steps_per_eval,
    steps_per_export=steps_per_export,
    tpu_type=tpu_type,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            train_step_stage1,
            train_step_stage2,
        ],
        description=f"Test training with varying mixtures",
    )
