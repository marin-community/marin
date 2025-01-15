"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

import os
from datetime import timedelta
import random
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

def full_training_stage(starting_code_portion, ending_code_portion, stage):
    BASE_DIR_STACK_PYTHON = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/python"
    BASE_DIR_DOLMA = "gs://marin-us-central2/raw/dolma/v1.7"

    # randomly split stack python parquet files into two seperate groups
    stack_file_ids = list(range(144))
    random.seed(42)
    random.shuffle(stack_file_ids)
    stack_file_ids_stage1 = stack_file_ids[0:72]
    stack_file_ids_stage2 = stack_file_ids[72:143]
    stack_file_ids_validation = stack_file_ids[143:144]

    # randomly split dolma c4 json.gz files into two seperate groups
    dolma_file_ids = list(range(171))
    random.shuffle(dolma_file_ids)
    dolma_file_ids_stage1 = dolma_file_ids[0:85]
    dolma_file_ids_stage2 = dolma_file_ids[85:170]
    dolma_file_ids_validation = dolma_file_ids[170:171]

    # Stage 1

    stack_dedup_stage1_tokenized = tokenize_train_validation(
        train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage1],
        validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
        name="stack_dedup_stage1",
        text_key="content"
    )

    dolma_c4_stage1_tokenized = tokenize_train_validation(
        train_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_stage1],
        validation_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_validation],
        name="dolma_c4_stage1",
    )

    data_config_stage1 = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage1_tokenized, "c4": dolma_c4_stage1_tokenized},
        weights={"stack_dedup": starting_code_portion, "c4": 1 - starting_code_portion},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    # Stage 2

    stack_dedup_stage2_tokenized = tokenize_train_validation(
        train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage2],
        validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
        name="stack_dedup_stage2",
        text_key="content"
    )

    dolma_c4_stage2_tokenized = tokenize_train_validation(
        train_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_stage2],
        validation_files=[f"{BASE_DIR_DOLMA}/c4-{id:04d}.json.gz" for id in dolma_file_ids_validation],
        name="dolma_c4_stage2",
    )

    data_config_stage2 = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage2_tokenized, "c4": dolma_c4_stage2_tokenized},
        weights={"stack_dedup": ending_code_portion, "c4": 1 - ending_code_portion},
    )

    pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = llama_150m
    train_batch_size=1024
    num_train_steps=3000 # 1024 * 1024 * 3000 = 3B tokens
    learning_rate=3e-3
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=num_train_steps // 2
    name_prefix = f"stack-dedup-c4-curriculum-3B-150m-halfsched"

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-{starting_code_portion}-stage1",
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
        name=f"{name_prefix}-{starting_code_portion}-{ending_code_portion}-stage2",
        pretraining_data=pretraining_data_stage2,
        evaluation_data=evaluation_data_stage2,
        model=model,
        model_checkpoint=f"gs://marin-us-central2/checkpoints/suhas/{name_prefix}-{starting_code_portion}-stage1/checkpoints/step-{num_train_steps // 2}",
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
    )

    if stage == "stage1":
        return train_step_stage1
    else:
        return train_step_stage2

############################################################

if __name__ == "__main__":
    stage = "stage2"

    executor_main(
        steps=[
            full_training_stage(starting_code_portion=0.001, ending_code_portion=0.009, stage=stage),
            full_training_stage(starting_code_portion=0.003, ending_code_portion=0.007, stage=stage),
            full_training_stage(starting_code_portion=0.005, ending_code_portion=0.005, stage=stage),
            full_training_stage(starting_code_portion=0.007, ending_code_portion=0.003, stage=stage),
            full_training_stage(starting_code_portion=0.009, ending_code_portion=0.001, stage=stage),
        ],
        description=f"Test training with varying mixtures",
    )
