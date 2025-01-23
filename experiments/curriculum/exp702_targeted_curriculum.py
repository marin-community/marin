"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

import os
from datetime import timedelta
import random
import math

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

def full_training_stage_varsched(total_code_portion, duration_frac_stage2, code_frac_alloc_stage2, stage, learning_rate=3e-3, cooldown_frac=None, schedule_type="cosine", version_tag="", additional_tags=[]):
    duration_frac_stage1 = 1 - duration_frac_stage2
    code_frac_alloc_stage1 = 1 - code_frac_alloc_stage2

    code_weight_stage1 = round(total_code_portion * code_frac_alloc_stage1 / duration_frac_stage1, 7)
    code_weight_stage2 = round(total_code_portion * code_frac_alloc_stage2 / duration_frac_stage2, 7)

    print('-' * 100)
    print(f"total_code_portion: {total_code_portion}")
    print(f"duration_frac_stage1: {duration_frac_stage1}, code_frac_alloc_stage1: {code_frac_alloc_stage1}, code_weight_stage1: {code_weight_stage1}")
    print(f"duration_frac_stage2: {duration_frac_stage2}, code_frac_alloc_stage2: {code_frac_alloc_stage2}, code_weight_stage2: {code_weight_stage2}")

    assert 0 <= code_weight_stage1 <= 1, f"code_weight_stage1: {code_weight_stage1}"
    assert 0 <= code_weight_stage2 <= 1, f"code_weight_stage2: {code_weight_stage2}"

    data_config_stage1 = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage1_tokenized, "c4": dolma_c4_stage1_tokenized},
        weights={"stack_dedup": code_weight_stage1, "c4": 1 - code_weight_stage1},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    # Stage 2

    data_config_stage2 = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage2_tokenized, "c4": dolma_c4_stage2_tokenized},
        weights={"stack_dedup": code_weight_stage2, "c4": 1 - code_weight_stage2},
    )

    pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = llama_150m
    train_batch_size=1024
    num_train_steps=3000 # 1024 * 1024 * 3000 = 3B tokens

    steps_stage1 = int(num_train_steps * duration_frac_stage1)
    steps_stage2 = num_train_steps - steps_stage1

    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=steps_stage1 if stage == "stage1" else num_train_steps
    name_prefix = f"varsched-bf-{code_frac_alloc_stage2}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "linear-sgd":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule="linear",
            decay=cooldown_frac,
            beta1=0.8,
            beta2=0.9,
        )
        name_prefix += f"-linear-sgd-0.8"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-{code_weight_stage1}-{code_weight_stage2}-stage1{version_tag}",
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
        optimizer_config=optimizer_config,
        additional_tags=additional_tags,
    )

    train_step_stage2 = train_executor_step(
        name=f"{name_prefix}-{code_weight_stage1}-{code_weight_stage2}-stage2{version_tag}",
        pretraining_data=pretraining_data_stage2,
        evaluation_data=evaluation_data_stage2,
        model=model,
        model_checkpoint=f"gs://marin-us-central2/checkpoints/suhas/{name_prefix}-{code_weight_stage1}-{code_weight_stage2}-stage1{version_tag}/checkpoints/step-{steps_stage1}",
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags,
    )

    if stage == "stage1":
        return train_step_stage1
    else:
        return train_step_stage2

def full_training_stage_halfsched(starting_code_portion, ending_code_portion, stage, num_train_steps=3000, model_size="150m", version_tag="", additional_tags=[]):
    data_config_stage1 = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage1_tokenized, "c4": dolma_c4_stage1_tokenized},
        weights={"stack_dedup": starting_code_portion, "c4": 1 - starting_code_portion},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    data_config_stage2 = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage2_tokenized, "c4": dolma_c4_stage2_tokenized},
        weights={"stack_dedup": ending_code_portion, "c4": 1 - ending_code_portion},
    )

    pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
    }[model_size]
    train_batch_size=1024
    learning_rate=3e-3
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=3000 // 2
    name_prefix = f"stack-dedup-c4-curriculum-{num_train_steps // 1000}B-{model_size}-halfsched"

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-{starting_code_portion}-stage1{version_tag}",
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
        additional_tags=additional_tags,
    )

    train_step_stage2 = train_executor_step(
        name=f"{name_prefix}-{starting_code_portion}-{ending_code_portion}-stage2{version_tag}",
        pretraining_data=pretraining_data_stage2,
        evaluation_data=evaluation_data_stage2,
        model=model,
        model_checkpoint=f"gs://marin-us-central2/checkpoints/suhas/{name_prefix}-{starting_code_portion}-stage1{version_tag}/checkpoints/step-{num_train_steps // 2}",
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        tpu_type=tpu_type,
        additional_tags=additional_tags,
    )

    if stage == "stage1":
        return train_step_stage1
    else:
        return train_step_stage2

def full_training_stage_baseline_sweep(learning_rate, schedule_type, cooldown_frac=None, num_train_steps=3000, model_size="150m", additional_tags=[], code_portion=0.005):
    data_config = lm_mixture_data_config(
        components={"stack_dedup": stack_dedup_stage1_tokenized, "c4": dolma_c4_stage1_tokenized},
        weights={"stack_dedup": code_portion, "c4": 1 - code_portion},
    )

    pretraining_data, evaluation_data = _prepare_data_config(data_config, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
    }[model_size]
    train_batch_size=1024
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=3000 // 2
    name_prefix = f"stack-c4-{num_train_steps // 1000}B-{model_size}-baseline-lr-sweep"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )

        schedule_type = f"linear-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    train_step = train_executor_step(
        name=f"{name_prefix}-{learning_rate}-{schedule_type}",
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
        additional_tags=additional_tags,
    )

    return train_step

############################################################

if __name__ == "__main__":
    # Do a sweep over length of stage 2 for all data in stage 2

    stage = "stage2"

    executor_main(
        steps=[
            full_training_stage_varsched(total_code_portion=0.005, duration_frac_stage2=duration_frac_stage2, code_frac_alloc_stage2=code_frac_alloc_stage2, schedule_type=schedule_type, cooldown_frac=cooldown_frac, stage=stage, additional_tags=["all-stage2-bruteforce", stage], version_tag="-v1")
            for duration_frac_stage2 in [0.2, 0.1, 0.05, 0.025, 0.00625]
            for schedule_type, cooldown_frac in [("linear", 0.01)]
            for code_frac_alloc_stage2 in [1.0]
        ],
        description=f"Test training with varying mixtures",
    )

    # # Fit scaling laws for no cooldown runs

    # executor_main(
    #     steps=[
    #         full_training_stage_baseline_sweep(learning_rate=3e-3, schedule_type="linear", cooldown_frac=cooldown_frac, num_train_steps=3000, model_size="150m", additional_tags=['zero-cooldown-scaling'], code_portion=code_portion)
    #         for cooldown_frac in [0.0]
    #         for code_portion in [0.001, 0.005, 0.01, 0.05, 0.1]
    #     ],
    #     description=f"Test training with varying mixtures",
    # )
