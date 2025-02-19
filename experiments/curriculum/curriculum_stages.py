"""
Test continued training from checkpoint to support different mixtures.
Issue: TODO
"""

import os
from datetime import timedelta
from typing import Optional

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_tokenize, default_train, _prepare_data_config
from experiments.llama import llama3_tokenizer, llama_150m, llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.pretraining_datasets import fineweb_edu, slimpajama_6b
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config

from marin.execution.executor import executor_main
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.processing.tokenize.tokenize import levanter_tokenize_sft
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    levanter_tokenize_supervised,
    lm_data_config,
    tokenize,
)

TAG = "702_targeted_curriculum"
USER = "suhas"

EVAL_TASKS = (
    EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    EvalTaskConfig("humaneval", 0),  # coding problems
    EvalTaskConfig("mbpp", 3),  # coding problems
    EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"),
    EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot"),
)

def tokenize_two_stages(
    base_file_path : str,
    train_file_ids_stage1 : list[int],
    train_file_ids_stage2 : list[int],
    validation_file_ids : list[int],
    name : str,
    **kwargs
):
    stage1_tokenized = tokenize_train_validation(
        base_file_path=base_file_path,
        train_file_ids=train_file_ids_stage1,
        validation_file_ids=validation_file_ids,
        name=f"{name}_stage1",
        **kwargs
    )

    stage2_tokenized = tokenize_train_validation(
        base_file_path=base_file_path,
        train_file_ids=train_file_ids_stage2,
        validation_file_ids=validation_file_ids,
        name=f"{name}_stage2",
        **kwargs
    )

    return stage1_tokenized, stage2_tokenized

def tokenize_train_validation(
    base_file_path : str,
    train_file_ids : list[int],
    validation_file_ids : list[int],
    name : str,
    **kwargs
) -> ExecutorStep:

    tokenizer_config = TokenizeConfig(
        train_paths=versioned([base_file_path.format(id=id) for id in train_file_ids]),
        validation_paths=versioned([base_file_path.format(id=id) for id in validation_file_ids]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
        **kwargs
    )

    return ExecutorStep(
        name=os.path.join("tokenized", "suhas", f"{name}"),
        description=f"Tokenize raw text using the llama3_tokenizer (with manual validation set).",
        fn=tokenize,
        config=tokenizer_config,
    )

def train_executor_step(
    name : str,
    pretraining_data : lm_data_config,
    model : LlamaConfig,
    model_checkpoint : str,
    train_batch_size : int,
    num_train_steps : int,
    learning_rate : float,
    weight_decay : float,
    steps_per_eval : int,
    steps_per_export_list : list[int],
    tpu_type : str,
    optimizer_config : AdamConfig = None,
    additional_tags : list[str] = [],
    steps_per_eval_task : Optional[int] = None,
) -> ExecutorStep:
    
    if optimizer_config is None:
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    if steps_per_eval_task:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(EVAL_TASKS))
    else:
        harness_config = None

    train_config = TrainLmOnPodConfig(
        output_path=this_output_path(),
        tpu_type=tpu_type,
        node_count=1,
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="suhas-curriculum",
                tags=[name, TAG, *additional_tags],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_batch_size,
            num_train_steps=num_train_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=steps_per_export, until=steps_per_export+1) for steps_per_export in steps_per_export_list],
            ),
            replica_dcn_axis_size=-1,
        ),
        z_loss_weight=None,
        model=model,
        optimizer=optimizer_config,
        data_seed=42,
        initialize_from_checkpoint_path=model_checkpoint,
        eval_harness_steps=steps_per_eval_task,
        eval_harness=harness_config,
    )

    executor_step_name = os.path.join("checkpoints", USER, name[:64])

    return ExecutorStep(
        name=executor_step_name,
        override_output_path=executor_step_name,
        fn=run_levanter_train_lm,
        description=f"Train a llama_300m model for "
        f"{num_train_steps} (steps) * "
        f"{train_batch_size} (batch_size) * "
        f"{model.seq_len} (seq_len) "
        f"= {num_train_steps * train_batch_size * model.seq_len:,} tokens.",
        config=train_config,
        pip_dependency_groups=["tokenize_train"],
    )
