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

from levanter.models.rotary import Llama3RotaryEmbeddingsConfig

TAG = "702_continued_training"
USER = "suhas"

EVAL_TASKS = (
    # EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    # EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    EvalTaskConfig("mathqa", 5, task_alias="mathqa_5shot"),
    # EvalTaskConfig("pubmedqa", 0, task_alias="pubmedqa"),
    # EvalTaskConfig("pubmedqa", 5, task_alias="pubmedqa_5shot"),
    # EvalTaskConfig("medqa", 0, task_alias="medqa"),
    # EvalTaskConfig("medqa", 5, task_alias="medqa_5shot")
)

def cpt_train_executor_step(
    name : str,
    pretraining_data : lm_data_config,
    model_name : str,
    train_batch_size : int,
    num_train_steps : int,
    learning_rate : float,
    weight_decay : float,
    steps_per_eval : int,
    steps_per_export_list : list[int],
    tpu_type : str,
    optimizer_config : Optional[AdamConfig] = None,
    additional_tags : list[str] = [],
    steps_per_eval_task : Optional[int] = None,
    data_seed : int = 42,
    project_name : str = "suhas-cpt",
    warmup_steps : float = 0.01,
) -> ExecutorStep:
    
    assert model_name == "meta-llama/Meta-Llama-3.1-8B"

    model_config=LlamaConfig(
        seq_len=4096,  # Seq len set to reproduce Tulu SFT
        hidden_dim=4096,
        intermediate_dim=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        use_bias=False,
        use_layer_norm_weight=True,
        initializer_range=0.02,
        use_flash_attention=True,
        flash_attention_block_size=512,
        rope=Llama3RotaryEmbeddingsConfig(
            # Using Llama3 defaults from the code
            theta=500000,
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    )
    
    if optimizer_config is None:
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup=warmup_steps,
        )

    if steps_per_eval_task:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(EVAL_TASKS))
    else:
        harness_config = None

    train_config = TrainLmOnPodConfig(
        output_path=this_output_path(),
        tpu_type=tpu_type,
        node_count=2,
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=project_name,
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
        model=model_config,
        initialize_from_hf=model_name,
        optimizer=optimizer_config,
        data_seed=data_seed,
        eval_harness_steps=steps_per_eval_task,
        eval_harness=harness_config,
        hf_save_steps=num_train_steps,
    )

    executor_step_name = os.path.join("checkpoints", USER, name[:64])

    return ExecutorStep(
        name=executor_step_name,
        override_output_path=executor_step_name,
        fn=run_levanter_train_lm,
        description=f"Train a {model_name} model for "
        f"{num_train_steps} (steps)",
        config=train_config,
        pip_dependency_groups=["tokenize_train"],
    )