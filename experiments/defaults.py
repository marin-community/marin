"""
This file represents the best practices for each stage of the pipeline.
"""

import os
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

############################################################
# Model sizes

llama_1_4b = LlamaConfig(
    seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_layers=16,
    num_heads=16,
    num_kv_heads=8,
)

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"

############################################################
# Tokenization

def default_tokenize(name: str, dataset: InputName | ExecutorStep, tokenizer: str) -> ExecutorStep:
    if isinstance(dataset, ExecutorStep):
        dataset = output_path_of(dataset)

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


############################################################
# Training

def default_train(name: str, tokenized: InputName | ExecutorStep, model: LmConfig) -> ExecutorStep:
    if isinstance(tokenized, ExecutorStep):
        tokenized = output_path_of(tokenized)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            tpu_type="v4-8",
            output_path=this_output_path(),
            data=LMDatasetConfig(
                train_urls=[tokenized],
            ),
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    project="marin",
                    tags=[name],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=versioned(4),
                num_train_steps=versioned(2000),
                steps_per_eval=500,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=[dict(every=500)],
                ),
            ),
            model=model,
            optimizer=AdamConfig(
                learning_rate=versioned(4e-4),
            ),
            hf_save_steps=1,
        ),
    )