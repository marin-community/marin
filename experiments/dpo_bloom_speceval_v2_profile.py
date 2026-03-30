# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DPO steady-state eval profiling run.

Targets the SECOND eval at step 400 to avoid first-eval JIT compilation overhead.
The per-batch eval function compiles on the first eval (step 200); by step 400 the
compiled program is cached and the trace shows pure steady-state execution.

Profiler captures steps 398-403 (2 training + eval at 400 + 3 post-eval).
Named scopes in the DPO loss function break down policy vs reference forward passes.
"""

from levanter.callbacks.profiler import ProfilerConfig

from experiments.dpo_bloom_speceval_v2 import tokenized_eval, tokenized_preferences, tokenized_train
from experiments.defaults import default_dpo
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM, SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-32", ram="256g"),
    per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-32"],
    train_batch_size=128,
    num_train_steps=450,
    learning_rate=5e-7,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    reference_model_path="marin-community/marin-8b-instruct",
    reference_is_hf=True,
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    steps_per_eval=200,
    steps_per_checkpoint=1000,
    steps_per_hf_export=1000,
    seed=0,
    profiler=ProfilerConfig(
        enabled=True,
        start_step=398,
        num_steps=6,
    ),
)

training_step = default_dpo(
    name="dpo/profile_bloom_speceval_v2_eval_steady_state",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=dpo_config,
    tags=["dpo", "bloom", "speceval-v2", "profiling", "eval-bottleneck", "steady-state"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            training_step,
        ]
    )
