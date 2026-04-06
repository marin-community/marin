# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for lm-eval failure on multi-host TPU (v4-32).

Trains a tiny Llama 30M on SlimPajama 6B for 20 steps with CORE_TASKS
eval at step 10 and 20.  The v4-32 forces a multi-host TPU slice where
the eval harness breaks.
"""
import dataclasses

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_30m
from experiments.pretraining_datasets.simple import tokenized
from experiments.simple_train_config import SimpleTrainConfig

RESOURCES = ResourceConfig.with_tpu("v4-32")

# Point at pre-existing tokenized SlimPajama-6B in us-central2
slimpajama_tokenized = tokenized["slimpajama_6b"].with_output_path("tokenized/SlimPajama-6B-499d45")

small_train_config = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=128,
    num_train_steps=20,
    learning_rate=6e-4,
    weight_decay=0.1,
    steps_per_eval=10,
    steps_per_task_eval=10,
    steps_per_export=20,
)

run = default_train(
    name="repro-lm-eval-multi-host-tpu",
    tokenized=slimpajama_tokenized,
    model_config=versioned(llama_30m),
    train_config=small_train_config,
    tags=["llama", "30m", "slimpajama_6b", "lm-eval-bug-repro"],
    eval_harness_tasks=CORE_TASKS,
)

if __name__ == "__main__":
    executor_main(steps=[run])
