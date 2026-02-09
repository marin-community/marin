# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a (HF-exported) checkpoint on a TPU slice via the executor.

We generally prefer to keep these small driver scripts configured in Python (constants below)
rather than adding custom CLI flags on top of `executor_main`.
"""

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

MODEL_PATH = "gs://marin-us-central1/checkpoints/example/hf/step-0/"
TPU_TYPE = "v5p-64"
SLICE_COUNT = 1


if __name__ == "__main__":
    eval_step = default_eval(
        MODEL_PATH,
        ResourceConfig.with_tpu(TPU_TYPE, slice_count=SLICE_COUNT),
        evals=list(CORE_TASKS_PLUS_MMLU),
    )
    executor_main(steps=[eval_step])
