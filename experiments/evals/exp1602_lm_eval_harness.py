# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive LM Evaluation Harness Testing
Reference: https://github.com/EleutherAI/lm-evaluation-harness
"""

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import (
    ACTION_TASKS,
    BIAS_SAFETY_TASKS,
    CODE_TASKS,
    EMOTIONAL_ETHICS_TASKS,
    KNOWLEDGE_TASKS,
    LANGUAGE_TASKS,
    MEDICAL_TASKS,
    REASONING_TASKS,
    SPECIALIZED_TASKS,
    TRUTHFULNESS_TASKS,
)
from experiments.models import qwen3_32b
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

from experiments.tootsie.exp1529_32b_mantis_cooldown import tootsie_32b_cooldown_mantis as marin_32b

# List of models to evaluate
MODELS_TO_EVALUATE = [
    marin_32b,
    qwen3_32b,
]

# Task configurations to run
TASK_CONFIGS = (
    REASONING_TASKS
    + EMOTIONAL_ETHICS_TASKS
    + LANGUAGE_TASKS
    + CODE_TASKS
    + MEDICAL_TASKS
    + KNOWLEDGE_TASKS
    + BIAS_SAFETY_TASKS
    + ACTION_TASKS
    + TRUTHFULNESS_TASKS
    + SPECIALIZED_TASKS
)

if __name__ == "__main__":
    # Comprehensive evaluation suite for multiple models
    eval_steps = []

    for model in MODELS_TO_EVALUATE:
        eval_steps.append(
            default_eval(
                step=model,
                resource_config=ResourceConfig.with_tpu("v5p-8"),
                evals=TASK_CONFIGS,
                discover_latest_checkpoint=False,
            )
        )
    executor_main(steps=eval_steps)
