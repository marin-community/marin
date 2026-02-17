# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Task configurations for Evalchemy-specific reasoning benchmarks.

Evalchemy (https://github.com/mlfoundations/evalchemy) provides specialized
reasoning tasks beyond standard lm-evaluation-harness. This module defines
self-contained tasks only (no LLM-as-judge benchmarks that require OPENAI_API_KEY).

Note: Judge-based benchmarks (MT-Bench, AlpacaEval, WildBench) are excluded
as they require OPENAI_API_KEY. These could be added later if needed.
"""

from marin.evaluation.evaluation_config import EvalTaskConfig

__all__ = [
    "AIME24",
    "AIME25",
    "ALICE_IN_WONDERLAND",
    "AMC23",
    "BIGCODEBENCH",
    "CODEELO",
    "CODEFORCES",
    "EVALCHEMY_CODE_TASKS",
    "EVALCHEMY_CORE_TASKS",
    "EVALCHEMY_MATH_TASKS",
    "EVALCHEMY_REASONING_TASKS",
    "EVALCHEMY_SCIENCE_TASKS",
    "GPQA_DIAMOND",
    "HMMT",
    "HUMANEVAL_PLUS",
    "HUMANITYS_LAST_EXAM",
    "JEEBENCH",
    "LIVECODEBENCH",
    "MATH500",
    "MBPP_PLUS",
]

# =============================================================================
# Math tasks
# =============================================================================
# Note: Evalchemy task names are case-sensitive and match the directory names.
# task_alias is set explicitly for clarity in logs/wandb, even when identical to name.
#
# Many evalchemy benchmarks hardcode n_repeat > 1 (running each problem multiple times with
# different seeds for averaged accuracy). We override n_repeat=1 and control the number of
# seeds from Marin instead, to avoid a hidden seeds x n_repeat multiplier on iteration count.
AIME24 = EvalTaskConfig(name="AIME24", num_fewshot=0, task_alias="AIME24", task_kwargs={"n_repeat": 1})
AIME25 = EvalTaskConfig(name="AIME25", num_fewshot=0, task_alias="AIME25", task_kwargs={"n_repeat": 1})
AMC23 = EvalTaskConfig(name="AMC23", num_fewshot=0, task_alias="AMC23", task_kwargs={"n_repeat": 1})

MATH500 = EvalTaskConfig(name="MATH500", num_fewshot=0, task_alias="MATH500")
HMMT = EvalTaskConfig(name="HMMT", num_fewshot=0, task_alias="HMMT", task_kwargs={"n_repeat": 1})

# =============================================================================
# Code tasks
# =============================================================================
HUMANEVAL_PLUS = EvalTaskConfig(name="HumanEvalPlus", num_fewshot=0, task_alias="HumanEvalPlus")
MBPP_PLUS = EvalTaskConfig(name="MBPPPlus", num_fewshot=0, task_alias="MBPPPlus")
LIVECODEBENCH = EvalTaskConfig(
    name="LiveCodeBench", num_fewshot=0, task_alias="LiveCodeBench", task_kwargs={"n_repeat": 1}
)
BIGCODEBENCH = EvalTaskConfig(name="BigCodeBench", num_fewshot=0, task_alias="BigCodeBench")
CODEFORCES = EvalTaskConfig(name="CodeForces", num_fewshot=0, task_alias="CodeForces", task_kwargs={"n_repeat": 1})
CODEELO = EvalTaskConfig(name="CodeElo", num_fewshot=0, task_alias="CodeElo", task_kwargs={"n_repeat": 1})

# =============================================================================
# Science tasks
# =============================================================================
GPQA_DIAMOND = EvalTaskConfig(name="GPQADiamond", num_fewshot=0, task_alias="GPQADiamond", task_kwargs={"n_repeat": 1})
JEEBENCH = EvalTaskConfig(name="JEEBench", num_fewshot=0, task_alias="JEEBench", task_kwargs={"n_repeat": 1})

# =============================================================================
# Reasoning tasks
# =============================================================================
ALICE_IN_WONDERLAND = EvalTaskConfig(name="AIW", num_fewshot=0, task_alias="AIW")
HUMANITYS_LAST_EXAM = EvalTaskConfig(
    name="HLE", num_fewshot=0, task_alias="HumanitysLastExam", task_kwargs={"n_repeat": 1}
)

# =============================================================================
# Task groups
# =============================================================================
EVALCHEMY_MATH_TASKS = (AIME24, AIME25, AMC23, MATH500, HMMT)
EVALCHEMY_CODE_TASKS = (HUMANEVAL_PLUS, MBPP_PLUS, LIVECODEBENCH, BIGCODEBENCH, CODEFORCES, CODEELO)
EVALCHEMY_SCIENCE_TASKS = (GPQA_DIAMOND, JEEBENCH)
EVALCHEMY_REASONING_TASKS = (ALICE_IN_WONDERLAND, HUMANITYS_LAST_EXAM)

# Combined core tasks for comprehensive evaluation
EVALCHEMY_CORE_TASKS = (
    *EVALCHEMY_MATH_TASKS,
    *EVALCHEMY_CODE_TASKS,
    *EVALCHEMY_SCIENCE_TASKS,
    *EVALCHEMY_REASONING_TASKS,
)
