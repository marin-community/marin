# Copyright The Marin Authors
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
    "AIME26",
    "ALICE_IN_WONDERLAND",
    "AMC23",
    "BIGCODEBENCH",
    "CODEELO",
    "CODEFORCES",
    "EVALCHEMY_CODE_TASKS",
    "EVALCHEMY_CODE_TTC_TASKS",
    "EVALCHEMY_CORE_TASKS",
    "EVALCHEMY_MATH_TASKS",
    "EVALCHEMY_MATH_TTC_TASKS",
    "EVALCHEMY_REASONING_TASKS",
    "EVALCHEMY_SCIENCE_TASKS",
    "EVALCHEMY_SCIENCE_TTC_TASKS",
    "GPQA_DIAMOND",
    "HMMT",
    "HUMANEVAL_PLUS",
    "HUMANITYS_LAST_EXAM",
    "JEEBENCH",
    "LIVECODEBENCH",
    "LIVECODEBENCH_V5_OFFICIAL",
    "LIVECODEBENCH_V6_OFFICIAL",
    "MATH500",
    "MBPP_PLUS",
    "OLYMPIADBENCH",
    "OLYMPIADBENCH_PHYSICS",
    # TTC variants
    "AIME24_TTC",
    "AIME25_TTC",
    "AIME26_TTC",
    "AMC23_TTC",
    "GPQA_DIAMOND_TTC",
    "HLE_TTC",
    "HMMT_TTC",
    "JEEBENCH_TTC",
    "LIVECODEBENCH_TTC",
    "LIVECODEBENCH_V5_OFFICIAL_TTC",
    "LIVECODEBENCH_V6_OFFICIAL_TTC",
    "MATH500_TTC",
    "OLYMPIADBENCH_PHYSICS_TTC",
    "OLYMPIADBENCH_TTC",
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
AIME26 = EvalTaskConfig(name="AIME26", num_fewshot=0, task_alias="AIME26", task_kwargs={"n_repeat": 1})
AMC23 = EvalTaskConfig(name="AMC23", num_fewshot=0, task_alias="AMC23", task_kwargs={"n_repeat": 1})

MATH500 = EvalTaskConfig(name="MATH500", num_fewshot=0, task_alias="MATH500")
HMMT = EvalTaskConfig(name="HMMT", num_fewshot=0, task_alias="HMMT", task_kwargs={"n_repeat": 1})
OLYMPIADBENCH = EvalTaskConfig(name="OlympiadBench", num_fewshot=0, task_alias="OlympiadBench")

# =============================================================================
# Code tasks
# =============================================================================
HUMANEVAL_PLUS = EvalTaskConfig(name="HumanEvalPlus", num_fewshot=0, task_alias="HumanEvalPlus")
MBPP_PLUS = EvalTaskConfig(name="MBPPPlus", num_fewshot=0, task_alias="MBPPPlus")
LIVECODEBENCH = EvalTaskConfig(
    name="LiveCodeBench", num_fewshot=0, task_alias="LiveCodeBench", task_kwargs={"n_repeat": 1}
)
LIVECODEBENCH_V5_OFFICIAL = EvalTaskConfig(
    name="LiveCodeBenchv5_official", num_fewshot=0, task_alias="LiveCodeBenchv5_official", task_kwargs={"n_repeat": 1}
)
LIVECODEBENCH_V6_OFFICIAL = EvalTaskConfig(
    name="LiveCodeBenchv6_official", num_fewshot=0, task_alias="LiveCodeBenchv6_official", task_kwargs={"n_repeat": 1}
)
BIGCODEBENCH = EvalTaskConfig(name="BigCodeBench", num_fewshot=0, task_alias="BigCodeBench")
CODEFORCES = EvalTaskConfig(name="CodeForces", num_fewshot=0, task_alias="CodeForces", task_kwargs={"n_repeat": 1})
CODEELO = EvalTaskConfig(name="CodeElo", num_fewshot=0, task_alias="CodeElo", task_kwargs={"n_repeat": 1})

# =============================================================================
# Science tasks
# =============================================================================
GPQA_DIAMOND = EvalTaskConfig(name="GPQADiamond", num_fewshot=0, task_alias="GPQADiamond", task_kwargs={"n_repeat": 1})
JEEBENCH = EvalTaskConfig(name="JEEBench", num_fewshot=0, task_alias="JEEBench", task_kwargs={"n_repeat": 1})
OLYMPIADBENCH_PHYSICS = EvalTaskConfig(
    name="OlympiadBench_Physics", num_fewshot=0, task_alias="OlympiadBench_Physics", task_kwargs={"n_repeat": 1}
)

# =============================================================================
# Reasoning tasks
# =============================================================================
ALICE_IN_WONDERLAND = EvalTaskConfig(name="AIW", num_fewshot=0, task_alias="AIW")
HUMANITYS_LAST_EXAM = EvalTaskConfig(name="HLE", num_fewshot=0, task_alias="HLE", task_kwargs={"n_repeat": 1})

# =============================================================================
# TTC (Test-Time Compute) tasks — generate N candidates + UQ filter
# =============================================================================
# TTC benchmarks use the same scoring as originals but generate 8 candidates
# per problem and apply 3-stage UQ validation to pick the best one.
# n_candidates can be overridden via task_kwargs.
AIME24_TTC = EvalTaskConfig(name="AIME24_TTC", num_fewshot=0, task_alias="AIME24_TTC")
AIME25_TTC = EvalTaskConfig(name="AIME25_TTC", num_fewshot=0, task_alias="AIME25_TTC")
AIME26_TTC = EvalTaskConfig(name="AIME26_TTC", num_fewshot=0, task_alias="AIME26_TTC")
AMC23_TTC = EvalTaskConfig(name="AMC23_TTC", num_fewshot=0, task_alias="AMC23_TTC")
MATH500_TTC = EvalTaskConfig(name="MATH500_TTC", num_fewshot=0, task_alias="MATH500_TTC")
HMMT_TTC = EvalTaskConfig(name="HMMT_TTC", num_fewshot=0, task_alias="HMMT_TTC")
OLYMPIADBENCH_TTC = EvalTaskConfig(name="OlympiadBench_TTC", num_fewshot=0, task_alias="OlympiadBench_TTC")

GPQA_DIAMOND_TTC = EvalTaskConfig(name="GPQADiamond_TTC", num_fewshot=0, task_alias="GPQADiamond_TTC")
JEEBENCH_TTC = EvalTaskConfig(name="JEEBench_TTC", num_fewshot=0, task_alias="JEEBench_TTC")
OLYMPIADBENCH_PHYSICS_TTC = EvalTaskConfig(
    name="OlympiadBench_Physics_TTC", num_fewshot=0, task_alias="OlympiadBench_Physics_TTC"
)
HLE_TTC = EvalTaskConfig(name="HLE_TTC", num_fewshot=0, task_alias="HLE_TTC")

LIVECODEBENCH_TTC = EvalTaskConfig(name="LiveCodeBench_TTC", num_fewshot=0, task_alias="LiveCodeBench_TTC")
LIVECODEBENCH_V5_OFFICIAL_TTC = EvalTaskConfig(
    name="LiveCodeBenchv5_official_TTC", num_fewshot=0, task_alias="LiveCodeBenchv5_official_TTC"
)
LIVECODEBENCH_V6_OFFICIAL_TTC = EvalTaskConfig(
    name="LiveCodeBenchv6_official_TTC", num_fewshot=0, task_alias="LiveCodeBenchv6_official_TTC"
)

# =============================================================================
# Task groups
# =============================================================================
EVALCHEMY_MATH_TASKS = (AIME24, AIME25, AIME26, AMC23, MATH500, HMMT, OLYMPIADBENCH)
EVALCHEMY_CODE_TASKS = (
    HUMANEVAL_PLUS,
    MBPP_PLUS,
    LIVECODEBENCH,
    LIVECODEBENCH_V5_OFFICIAL,
    LIVECODEBENCH_V6_OFFICIAL,
    BIGCODEBENCH,
    CODEFORCES,
    CODEELO,
)
EVALCHEMY_SCIENCE_TASKS = (GPQA_DIAMOND, JEEBENCH)
EVALCHEMY_REASONING_TASKS = (ALICE_IN_WONDERLAND, HUMANITYS_LAST_EXAM)

# TTC task groups
EVALCHEMY_MATH_TTC_TASKS = (AIME24_TTC, AIME25_TTC, AIME26_TTC, AMC23_TTC, MATH500_TTC, HMMT_TTC, OLYMPIADBENCH_TTC)
EVALCHEMY_SCIENCE_TTC_TASKS = (GPQA_DIAMOND_TTC, JEEBENCH_TTC, HLE_TTC, OLYMPIADBENCH_PHYSICS_TTC)
EVALCHEMY_CODE_TTC_TASKS = (LIVECODEBENCH_TTC, LIVECODEBENCH_V5_OFFICIAL_TTC, LIVECODEBENCH_V6_OFFICIAL_TTC)

# Combined core tasks for comprehensive evaluation
EVALCHEMY_CORE_TASKS = (
    *EVALCHEMY_MATH_TASKS,
    *EVALCHEMY_CODE_TASKS,
    *EVALCHEMY_SCIENCE_TASKS,
    *EVALCHEMY_REASONING_TASKS,
)
