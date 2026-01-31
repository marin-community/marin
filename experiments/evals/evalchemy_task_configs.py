# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    # Math tasks
    "AIME24",
    "AIME25",
    "AMC23",
    "MATH500",
    "HMMT",
    # Code tasks
    "HUMANEVAL_PLUS",
    "MBPP_PLUS",
    "LIVECODEBENCH",
    "BIGCODEBENCH",
    "CODEFORCES",
    "CODEELO",
    # Science tasks
    "GPQA_DIAMOND",
    "JEEBENCH",
    # Reasoning tasks
    "ALICE_IN_WONDERLAND",
    "HUMANITYS_LAST_EXAM",
    # Task groups
    "EVALCHEMY_MATH_TASKS",
    "EVALCHEMY_CODE_TASKS",
    "EVALCHEMY_SCIENCE_TASKS",
    "EVALCHEMY_REASONING_TASKS",
    "EVALCHEMY_CORE_TASKS",
]

# =============================================================================
# Math tasks
# =============================================================================
# Note: Evalchemy task names are case-sensitive and match the directory names.
# task_alias is set explicitly for clarity in logs/wandb, even when identical to name.
#
# For AIME/AMC benchmarks, we set n_repeat=1 by default. Evalchemy's originally hardcoded
# n_repeat=10 (running each problem 10 times with different seeds for averaged accuracy),
# Here we run once but control the number of seeds from Marin.
AIME24 = EvalTaskConfig(name="AIME24", num_fewshot=0, task_alias="AIME24", task_kwargs={"n_repeat": 1})
AIME25 = EvalTaskConfig(name="AIME25", num_fewshot=0, task_alias="AIME25", task_kwargs={"n_repeat": 1})
AMC23 = EvalTaskConfig(name="AMC23", num_fewshot=0, task_alias="AMC23", task_kwargs={"n_repeat": 1})

MATH500 = EvalTaskConfig(name="MATH500", num_fewshot=0, task_alias="MATH500")
HMMT = EvalTaskConfig(name="HMMT", num_fewshot=0, task_alias="HMMT")  # Harvard-MIT Math Tournament

# =============================================================================
# Code tasks
# =============================================================================
HUMANEVAL_PLUS = EvalTaskConfig(name="HumanEvalPlus", num_fewshot=0, task_alias="HumanEvalPlus")
MBPP_PLUS = EvalTaskConfig(name="MBPPPlus", num_fewshot=0, task_alias="MBPPPlus")
LIVECODEBENCH = EvalTaskConfig(name="LiveCodeBench", num_fewshot=0, task_alias="LiveCodeBench")
BIGCODEBENCH = EvalTaskConfig(name="BigCodeBench", num_fewshot=0, task_alias="BigCodeBench")
CODEFORCES = EvalTaskConfig(name="CodeForces", num_fewshot=0, task_alias="CodeForces")
CODEELO = EvalTaskConfig(name="CodeElo", num_fewshot=0, task_alias="CodeElo")

# =============================================================================
# Science tasks
# =============================================================================
GPQA_DIAMOND = EvalTaskConfig(name="GPQADiamond", num_fewshot=0, task_alias="GPQADiamond")
JEEBENCH = EvalTaskConfig(name="JEEBench", num_fewshot=0, task_alias="JEEBench")  # IIT JEE problems

# =============================================================================
# Reasoning tasks
# =============================================================================
ALICE_IN_WONDERLAND = EvalTaskConfig(name="AIW", num_fewshot=0, task_alias="AIW")
HUMANITYS_LAST_EXAM = EvalTaskConfig(name="HumanitysLastExam", num_fewshot=0, task_alias="HumanitysLastExam")

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
