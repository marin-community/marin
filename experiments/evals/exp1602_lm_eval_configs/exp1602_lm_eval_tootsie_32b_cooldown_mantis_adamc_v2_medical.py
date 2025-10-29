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
Comprehensive LM Evaluation Harness Testing for tootsie_32b_cooldown_mantis_adamc_v2 - MEDICAL_TASKS
Reference: https://github.com/EleutherAI/lm-evaluation-harness
"""

from experiments.evals.evals import default_eval
from experiments.evals.resource_configs import *
from experiments.evals.task_configs import MEDICAL_TASKS
from experiments.models import tootsie_32b_cooldown_mantis_adamc_v2
from marin.execution.executor import ExecutorMainConfig, executor_main

# from experiments.tootsie.exp1529_32b_mantis_cooldown import tootsie_32b_cooldown_mantis as marin_32b

TPU = SINGLE_TPU_V5p_8_FULL
MODEL = tootsie_32b_cooldown_mantis_adamc_v2
TASK_CONFIG = MEDICAL_TASKS

if __name__ == "__main__":
    eval_steps = [
        default_eval(
            step=MODEL,
            resource_config=TPU,
            evals=TASK_CONFIG,
            discover_latest_checkpoint=False,
        )
    ]
    executor_main(steps=eval_steps)
