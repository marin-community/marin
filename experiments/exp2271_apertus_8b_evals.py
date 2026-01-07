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
Evaluate Apertus-8B on MMLU 0-shot and 5-shot.
"""

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_0_SHOT, MMLU_5_SHOT
from experiments.models import apertus_8b
from marin.execution.executor import executor_main

if __name__ == "__main__":
    mmlu_eval = default_eval(
        step=apertus_8b,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        evals=(MMLU_0_SHOT, MMLU_5_SHOT),
        discover_latest_checkpoint=False,
    )
    executor_main(steps=[mmlu_eval])
