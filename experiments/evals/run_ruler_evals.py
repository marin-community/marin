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
Evaluate Llama-3.1-8B-Instruct on the RULER long-context benchmark.
"""

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import LONG_CONTEXT_TASKS
from experiments.models import llama_3_1_8b_instruct
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

ruler_eval_step = default_eval(
    step=llama_3_1_8b_instruct,
    resource_config=ResourceConfig.with_tpu("v4-8"),
    evals=LONG_CONTEXT_TASKS,
    apply_chat_template=True,
)

if __name__ == "__main__":
    executor_main(steps=[ruler_eval_step])
