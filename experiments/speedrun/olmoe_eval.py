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
