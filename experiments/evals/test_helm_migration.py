#!/usr/bin/env python3
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

"""Test HELM evaluation with new pool-based architecture."""

import os

from fray.cluster.base import CpuConfig, ResourceConfig, TpuConfig
from marin.execution.executor import executor_main

from experiments.evals.evals import evaluate_helm
from experiments.evals.task_configs import EvalTaskConfig

backend_type = os.environ.get("backend_type", "tpu")

if backend_type == "cpu":
    resource_config = ResourceConfig(
        cpu=1,
        ram="1g",
        disk="10g",
        device=CpuConfig(),
        replicas=1,
    )
else:
    resource_config = ResourceConfig(
        cpu=1,
        ram="16",
        disk="10g",
        device=TpuConfig(type="v5litepod-4", count=4),
        replicas=1,
        regions=["eu-west4"],
    )

step = evaluate_helm(
    model_name="HuggingFaceTB/SmolLM2-135M",
    model_path="HuggingFaceTB/SmolLM2-135M",
    evals=[EvalTaskConfig(name="mmlu", num_fewshot=0)],
    resource_config=resource_config,
    max_eval_instances=10,
)

if __name__ == "__main__":
    executor_main(steps=[step])
