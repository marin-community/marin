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
Run a small lm-eval-harness CPU smoke test against the tiny TinyStories checkpoint.
"""

from experiments.evals.task_configs import LONG_CONTEXT_TASKS
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

MODEL_PATH = "local_store/checkpoints/marin-nano-tinystories-57591f/hf/step-99"

tiny_long_context_eval = ExecutorStep(
    name="tutorials/eval_tiny_model_cpu/long_context",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="lm_evaluation_harness_hf",
        model_name="marin-nano-tinystories-cpu",
        model_path=MODEL_PATH,
        evaluation_path=this_output_path(),
        evals=list(LONG_CONTEXT_TASKS),
        max_eval_instances=2,
        launch_with_ray=False,
        discover_latest_checkpoint=False,
    ),
    pip_dependency_groups=["cpu", "eval"],
)

if __name__ == "__main__":
    executor_main(steps=[tiny_long_context_eval])


