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
Experiment to test environment evaluation using evaluate_environment.py
"""

import logging

from marin.execution.executor import InputName, executor_main
from marin.rl.environments.prime_intellect_env import PrimeIntellectEnv
from marin.rl.evaluate_environment import evaluate_environment

logger = logging.getLogger("ray")


def create_eval_step():
    """Main function to run environment evaluation experiments."""

    # Example model checkpoint path - update this to a real checkpoint
    model_checkpoint = "Qwen/Qwen3-4B"

    # Configure PrimeIntellect environment for testing
    env = PrimeIntellectEnv(
        env_id="willb/gsm8k",
        env_args={
            "num_train_examples": 0,
            "num_eval_examples": 100,
        },
    )

    # Create evaluation step
    # Use InputName.hardcoded to create a path relative to the prefix
    output_path = InputName.hardcoded("env_evals/test_evaluation")

    eval_step = evaluate_environment(
        model=model_checkpoint,
        env=env,
        name="evaluate-test-environment-4B-v18",
        output_path=output_path,
        tpu_type="v5litepod-4",  # Use smaller TPU for testing
    )

    return eval_step


if __name__ == "__main__":
    eval_step = create_eval_step()
    executor_main(steps=[eval_step])
