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

from marin.execution.executor import executor_main
from marin.post_training.environments.prime_intellect_env import PrimeIntellectEnv
from marin.rl.evaluate_environment import evaluate_environment

logger = logging.getLogger(__name__)


def main():
    """Main function to run environment evaluation experiments."""

    # Example model checkpoint path - update this to a real checkpoint
    model_checkpoint = "Qwen/Qwen3-32B"

    # Configure PrimeIntellect environment for testing
    env = PrimeIntellectEnv(
        env_id="willb/gsm8k",
        env_args={
            "num_train_examples": -1,
            "num_eval_examples": 100,
        },
    )

    # Create evaluation step
    eval_step = evaluate_environment(
        model=model_checkpoint,
        env=env,
    )

    # Run the evaluation
    executor_main(
        steps=[eval_step],
        description="Test environment evaluation on math tasks",
    )


if __name__ == "__main__":
    main()
