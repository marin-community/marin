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

import dataclasses
import logging

from levanter.models.qwen import Qwen3Config
from transformers import AutoConfig

from marin.execution.executor import InputName, executor_main
from marin.rl.environments.prime_intellect_env import PrimeIntellectEnv
from marin.rl.evaluate_environment import evaluate_environment

logger = logging.getLogger("ray")


def create_eval_step():
    """Main function to run environment evaluation experiments."""

    # Example model checkpoint path - update this to a real checkpoint
    model_checkpoint = "Qwen/Qwen3-4B"

    # Create model config directly from HF config
    hf_config = AutoConfig.from_pretrained(model_checkpoint)
    model_config = Qwen3Config.from_hf_config(hf_config)

    # Set head_dim for Qwen3-4B if needed
    if hasattr(hf_config, "head_dim"):
        model_config = dataclasses.replace(model_config, head_dim=hf_config.head_dim)

    # Set seq_len and tokenizer
    model_config = dataclasses.replace(
        model_config,
        seq_len=10240,  # 2048 input + 2048 output
        tokenizer=model_checkpoint,
    )

    # Configure PrimeIntellect environment for testing
    env = PrimeIntellectEnv(
        env_id="will/gsm8k",
        env_args={"num_train_examples": -1, "num_eval_examples": -1},
    )

    # Create evaluation step
    # Use InputName.hardcoded to create a path relative to the prefix
    output_path = InputName.hardcoded("env_evals/gsm8k")

    eval_step = evaluate_environment(
        model_config=model_config,
        env=env,
        name="prime-intellect/gsm8k-test-environment-4B-will-v22",
        output_path=output_path,
        tpu_type="v4-8",  # Use v4-8 for smaller memory footprint
    )

    return eval_step


if __name__ == "__main__":
    eval_step = create_eval_step()
    executor_main(steps=[eval_step])
