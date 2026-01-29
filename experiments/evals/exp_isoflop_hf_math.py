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
Evaluate all ISOFlop and HF models on MATH-500 using MathEnv.

This script evaluates models by generating full responses and grading them,
rather than using log-probability scoring like lm-eval harness.
"""

import logging
import warnings

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)  # Root logger

import os

from experiments.isoflop_sweep import MARIN_SCALING_SUITES
from marin.execution.executor import executor_main, output_path_of, this_output_path
from marin.execution.executor import Executor, ExecutorStep
from marin.evaluation.utils import discover_hf_checkpoints
from fray.cluster import ResourceConfig
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.evals.exp1600_uncheatable_evals import (
    models,
)
from marin.rl.environments.base import EnvConfig
from marin.rl.scripts.evaluate_environment import evaluate_environment

logger = logging.getLogger(__name__)

models = models[:1]


def build_steps(model_types: list[str]):
    steps = []
    isoflop_steps, isoflop_candidates = MARIN_SCALING_SUITES["nemotron"]

    env_config = EnvConfig(
        env_class="marin.rl.environments.math_env.MathEnv",
        env_args={
            "max_eval_examples": 100,
            "seed": 42,
        },
    )

    if "iso" in model_types:
        for isoflop_step, candidate in zip(isoflop_steps, isoflop_candidates):
            checkpoint_path = get_isoflop_hf_model(
                isoflop_step=isoflop_step,
                prefix="gs://marin-us-central1"
            )
            steps.append(
                evaluate_environment(
                    checkpoint=checkpoint_path,
                    checkpoint_is_hf=True,
                    env_config=env_config,
                    output_path=this_output_path(),
                    tpu_type="v5p-8",
                )
            )

    if "hf" in model_types:
        for model_config in models:
            model_instance = download_model_step(
                HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
            )
            steps.append(
                evaluate_environment(
                    checkpoint=output_path_of(model_instance),
                    checkpoint_is_hf=True,
                    env_config=env_config,
                    output_path=this_output_path(),
                    tpu_type="v5p-8",
                )
            )

    return steps


def get_step_output_path(step: ExecutorStep, prefix: str) -> str:
    """
    Get the output path for an ExecutorStep.
    
    Args:
        step: The ExecutorStep to get the output path for
        prefix: The prefix to use. If None, uses MARIN_PREFIX env var.
    
    Returns:
        The output path as a string.
    """
    
    executor_info_base_path = os.path.join(prefix, "experiments")
    executor = Executor(
        prefix=prefix,
        executor_info_base_path=executor_info_base_path,
    )
    executor.compute_version(step, is_pseudo_dep=False)
    return executor.output_paths[step]


def get_isoflop_hf_model(isoflop_step: ExecutorStep, prefix: str):
    path = get_step_output_path(isoflop_step, prefix)
    hf_path = os.path.join(path,"hf")

    checkpoints = discover_hf_checkpoints(base_path=hf_path)
    def get_step(checkpoint):
        return int(checkpoint.rsplit("step-", 1)[-1])

    return sorted(checkpoints, key=get_step)[-1]


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    import warnings
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    model_types = ["hf"]  # ["iso", "hf"]
    steps = build_steps(model_types=model_types)

    # bsz = 10
    # n_batches = math.ceil(len(steps) / bsz)

    # for i in range(n_batches):
    #     batch = steps[i*bsz:(i+1)*bsz]
    #     executor_main(steps=batch)

    executor_main(
        steps=steps,
        description="ISOFlop and HF model MATH-500 evaluations."
    )


if __name__ == "__main__":
    main()
