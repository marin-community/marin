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

import logging
import os
from abc import ABC

import ray

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.run.ray_deps import build_runtime_env_for_packages
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


class LevanterTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with Levanter (primarily Lm Eval Harness) on TPUs."""

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/opt/gcsfuse_mount/models"

    @staticmethod
    def download_model(model: ModelConfig) -> str:
        """
        Download the model if it's not already downloaded
        """
        downloaded_path: str | None = model.ensure_downloaded(
            local_path=os.path.join(LevanterTpuEvaluator.CACHE_PATH, model.name)
        )

        print(f"IN TPU: {downloaded_path}")
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = model.name if downloaded_path is None else downloaded_path

        return model_name_or_path

    @staticmethod
    def cleanup(model: ModelConfig) -> None:
        """
        Clean up resources.
        """
        logger.info("Cleaning up resources.")

        # Delete the checkpoint
        model.destroy()

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval", "tpu"],
            env_vars={
                "HF_DATASETS_TRUST_REMOTE_CODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        resource_config: ResourceConfig | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Launches the evaluation run with Ray.
        """

        @ray.remote(
            resources={
                "TPU": resource_config.num_tpu,
                f"{resource_config.tpu_type}-head": 1,
            },
            runtime_env=self.get_runtime_env(),
            max_calls=1,
        )
        @remove_tpu_lockfile_on_exit
        def launch(
            model: ModelConfig,
            evals: list[EvalTaskConfig],
            output_path: str,
            max_eval_instances: int | None = None,
            wandb_tags: list[str] | None = None,
        ) -> None:
            self.evaluate(model, evals, output_path, max_eval_instances, wandb_tags)

        ray.get(launch.remote(model, evals, output_path, max_eval_instances, wandb_tags))
