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
from urllib.parse import urlparse

import ray
from fray.cluster import ResourceConfig
from fray.cluster.ray.deps import build_runtime_env_for_packages
from fray.cluster.ray.resources import get_scheduling_strategy

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.utils import is_remote_path
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


class LevanterTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with Levanter (primarily Lm Eval Harness) on TPUs."""

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/opt/gcsfuse_mount/models"

    @staticmethod
    def _looks_like_url(path: str) -> bool:
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)

    @staticmethod
    def download_model_if_necessary(model: ModelConfig) -> str:
        """Return a path or identifier Levanter can read without copying checkpoints needlessly."""

        if model.path:
            if (
                is_remote_path(model.path)
                or os.path.isdir(model.path)
                or LevanterTpuEvaluator._looks_like_url(model.path)
            ):
                return model.path

        downloaded_path: str | None = model.ensure_downloaded(
            local_path=os.path.join(LevanterTpuEvaluator.CACHE_PATH, model.name)
        )

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
        scheduling_strategy = get_scheduling_strategy(resource_config)

        @ray.remote(
            runtime_env=self.get_runtime_env(),
            max_calls=1,
            scheduling_strategy=scheduling_strategy,
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
