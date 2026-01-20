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
from abc import ABC

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


class LevanterTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with Levanter (primarily Lm Eval Harness) on TPUs."""

    @staticmethod
    def model_name_or_path(model: ModelConfig) -> str:
        """Return a reference Levanter can read without staging to local disk."""
        if model.path is None:
            return model.name
        return model.path

    @staticmethod
    def cleanup(model: ModelConfig) -> None:
        """
        Clean up resources.
        """
        logger.info("Cleaning up resources.")

        # Delete the checkpoint
        model.cleanup()

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Launches the evaluation run with Fray.
        """

        def _run():
            with remove_tpu_lockfile_on_exit():
                self.evaluate(model, evals, output_path, max_eval_instances, wandb_tags)

        job_request = JobRequest(
            name="levanter-tpu-eval",
            entrypoint=Entrypoint.from_callable(_run),
            resources=resource_config,
            environment=EnvironmentConfig.create(
                extras=["eval", "tpu"],
            ),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
