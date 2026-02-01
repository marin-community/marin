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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from fray.cluster.ray import get_scheduling_strategy

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.utils import remove_tpu_lockfile_on_exit


@dataclass(frozen=True)
class Dependency:
    """Represents a Python dependency e.g., transformers==4.9.2"""

    name: str
    """The name of the dependency e.g., transformers"""

    version: str | None = None
    """The version of the dependency e.g., 4.9.2"""

    def __str__(self):
        return f"{self.name}=={self.version}" if self.version else self.name


@dataclass
class ModelConfig:
    name: str
    """The name of the model e.g., allenai/olmo-7b"""

    path: str | None
    """
    The path to the model checkpoint. Can be a local path or a path on GCS.
    """

    engine_kwargs: dict[str, Any]
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    generation_params: dict | None = None
    """
    Additional keyword arguments passed to the SamplingParams for the vLLM engine
    """

    apply_chat_template: bool = False
    """
    Whether or not this model was trained with a Chat Template in the tokenizer
    """


class Evaluator(ABC):
    def _get_scheduling_strategy(self, resource_config: ResourceConfig | None):
        if resource_config is None:
            return None
        return get_scheduling_strategy(resource_config)

    @abstractmethod
    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        max_length: int | None = None,
    ) -> None:
        """
        Launches the evaluation run with Ray.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            step (ExecutorStep | None): The step to evaluate. Used to get the config for the model and the trainer.
            wandb_tags (list[str] | None): The tags to add to the wandb run.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        max_length: int | None = None,
    ) -> None:
        """What to run to evaluate."""
        pass


def launch_evaluate_with_ray(
    *,
    evaluator: Evaluator,
    job_name: str,
    model: ModelConfig,
    evals: list[EvalTaskConfig],
    output_path: str,
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
    wandb_tags: list[str] | None = None,
    extras: Sequence[str] = (),
    pip_packages: Sequence[str] = (),
    env_vars: dict[str, str] | None = None,
    configure_logging: bool = True,
) -> None:
    """Launch an evaluator on the Ray/Fray cluster."""

    def launch(
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        if configure_logging:
            import logging

            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
        evaluator.evaluate(model, evals, output_path, max_eval_instances, wandb_tags)

    def _run() -> None:
        with remove_tpu_lockfile_on_exit():
            launch(model, evals, output_path, max_eval_instances, wandb_tags)

    if resource_config is None:
        resource_config = ResourceConfig()

    job_request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(_run),
        resources=resource_config,
        environment=EnvironmentConfig.create(
            extras=list(extras),
            pip_packages=list(pip_packages),
            env_vars=env_vars,
        ),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
