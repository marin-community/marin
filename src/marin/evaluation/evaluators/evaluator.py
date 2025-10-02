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

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.utils import download_from_gcs, is_remote_path
from marin.generation.ray_utils import scheduling_strategy_fn


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

    engine_kwargs: dict | None = None
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

    def ensure_downloaded(self, local_path: str | None = None) -> str | None:
        """
        Ensures that the model checkpoint is downloaded to `local_path` if necessary.
        """
        if self.path is None:
            return None
        elif is_remote_path(self.path):
            assert local_path is not None
            download_from_gcs(gcs_path=self.path, destination_path=local_path)
            self.path = local_path
            # Show the contents of self.path
            print(f"Downloaded model checkpoint to {self.path}: {os.listdir(self.path)}")
            return local_path

    def destroy(self) -> None:
        """Deletes the model checkpoint."""
        if self.path and os.path.exists(self.path) and "gcsfuse" not in self.path:
            shutil.rmtree(self.path, ignore_errors=True)
            print(f"Deleted local checkpoint at {self.path}.")


class Evaluator(ABC):
    def _get_scheduling_strategy(self, resource_config: ResourceConfig | None):
        if resource_config is None:
            fn = None
        else:
            fn = scheduling_strategy_fn(
                resource_config.num_tpu,
                resource_config.strategy,
                resource_config.tpu_type,
                resource_config.include_head_in_scheduling_strategy,
            )

        return fn

    @abstractmethod
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
    ) -> None:
        """What to run to evaluate."""
        pass
