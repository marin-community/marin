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
from dataclasses import dataclass

from marin.evaluation.evaluation_config import EvalTaskConfig, ModelConfig


@dataclass(frozen=True)
class Dependency:
    """Represents a Python dependency e.g., transformers==4.9.2"""

    name: str
    """The name of the dependency e.g., transformers"""

    version: str | None = None
    """The version of the dependency e.g., 4.9.2"""

    def __str__(self):
        return f"{self.name}=={self.version}" if self.version else self.name


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        openai_base_url: str,
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """What to run to evaluate.

        Args:
            model: Model configuration
            evals: List of evaluation tasks to run
            openai_base_url: Base URL for the OpenAI-compatible inference pool API
            output_path: Path to write evaluation results
            max_eval_instances: Maximum number of evaluation instances
            wandb_tags: Tags to add to wandb run
        """
        pass
