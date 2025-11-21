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

"""Base classes for evaluation tasks."""

from abc import ABC, abstractmethod

from marin.evaluation.types import EvaluationConfig


class EvaluationTask(ABC):
    """Base class for evaluation tasks.

    Evaluation tasks use the controller to run inference and handle their
    own logic for preparing requests, processing results, and writing outputs
    to GCS or local storage.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    @abstractmethod
    def run(self) -> None:
        """Run the evaluation task.

        Tasks should:
        1. Create a Controller instance
        2. Prepare and submit inference requests
        3. Process results as needed
        4. Write outputs to GCS or local storage (via config.evaluation_path)
        """
        pass
