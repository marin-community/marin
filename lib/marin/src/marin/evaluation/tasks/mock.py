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

"""Mock evaluation task for testing."""

import logging

from marin.evaluation.controller import Controller
from marin.evaluation.tasks.base import EvaluationTask

logger = logging.getLogger(__name__)


class MockTask(EvaluationTask):
    """Simple mock task that generates one word completions.

    This is primarily for integration testing the controller and worker infrastructure.
    """

    def __init__(self, prompts: list[str], *args, **kwargs):
        """Initialize mock task with prompts.

        Args:
            prompts: List of prompts to complete
        """
        super().__init__(*args, **kwargs)
        self.prompts = prompts

    def run(self) -> dict:
        """Run mock evaluation.

        Returns:
            Dictionary mapping request IDs to results.
        """
        logger.info(f"Running mock task with {len(self.prompts)} prompts")

        controller = Controller(config=self.config, model_config=self.model_config)
        results = controller.run()

        logger.info(f"Mock task completed with {len(results)} results")
        return results
