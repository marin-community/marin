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

from marin.evaluation.pool import InferencePool
from marin.evaluation.tasks.base import EvaluationTask
from marin.evaluation.types import EvaluationConfig, InferenceResult

logger = logging.getLogger(__name__)


class MockTask(EvaluationTask):
    def __init__(self, config: EvaluationConfig, prompts: list[str]):
        super().__init__(config)
        self.prompts = prompts

    def run(self) -> list[InferenceResult]:
        logger.info(f"Running mock task with {len(self.prompts)} prompts")
        with InferencePool(self.config) as pool:
            results = pool.map(self.prompts)

        logger.info(f"Mock task completed with {len(results)} results")
        return results
