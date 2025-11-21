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

"""Inference server worker implementation."""

from __future__ import annotations

import logging

from fray.types import InferenceRequest, InferenceResult
from marin.evaluation.evaluation_config import ModelConfig

logger = logging.getLogger(__name__)


class InferenceServer:
    """Worker class that loads a model and serves inference requests.

    This class is intended to be used with WorkerPool. It initializes the
    inference engine (e.g., vLLM) on startup and processes requests.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize the inference server.

        Args:
            model_config: Configuration for the model to load
        """
        self.model_config = model_config
        self.llm = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load the model using vLLM."""
        try:
            # Import vLLM here to avoid dependency on coordinator
            from vllm import LLM

            logger.info(f"Initializing vLLM with model: {self.model_config.name}")

            # Use path if provided, otherwise use name (for HF models)
            # vLLM can load directly from GCS paths
            model_path = self.model_config.path or self.model_config.name

            # Initialize LLM
            # Note: We assume this runs in a process with appropriate resources (TPU/GPU)
            self.llm = LLM(model=model_path, trust_remote_code=True, **self.model_config.engine_kwargs)
            logger.info("vLLM initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    def __call__(self, request: InferenceRequest | dict) -> InferenceResult:
        """Process an inference request.

        Args:
            request: InferenceRequest object or dictionary

        Returns:
            InferenceResult object
        """
        try:
            from vllm import SamplingParams

            if isinstance(request, dict):
                request = InferenceRequest(**request)

            logger.debug(f"Processing request: {request.request_id}")

            # Merge sampling params
            params = self.model_config.generation_params or {}
            if request.sampling_params:
                params.update(request.sampling_params)

            sampling_params = SamplingParams(**params)

            # Run inference
            outputs = self.llm.generate([request.prompt], sampling_params)

            generated_text = []
            for output in outputs:
                for prompt_output in output.outputs:
                    generated_text.append(prompt_output.text)

            return InferenceResult(text=generated_text, request_id=request.request_id)

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return InferenceResult(
                text=[], request_id=request.request_id if isinstance(request, InferenceRequest) else None, error=str(e)
            )
