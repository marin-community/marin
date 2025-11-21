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
import time

import torch
from fray.queues.base import Queue
from fray.types import InferenceRequest, InferenceResult
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin.evaluation.evaluation_config import ModelConfig

logger = logging.getLogger(__name__)


class TransformersWorker:
    model_config: ModelConfig
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the model and tokenizer using HuggingFace Transformers.
        """
        logger.info(f"Initializing TransformersWorker for model {self.model_config.name}")

        # Use path if provided, otherwise use name (for HF models)
        model_path = self.model_config.path or self.model_config.name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32, **self.model_config.engine_kwargs
        )
        self.model.eval()
        logger.info("Transformers initialization complete")

    def run(self, task_queue: Queue, result_queue: Queue):
        """Run the worker loop processing tasks from the queue."""
        logger.info("TransformersWorker starting run loop")

        while True:
            # Pop task
            lease = task_queue.pop(lease_timeout=30.0)
            if lease is None:
                time.sleep(0.1)
                continue

            # Check for shutdown
            if lease.item == "__FRAY_WORKER_SHUTDOWN__":
                task_queue.done(lease)
                logger.info("TransformersWorker received shutdown signal")
                break

            request = lease.item
            logger.info(f"Processing request: {request.request_id}")

            try:
                # Process request
                result = self._process_request(request)

                # Push result
                result_queue.push(result)
                task_queue.done(lease)

            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {e}")
                # Release lease
                task_queue.release(lease)

                # Push error result
                error_result = InferenceResult(text=[], request_id=request.request_id, error=str(e))
                result_queue.push(error_result)

    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process a single inference request."""
        try:
            params = self.model_config.generation_params or {}
            max_tokens = params.get("max_tokens", 50)
            inputs = self.tokenizer(request.prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.pad_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(request.prompt) :]

            return InferenceResult(text=[generated_text], request_id=request.request_id)

        except Exception as e:
            logger.error(f"Error during Transformers inference: {e}")
            raise
