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
import shutil
import time

import haliax as hax
from fray.queues.base import Queue
from fray.types import InferenceRequest, InferenceResult
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request, SeqDecodingParams
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig, AutoTokenizer

from marin.evaluation.evaluation_config import ModelConfig

logger = logging.getLogger(__name__)


class LevanterWorker:
    """
    Worker that runs inference with Levanter on TPUs (or CPUs for testing).
    """

    CACHE_PATH: str = "/tmp/levanter_cache"

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.engine: InferenceEngine | None = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the model and inference engine.
        """
        logger.info(f"Initializing LevanterWorker for model {self.model_config.name}")

        # 1. Load Tokenizer
        # We use the model name or path to load the tokenizer
        tokenizer_path = self.model_config.path or self.model_config.name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load HF config and convert to Levanter config
        hf_config = AutoConfig.from_pretrained(tokenizer_path)
        llama_config = LlamaConfig.from_hf_config(hf_config)

        # 3. Create trainer config (using CPU for testing)
        trainer_config = TrainerConfig(
            model_axis_size=1,
            tensor_parallel_axes=[],
            fsdp_axis=None,
            batch_axis="batch",
        )

        with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
            logger.info(f"Loading model from HF checkpoint {tokenizer_path}")
            converter = HFCheckpointConverter(
                type(llama_config),
                reference_checkpoint=tokenizer_path,
                tokenizer=self.tokenizer,
            )

            model = converter.load_pretrained(
                llama_config.model_type,
                ref=tokenizer_path,
                dtype=trainer_config.mp.compute_dtype,
                axis_mapping=trainer_config.parameter_axis_mapping,
            )

            # 5. Create inference engine config and engine
            engine_config = InferenceEngineConfig(
                max_seq_len=llama_config.seq_len or 128,
                max_seqs=2,
                page_size=8,
                max_rounds=4,
            )

            # Merge any engine_kwargs from model_config
            if self.model_config.engine_kwargs:
                import dataclasses

                engine_config = dataclasses.replace(engine_config, **self.model_config.engine_kwargs)

            self.engine = InferenceEngine.from_model_with_config(
                model=model,
                tokenizer=self.tokenizer,
                config=engine_config,
            )

    def run(self, task_queue: Queue, result_queue: Queue):
        """Run the worker loop processing tasks from the queue."""
        logger.info("LevanterWorker starting run loop")

        while True:
            # Pop task
            lease = task_queue.pop(lease_timeout=30.0)
            if lease is None:
                time.sleep(0.1)
                continue

            # Check for shutdown
            if lease.item == "__FRAY_WORKER_SHUTDOWN__":  # Hardcoded sentinel for now to match pool
                task_queue.done(lease)
                logger.info("LevanterWorker received shutdown signal")
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

                # Optionally push error result
                error_result = InferenceResult(text=[], request_id=request.request_id, error=str(e))
                result_queue.push(error_result)

    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process a single inference request."""
        try:
            # Tokenize
            prompt_ids = self.tokenizer.encode(request.prompt)

            # Create Levanter Request
            levanter_request = Request(
                prompt_tokens=prompt_ids,
                request_id=(
                    int(request.request_id)
                    if request.request_id and request.request_id.isdigit()
                    else hash(request.prompt)
                ),
                decode_params=SeqDecodingParams.default(),  # Use defaults for now
                n_generations=1,
            )

            # Generate - engine.generate expects a sequence of requests
            result = self.engine.generate([levanter_request])

            # Decode the generated tokens
            # result.tokens is a list of token lists (one per generation)
            if result.tokens and len(result.tokens) > 0:
                generated_tokens = result.tokens[0]  # Get first generation
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""

            return InferenceResult(text=[generated_text], request_id=request.request_id)

        except Exception as e:
            logger.error(f"Error during Levanter inference: {e}")
            raise

    @staticmethod
    def cleanup(model_config: ModelConfig):
        """Clean up resources associated with the model."""
        # TODO: Implement cleanup logic (e.g. close engine, free memory)
        pass
        if os.path.exists(LevanterWorker.CACHE_PATH) and "gcsfuse" not in LevanterWorker.CACHE_PATH:
            shutil.rmtree(LevanterWorker.CACHE_PATH)
