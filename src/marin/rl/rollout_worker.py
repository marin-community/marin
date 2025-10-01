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

"""
Inference worker for RL/post-training rollout generation.

This worker loads model checkpoints, generates rollouts from a single environment,
and writes the rollout data to files for training workers to consume.
"""

import asyncio
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import haliax as hax
import jax
import jax.random as jrandom
import levanter
import numpy as np
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import barrier_sync
from openai import AsyncOpenAI
from transformers import AutoTokenizer, PreTrainedTokenizer

from marin.rl.environments import EnvConfig, load_environment_from_spec
from marin.rl.model_utils import load_model_from_checkpoint

from .rollout_storage import RolloutStorageConfig, RolloutWriter
from .types import (
    InferenceChoice,
    InferenceContext,
    InferenceResponse,
    RolloutBatch,
    RolloutMetadata,
)
from .weight_transfer import WeightTransferClient, WeightTransferConfig, create_weight_transfer_client

logger = logging.getLogger(__name__)


@dataclass
class RolloutWorkerConfig:
    """Configuration for RolloutWorker."""

    inference_server_config: InferenceServerConfig

    trainer: TrainerConfig
    model: LmConfig
    environment_spec: EnvConfig
    rollout_storage: RolloutStorageConfig
    max_input_length: int
    max_output_length: int
    # TODO(power) Lift these out into SamplingConfig
    pad_token_id: int
    n_prompts_per_step: int
    n_generations: int
    temperature: float
    log_freq: int
    weight_transfer: WeightTransferConfig

    run_id: str
    """Run ID to pass into the tracker. (unclear why this can't be passed directly)"""

    max_rollouts: int | None = None
    """Maximum number of rollouts to generate before stopping. Defaults to running forever."""

    stop_tokens: list[int] | None = None
    """List of stop tokens to supply when performing auto-regression."""

    # Initial checkpoint for the reference model (auto-detects HF repo vs local path)
    initial_checkpoint: str | None = None


def find_open_port() -> int:
    """Find an open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class LevanterInferenceContext(InferenceContext):
    """Context that uses Levanter model and inference server."""

    inference_server: InferenceServer
    max_tokens: int
    _tokenizer: Any
    _stop_tokens: list[int] | None = None

    def __init__(
        self,
        tokenizer,
        stop_tokens: list[int] | None,
        inference_server: InferenceServer,
        max_tokens: int,
    ):
        self.inference_server = inference_server
        self.max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._stop_tokens = stop_tokens

    @property
    def tokenizer(self):
        return self._tokenizer

    def openai_client(self):
        base_url = f"http://{self.inference_server.config.host}:{self.inference_server.config.port}/v1"
        return AsyncOpenAI(base_url=base_url, api_key="marin")

    def generate(
        self,
        prompts: list[str],
        temperature: float,
        n_generations: int,
    ) -> list[InferenceResponse]:
        """Generate responses for a batch of prompts."""
        stop_strings = None
        if self._stop_tokens is not None:
            stop_strings = [self._tokenizer.decode([token]) for token in self._stop_tokens]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.openai_client()

        def _process_batch(batch_prompts: list[str]) -> list[InferenceResponse]:
            batch_completions = []

            for prompt in batch_prompts:
                completion = client.chat.completions.create(
                    model=getattr(self.inference_server.config, "model_name", "test-model"),
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=True,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    n=n_generations,
                    stop=stop_strings,
                    timeout=30,
                )
                batch_completions.append(completion)

            completions = loop.run_until_complete(asyncio.gather(*batch_completions, return_exceptions=True))

            batch_results = []
            for prompt, completion in zip(batch_prompts, completions, strict=True):
                choices = []
                # drop responses that failed.
                if isinstance(completion, Exception):
                    logger.error(f"Error during generation: {completion}")
                else:
                    for choice in completion.choices:
                        content = choice.message.content
                        tokens = self.tokenizer.encode(content)
                        logprobs = [t.logprob for t in choice.logprobs.content]
                        choices.append(
                            InferenceChoice(
                                response_text=content,
                                response_tokens=np.array(tokens, dtype=np.int32),
                                logprobs=np.array(logprobs, dtype=np.float32),
                            )
                        )

                # Create InferenceResponse with prompt tokens
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                batch_results.append(
                    InferenceResponse(
                        prompt=prompt,
                        prompt_tokens=np.array(prompt_tokens, dtype=np.int32),
                        choices=choices,
                    )
                )
            return batch_results

        # Process prompts in batches to limit concurrent requests
        # Each prompt with n_generations counts as n_generations requests
        max_concurrent_requests = 8
        batch_size = max(1, max_concurrent_requests // n_generations)
        all_results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_results = _process_batch(batch_prompts)
            all_results.extend(batch_results)

        loop.run_until_complete(client.close())

        loop.close()
        return all_results


class RolloutWorker:
    """Asynchronous inference & rollout worker for RL training.

    Inference workers periodically load model checkpoints generated by the training job,
    and continously generate rollouts from a single environment. Rollouts are communicated to the
    training job via a rollout queue.
    """

    _inference_thread: threading.Thread
    _inference_server: InferenceServer
    _policy_model: Any
    _transfer_client: WeightTransferClient
    _rollout_writer: RolloutWriter
    _tokenizer: PreTrainedTokenizer

    def __init__(self, config: RolloutWorkerConfig):
        config.trainer.id = f"{config.run_id}-rollout"
        levanter.initialize(config.trainer)
        self.tracker = levanter.current_tracker()
        self.config = config
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()
        self._current_weight_step: int = 0

        # for testing, we accept a tokenizer instance or a string
        if isinstance(self.config.model.tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer)
        else:
            self._tokenizer = cast(PreTrainedTokenizer, self.config.model.tokenizer)

        self._environment = load_environment_from_spec(config.environment_spec)

        logger.info("Starting weight transfer client with config %s", self.config.weight_transfer)
        self._transfer_client = create_weight_transfer_client(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

        self._rollout_writer = config.rollout_storage.create_writer()
        self._build_models()
        self._inference_server = InferenceServer.create(
            config.inference_server_config,
            model=self._policy_model,
            tokenizer=self._tokenizer,
        )
        self._inference_thread = threading.Thread(target=lambda: self._inference_server.serve(), daemon=True)
        self._inference_thread.start()

    def _build_models(self):
        """Build policy model after levanter initialization."""

        if self.config.initial_checkpoint is not None:
            logger.info(f"Loading initial policy model from checkpoint: {self.config.initial_checkpoint}")
        else:
            logger.info("Building new policy model from scratch")

        key = jrandom.PRNGKey(42)
        vocab_size = self._tokenizer.vocab_size
        Vocab = hax.Axis("vocab", vocab_size)

        initial_model = load_model_from_checkpoint(
            checkpoint=self.config.initial_checkpoint,
            model_config=self.config.model,
            trainer_config=self.config.trainer,
            mesh=self.config.trainer.device_mesh,
            # use the compute axis mapping for inference
            axis_mapping=self.config.trainer.compute_axis_mapping,
            vocab_axis=Vocab,
            tokenizer=self._tokenizer,
            key=key,
        )

        update = self._transfer_client.receive_weights(initial_model)
        if update:
            logger.info("Loaded initial policy model from weight transfer")
            self._policy_model = update.model
            self._current_weight_step = update.weight_id
        else:
            logger.info("Initializing policy model from initial checkpoint")
            self._policy_model = initial_model
        logger.info("Loaded/built policy model")

    def stop(self):
        """Stop the inference worker loop and server."""
        with self._shutdown_condition:
            self._running = False
            self._transfer_client.cleanup()
            self._shutdown_condition.notify()

        # Wait for the main loop to finish
        self._shutdown_complete.wait()

        # Now shutdown the inference server
        if self._inference_server:
            self._inference_server.shutdown()

    def _generate_rollout_batch(self, rng) -> tuple[RolloutBatch | None, dict | None]:
        barrier_sync()

        # Create policy inference context for sampling from the inference server
        policy_ctx = LevanterInferenceContext(
            tokenizer=self._tokenizer,
            inference_server=self._inference_server,
            max_tokens=self.config.max_input_length + self.config.max_output_length,
            stop_tokens=self.config.stop_tokens,
        )

        with (
            self.config.trainer.device_mesh,
            hax.axis_mapping(self.config.trainer.compute_axis_mapping),
        ):
            # Sample examples, generate responses, and create rollouts
            rollout_groups, metrics = self._environment.sample(
                inference_ctx=policy_ctx,
                n_examples=self.config.n_prompts_per_step,
                n_generations=self.config.n_generations,
                temperature=self.config.temperature,
                prng_key=rng,
                mode="train",
            )

        if len(rollout_groups) == 0:
            logger.warning("No valid rollouts generated in this batch, retrying...")
            return None, None

        rollout_batch = RolloutBatch(
            groups=rollout_groups,
            metadata=RolloutMetadata(
                worker_id=f"{socket.gethostname()}_{os.getpid()}",
                timestamp=time.time(),
                weight_step=self._current_weight_step,
            ),
        )

        barrier_sync()
        return rollout_batch, metrics

    def _sync_weights(self):
        logger.info("Checking for new weights...")
        update = self._transfer_client.receive_weights(self._policy_model)

        if update:
            self._current_weight_step = update.weight_id
            logger.info(f"Received new weights from step {update.weight_id}")
            self._policy_model = update.model
            self._inference_server.reload(lambda model: self._policy_model)
            return update.model
        else:
            logger.info("No new weights available")
            return None

    def run(self):
        """Main inference worker loop."""
        logger.info("Starting inference worker...")

        step = 0
        seed = 0
        logger.info(f"Starting rollout worker with seed {seed}")
        rng = jax.random.PRNGKey(seed)

        last_weight_check = time.time()

        while self._running:
            barrier_sync()

            if self.config.max_rollouts is not None and step >= self.config.max_rollouts:
                logger.info(f"Reached max rollouts ({self.config.max_rollouts}), stopping")
                break

            if time.time() - last_weight_check > self.config.weight_transfer.poll_interval_seconds:
                self._sync_weights()
                last_weight_check = time.time()

            rng, input_rng = jax.random.split(rng)
            logger.info("Generating rollout batch...")
            rollout_batch, metrics = self._generate_rollout_batch(input_rng)
            if rollout_batch is None:
                continue
            step += 1
            self._rollout_writer.write_batch(rollout_batch)

            if jax.process_index() == 0:
                if self.config.log_freq > 0 and step % self.config.log_freq == 0:
                    log_metrics = {"step": step}
                    log_metrics.update(jax.device_get(metrics))
                    log_metrics.update(self._transfer_client.get_metrics())
                    log_metrics = {"inference." + k: v for k, v in log_metrics.items()}
                    logger.info(f"Logging metrics at step {step}... {log_metrics}")
                    self.tracker.log(log_metrics, step=step)

        logger.info(f"Inference worker completed after generating {step} rollouts")
        barrier_sync()
        self._shutdown_complete.set()
