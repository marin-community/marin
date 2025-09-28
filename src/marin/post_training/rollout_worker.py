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
import jax.numpy as jnp
import jax.random as jrandom
import levanter
import numpy as np
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import barrier_sync
from openai import AsyncOpenAI
from optax import softmax_cross_entropy_with_integer_labels
from transformers import AutoTokenizer, PreTrainedTokenizer

from marin.post_training.environments.load_environments import load_environment_from_spec
from marin.post_training.environments.marin_env import InferenceContext

from . import weight_transfer
from .model_utils import load_model_from_checkpoint
from .rl_dataset import create_dataset_from_environment
from .rollout_storage import RolloutBatch, RolloutStorageConfig, RolloutWriter, TaggedRolloutBatch
from .weight_transfer import WeightTransferConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutWorkerConfig:
    """Configuration for RolloutWorker."""

    inference_server_config: InferenceServerConfig

    trainer: TrainerConfig
    model: LmConfig
    environment_spec: str
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


@jax.jit
def compute_model_logprobs(
    model,
    input_tokens: np.ndarray,
    input_attention_mask: np.ndarray,
    target_tokens: np.ndarray,
    target_attention_mask: np.ndarray,
) -> np.ndarray:
    """Compute log probabilities for target tokens.

    N.B. This does not compute full log-probs, just the log-probs of the target
    tokens themselves.

    Args:
        model: Haliax model to use for computation
        input_tokens: (batch_size, input_length) input token IDs
        input_attention_mask: (batch_size, input_length) attention mask for input
        target_tokens: (batch_size, target_length) target token IDs
        target_attention_mask: (batch_size, target_length) attention mask for target

    Returns:
        np.ndarray: (batch_size, target_length) log probabilities for target tokens
    """
    # Concatenate input and target tokens
    full_tokens = jnp.concatenate([input_tokens, target_tokens], axis=1)
    full_attention_mask = jnp.concatenate([input_attention_mask, target_attention_mask], axis=1)
    full_position_ids = jnp.maximum(jnp.cumsum(full_attention_mask, axis=1) - 1, 0)

    # Convert to Haliax named arrays
    Batch = hax.Axis("batch", full_tokens.shape[0])
    SeqLen = hax.Axis("position", full_tokens.shape[1])

    tokens_named = hax.named(full_tokens, (Batch, SeqLen))
    attn_mask_named = hax.named(full_attention_mask.astype(jnp.bool_), (Batch, SeqLen))
    position_ids_named = hax.named(full_position_ids, (Batch, SeqLen))

    # Compute logits using the model
    input_tokens_named = tokens_named[SeqLen, hax.ds(0, SeqLen.size - 1)]
    input_mask_named = attn_mask_named[SeqLen, hax.ds(0, SeqLen.size - 1)]
    input_position_ids_named = position_ids_named[SeqLen, hax.ds(0, SeqLen.size - 1)]

    # Run forward pass through the model
    logits = model(
        input_tokens_named, attn_mask=input_mask_named, pos_ids=input_position_ids_named
    )  # Shape: (batch, seq_len-1, vocab_size)

    # Extract logits corresponding to target positions
    logits_array = logits.array[:, input_tokens.shape[1] - 1 :]

    logprobs = -softmax_cross_entropy_with_integer_labels(
        logits_array.astype(jnp.float32), target_tokens.astype(jnp.int32)
    )
    return logprobs


def find_open_port() -> int:
    """Find an open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class LevanterInferenceContext(InferenceContext):
    """Context that uses Levanter model and inference server."""

    model: Any
    inference_server: InferenceServer
    max_tokens: int
    _tokenizer: Any
    _stop_tokens: list[int] | None = None

    def __init__(
        self,
        model,
        tokenizer,
        stop_tokens: list[int] | None,
        inference_server: InferenceServer,
        max_tokens: int,
    ):
        self.model = model
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
    ) -> list[list[dict]]:
        """Generate responses for a batch of prompts."""
        self.inference_server.reload(lambda model: self.model)

        # Convert stop tokens to strings for OpenAI API
        stop_strings = None
        if self._stop_tokens is not None:
            stop_strings = [self._tokenizer.decode([token]) for token in self._stop_tokens]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.openai_client()

        def _process_batch(batch_prompts: list[str]) -> list[list[dict]]:
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
            for completion in completions:
                if isinstance(completion, Exception):
                    logger.error(f"Error during generation: {completion}")
                    prompt_results = [{"tokens": [], "logprobs": []} for _ in range(n_generations)]
                    batch_results.append(prompt_results)
                    continue

                prompt_results = []
                for choice in completion.choices:
                    content = choice.message.content
                    tokens = self.tokenizer.encode(content)
                    logprobs = [t.logprob for t in choice.logprobs.content]
                    prompt_results.append({"tokens": tokens, "logprobs": logprobs})
                batch_results.append(prompt_results)
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

    def compute_logprobs(
        self,
        input_tokens: np.ndarray,
        input_attention_mask: np.ndarray,
        target_tokens: np.ndarray,
        target_attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute log probabilities for given input/target pairs."""
        self.inference_server.unload()

        # break into batches of size 8 to avoid OOM
        input_batches = np.array_split(input_tokens, 8)
        input_attention_batches = np.array_split(input_attention_mask, 8)
        target_batches = np.array_split(target_tokens, 8)
        target_attention_batches = np.array_split(target_attention_mask, 8)
        logprobs_list = []
        for ib, iam, tb, tam in zip(
            input_batches,
            input_attention_batches,
            target_batches,
            target_attention_batches,
            strict=True,
        ):
            logprobs_batch = compute_model_logprobs(
                self.model,
                ib,
                iam,
                tb,
                tam,
            )
            logprobs_list.append(logprobs_batch)
        return np.concatenate(logprobs_list, axis=0)


class RolloutWorker:
    """Asynchronous inference & rollout worker for RL training.

    Inference workers periodically load model checkpoints generated by the training job,
    and continously generate rollouts from a single environment. Rollouts are communicated to the
    training job via a rollout queue.
    """

    _server_thread: threading.Thread
    inference_server: InferenceServer
    policy_model: Any
    reference_model: Any
    transfer_client: weight_transfer.WeightTransferClient
    rollout_writer: RolloutWriter
    _tokenizer: PreTrainedTokenizer

    def __init__(self, config: RolloutWorkerConfig):
        config.trainer.id = f"{config.run_id}-rollout"
        levanter.initialize(config.trainer)
        self.tracker = levanter.current_tracker()
        self.config = config
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()

        if isinstance(self.config.model.tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer)
        else:
            self._tokenizer = cast(PreTrainedTokenizer, self.config.model.tokenizer)

        self._environment = load_environment_from_spec(config.environment_spec, tokenizer=self._tokenizer)

        config.inference_server_config.port = find_open_port()
        self.inference_server = InferenceServer.create(config.inference_server_config)

        self._start_inference_server()

        self.transfer_client = weight_transfer.create_weight_transfer_client(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

        self.rollout_writer = config.rollout_storage.create_writer()
        self._build_models()

    def _build_models(self):
        """Build policy and reference models after levanter initialization."""

        if self.config.initial_checkpoint is not None:
            logger.info(f"Loading initial reference model from checkpoint: {self.config.initial_checkpoint}")
        else:
            logger.info("Building new reference model from scratch")

        key = jrandom.PRNGKey(42)
        vocab_size = self._tokenizer.vocab_size
        Vocab = hax.Axis("vocab", vocab_size)

        self.reference_model = load_model_from_checkpoint(
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

        self.policy_model = self.transfer_client.receive_weights(self.reference_model)
        if self.policy_model:
            logger.info("Loaded initial policy model from weight transfer")
        else:
            logger.info("Initializing policy model from reference model")
            self.policy_model = self.reference_model
        logger.info("Loaded/built policy and reference models")

    def _start_inference_server(self):
        """Start the inference server in a background thread."""

        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.inference_server.serve_async())
            except Exception as e:
                logger.error(f"Inference server error: {e}")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

    def stop(self):
        """Stop the inference worker loop and server."""
        with self._shutdown_condition:
            self._running = False
            self._shutdown_condition.notify()

        # Wait for the main loop to finish
        self._shutdown_complete.wait()

        # Now shutdown the inference server
        if self.inference_server:
            self.inference_server.shutdown()

    def _generate_rollout_batch(self, rng) -> tuple[list[dict], dict]:
        """Generate a set of rollout batches from the environment."""
        barrier_sync()

        # Create Levanter inference contexts
        policy_ctx = LevanterInferenceContext(
            self.policy_model,
            tokenizer=self._tokenizer,
            inference_server=self.inference_server,
            max_tokens=self.config.max_input_length + self.config.max_output_length,
            stop_tokens=self.config.stop_tokens,
        )
        reference_ctx = LevanterInferenceContext(
            self.reference_model,
            tokenizer=self._tokenizer,
            inference_server=self.inference_server,
            max_tokens=self.config.max_input_length + self.config.max_output_length,
            stop_tokens=self.config.stop_tokens,
        )

        with (
            self.config.trainer.device_mesh,
            hax.axis_mapping(self.config.trainer.compute_axis_mapping),
        ):
            rl_dataset, dataset_metrics = create_dataset_from_environment(
                environment=self._environment,
                policy_ctx=policy_ctx,
                reference_ctx=reference_ctx,
                n_examples=self.config.n_prompts_per_step,
                prng_key=rng,
                n_generations=self.config.n_generations,
                max_input_length=self.config.max_input_length,
                max_output_length=self.config.max_output_length,
                pad_token_id=self.config.pad_token_id,
                mode="train",
                temperature=self.config.temperature,
            )
        barrier_sync()

        return (
            list(
                rl_dataset.iterate_batches(
                    batch_size=self.config.n_generations * self.config.n_prompts_per_step,
                    shuffle=True,
                    loop=False,
                )
            ),
            dataset_metrics,
        )

    def _sync_weights(self):
        logger.info("Checking for new weights...")
        weights = self.transfer_client.receive_weights(self.policy_model)

        if weights:
            logger.info("Received new weights for policy model")
            self.policy_model = weights
            return weights
        else:
            logger.info("No new weights available for policy model")
            return None

    def run(self):
        """Main inference worker loop."""
        logger.info("Starting inference worker...")

        rollouts_generated = 0
        step = 0
        rng = jax.random.PRNGKey(0)

        last_weight_check = time.time()

        while self._running:
            barrier_sync()

            if self.config.max_rollouts is not None and rollouts_generated >= self.config.max_rollouts:
                logger.info(f"Reached max rollouts ({self.config.max_rollouts}), stopping")
                break

            if time.time() - last_weight_check > self.config.weight_transfer.poll_interval_seconds:
                self._sync_weights()
                last_weight_check = time.time()

            rng, input_rng = jax.random.split(rng)
            logger.info("Generating rollout batch...")
            rollout_batches, metrics = self._generate_rollout_batch(input_rng)
            step += 1
            for batch_data in rollout_batches:
                rollout_batch = TaggedRolloutBatch(
                    batch=RolloutBatch(
                        input_ids=batch_data["input_ids"],
                        attention_mask=batch_data["attention_mask"],
                        position_ids=batch_data["position_ids"],
                        target_ids=batch_data["target_ids"],
                        loss_weights=batch_data["loss_weights"],
                        loss_masks=batch_data["loss_masks"],
                        policy_logprobs=batch_data["policy_logprobs"],
                    ),
                    env_name=self.config.environment_spec,
                    worker_id=f"{socket.gethostname()}_{os.getpid()}",
                    timestamp=time.time(),
                    rollout_id=f"{socket.gethostname()}_{int(time.time() * 1000000)}_{step}",
                )
                self.rollout_writer.write_batch(rollout_batch)

            if jax.process_index() == 0:
                if self.config.log_freq > 0 and step % self.config.log_freq == 0:
                    log_metrics = {"step": step}
                    log_metrics.update(jax.device_get(metrics))
                    log_metrics.update(self.transfer_client.get_metrics())
                    log_metrics = {"inference." + k: v for k, v in log_metrics.items()}
                    logger.info(f"Logging metrics at step {step}... {log_metrics}")
                    self.tracker.log(log_metrics, step=step)

            rollouts_generated += 1
        logger.info(f"Inference worker completed after generating {rollouts_generated} rollouts")
        barrier_sync()
        self._shutdown_complete.set()
