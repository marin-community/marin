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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import requests
from levanter.inference.openai import InferenceServer, InferenceServerConfig

from marin.post_training.environments.marin_env import InferenceContext

from .flax.utils import (
    jax_distributed_barrier,
)
from .rl_dataset import create_dataset_from_environment
from .rollout_storage import RolloutBatch, RolloutWriter, TaggedRolloutBatch

logger = logging.getLogger(__name__)


@dataclass
class InferenceWorkerConfig:
    """Configuration for InferenceWorker."""

    inference_server_config: InferenceServerConfig
    policy_model: Any  # We'll type this properly later
    reference_model: Any  # We'll type this properly later
    environment_spec: str
    rollout_writer: RolloutWriter
    environment: Any
    environment_name: str
    max_input_length: int
    max_output_length: int
    pad_token_id: int
    n_prompts_per_step: int
    n_generations: int
    temperature: float
    log_freq: int
    rollout_batch_size: int = 32
    max_rollouts: int | None = None


class LevanterInferenceContext(InferenceContext):
    """Context that uses Levanter model and inference server."""

    model: Any
    inference_server: InferenceServer
    max_tokens: int
    _tokenizer: Any

    def __init__(self, model, inference_server: InferenceServer, max_tokens: int):
        self.model = model
        self.inference_server = inference_server
        self.max_tokens = max_tokens
        self._tokenizer = inference_server.inference_context.tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        n_generations: int = 1,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """Generate responses for a batch of prompts."""
        host = self.inference_server.config.host
        port = self.inference_server.config.port
        url = f"http://{host}:{port}/v1/chat/completions"
        responses = []
        for prompt in prompts:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": getattr(self.inference_server.config, "model_name", "test-model"),
                    "messages": [{"role": "user", "content": prompt}],
                    "logprobs": True,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "n": n_generations,
                    "top_p": top_p,
                    "top_k": top_k,
                },
            )
            responses.append(response)

        all_results = []
        for response in responses:
            if response.status_code != 200:
                raise RuntimeError(f"Inference server error: {response.status_code} - {response.text}")

            from openai.types.chat.chat_completion import ChatCompletion

            chat_response = ChatCompletion.model_validate_json(response.text)
            prompt_results = []
            for choice in chat_response.choices:
                content = choice.message.content
                tokens = self._tokenizer.encode(content)
                logprobs = [t.logprob for t in choice.logprobs.content]
                prompt_results.append({"tokens": tokens, "logprobs": logprobs})
            all_results.append(prompt_results)

        return all_results

    def compute_logprobs(
        self,
        input_tokens: np.ndarray,
        input_attention_mask: np.ndarray,
        target_tokens: np.ndarray,
        target_attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute log probabilities for given input/target pairs."""
        import haliax as hax
        from jax.nn import log_softmax

        # Concatenate input and target tokens
        full_tokens = jnp.concatenate([input_tokens, target_tokens], axis=1)
        full_attention_mask = jnp.concatenate([input_attention_mask, target_attention_mask], axis=1)

        # Convert to Haliax named arrays
        # Get the axis names from the model config
        Batch = hax.Axis("batch", full_tokens.shape[0])
        SeqLen = hax.Axis("position", full_tokens.shape[1])

        tokens_named = hax.named(full_tokens, (Batch, SeqLen))
        attn_mask_named = hax.named(full_attention_mask.astype(jnp.bool_), (Batch, SeqLen))

        # Compute logits using the model
        # For causal language modeling, we need tokens[:-1] to predict tokens[1:]
        input_tokens_named = tokens_named.slice(SeqLen, start=0, length=SeqLen.size - 1)
        input_mask_named = attn_mask_named.slice(SeqLen, start=0, length=SeqLen.size - 1)

        # Run forward pass through the model
        logits = self.model(input_tokens_named, attn_mask=input_mask_named)

        # Convert logits to log probabilities
        log_probs = log_softmax(logits.array, axis=-1)  # Shape: (batch, seq_len-1, vocab_size)

        # Extract log probabilities for the target tokens
        target_start_idx = input_tokens.shape[1]
        target_logits = log_probs[:, target_start_idx - 1 :]  # Adjust for shift

        batch_indices = jnp.arange(target_logits.shape[0])[:, None]  # (batch, 1)
        seq_indices = jnp.arange(target_logits.shape[1])[None, :]  # (1, target_len)

        # Expand batch indices to match target shape
        batch_indices = jnp.broadcast_to(batch_indices, target_tokens.shape)  # (batch, target_len)

        # Extract the log probabilities for target tokens
        target_logprobs = target_logits[batch_indices, seq_indices, target_tokens]

        # Apply target attention mask to zero out padded positions
        target_logprobs = target_logprobs * target_attention_mask

        return np.array(target_logprobs)


class InferenceWorker:
    """Asynchonous inference & rollout worker for RL training.

    Inference workers periodically load model checkpoints generated by the training job,
    and continously generate rollouts from a single environment. Rollouts are communicated to the
    training job via a rollout queue.
    """

    _running: bool = True
    rollout_writer: RolloutWriter

    def __init__(
        self,
        config: InferenceWorkerConfig,
        coordinator=None,
    ):
        """Initialize inference worker.

        Args:
            config: Inference worker configuration.
            coordinator: Coordinator for weight transfer (required for RAY_REMOTING and JAX_TRANSFER_SERVER modes).
        """
        # Create and start the inference server
        self.inference_server_config = config.inference_server_config
        self.inference_server = InferenceServer.create(self.inference_server_config)
        self._server_thread = None
        self._start_inference_server()
        self.policy_model = config.policy_model
        self.reference_model = config.reference_model
        self.environment_spec = config.environment_spec
        self.rollout_writer = config.rollout_writer
        self.environment = config.environment
        self.environment_name = config.environment_name
        self.max_input_length = config.max_input_length
        self.max_output_length = config.max_output_length
        self.pad_token_id = config.pad_token_id
        self.n_prompts_per_step = config.n_prompts_per_step
        self.n_generations = config.n_generations
        self.temperature = config.temperature
        self.log_freq = config.log_freq
        self.rollout_batch_size = config.rollout_batch_size
        self.max_rollouts = config.max_rollouts
        self.coordinator = coordinator

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
        self._running = False
        if self.inference_server:
            self.inference_server.shutdown()

    def _generate_rollout_batch(self, rng) -> tuple[list[dict], dict]:
        """Generate a set of rollout batches from the environment."""
        jax_distributed_barrier()

        # Create Levanter inference contexts
        policy_ctx = LevanterInferenceContext(
            self.policy_model, self.inference_server, max_tokens=self.max_output_length
        )
        reference_ctx = LevanterInferenceContext(
            self.reference_model, self.inference_server, max_tokens=self.max_output_length
        )

        rl_dataset, dataset_metrics = create_dataset_from_environment(
            environment=self.environment,
            policy_ctx=policy_ctx,
            reference_ctx=reference_ctx,
            n_examples=self.n_prompts_per_step,
            prng_key=rng,
            n_generations=self.n_generations,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            pad_token_id=self.pad_token_id,
            mode="train",
            temperature=self.temperature,
        )
        jax_distributed_barrier()

        return (
            list(rl_dataset.iterate_batches(batch_size=self.rollout_batch_size, shuffle=True, loop=False)),
            dataset_metrics,
        )

    def run(self):
        """Main inference worker loop."""
        logger.info("Starting inference worker...")

        rollouts_generated = 0
        step = 0
        rng = jax.random.PRNGKey(0)

        while self._running:
            jax_distributed_barrier()

            if self.max_rollouts is not None and rollouts_generated >= self.max_rollouts:
                logger.info(f"Reached max rollouts ({self.max_rollouts}), stopping")
                break

            rng, input_rng = jax.random.split(rng)
            rollout_batches, metrics = self._generate_rollout_batch(input_rng)
            for batch_data in rollout_batches:
                step += 1

                if self.log_freq > 0 and step % self.log_freq == 0:
                    log_metrics = {"step": step}
                    log_metrics.update(jax.device_get(metrics))
                    log_metrics = {"inference." + k: v for k, v in log_metrics.items()}
                    logger.info(f"Logging metrics at step {step}... {log_metrics}")

                rollout_batch = TaggedRolloutBatch(
                    batch=RolloutBatch(
                        input_ids=batch_data["input_ids"],
                        attention_mask=batch_data["attention_mask"],
                        position_ids=batch_data["position_ids"],
                        target_ids=batch_data["target_ids"],
                        loss_weights=batch_data["loss_weights"],
                        loss_masks=batch_data["loss_masks"],
                        reference_logprobs=batch_data["reference_logprobs"],
                        policy_logprobs=batch_data["policy_logprobs"],
                    ),
                    env_name=self.environment_name,
                    worker_id=f"{socket.gethostname()}_{os.getpid()}",
                    timestamp=time.time(),
                    rollout_id=f"{socket.gethostname()}_{int(time.time() * 1000000)}_{step}",
                )
                self.rollout_writer.write_batch(rollout_batch)
                rollouts_generated += 1
            logger.info(f"Generating rollout batch {rollouts_generated}")

        logger.info(f"Inference worker completed after generating {rollouts_generated} rollouts")
        jax_distributed_barrier()
