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

import logging
import os
import socket
import time

import jax
import jax.numpy as jnp
import numpy as np

from marin.post_training.inference_loader import InferenceServer

from .flax.utils import (
    jax_distributed_barrier,
)
from .rl_dataset import create_dataset_from_environment
from .rollout_storage import RolloutBatch, RolloutWriter, TaggedRolloutBatch
from .training_config import TrainingConfig

logger = logging.getLogger(__name__)

COMPLETION_EXAMPLE = """
curl http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer $OPENAI_API_KEY" -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How many words in the next sentence? \"I like cats!\""}],
    "logprobs": false,
    "max_tokens": 128,
    "temperature": 0.1
  }'
"""


class LevanterInferenceContext:
    """Context that uses Levanter model and inference server."""

    def __init__(self, model, inference_server: InferenceServer):
        self.model = model
        self.inference_server = inference_server
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
        # use requests to call the inference server using the OpenAI chat/completions API
        import requests

        host = self.inference_server.config.host
        port = self.inference_server.config.port
        url = f"http://{host}:{port}/v1/chat/completions"
        requests.get(
            url,
            headers={"Content-Type": "application/json"},
            data={
                "model": self.inference_server.config.model_name,
                "messages": [{"role": "user", "content": prompt} for prompt in prompts],
                "logprobs": True,
                "max_tokens": 1024,
                "temperature": temperature,
                "n": n_generations,
                "top_p": top_p,
                "top_k": top_k,
            },
        )

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
        input_tokens_named = tokens_named[:, :-1]  # Remove last token
        input_mask_named = attn_mask_named[:, :-1]  # Remove last mask

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
    training_config: TrainingConfig
    rollout_writer: RolloutWriter

    def __init__(
        self,
        training_config: TrainingConfig,
        inference_server: InferenceServer,
        policy_model,
        reference_model,
        environment_spec: str,
        rollout_writer: RolloutWriter,
        rollout_batch_size: int = 32,
        max_rollouts: int | None = None,
        coordinator=None,
    ):
        """Initialize inference worker.

        Args:
            training_config: Training configuration.
            inference_server: Levanter inference server.
            policy_model: Policy model for generation.
            reference_model: Reference model for logprobs.
            environment_spec: Environment specification string.
            rollout_writer: Writer for rollout output.
            rollout_batch_size: Size of rollout batches.
            max_rollouts: Maximum number of rollouts to generate. None for unlimited.
            coordinator: Coordinator for weight transfer (required for RAY_REMOTING and JAX_TRANSFER_SERVER modes).
        """
        self.training_config = training_config
        self.inference_server = inference_server
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.environment_spec = environment_spec
        self.rollout_batch_size = rollout_batch_size
        self.max_rollouts = max_rollouts
        self.coordinator = coordinator

    def stop(self):
        """Stop the inference worker loop."""
        self._running = False

    def _generate_rollout_batch(self, rng) -> tuple[list[dict], dict]:
        """Generate a set of rollout batches from the environment."""
        jax_distributed_barrier()

        # Create Levanter inference contexts
        policy_ctx = LevanterInferenceContext(self.policy_model, self.inference_server)
        reference_ctx = LevanterInferenceContext(self.reference_model, self.inference_server)

        rl_dataset, dataset_metrics = create_dataset_from_environment(
            environment=self.environment,
            policy_ctx=policy_ctx,
            reference_ctx=reference_ctx,
            n_examples=self.training_config.hyperparameters.n_prompts_per_step,
            prng_key=rng,
            n_generations=self.training_config.generation_config.n_generations,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            pad_token_id=self.pad_token_id,
            mode="train",
            temperature=self.training_config.generation_config.temperature,
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

                if self.training_config.logging.log_freq > 0 and step % self.training_config.logging.log_freq == 0:
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
