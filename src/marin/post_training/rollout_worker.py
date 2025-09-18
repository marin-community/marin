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
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizer

from marin.post_training.environments.load_environments import load_environment_from_spec
from marin.post_training.environments.marin_env import InferenceContext

from . import weight_transfer_manager
from .flax.utils import (
    jax_distributed_barrier,
)
from .rl_dataset import create_dataset_from_environment
from .rollout_storage import RolloutBatch, RolloutWriter, TaggedRolloutBatch
from .weight_transfer_manager import WeightTransferConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutWorkerConfig:
    """Configuration for RolloutWorker."""

    inference_server_config: InferenceServerConfig

    # used for initialization
    trainer: TrainerConfig
    model: LlamaConfig
    environment_spec: str
    rollout_writer: RolloutWriter
    max_input_length: int
    max_output_length: int
    pad_token_id: int
    n_prompts_per_step: int
    n_generations: int
    temperature: float
    log_freq: int
    weight_transfer: WeightTransferConfig
    rollout_batch_size: int = 32
    max_rollouts: int | None = None


def compute_model_logprobs(
    model,
    input_tokens: np.ndarray,
    input_attention_mask: np.ndarray,
    target_tokens: np.ndarray,
    target_attention_mask: np.ndarray,
) -> np.ndarray:
    """Compute log probabilities for target tokens given input using a Haliax model.

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

    from optax import softmax_cross_entropy_with_integer_labels

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

    def __init__(self, model, tokenizer, inference_server: InferenceServer, max_tokens: int):
        self.model = model
        self.inference_server = inference_server
        self.max_tokens = max_tokens
        self._tokenizer = tokenizer

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
        base_url = f"http://{host}:{port}/v1"

        client = OpenAI(base_url=base_url, api_key="dummy")

        all_results = []
        for prompt in prompts:
            completion = client.chat.completions.create(
                model=getattr(self.inference_server.config, "model_name", "test-model"),
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                max_tokens=self.max_tokens,
                temperature=temperature,
                n=n_generations,
                top_p=top_p,
                extra_body={"top_k": top_k} if top_k is not None else None,
            )

            prompt_results = []
            for choice in completion.choices:
                content = choice.message.content
                tokens = self.tokenizer.encode(content)
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
        return compute_model_logprobs(
            self.model,
            input_tokens,
            input_attention_mask,
            target_tokens,
            target_attention_mask,
        )


class RolloutWorker:
    """Asynchonous inference & rollout worker for RL training.

    Inference workers periodically load model checkpoints generated by the training job,
    and continously generate rollouts from a single environment. Rollouts are communicated to the
    training job via a rollout queue.
    """

    _server_thread: threading.Thread
    inference_server: InferenceServer
    policy_model: Any
    reference_model: Any
    transfer_client: weight_transfer_manager.WeightTransferClient
    _tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        config: RolloutWorkerConfig,
    ):
        """Initialize inference worker.

        Args:
            config: Inference worker configuration.
            coordinator: Coordinator for weight transfer (required for RAY_REMOTING and
                JAX_TRANSFER_SERVER modes).
        """
        levanter.initialize(config.trainer)
        self.config = config
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()

        if isinstance(self.config.model.tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer)
        else:
            self._tokenizer = cast(PreTrainedTokenizer, self.config.model.tokenizer)

        self._environment = load_environment_from_spec(config.environment_spec, tokenizer=self._tokenizer)

        self._build_models()

        config.inference_server_config.port = find_open_port()
        self.inference_server = InferenceServer.create(config.inference_server_config)
        self._start_inference_server()

        self.transfer_client = weight_transfer_manager.create_weight_transfer_client(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

    def _build_models(self):
        """Build policy and reference models after levanter initialization."""

        key = jrandom.PRNGKey(42)
        vocab_size = self._tokenizer.vocab_size
        Vocab = hax.Axis("vocab", vocab_size)

        with (
            self.config.trainer.device_mesh,
            hax.axis_mapping(self.config.trainer.compute_axis_mapping),
        ):
            self.policy_model = self.config.model.build(Vocab, key=key)
            self.reference_model = self.config.model.build(Vocab, key=key)

        logger.info("Built policy and reference models after levanter initialization")

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
        jax_distributed_barrier()

        # Create Levanter inference contexts
        policy_ctx = LevanterInferenceContext(
            self.policy_model,
            tokenizer=self._tokenizer,
            inference_server=self.inference_server,
            max_tokens=self.config.max_output_length,
        )
        reference_ctx = LevanterInferenceContext(
            self.reference_model,
            tokenizer=self._tokenizer,
            inference_server=self.inference_server,
            max_tokens=self.config.max_output_length,
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
        jax_distributed_barrier()

        return (
            list(rl_dataset.iterate_batches(batch_size=self.config.rollout_batch_size, shuffle=True, loop=False)),
            dataset_metrics,
        )

    def _sync_weights(self):
        logger.info("Checking for new weights...")
        weights = self.transfer_client.receive_weights(self.policy_model)
        if weights:
            logger.info("Received new weights for policy model")

            self.policy_model = weights
            self.inference_server.reload(lambda model: self.policy_model)
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
            jax_distributed_barrier()

            if self.config.max_rollouts is not None and rollouts_generated >= self.config.max_rollouts:
                logger.info(f"Reached max rollouts ({self.config.max_rollouts}), stopping")
                break

            if time.time() - last_weight_check > self.config.weight_transfer.poll_interval_seconds:
                self._sync_weights()
                last_weight_check = time.time()

            rng, input_rng = jax.random.split(rng)
            rollout_batches, metrics = self._generate_rollout_batch(input_rng)
            for batch_data in rollout_batches:
                step += 1

                if self.config.log_freq > 0 and step % self.config.log_freq == 0:
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
                    env_name=self.config.environment_spec,
                    worker_id=f"{socket.gethostname()}_{os.getpid()}",
                    timestamp=time.time(),
                    rollout_id=f"{socket.gethostname()}_{int(time.time() * 1000000)}_{step}",
                )
                self.config.rollout_writer.write_batch(rollout_batch)
                rollouts_generated += 1
            logger.info(f"Generating rollout batch {rollouts_generated}")

        logger.info(f"Inference worker completed after generating {rollouts_generated} rollouts")
        jax_distributed_barrier()
        self._shutdown_complete.set()
