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

import haliax as hax
import jax
import jax.random as jrandom
import levanter
import numpy as np
import ray
from jax.experimental import multihost_utils
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import barrier_sync
from openai import AsyncOpenAI
from transformers import PreTrainedTokenizer

from marin.rl.curriculum import CurriculumConfig, get_or_create_curriculum_actor
from marin.rl.environments import MarinEnv
from marin.rl.environments.base import load_environment_from_spec
from marin.rl.model_utils import load_model_from_checkpoint

from .rollout_storage import RolloutStorageConfig, RolloutWriter
from .types import (
    InferenceChoice,
    InferenceContext,
    InferenceResponse,
    RolloutBatch,
    RolloutMetadata,
    RolloutStats,
)
from .weight_transfer import WeightTransferClient, WeightTransferConfig, create_weight_transfer_client

logger = logging.getLogger(__name__)


@dataclass
class RolloutWorkerConfig:
    """Configuration for RolloutWorker."""

    inference_server_config: InferenceServerConfig
    trainer: TrainerConfig
    model: LmConfig
    curriculum_config: CurriculumConfig
    rollout_storage: RolloutStorageConfig
    weight_transfer: WeightTransferConfig
    tokenizer: PreTrainedTokenizer
    run_id: str

    max_rollouts: int | None = None
    """Maximum number of rollouts to generate before stopping. Defaults to running forever."""

    initial_checkpoint: str | None = None
    """Initial checkpoint for the reference model (auto-detects HF repo vs local path)."""

    log_freq: int = 10


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


@dataclass
class RolloutBatchStats:
    total_count: int
    success_count: int
    rollout_stats: list[RolloutStats]
    avg_reward: float


def _compute_batch_stats(batch: RolloutBatch, lesson_id: str):
    rollout_stats_list = []
    total_count = 0
    success_count = 0
    reward_sum = 0.0

    for group in batch.groups:
        for rollout in group.rollouts:
            rollout_stats_list.append(
                RolloutStats(
                    lesson_id=lesson_id,
                    episode_reward=rollout.episode_reward,
                    env_example_id=rollout.env_example_id,
                )
            )

            total_count += 1
            if rollout.episode_reward > 0:
                success_count += 1
            reward_sum += rollout.episode_reward

    return RolloutBatchStats(
        total_count=total_count,
        success_count=success_count,
        rollout_stats=rollout_stats_list,
        avg_reward=(reward_sum / total_count) if total_count > 0 else 0.0,
    )


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
    _environments: dict[str, MarinEnv]

    def __init__(self, config: RolloutWorkerConfig):
        config.trainer.id = f"{config.run_id}-rollout"
        levanter.initialize(config.trainer)
        self.tracker = levanter.current_tracker()
        self.config = config
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()
        self._current_weight_step: int = 0

        self._tokenizer = config.tokenizer
        self._curriculum_actor = get_or_create_curriculum_actor(config.curriculum_config)

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

        self._environments = {}

    def _load_environment(self, lesson_id: str) -> MarinEnv:
        """Load environment from lesson ID."""
        if lesson_id in self._environments:
            return self._environments[lesson_id]

        lesson_config = self.config.curriculum_config.lessons[lesson_id]
        env = load_environment_from_spec(lesson_config.env_config)
        self._environments[lesson_id] = env
        return env

    def _sample_batch(self, lesson_id: str, mode: str, rng) -> tuple[RolloutBatch | None, dict | None]:
        """Sample a batch of rollouts from the environment for the given lesson ID."""
        env = self._load_environment(lesson_id)
        lesson_config = self.config.curriculum_config.lessons[lesson_id]

        # Get sampling params from lesson or use eval defaults
        if mode == "eval":
            n_examples = self.config.curriculum_config.eval_n_examples
            n_generations = self.config.curriculum_config.eval_n_generations
            temperature = lesson_config.sampling_params.temperature
            stop_tokens = lesson_config.sampling_params.stop_tokens
        else:  # train
            n_examples = lesson_config.sampling_params.n_prompts
            n_generations = lesson_config.sampling_params.n_generations_per_prompt
            temperature = lesson_config.sampling_params.temperature
            stop_tokens = lesson_config.sampling_params.stop_tokens

        # Get max_tokens from lesson
        max_tokens = lesson_config.sampling_params.max_tokens

        policy_ctx = LevanterInferenceContext(
            tokenizer=self._tokenizer,
            inference_server=self._inference_server,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )

        with (
            self.config.trainer.device_mesh,
            hax.axis_mapping(self.config.trainer.compute_axis_mapping),
        ):
            # Sample examples, generate responses, and create rollouts from selected lesson
            rollout_groups, metrics = env.sample(
                inference_ctx=policy_ctx,
                n_examples=n_examples,
                n_generations=n_generations,
                temperature=temperature,
                prng_key=rng,
                mode=mode,
            )

        if len(rollout_groups) == 0:
            logger.warning("No valid rollouts generated in this batch...")
            return None, None

        rollout_batch = RolloutBatch(
            groups=rollout_groups,
            metadata=RolloutMetadata(
                worker_id=f"{socket.gethostname()}_{os.getpid()}",
                timestamp=time.time(),
                weight_step=self._current_weight_step,
            ),
        )
        return rollout_batch, metrics

    def _build_models(self):
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

    def _evaluate_curriculum(self, rng, step: int) -> dict:
        """Evaluate all lessons and update the curriculum actor.

        Returns:
            Dictionary of evaluation metrics for logging.
        """
        barrier_sync()

        # Evaluate all lessons, not just active ones
        lesson_names = list(self.config.curriculum_config.lessons.keys())

        if not lesson_names:
            logger.info("No lessons to evaluate")
            return {}

        logger.info(f"Evaluating {len(lesson_names)} lessons")

        eval_metrics = {}

        for lesson_id in lesson_names:
            batch, _ = self._sample_batch(lesson_id, mode="eval", rng=rng)
            stats = _compute_batch_stats(batch, lesson_id)

            self._curriculum_actor.update_lesson_stats.remote(stats.rollout_stats, mode="eval", current_step=step)

            if stats.total_count > 0:
                success_rate = stats.success_count / stats.total_count

                logger.info(
                    f"Eval {lesson_id}: success={stats.success_count}/{stats.total_count} "
                    f"({100 * success_rate:.1f}%), reward={stats.avg_reward:.3f}"
                )

                eval_metrics[f"eval/{lesson_id}/success_rate"] = success_rate
                eval_metrics[f"eval/{lesson_id}/avg_reward"] = stats.avg_reward
                eval_metrics[f"eval/{lesson_id}/total_count"] = stats.total_count

        barrier_sync()
        return eval_metrics

    def run(self):
        """Main inference worker loop."""
        logger.info("Starting inference worker...")

        step = 0

        # compute the seed as the all-reduce across all hosts in the jax process
        seed = abs(hash(f"{socket.gethostname()}-{os.getpid()}")) % (2**31 - 1)
        rng = jax.random.PRNGKey(seed)
        rng = multihost_utils.broadcast_one_to_all(rng)
        logger.info(f"Starting rollout worker with seed {seed}")

        last_weight_check = time.time()

        while self._running:
            barrier_sync()

            if self.config.max_rollouts is not None and step >= self.config.max_rollouts:
                logger.info(f"Reached max rollouts ({self.config.max_rollouts}), stopping")
                break

            if time.time() - last_weight_check > self.config.weight_transfer.poll_interval_seconds:
                self._sync_weights()
                last_weight_check = time.time()

            if step > 0 and step % self.config.curriculum_config.eval_frequency == 0:
                rng, eval_rng = jrandom.split(rng)
                eval_metrics = self._evaluate_curriculum(eval_rng, step)

                # Log eval metrics
                if jax.process_index() == 0 and eval_metrics:
                    log_metrics = {"step": step}
                    log_metrics.update(jax.device_get(eval_metrics))
                    log_metrics = {"inference." + k: v for k, v in log_metrics.items()}
                    logger.info(f"Logging eval metrics at step {step}... {log_metrics}")
                    self.tracker.log(log_metrics, step=step)

            logger.info("Generating rollout batch...")
            rng, seed_key = jax.random.split(rng)
            seed = int(seed_key[0])
            lesson_id = ray.get(self._curriculum_actor.sample_lesson.remote(seed))
            logger.info(f"Sampled lesson '{lesson_id}' from curriculum")

            rng, input_rng = jax.random.split(rng)
            rollout_batch, metrics = self._sample_batch(lesson_id, mode="train", rng=input_rng)
            if rollout_batch is None:
                continue
            barrier_sync()

            stats = _compute_batch_stats(rollout_batch, lesson_id)
            self._curriculum_actor.update_lesson_stats.remote(stats.rollout_stats, mode="training", current_step=step)

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
