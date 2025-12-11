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

import dataclasses
import logging
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import wandb

import equinox as eqx
import haliax as hax
import jax
import jax.random as jrandom
import levanter
from jax.experimental import multihost_utils
from levanter.inference.openai import InferenceServer
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import barrier_sync
from transformers import PreTrainedTokenizer
from typing import Literal

from marin.rl.curriculum import CurriculumConfig, get_or_create_curriculum_actor
from marin.rl.environments import MarinEnv
from marin.rl.environments.base import load_environment_from_spec
from marin.rl.environments.inference_ctx import (
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    vLLMInferenceContextConfig,
    vLLMInferenceContext,
    AsyncvLLMInferenceContext,
    BaseInferenceContext,
)
from marin.rl.model_utils import load_model_from_checkpoint

from .rollout_storage import RolloutStorageConfig, RolloutWriter
from .types import (
    RolloutBatch,
    RolloutGroup,
    RolloutMetadata,
    RolloutStats,
)
from .weight_transfer.base import WeightUpdate
from .weight_transfer import WeightTransferClient, WeightTransferConfig, create_weight_transfer_client

logger = logging.getLogger(__name__)


@dataclass
class RolloutTrackerConfig:
    """Configuration for the rollout worker's WandB tracker.

    This is a standalone tracker that doesn't depend on JAX initialization,
    avoiding deadlocks when running vLLM inference workers.
    """

    project: str
    """WandB project name."""

    name: str | None = None
    """WandB run name."""

    tags: list[str] = field(default_factory=list)
    """Tags to attach to the WandB run."""

    entity: str | None = None
    """WandB entity (team or username)."""

    mode: str = "online"
    """WandB mode: 'online', 'offline', or 'disabled'."""


class RolloutTracker:
    """A simple WandB tracker for rollout workers.

    Unlike Levanter's tracker, this doesn't call jax.process_index() during init,
    avoiding JAX distributed initialization deadlocks.
    """

    def __init__(self, config: RolloutTrackerConfig, run_id: str):
        self._run = wandb.init(
            entity=config.entity,
            project=config.project,
            name=config.name,
            tags=config.tags,
            id=run_id,
            resume="allow",
            mode=config.mode,
        )

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None):
        self._run.log(metrics, step=step)

    def finish(self):
        self._run.finish()


@dataclass
class RolloutWorkerConfig:
    """Configuration for RolloutWorker."""

    curriculum_config: CurriculumConfig
    rollout_storage: RolloutStorageConfig
    weight_transfer: WeightTransferConfig
    tokenizer: PreTrainedTokenizer
    run_id: str
    trainer: TrainerConfig
    model: LmConfig
    inference_type: Literal["levanter", "vllm"]
    """Type of inference to use."""

    inference_config: LevanterInferenceContextConfig | vLLMInferenceContextConfig
    """Configuration for inference context."""

    tracker_config: RolloutTrackerConfig | None = None
    """Configuration for the rollout worker's tracker. If None, tracking is disabled."""

    seed: int = 0
    """Random seed to use for sampling."""
    max_rollouts: int | None = None
    """Maximum number of rollouts to generate before stopping. Defaults to running forever."""

    log_freq: int = 10

    initial_checkpoint: str | None = None
    """Initial checkpoint for the reference model (auto-detects HF repo vs local path)."""

    system_prompt: str | None = None
    """System prompt to use for inference."""

    inflight_weight_updates: bool = False
    """Whether to use inflight weight updates."""


def find_open_port() -> int:
    """Find an open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class RolloutBatchStats:
    total_count: int
    success_count: int
    rollout_stats: list[RolloutStats]
    avg_reward: float
    pass_at_one: float | None = None
    pass_at_k: float | None = None
    avg_at_k: float | None = None


def _compute_batch_stats(batch: RolloutBatch, lesson_id: str):
    rollout_stats_list = []
    total_count = 0
    success_count = 0
    reward_sum = 0.0
    pass_at_k = 0.0
    pass_at_one = 0.0
    avg_at_k = 0.0

    for group in batch.groups:
        pass_at_k_for_current_group = 0.0
        avg_at_k_for_current_group = 0.0
        for rollout in group.rollouts:
            rollout_stats_list.append(
                RolloutStats(
                    lesson_id=lesson_id,
                    episode_reward=rollout.episode_reward,
                    env_example_id=rollout.env_example_id,
                )
            )

            if rollout.correctness_reward is not None:
                pass_at_k_for_current_group = max(pass_at_k_for_current_group, rollout.correctness_reward)
                avg_at_k_for_current_group += rollout.correctness_reward

            total_count += 1
            if rollout.episode_reward > 0:
                success_count += 1
            reward_sum += rollout.episode_reward

        pass_at_k += pass_at_k_for_current_group
        if group.rollouts[0].correctness_reward is not None:
            pass_at_one += group.rollouts[0].correctness_reward

        avg_at_k += avg_at_k_for_current_group / len(group.rollouts)

    return RolloutBatchStats(
        total_count=total_count,
        success_count=success_count,
        rollout_stats=rollout_stats_list,
        avg_reward=(reward_sum / total_count) if total_count > 0 else 0.0,
        pass_at_one=(pass_at_one / len(batch.groups)) if len(batch.groups) > 0 else 0.0,
        pass_at_k=(pass_at_k / len(batch.groups)) if len(batch.groups) > 0 else 0.0,
        avg_at_k=(avg_at_k / len(batch.groups)) if len(batch.groups) > 0 else 0.0,
    )


def create_inference_context(
    inference_type: str,
    inference_config: LevanterInferenceContextConfig | vLLMInferenceContextConfig,
    inflight_weight_updates: bool,
) -> BaseInferenceContext:
    """Create an inference context based on the configuration."""
    if inference_type == "levanter":
        # Infer model_axis_size from the actual TPU configuration now that JAX is initialized.
        # For inference servers, we shard across all local devices on a single host.
        inference_config.inference_server_config = dataclasses.replace(
            inference_config.inference_server_config,
            trainer=dataclasses.replace(
                inference_config.inference_server_config.trainer,
                model_axis_size=jax.local_device_count(),
            ),
        )
        return LevanterInferenceContext(
            inference_config=inference_config,
        )
    elif inference_type == "vllm" and not inflight_weight_updates:
        return vLLMInferenceContext(
            inference_config=inference_config,
        )
    elif inference_type == "vllm" and inflight_weight_updates:
        return AsyncvLLMInferenceContext(
            inference_config=inference_config,
        )

    raise ValueError(f"Invalid inference type: {inference_type}")


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
    tracker: Any  # levanter.Tracker or RolloutTracker

    def __init__(self, config: RolloutWorkerConfig):
        config.trainer.id = f"{config.run_id}-rollout"

        # Infer model_axis_size from the actual TPU configuration now that JAX is initialized.
        # For inference servers, we shard across all local devices on a single host.
        if config.inference_type == "levanter":
            self.tracker = levanter.current_tracker()
        else:
            # Initialize our own tracker to avoid JAX distributed initialization deadlocks.
            # Levanter's tracker calls jax.process_index() which forces JAX initialization.
            self.tracker = RolloutTracker(config.tracker_config, config.run_id)

        self.config = config
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()
        self._current_weight_step: int = -2

        self._tokenizer = config.tokenizer

        # Event to signal that the first weight transfer has completed.
        # For inflight weight updates, we block inference until initial weights are received.
        self._first_weights_received = threading.Event()

        logger.info("Starting weight transfer client with config %s", self.config.weight_transfer)

        self._rollout_writer = config.rollout_storage.create_writer()
        self._policy_ctx = create_inference_context(
            self.config.inference_type, self.config.inference_config, self.config.inflight_weight_updates
        )

        # Need to build the policy model and then use that to start the inference server
        self._build_models()
        self._policy_ctx.start_server(self._policy_model)

        self._transfer_client = create_weight_transfer_client(
            config.weight_transfer,
            mesh=self._policy_ctx.mesh,
            axis_mapping=self._policy_ctx.axis_mapping,
        )

        # TODO(power) -- replace this with a wait_until_ready() on the levanter inference server
        time.sleep(1.0)

        self._environments = {}

        # Create curriculum actor (no checkpoint path for rollout workers)
        self._curriculum_actor = get_or_create_curriculum_actor(self.config.curriculum_config)

        self.weight_transfer_thread: threading.Thread | None = None
        if self.config.inflight_weight_updates:
            self.weight_transfer_thread = threading.Thread(
                target=self._sync_weights_loop,
                name=f"{self.config.run_id}-weight-sync",
                daemon=True,
            )
            self.weight_transfer_thread.start()

    def _load_environment(self, lesson_id: str) -> MarinEnv:
        """Load environment from lesson ID."""
        if lesson_id in self._environments:
            return self._environments[lesson_id]

        lesson_config = self.config.curriculum_config.lessons[lesson_id]
        env = load_environment_from_spec(lesson_config.env_config)
        self._environments[lesson_id] = env
        return env

    def _sample_batch(
        self, lesson_id: str, n_examples: int, n_generations: int, mode: str, rng
    ) -> tuple[RolloutBatch | None, dict | None]:
        """Sample a batch of rollouts from the environment for the given lesson ID."""
        env = self._load_environment(lesson_id)
        lesson_config = self.config.curriculum_config.lessons[lesson_id]

        # Get sampling params from lesson config
        temperature = lesson_config.sampling_params.temperature
        stop_tokens = lesson_config.sampling_params.stop_tokens
        max_tokens = lesson_config.sampling_params.max_tokens

        rollout_groups, metrics = env.sample(
            inference_ctx=self._policy_ctx,
            n_examples=n_examples,
            n_generations=n_generations,
            temperature=temperature,
            prng_key=rng,
            mode=mode,
            max_tokens=max_tokens,
            stop=stop_tokens,
            system_prompt=self.config.system_prompt,
        )

        if len(rollout_groups) == 0:
            logger.warning("No valid rollouts generated in this batch...")
            return None, None

        logger.info(
            "Generated rollout with %d groups from lesson %s at step %d",
            len(rollout_groups),
            lesson_id,
            self._current_weight_step,
        )

        # Create metadata once for this batch
        batch_metadata = RolloutMetadata(
            worker_id=f"{socket.gethostname()}_{os.getpid()}",
            timestamp=time.time(),
            weight_step=self._current_weight_step,
        )

        # Attach metadata to each rollout in each group
        rollout_groups_with_metadata = []
        for group in rollout_groups:
            rollouts_with_metadata = []
            for rollout in group.rollouts:
                # Create new rollout with metadata attached
                rollout_with_meta = eqx.tree_at(lambda r: r.metadata, rollout, batch_metadata)
                rollouts_with_metadata.append(rollout_with_meta)

            rollout_groups_with_metadata.append(RolloutGroup(rollouts=rollouts_with_metadata))

        rollout_batch = RolloutBatch(
            groups=rollout_groups_with_metadata,
            metadata=batch_metadata,
        )
        return rollout_batch, metrics

    def _build_models(self):
        if self.config.initial_checkpoint is not None:
            logger.info(f"Loading initial policy model from checkpoint: {self.config.initial_checkpoint}")
        else:
            logger.info("Building new policy model from scratch")

        if self.config.inference_type == "levanter":
            key = jrandom.PRNGKey(self.config.seed)
            vocab_size = self._tokenizer.vocab_size
            Vocab = hax.Axis("vocab", vocab_size)

            initial_model = load_model_from_checkpoint(
                checkpoint=self.config.initial_checkpoint,
                model_config=self.config.model,
                trainer_config=self.config.trainer,
                vocab_axis=Vocab,
                mesh=self._policy_ctx.mesh,
                # use the compute axis mapping for inference
                axis_mapping=self._policy_ctx.axis_mapping,
                tokenizer=self._tokenizer,
                key=key,
            )

            logger.info("Initializing policy model from initial checkpoint")
            self._policy_model = initial_model
        elif self.config.inference_type == "vllm":
            # TODO(chris): Remove this completely
            self._policy_model = None

    def stop(self):
        """Stop the inference worker loop and server."""
        with self._shutdown_condition:
            self._running = False
            self._transfer_client.cleanup()
            self._shutdown_condition.notify()

        # Wait for the main loop to finish
        self._shutdown_complete.wait()

        # Now shutdown the inference server
        self._policy_ctx.shutdown()

        if self.weight_transfer_thread:
            self.weight_transfer_thread.join()

    def _apply_weight_update(self, update: WeightUpdate):
        """Apply a newly received weight update to the inference context."""
        self._current_weight_step = update.weight_id
        logger.info("Received new weights from step %d", update.weight_id)
        self._policy_model = self._policy_ctx.reload_model(update.model, update.state_dict)

        # Signal that we've received the first weights
        if not self._first_weights_received.is_set():
            logger.info("First weight transfer complete, inference can proceed")
            self._first_weights_received.set()

    def _sync_weights(self):
        """Attempt to receive updated weights, optionally waiting for them."""
        max_wait_time = self.config.weight_transfer.max_weight_transfer_wait_time

        def _receive_once():
            try:
                return self._transfer_client.receive_weights(self._policy_model)
            except Exception:
                logger.exception("Weight transfer client failed while receiving weights.")
                return None

        if max_wait_time <= 0:
            update = _receive_once()
            if update:
                self._apply_weight_update(update)

            return None

        start_time = time.time()

        while self._running:
            update = _receive_once()
            if update:
                self._apply_weight_update(update)
                break

            elapsed = time.time() - start_time
            if elapsed >= max_wait_time:
                logger.info(
                    "Waited %.1fs for new weights, proceeding with current weights",
                    elapsed,
                )
                break

            time.sleep(1.0)

        return None

    def _sync_weights_loop(self):
        """Continuously sync weights in a background thread."""
        logger.info("Starting background weight sync loop")
        try:
            while self._running:
                self._sync_weights()
                if not self._running:
                    break

                if self.config.weight_transfer.max_weight_transfer_wait_time <= 0:
                    time.sleep(1.0)
        except Exception:
            logger.exception("Background weight sync loop crashed")
        finally:
            logger.info("Background weight sync loop exiting")

    def _log_prompt_example(self, lesson_id: str, batch: RolloutBatch, step: int, eval_type: str = "eval") -> None:
        """Log representative samples from an evaluation batch.

        Args:
            lesson_id: ID of the evaluated lesson
            batch: The rollout batch containing samples
            step: Current training step
            mode: Either "eval" or "micro_eval"
        """
        if not batch or not batch.groups:
            return

        sample_groups = min(1000, len(batch.groups))
        selected_group_indices = random.sample(range(len(batch.groups)), k=sample_groups)

        rows = []
        for group_idx in selected_group_indices:
            group = batch.groups[group_idx]
            if not group.rollouts:
                continue
            rollout = random.choice(group.rollouts)
            prompt_text = self._tokenizer.decode(rollout.prompt_tokens, skip_special_tokens=False)
            response_text = self._tokenizer.decode(rollout.response_tokens, skip_special_tokens=False)
            rows.append(
                {"prompt": prompt_text, "response": response_text, "reward": rollout.episode_reward, "step": step}
            )

        if not rows:
            return

        table = wandb.Table(columns=["prompt", "response", "reward", "step"])
        for row in rows:
            table.add_data(row["prompt"], row["response"], row["reward"], row["step"])

        prefix = f"inference.{eval_type}/{lesson_id}"
        metrics = {f"{prefix}/sample_table": table}
        self.tracker.log(metrics, step=step)
        logger.info(f"Logged {len(rows)} eval samples for lesson {lesson_id} at step {step}")

    def _build_eval_metrics(
        self, prefix: str, lesson_id: str, batch: RolloutBatch, n_generations: int
    ) -> dict[str, Any]:
        metrics = {}
        stats = _compute_batch_stats(batch, lesson_id)
        if stats.total_count == 0:
            return metrics
        success_rate = stats.success_count / stats.total_count
        metrics[f"{prefix}/{lesson_id}/success_rate"] = success_rate
        metrics[f"{prefix}/{lesson_id}/avg_reward"] = stats.avg_reward
        metrics[f"{prefix}/{lesson_id}/total_count"] = stats.total_count
        metrics[f"{prefix}/{lesson_id}/pass_at_one"] = stats.pass_at_one
        metrics[f"{prefix}/{lesson_id}/pass_at_{n_generations}"] = stats.pass_at_k
        metrics[f"{prefix}/{lesson_id}/avg_at_{n_generations}"] = stats.avg_at_k
        return metrics

    def _evaluate_lesson(self, lesson_id: str, n_examples: int, eval_type: str, rng, step: int) -> dict:
        """Evaluate a single lesson and log metrics."""
        N_EVAL_GENERATIONS = 1

        batch, _ = self._sample_batch(
            lesson_id=lesson_id,
            n_examples=n_examples,
            n_generations=N_EVAL_GENERATIONS,
            mode="eval",
            rng=rng,
        )
        stats = _compute_batch_stats(batch, lesson_id)
        self._log_prompt_example(lesson_id, batch, step, eval_type=eval_type)
        metrics = self._build_eval_metrics(
            prefix=f"inference.{eval_type}", lesson_id=lesson_id, batch=batch, n_generations=N_EVAL_GENERATIONS
        )
        self.tracker.log(metrics, step=step)
        logger.info("Eval metrics for lesson %s at step %d: %s", lesson_id, step, metrics)
        # only update curriculum for full evals
        if eval_type == "eval":
            self._curriculum_actor.update_lesson_stats.options(enable_task_events=False).call(
                stats.rollout_stats, mode="eval", current_step=step
            )
        return stats

    def _evaluate_curriculum(self, rng, step: int) -> dict:
        """Evaluate all lessons and update the curriculum actor."""
        lesson_names = list(self.config.curriculum_config.lessons.keys())
        if not lesson_names:
            logger.info("No lessons to evaluate")
            return {}

        logger.info(f"Evaluating {len(lesson_names)} lessons")

        for lesson_id in lesson_names:
            self._evaluate_lesson(
                lesson_id, self.config.curriculum_config.eval_n_examples, eval_type="eval", rng=rng, step=step
            )

        # NOTE(chris): why do we need this?
        # barrier_sync()

    def run(self):
        """Main inference worker loop."""
        logger.info("Starting inference worker...")

        # For inflight weight updates, wait for first weights before generating rollouts
        if self.config.inflight_weight_updates:
            logger.info("Waiting for first weight transfer before starting inference...")
            while not self._first_weights_received.wait(timeout=10.0):
                if not self._running:
                    logger.info("Shutdown requested while waiting for first weights")
                    self._shutdown_complete.set()
                    return
                logger.info("Still waiting for first weight transfer...")
            logger.info("First weights received, starting inference loop")

        step = 0
        use_jax_rng = self.config.inference_type == "levanter"

        # Initialize RNG - use Python's random for vLLM to avoid JAX device access
        if use_jax_rng:
            rng = jax.random.PRNGKey(self.config.seed)
            rng = multihost_utils.broadcast_one_to_all(rng)
        else:
            py_rng = random.Random(self.config.seed)

        logger.info(f"Starting rollout worker with seed {self.config.seed}")

        while self._running:
            # Synchronize weights on main thread unless using inflight weight updates
            if not self.config.inflight_weight_updates:
                self._sync_weights()

            if self.config.max_rollouts is not None and step >= self.config.max_rollouts:
                logger.info(f"Reached max rollouts ({self.config.max_rollouts}), stopping")
                break

            logger.info("Generating rollout batch...")

            if use_jax_rng:
                rng, seed_key = jax.random.split(rng)
                seed = int(seed_key[0])
            else:
                seed = py_rng.randint(0, 2**31 - 1)

            try:
                lesson_id = self._curriculum_actor.sample_lesson.call(seed)
            except Exception as e:
                logger.warning(f"Failed to sample lesson from curriculum: {e}, will try again...")
                time.sleep(10.0)
                continue

            # Micro-eval: feedback on current lesson
            if step > 0 and step % self.config.curriculum_config.micro_eval_frequency == 0:
                if use_jax_rng:
                    rng, micro_eval_rng = jrandom.split(rng)
                else:
                    micro_eval_rng = py_rng.randint(0, 2**31 - 1)
                self._evaluate_lesson(
                    lesson_id,
                    self.config.curriculum_config.micro_eval_n_examples,
                    eval_type="micro_eval",
                    rng=micro_eval_rng,
                    step=step,
                )

            # Full eval: comprehensive check on all lessons
            # Evaluate based on the train worker step
            if step % self.config.curriculum_config.eval_frequency == 0:
                if use_jax_rng:
                    rng, eval_rng = jrandom.split(rng)
                else:
                    eval_rng = py_rng.randint(0, 2**31 - 1)
                self._evaluate_curriculum(eval_rng, step)

            logger.info(f"Sampled lesson '{lesson_id}' from curriculum")

            if use_jax_rng:
                rng, input_rng = jax.random.split(rng)
            else:
                input_rng = py_rng.randint(0, 2**31 - 1)

            lesson_config = self.config.curriculum_config.lessons[lesson_id]
            rollout_batch, env_metrics = self._sample_batch(
                lesson_id=lesson_id,
                n_examples=lesson_config.sampling_params.n_prompts,
                n_generations=lesson_config.sampling_params.n_generations_per_prompt,
                mode="train",
                rng=input_rng,
            )
            if rollout_batch is None:
                continue

            self._rollout_writer.write_batch(rollout_batch)

            stats = _compute_batch_stats(rollout_batch, lesson_id)
            self._curriculum_actor.update_lesson_stats.options(enable_task_events=False).call(
                stats.rollout_stats, mode="training", current_step=step
            )
            eval_metrics = self._build_eval_metrics(
                prefix="rollout",
                lesson_id=lesson_id,
                batch=rollout_batch,
                n_generations=lesson_config.sampling_params.n_generations_per_prompt,
            )

            step += 1

            if self.config.log_freq > 0 and step % self.config.log_freq == 0:
                log_metrics = eval_metrics
                log_metrics.update(self._transfer_client.get_metrics())
                log_metrics.update({f"env.{k}": v for k, v in (env_metrics or {}).items()})
                log_metrics = {"inference." + k: v for k, v in log_metrics.items()}
                logger.info(f"Logging metrics at step {step}... {log_metrics}")
                self.tracker.log(log_metrics, step=step)

        logger.info(f"Inference worker completed after generating {step} rollouts")
        if use_jax_rng:
            barrier_sync()
        self._shutdown_complete.set()
