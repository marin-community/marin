# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Inference worker for RL/post-training rollout generation.

This worker loads model checkpoints, generates rollouts from a single environment,
and writes the rollout data to files for training workers to consume.
"""

import dataclasses
import faulthandler
import logging
import os
import random
import signal
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jax.random as jrandom
import levanter
import wandb
from jax.experimental import multihost_utils
from levanter.inference.openai import InferenceServer
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import barrier_sync
from levanter.tokenizers import MarinTokenizer
from typing import Literal

from levanter.utils.mesh import MeshConfig
from marin.rl.curriculum import CurriculumConfig
from marin.rl.runtime import RLRuntimeHandles
from marin.rl.run_state import RolloutTransferCounters
from marin.rl.environments import MarinEnv
from marin.rl.environments.base import load_environment_from_spec
from marin.rl.environments.inference_ctx import (
    AsyncvLLMInferenceContext,
    BaseInferenceContext,
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)
from marin.rl.environments.inference_ctx.staging import (
    prepare_vllm_inference_config_for_inflight,
)
from marin.rl.metrics import pass_at_k_estimator
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


@dataclass(frozen=True)
class RolloutTransferCounterSnapshot:
    """Attempt-local weight receive counters for a rollout worker process."""

    total_polls: int = 0
    successful_receives: int = 0
    failed_receives: int = 0


def _rollout_transfer_counter_delta(
    current: RolloutTransferCounterSnapshot,
    previous: RolloutTransferCounterSnapshot,
) -> RolloutTransferCounterSnapshot:
    """Return non-negative deltas for possibly-reset attempt-local counters."""

    def _delta(metric_name: str, current_value: int, previous_value: int) -> int:
        if current_value < previous_value:
            logger.warning(
                "Rollout transfer metric %s decreased from %d to %d; "
                "treating current value as a fresh attempt-local counter",
                metric_name,
                previous_value,
                current_value,
            )
            return current_value
        return current_value - previous_value

    return RolloutTransferCounterSnapshot(
        total_polls=_delta("total_polls", current.total_polls, previous.total_polls),
        successful_receives=_delta(
            "successful_receives",
            current.successful_receives,
            previous.successful_receives,
        ),
        failed_receives=_delta("failed_receives", current.failed_receives, previous.failed_receives),
    )


def _rollout_transfer_counter_snapshot(metrics: Mapping[str, Any]) -> RolloutTransferCounterSnapshot:
    """Extract cumulative counter fields from transfer-client metrics."""

    return RolloutTransferCounterSnapshot(
        total_polls=int(metrics["total_polls"]),
        successful_receives=int(metrics["successful_receives"]),
        failed_receives=int(metrics["failed_receives"]),
    )


def _rollout_transfer_metrics_for_logging(
    metrics: Mapping[str, Any],
    cumulative_counters: RolloutTransferCounters,
) -> dict[str, float | int]:
    """Format rollout transfer metrics for W&B logging."""

    return {
        "attempt_total_polls": int(metrics["total_polls"]),
        "attempt_successful_receives": int(metrics["successful_receives"]),
        "attempt_failed_receives": int(metrics["failed_receives"]),
        "total_polls": cumulative_counters.total_polls,
        "successful_receives": cumulative_counters.successful_receives,
        "failed_receives": cumulative_counters.failed_receives,
        "total_receive_bytes": int(metrics["total_receive_bytes"]),
        "receive_bytes": int(metrics["receive_bytes"]),
        "param_count": int(metrics["param_count"]),
        "largest_param_bytes": int(metrics["largest_param_bytes"]),
        "fetch_time": metrics["fetch_time"],
        "decode_time": metrics["decode_time"],
        "poll_time": metrics["poll_time"],
        "fetch_mib_per_second": metrics["fetch_mib_per_second"],
        "decode_mib_per_second": metrics["decode_mib_per_second"],
    }


class _NoOpTracker:
    """Minimal tracker that discards all metrics. Used when no tracker config is available."""

    def log(self, metrics, step=None):
        pass

    def finish(self):
        pass


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
        run_name = config.name or run_id
        self._run = wandb.init(
            entity=config.entity,
            project=config.project,
            name=run_name,
            tags=config.tags,
            id=run_id,
            resume="allow",
            mode=config.mode,
        )

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None):
        if step is None:
            self._run.log(metrics)
            return
        self._run.log(metrics, step=step)

    def finish(self):
        self._run.finish()


@dataclass
class RolloutWorkerConfig:
    """Configuration for RolloutWorker."""

    curriculum_config: CurriculumConfig
    rollout_storage: RolloutStorageConfig
    weight_transfer: WeightTransferConfig
    tokenizer: MarinTokenizer
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

    vocab_size: int | None = None
    """Vocab size for model construction. Should match the checkpoint's vocab dimension.
    If None, falls back to tokenizer.vocab_size."""

    system_prompt: str | None = None
    """System prompt to use for inference."""

    inflight_weight_updates: bool = False
    """Whether to use inflight weight updates."""

    worker_index: int = 0
    """Index of this worker among all rollout workers."""


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
        pass_at_one_for_current_group = 0.0
        avg_at_k_for_current_group = 0.0
        correct_flags: list[bool] = []
        for rollout in group.rollouts:
            rollout_stats_list.append(
                RolloutStats(
                    lesson_id=lesson_id,
                    episode_reward=rollout.episode_reward,
                    env_example_id=rollout.env_example_id,
                    temperature=rollout.temperature,
                    top_k=rollout.top_k,
                )
            )

            if rollout.correctness_reward is not None:
                avg_at_k_for_current_group += rollout.correctness_reward
                correct_flags.append(rollout.correctness_reward > 0.0)
            else:
                correct_flags.append(False)

            total_count += 1
            if rollout.episode_reward > 0:
                success_count += 1
            reward_sum += rollout.episode_reward

        if correct_flags:
            pass_at_k_for_current_group = pass_at_k_estimator(correct_flags, len(correct_flags))
            pass_at_one_for_current_group = pass_at_k_estimator(correct_flags, 1)

        pass_at_k += pass_at_k_for_current_group
        pass_at_one += pass_at_one_for_current_group

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
    """Create an inference context based on the configuration.

    IMPORTANT: When inflight_weight_updates=True with vLLM, this creates an
    AsyncvLLMInferenceContext which uses AsyncLLM. AsyncLLM always forks an
    EngineCore subprocess via os.fork(). If JAX's TPU runtime has been fully
    initialized before this fork (e.g., by calling jax.devices()), the forked
    subprocess will deadlock on TPU device locks (/dev/vfio). This function
    must be called BEFORE any code that initializes JAX on TPU.
    """
    if inference_type == "levanter":
        # Infer model_axis_size from the actual TPU configuration now that JAX is initialized.
        # For inference servers, we shard across all local devices on a single host.
        inference_config.inference_server_config = dataclasses.replace(
            inference_config.inference_server_config,
            trainer=dataclasses.replace(
                inference_config.inference_server_config.trainer, mesh=MeshConfig(axes={"data": 1, "model": -1})
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
        inference_config = prepare_vllm_inference_config_for_inflight(inference_config)
        return AsyncvLLMInferenceContext(
            inference_config=inference_config,
        )

    raise ValueError(f"Invalid inference type: {inference_type}")


def _should_run_curriculum_eval(
    *,
    current_train_step: int,
    last_eval_train_step: int | None,
    eval_frequency: int,
    worker_index: int,
) -> bool:
    """Return whether full eval should run for the current completed trainer step."""
    if eval_frequency <= 0:
        raise ValueError("eval_frequency must be positive")
    if worker_index != 0:
        return False
    if current_train_step < 0:
        return False
    if current_train_step % eval_frequency != 0:
        return False
    return current_train_step != last_eval_train_step


def _should_run_micro_eval(
    *,
    rollout_step: int,
    micro_eval_frequency: int | None,
    worker_index: int,
) -> bool:
    """Return whether micro-eval should run for the current rollout step."""
    if micro_eval_frequency is None:
        return False
    if micro_eval_frequency <= 0:
        raise ValueError("micro_eval_frequency must be positive when enabled")
    if worker_index != 0:
        return False
    if rollout_step <= 0:
        return False
    return rollout_step % micro_eval_frequency == 0


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
    _tokenizer: MarinTokenizer
    _environments: dict[str, MarinEnv]
    tracker: Any  # levanter.Tracker or RolloutTracker

    def __init__(self, config: RolloutWorkerConfig, runtime: RLRuntimeHandles):
        config.trainer.id = f"{config.run_id}-rollout"

        # Infer model_axis_size from the actual TPU configuration now that JAX is initialized.
        # For inference servers, we shard across all local devices on a single host.
        if config.inference_type == "levanter":
            try:
                self.tracker = levanter.current_tracker()
            except RuntimeError:
                # No global tracker set (e.g. in tests or standalone rollout workers)
                if config.tracker_config is not None:
                    self.tracker = RolloutTracker(config.tracker_config, config.run_id)
                else:
                    self.tracker = _NoOpTracker()
        else:
            # Initialize our own tracker to avoid JAX distributed initialization deadlocks.
            # Levanter's tracker calls jax.process_index() which forces JAX initialization.
            if config.tracker_config is not None:
                self.tracker = RolloutTracker(config.tracker_config, config.run_id)
            else:
                self.tracker = _NoOpTracker()

        self.config = config
        self._runtime = runtime
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()
        self._current_weight_step: int = -2
        self._current_train_step: int = -1
        self._last_transfer_counters = RolloutTransferCounterSnapshot()
        self._last_eval_train_step: int | None = None

        self._tokenizer = config.tokenizer

        # Event to signal that the first weight transfer has completed.
        # For inflight weight updates, we block inference until initial weights are received.
        self._first_weights_received = threading.Event()

        logger.info("Starting rollout policy context with weight transfer config %s", self.config.weight_transfer)

        self._rollout_writer = config.rollout_storage.create_writer()
        self._policy_ctx = create_inference_context(
            self.config.inference_type,
            self.config.inference_config,
            self.config.inflight_weight_updates,
        )

        # Need to build the policy model and then use that to start the inference server
        self._build_models()
        self._policy_ctx.start_server(self._policy_model)

        self._transfer_client = create_weight_transfer_client(
            config.weight_transfer,
            mesh=self._policy_ctx.mesh,
            axis_mapping=self._policy_ctx.axis_mapping,
            coordinator_handle=runtime.weight_transfer.arrow_flight_coordinator,
        )

        # TODO(power) -- replace this with a wait_until_ready() on the levanter inference server
        time.sleep(1.0)

        self._environments = {}
        self._curriculum_actor = runtime.curriculum

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
        self,
        lesson_id: str,
        n_examples: int,
        n_generations: int,
        mode: str,
        rng,
    ) -> tuple[RolloutBatch | None, dict | None]:
        """Sample a batch of rollouts from the environment for the given lesson ID."""
        env = self._load_environment(lesson_id)
        lesson_config = self.config.curriculum_config.lessons[lesson_id]

        # Get sampling params from lesson config
        temperature = lesson_config.sampling_params.temperature
        top_k = lesson_config.sampling_params.top_k
        stop_tokens = lesson_config.sampling_params.stop_tokens
        max_tokens = lesson_config.sampling_params.max_output_tokens

        rollout_groups, metrics = env.sample(
            inference_ctx=self._policy_ctx,
            n_examples=n_examples,
            n_generations=n_generations,
            temperature=temperature,
            prng_key=rng,
            mode=mode,
            max_tokens=max_tokens,
            top_k=top_k,
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
            vocab_size = self.config.vocab_size if self.config.vocab_size is not None else len(self._tokenizer)
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

            self._policy_model = None

    def stop(self):
        """Stop the inference worker loop and server."""
        with self._shutdown_condition:
            self._running = False
            if self._transfer_client is not None:
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

    def _check_run_state(self) -> bool:
        """Check if the RL run is still active. Returns False if should stop."""
        try:
            snapshot = self._runtime.run_state.get_snapshot.remote().result(timeout=5.0)
            self._current_train_step = snapshot.train_step
            if snapshot.status in ("completed", "failed"):
                logger.info("Run state is '%s', stopping rollout worker", snapshot.status)
                self._running = False
                return False
        except Exception:
            pass  # run_state check is best-effort; don't crash on transient failures
        return True

    def _sync_weights(self):
        """Attempt to receive updated weights, optionally waiting for them."""
        if self._transfer_client is None:
            raise RuntimeError("Parent-managed weight sync was requested without a transfer client")
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

            # Check if training finished while we're waiting for weights
            if not self._check_run_state():
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

    def _build_prompt_example_metrics(
        self,
        lesson_id: str,
        batch: RolloutBatch,
        step: int,
        eval_type: str = "eval",
    ) -> dict[str, Any]:
        """Build representative sample-table metrics from an evaluation batch.

        Args:
            lesson_id: ID of the evaluated lesson
            batch: The rollout batch containing samples
            step: Semantic training/weight step to include in the table rows
            eval_type: Either "eval" or "micro_eval"
        """
        if not batch or not batch.groups:
            return {}

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
            return {}

        table = wandb.Table(columns=["prompt", "response", "reward", "step"])
        for row in rows:
            table.add_data(row["prompt"], row["response"], row["reward"], row["step"])

        prefix = f"inference.{eval_type}/{lesson_id}"
        logger.info("Prepared %d eval samples for lesson %s at step %d", len(rows), lesson_id, step)
        return {f"{prefix}/sample_table": table}

    def _build_eval_metrics(
        self,
        prefix: str,
        lesson_id: str,
        batch: RolloutBatch,
        n_generations: int,
        temperature: float,
        top_k: int | None,
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
        metrics[f"{prefix}/{lesson_id}/temperature"] = temperature
        metrics[f"{prefix}/{lesson_id}/top_k"] = top_k if top_k is not None else -1
        return metrics

    def _log_lesson_eval(
        self,
        lesson_id: str,
        eval_type: Literal["eval", "micro_eval"],
        batch: RolloutBatch,
        step: int,
        weight_step: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """Log one completed evaluation batch."""
        log_metrics = self._build_prompt_example_metrics(
            lesson_id,
            batch,
            step,
            eval_type=eval_type,
        )
        log_metrics.update(metrics)
        log_metrics["inference.weight_step"] = weight_step
        if eval_type == "eval":
            log_metrics["inference.train_step"] = step
        self.tracker.log(log_metrics)
        logger.info(
            "Eval metrics for lesson %s at weight_step=%d: %s",
            lesson_id,
            weight_step,
            metrics,
        )

    def _evaluate_lesson(
        self,
        lesson_id: str,
        n_examples: int,
        eval_type: Literal["eval", "micro_eval"],
        rng,
        step: int,
    ) -> RolloutBatchStats:
        """Evaluate a single lesson and log metrics."""
        n_eval_generations = 1
        batch, _ = self._sample_batch(
            lesson_id=lesson_id,
            n_examples=n_examples,
            n_generations=n_eval_generations,
            mode="eval",
            rng=rng,
        )
        if batch is None:
            raise RuntimeError(f"Eval batch for lesson {lesson_id} produced no rollouts")

        stats = _compute_batch_stats(batch, lesson_id)
        sampling_params = self.config.curriculum_config.lessons[lesson_id].sampling_params
        metrics = self._build_eval_metrics(
            prefix=f"inference.{eval_type}",
            lesson_id=lesson_id,
            batch=batch,
            n_generations=n_eval_generations,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
        )
        self._log_lesson_eval(
            lesson_id=lesson_id,
            eval_type=eval_type,
            batch=batch,
            step=step,
            weight_step=self._current_weight_step,
            metrics=metrics,
        )
        if eval_type == "eval":
            self._curriculum_actor.update_lesson_stats.remote(
                stats.rollout_stats,
                mode="eval",
                current_step=step,
            ).result()
        return stats

    def _evaluate_curriculum(self, rng, step: int) -> None:
        """Evaluate all lessons and update the curriculum actor."""
        lesson_names = list(self.config.curriculum_config.lessons.keys())
        if not lesson_names:
            logger.info("No lessons to evaluate")
            return

        logger.info("Evaluating %d lessons", len(lesson_names))
        for lesson_id in lesson_names:
            self._evaluate_lesson(
                lesson_id=lesson_id,
                n_examples=self.config.curriculum_config.eval_n_examples,
                eval_type="eval",
                rng=rng,
                step=step,
            )

    def _resume_safe_transfer_metrics(self) -> dict[str, float | int]:
        current_metrics = self._transfer_client.get_metrics()
        current_counters = _rollout_transfer_counter_snapshot(current_metrics)
        delta = _rollout_transfer_counter_delta(current_counters, self._last_transfer_counters)
        cumulative_counters = self._runtime.run_state.add_rollout_transfer_counters.remote(
            self.config.worker_index,
            delta.total_polls,
            delta.successful_receives,
            delta.failed_receives,
        ).result()
        self._last_transfer_counters = current_counters
        return _rollout_transfer_metrics_for_logging(current_metrics, cumulative_counters)

    def _log_rollout_metrics(
        self,
        *,
        rollout_metrics: Mapping[str, Any],
        env_metrics: Mapping[str, Any] | None,
        throughput_metrics: Mapping[str, Any],
        rollout_step: int,
    ) -> None:
        log_metrics = dict(rollout_metrics)
        log_metrics.update(self._resume_safe_transfer_metrics())
        log_metrics.update(self._policy_ctx.get_metrics())
        log_metrics.update({f"env.{k}": v for k, v in (env_metrics or {}).items()})
        if hasattr(self._rollout_writer, "get_metrics"):
            log_metrics.update(self._rollout_writer.get_metrics())
        log_metrics = {"inference." + k: v for k, v in log_metrics.items()}
        log_metrics.update(throughput_metrics)
        log_metrics["inference.weight_step"] = self._current_weight_step
        if self._current_train_step >= 0:
            log_metrics["inference.train_step"] = self._current_train_step
        logger.info(
            "Logging metrics at rollout_step=%d weight_step=%d",
            rollout_step,
            self._current_weight_step,
        )
        self.tracker.log(log_metrics)

    def run(self):
        """Main inference worker loop."""
        faulthandler.enable()
        if hasattr(signal, "SIGUSR2"):
            faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)

        logger.info("Starting inference worker...")

        step = 0

        try:
            # For inflight weight updates, wait for first weights before generating rollouts
            if self.config.inflight_weight_updates:
                max_wait_time = self.config.weight_transfer.max_weight_transfer_wait_time
                if max_wait_time <= 0:
                    max_wait_time = 1200.0  # 20 minutes default for first-weight wait
                logger.info(
                    "Waiting for first weight transfer before starting inference (timeout %.1fs)...",
                    max_wait_time,
                )
                start_time = time.time()
                while True:
                    if self._first_weights_received.wait(timeout=10.0):
                        break

                    if not self._running:
                        logger.info("Shutdown requested while waiting for first weights")
                        return

                    elapsed = time.time() - start_time
                    if max_wait_time - elapsed <= 0:
                        raise RuntimeError("Timed out waiting for initial weight transfer.")

                    logger.info("Still waiting for first weight transfer (elapsed: %.1fs)", elapsed)
                logger.info("First weights received, starting inference loop")

            use_jax_rng = self.config.inference_type == "levanter"

            # Initialize RNG - use Python's random for vLLM to avoid JAX device access
            if use_jax_rng:
                rng = jax.random.PRNGKey(self.config.seed)
                rng = multihost_utils.broadcast_one_to_all(rng)
            else:
                py_rng = random.Random(self.config.seed)

            logger.info(f"Starting rollout worker with seed {self.config.seed}")

            while self._running:
                step_start = time.time()

                # Check if training is done before generating more rollouts
                if not self._check_run_state():
                    break

                # Synchronize weights on main thread unless using inflight weight updates
                if not self.config.inflight_weight_updates:
                    logger.info("PHASE: SYNC_WEIGHTS step=%d", step)
                    faulthandler.dump_traceback_later(600, repeat=False, file=sys.stderr, exit=True)
                    self._sync_weights()

                # Re-check after potentially long weight sync wait
                if not self._check_run_state():
                    break

                # Guard: never generate rollouts with dummy weights
                if self._current_weight_step < -1:
                    logger.warning(
                        "No valid weights received yet (weight_step=%d), retrying sync...",
                        self._current_weight_step,
                    )
                    faulthandler.cancel_dump_traceback_later()
                    time.sleep(5.0)
                    continue

                if self.config.max_rollouts is not None and step >= self.config.max_rollouts:
                    logger.info(f"Reached max rollouts ({self.config.max_rollouts}), stopping")
                    break

                if use_jax_rng:
                    # pyrefly: ignore[unbound-name] - rng is set in the matching use_jax_rng init branch above
                    rng, seed_key = jax.random.split(rng)
                    seed = int(seed_key[0])
                else:
                    # pyrefly: ignore[unbound-name] - py_rng is set in the matching else init branch above
                    seed = py_rng.randint(0, 2**31 - 1)

                try:
                    future = self._curriculum_actor.sample_lesson.remote(seed)
                    lesson_id = future.result()
                except Exception as e:
                    logger.warning(f"Failed to sample lesson from curriculum: {e}, will try again...")
                    time.sleep(10.0)
                    continue

                # Micro-eval: feedback on current lesson
                if _should_run_micro_eval(
                    rollout_step=step,
                    micro_eval_frequency=self.config.curriculum_config.micro_eval_frequency,
                    worker_index=self.config.worker_index,
                ):
                    if use_jax_rng:
                        # pyrefly: ignore[unbound-name] - rng set in matching init branch above
                        rng, micro_eval_rng = jrandom.split(rng)
                    else:
                        # pyrefly: ignore[unbound-name] - py_rng set in matching init branch above
                        micro_eval_rng = py_rng.randint(0, 2**31 - 1)
                    self._evaluate_lesson(
                        lesson_id=lesson_id,
                        n_examples=self.config.curriculum_config.micro_eval_n_examples,
                        eval_type="micro_eval",
                        rng=micro_eval_rng,
                        step=self._current_weight_step,
                    )

                # Full eval: comprehensive check on all lessons once per completed trainer step.
                if _should_run_curriculum_eval(
                    current_train_step=self._current_train_step,
                    last_eval_train_step=self._last_eval_train_step,
                    eval_frequency=self.config.curriculum_config.eval_frequency,
                    worker_index=self.config.worker_index,
                ):
                    if use_jax_rng:
                        # pyrefly: ignore[unbound-name] - rng set in matching init branch above
                        rng, eval_rng = jrandom.split(rng)
                    else:
                        # pyrefly: ignore[unbound-name] - py_rng set in matching init branch above
                        eval_rng = py_rng.randint(0, 2**31 - 1)
                    self._evaluate_curriculum(eval_rng, self._current_train_step)
                    self._last_eval_train_step = self._current_train_step

                logger.info(f"Sampled lesson '{lesson_id}' from curriculum")

                if use_jax_rng:
                    # pyrefly: ignore[unbound-name] - rng set in matching init branch above
                    rng, input_rng = jax.random.split(rng)
                else:
                    # pyrefly: ignore[unbound-name] - py_rng set in matching init branch above
                    input_rng = py_rng.randint(0, 2**31 - 1)

                lesson_config = self.config.curriculum_config.lessons[lesson_id]

                # Time the batch sampling for throughput metrics
                logger.info("PHASE: GENERATE step=%d lesson=%s", step, lesson_id)
                faulthandler.dump_traceback_later(1200, repeat=False, file=sys.stderr, exit=True)

                batch_start_time = time.time()
                rollout_batch, env_metrics = self._sample_batch(
                    lesson_id=lesson_id,
                    n_examples=lesson_config.sampling_params.n_prompts,
                    n_generations=lesson_config.sampling_params.n_generations_per_prompt,
                    mode="train",
                    rng=input_rng,
                )
                batch_time = time.time() - batch_start_time

                if rollout_batch is None:
                    faulthandler.cancel_dump_traceback_later()
                    continue

                # Count tokens for throughput calculation
                total_output_tokens = 0
                total_prompts = 0
                for group in rollout_batch.groups:
                    for rollout in group.rollouts:
                        total_output_tokens += len(rollout.response_tokens)
                        total_prompts += 1

                # Calculate throughput metrics
                throughput_metrics = {
                    "inference.throughput/tokens_per_second": total_output_tokens / batch_time if batch_time > 0 else 0,
                    "inference.throughput/requests_per_second": total_prompts / batch_time if batch_time > 0 else 0,
                    "inference.throughput/batch_time_seconds": batch_time,
                }

                logger.info("PHASE: WRITE_ROLLOUT step=%d", step)
                faulthandler.dump_traceback_later(300, repeat=False, file=sys.stderr, exit=True)

                self._rollout_writer.write_batch(rollout_batch)

                stats = _compute_batch_stats(rollout_batch, lesson_id)
                self._curriculum_actor.update_lesson_stats.remote(
                    stats.rollout_stats, mode="training", current_step=self._current_weight_step
                ).result()
                eval_metrics = self._build_eval_metrics(
                    prefix="rollout",
                    lesson_id=lesson_id,
                    batch=rollout_batch,
                    n_generations=lesson_config.sampling_params.n_generations_per_prompt,
                    temperature=lesson_config.sampling_params.temperature,
                    top_k=lesson_config.sampling_params.top_k,
                )

                step += 1

                logger.info("PHASE: IDLE step=%d elapsed=%.1fs", step, time.time() - step_start)
                faulthandler.cancel_dump_traceback_later()

                if self.config.log_freq > 0 and step % self.config.log_freq == 0:
                    self._log_rollout_metrics(
                        rollout_metrics=eval_metrics,
                        env_metrics=env_metrics,
                        throughput_metrics=throughput_metrics,
                        rollout_step=step,
                    )

            logger.info(f"Inference worker completed after generating {step} rollouts")
            if use_jax_rng:
                barrier_sync()

        except Exception:
            logger.exception("ROLLOUT WORKER CRASHED at step=%d, weight_step=%d", step, self._current_weight_step)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
            self._running = False
            try:
                if hasattr(self.tracker, "finish"):
                    self.tracker.finish()
            except Exception:
                logger.exception("Failed to finish tracker")
            self._shutdown_complete.set()
