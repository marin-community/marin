# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Training worker for RL/post-training tasks.

This worker reads rollout information from a queue which is populated by the
rollout workers, and periodically dumps new checkpoints to disk. These
checkpoints are read by the rollout workers to update their models.
"""

import dataclasses
import faulthandler
import logging
import signal
import sys
import time
from dataclasses import dataclass

import haliax as hax
import jax
import jax.random as jrandom
import levanter
import wandb
from levanter import callbacks
from levanter.checkpoint import (
    register_debug_checkpointer_state_provider,
    unregister_debug_checkpointer_state_provider,
)
from levanter.callbacks.tensorstore_callbacks import install_tensorstore_metrics_hook
from levanter.layers.attention import DEFAULT_SPLASH_BLOCK_SIZE, AttentionBackend
from levanter.models.flash_attention import BLOCK_SIZE as DEFAULT_FLASH_BLOCK_SIZE
from levanter.models.lm_model import LmConfig
from levanter.models.lm_model import LmHeadModel
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.tokenizers import MarinTokenizer

from marin.rl import weight_transfer
from marin.rl.curriculum import CurriculumConfig
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.objectives import ObjectiveRuntime, ObjectiveRuntimeConfig, ObjectiveSpec, build_objective_runtime
from marin.rl.runtime import RLRuntimeHandles
from marin.rl.weight_transfer import WeightTransferConfig

from .replay_buffer import ReplayBuffer, ReplayBufferConfig, ReplayDataLoader, StoredTrajectory
from .rollout_storage import RolloutStorageConfig
from .train_batch import create_sequence_batch_from_trajectories

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InitialRolloutState:
    """Startup rollout-sync state derived from the recovered trainer step."""

    weight_step: int
    published_train_step: int | None


def _resume_safe_weight_transfer_metrics(step: int, sync_interval_steps: int) -> dict[str, int]:
    """Return monotonic weight-transfer counters for a resumed trainer run.

    The trainer serves one bootstrap weight update at step `-1`, then serves one
    update every `sync_interval_steps` training steps starting at step `0`.
    These metrics are intended for run-global W&B charts, so they are derived
    from the restored trainer step instead of from process-local server state.
    """

    if step < 0:
        raise ValueError(f"weight transfer metrics require non-negative step, got {step}")
    if sync_interval_steps <= 0:
        raise ValueError(f"sync_interval_steps must be positive, got {sync_interval_steps}")
    if step % sync_interval_steps != 0:
        raise ValueError(
            "weight transfer hook ran at step "
            f"{step}, which is not aligned with sync_interval_steps={sync_interval_steps}"
        )

    successful_transfers = 2 + step // sync_interval_steps
    return {
        "total_transfers": successful_transfers,
        "successful_transfers": successful_transfers,
    }


def _initial_rollout_state(train_step: int) -> InitialRolloutState:
    """Return rollout startup semantics for a fresh or resumed trainer state.

    Fresh runs still use the historical bootstrap weight id ``-1``. Resumed runs
    reuse the latest completed trainer step as the startup weight id so manual
    relaunch and automatic retry converge on the same resume behavior.
    """

    if train_step < 0:
        raise ValueError(f"train_step must be non-negative, got {train_step}")

    if train_step == 0:
        return InitialRolloutState(weight_step=-1, published_train_step=None)

    return InitialRolloutState(weight_step=train_step, published_train_step=train_step)


@dataclass(frozen=True)
class BatchPrepTiming:
    """Timing breakdown for preparing one trainer batch."""

    fetch_time: float = 0.0
    batch_time: float = 0.0
    shard_time: float = 0.0

    @property
    def total_time(self) -> float:
        return self.fetch_time + self.batch_time + self.shard_time


def _training_step_timing_metrics(step_duration: float, batch_prep_timing: BatchPrepTiming) -> dict[str, float]:
    """Return RL trainer timing metrics with non-overlapping phase semantics.

    `step_duration` comes from Levanter and measures only the train-step compute path.
    Batch preparation happens outside that interval, so the correct end-to-end
    iteration time is `batch_prep_timing.total_time + step_duration`.
    """

    return {
        "throughput/train_step_duration_seconds": step_duration,
        "throughput/forward_backward_duration_seconds": step_duration,
        "throughput/rollout_wait_duration_seconds": batch_prep_timing.fetch_time,
        "throughput/batch_create_duration_seconds": batch_prep_timing.batch_time,
        "throughput/batch_shard_duration_seconds": batch_prep_timing.shard_time,
        "throughput/batch_prep_duration_seconds": batch_prep_timing.total_time,
        "throughput/iteration_duration_seconds": batch_prep_timing.total_time + step_duration,
    }


@dataclass
class TrainWorkerConfig:
    """Configuration for Levanter-based RL training worker."""

    rollout_storage: RolloutStorageConfig
    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    replay_buffer: ReplayBufferConfig
    weight_transfer: WeightTransferConfig
    curriculum_config: CurriculumConfig
    objective: ObjectiveSpec
    scorer_vocab_tile_size: int | None
    tokenizer: MarinTokenizer
    run_id: str

    initial_checkpoint: str | None = None
    """Initial checkpoint for the reference model (auto-detects HF repo vs local path)."""

    vocab_size: int | None = None
    """Vocab size for model construction. Should match the checkpoint's vocab dimension.
    If None, falls back to tokenizer.vocab_size."""

    seed: int = 0
    """Random seed for replay buffer sampling and model construction."""


class StreamingRolloutLoader:
    """Direct loader for streaming rollout data.

    Rollouts are a continous stream of data, not really well modeled by the
    default Levanter indexing API. Instead of implemented a Dataset, we
    implement the expected data loader interface directly.
    """

    config: TrainWorkerConfig

    def __init__(
        self,
        data_loader: ReplayDataLoader,
        config: TrainWorkerConfig,
        objective_runtime: ObjectiveRuntime,
    ):
        """Initialize the streaming rollout loader.

        Args:
            data_loader: The replay data loader to get rollouts from
            config: Train worker config with tokenizer and curriculum information
        """
        self.data_loader = data_loader
        self.config = config
        self.objective_runtime = objective_runtime
        self.timeout = 60.0

        # Get max_seq_len from curriculum (total sequence length for prompt + response)
        self.max_tokens = self.config.curriculum_config.max_seq_len

        is_splash = getattr(self.config.model, "attn_backend", None) == AttentionBackend.SPLASH
        flash_block_size = getattr(self.config.model, "flash_attention_block_size", None)

        if is_splash:
            self.pad_to_multiple = flash_block_size or DEFAULT_SPLASH_BLOCK_SIZE
        else:
            self.pad_to_multiple = flash_block_size or DEFAULT_FLASH_BLOCK_SIZE

        self.pad_token_id = self.config.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.config.tokenizer.eos_token_id

        # Track batch preparation timing for RL throughput diagnostics.
        self._last_batch_prep_timing = BatchPrepTiming()
        self._last_trajectories: list[StoredTrajectory] | None = None

    def __iter__(self):
        """Yield batches continuously from the replay buffer."""
        cumulative_wait = 0.0
        max_cumulative_wait = 3600.0  # 1 hour

        while True:
            fetch_start = time.time()
            trajectories = self.data_loader.get_trajectories(timeout=self.timeout)
            fetch_time = time.time() - fetch_start

            self._last_trajectories = trajectories

            if not trajectories:
                cumulative_wait += fetch_time
                if cumulative_wait >= max_cumulative_wait:
                    raise TimeoutError(f"No rollouts received after {cumulative_wait:.0f}s total wait")
                logger.warning(
                    "No rollouts received from data loader within timeout (cumulative_wait=%.0fs), retrying...",
                    cumulative_wait,
                )
                continue

            cumulative_wait = 0.0

            # Measure batch creation time
            batch_start = time.time()
            sequence_batch, batch_info = create_sequence_batch_from_trajectories(
                [trajectory.trajectory for trajectory in trajectories],
                self.max_tokens,
                self.pad_token_id,
                self.pad_to_multiple,
            )
            batch = self.objective_runtime.prepare_batch(sequence_batch, batch_info)
            batch_time = time.time() - batch_start

            # Measure sharding time
            shard_start = time.time()
            with hax.set_mesh(self.config.trainer.device_mesh):
                sharded_batch = hax.shard(batch, self.config.trainer.compute_axis_mapping)
            shard_time = time.time() - shard_start

            timing = BatchPrepTiming(fetch_time=fetch_time, batch_time=batch_time, shard_time=shard_time)
            self._last_batch_prep_timing = timing
            logger.info(
                "Batch prep: fetch=%.3fs, create=%.3fs, shard=%.3fs, total=%.3fs, trajectories=%d",
                fetch_time,
                batch_time,
                shard_time,
                timing.total_time,
                len(trajectories),
            )

            yield sharded_batch


class StopTrainerException(Exception):
    """Exception to signal stopping the trainer."""

    pass


class TrainWorker:
    """Training worker that reads rollout data from a queue and trains the model using Levanter."""

    config: TrainWorkerConfig
    replay_buffer: ReplayBuffer
    replay_loader: ReplayDataLoader
    transfer_server: weight_transfer.WeightTransferServer
    tokenizer: MarinTokenizer
    objective_runtime: ObjectiveRuntime
    initial_model: LmHeadModel | None
    reference_model: LmHeadModel | None

    def __init__(
        self,
        config: TrainWorkerConfig,
        runtime: RLRuntimeHandles,
    ):
        """Initialize training worker.

        Args:
            config: Training worker configuration with Levanter components.
            runtime: RLRuntimeHandles with actor handles for curriculum, run_state,
                     and weight transfer.
        """

        print("Run id: ", config.run_id)

        config.trainer.id = f"{config.run_id}-train"
        levanter.initialize(config.trainer)

        self.config = config
        self._runtime = runtime
        self._should_stop = False
        self.tokenizer = config.tokenizer
        self.objective_runtime = build_objective_runtime(
            ObjectiveRuntimeConfig(
                objective=config.objective,
                vocab_tile_size=config.scorer_vocab_tile_size,
            )
        )
        logger.info(
            "Using objective runtime backend=%s requirements=%s scorer_vocab_tile_size=%s",
            self.objective_runtime.score_source.backend_name,
            self.objective_runtime.score_requirements,
            config.scorer_vocab_tile_size,
        )

        self.rollout_reader = config.rollout_storage.create_reader()

        self.replay_buffer = ReplayBuffer.from_config(
            config=config.replay_buffer,
            local_batch_size=config.trainer.train_batch_size,
            total_processes=jax.process_count(),
            seed=config.seed,
        )

        self.replay_loader = ReplayDataLoader(
            rollout_reader=self.rollout_reader,
            replay_buffer=self.replay_buffer,
            rollout_fetch_interval=0.1,
        )

        self.data_loader = StreamingRolloutLoader(
            self.replay_loader,
            config,
            self.objective_runtime,
        )

        self.transfer_server = weight_transfer.create_weight_transfer_server(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
            coordinator_handle=runtime.weight_transfer.arrow_flight_coordinator,
        )

        self._curriculum_actor = runtime.curriculum
        checkpoint_dir = config.trainer.checkpointer.expanded_path(config.run_id)
        try:
            self._curriculum_actor.restore_checkpoint.remote(checkpoint_dir).result()
        except Exception as e:
            logger.warning("Failed to restore curriculum checkpoint from %s: %s, starting fresh", checkpoint_dir, e)

        logger.info("Connected to curriculum actor: %s", config.curriculum_config.actor_name)

        self._build_models()

    def _build_models(self):
        """Build the initial policy model and optional retained reference model."""
        config = self.config
        model_key = jrandom.PRNGKey(config.seed)
        vocab_size = config.vocab_size if config.vocab_size is not None else self.tokenizer.vocab_size
        Vocab = hax.Axis("vocab", vocab_size)

        if config.initial_checkpoint is not None:
            logger.info(f"Loading initial model from checkpoint: {config.initial_checkpoint}")
        else:
            logger.info("Building new model from scratch")

        def _load_model():
            return load_model_from_checkpoint(
                checkpoint=config.initial_checkpoint,
                model_config=config.model,
                trainer_config=config.trainer,
                vocab_axis=Vocab,
                tokenizer=self.tokenizer,
                mesh=config.trainer.device_mesh,
                axis_mapping=self.config.trainer.parameter_axis_mapping,
                key=model_key,
            )

        self.initial_model = _load_model()
        # Keep compatibility for callers that inspect the worker immediately after construction.
        # The zero-KL path clears this alias once trainer state has been materialized.
        self.reference_model = self.initial_model

    def _drop_bootstrap_model_references(self) -> None:
        """Release one-shot bootstrap model references once trainer state exists."""
        self.initial_model = None
        if not self._needs_reference_model():
            self.reference_model = None

    def _needs_reference_model(self) -> bool:
        """Return whether the active objective runtime needs a fixed reference model."""
        return self.objective_runtime.score_requirements.reference_logprobs

    def _seed_initial_rollout_state(self, rollout_state: InitialRolloutState) -> None:
        """Seed replay/run state before replay ingestion starts."""
        self.replay_buffer.set_current_step(rollout_state.weight_step)
        if rollout_state.published_train_step is not None:
            self._runtime.run_state.update_train_step.remote(rollout_state.published_train_step)

    def _wait_for_initial_rollouts(
        self,
        *,
        weight_step: int,
        max_wait_time: float = 7200.0,
        poll_interval: float = 5.0,
    ) -> bool:
        """Wait for startup rollouts for the current startup weight step.

        Args:
            weight_step: Weight step rollout workers should generate against before training continues.
            max_wait_time: Maximum time to wait in seconds (default: 2 hours).
                On Iris, rollout workers may take a long time to get scheduled.
            poll_interval: How often to check for rollouts in seconds (default: 5 seconds)

        Returns:
            True if initial rollouts were received, False if timeout
        """
        logger.info("Waiting for initial rollouts from step %d...", weight_step)
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            buffer_size = self.replay_buffer.size()
            if buffer_size > 0:
                elapsed = time.time() - start_time
                logger.info(
                    "Received initial rollouts for step %d! Buffer size: %d (waited %.1fs)",
                    weight_step,
                    buffer_size,
                    elapsed,
                )
                return True

            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed > 0:  # Log every 10 seconds
                logger.info(
                    "Still waiting for initial rollouts from step %d (elapsed: %.0fs, buffer size: %d)",
                    weight_step,
                    elapsed,
                    buffer_size,
                )

            time.sleep(poll_interval)

        logger.warning("Timeout waiting for initial rollouts from step %d after %.1fs", weight_step, max_wait_time)
        return False

    def _checkpoint_debug_snapshot(self) -> dict[str, object]:
        replay_stats = self.replay_buffer.get_stats()
        replay_stats["current_step"] = self.replay_buffer._current_step

        transfer_snapshot: dict[str, object] = {}
        if hasattr(self.transfer_server, "get_debug_snapshot"):
            maybe_snapshot = self.transfer_server.get_debug_snapshot()
            transfer_snapshot = dict(maybe_snapshot)

        return {
            "replay_buffer": replay_stats,
            "weight_transfer": transfer_snapshot,
        }

    def train(self):
        """Main training method using Levanter's standard train_lm infrastructure."""
        faulthandler.enable()
        debug_checkpointer = self.config.trainer.checkpointer.debug.enabled
        debug_weight_transfer = self.config.weight_transfer.debug_weight_transfer
        if (debug_checkpointer or debug_weight_transfer) and hasattr(signal, "SIGUSR2"):
            faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        logger.info("Starting objective-driven RL training with Levanter...")

        checkpoint_debug_provider_name = f"{self.config.run_id}-checkpoint-debug"
        if debug_checkpointer:
            register_debug_checkpointer_state_provider(
                checkpoint_debug_provider_name,
                self._checkpoint_debug_snapshot,
            )

        try:
            config = self.config
            optimizer = config.optimizer.build(config.trainer.num_train_steps)
            loss_fn = self.objective_runtime.create_loss_fn(reference_model=self.reference_model)

            @jax.jit
            def _loss_function(model, batch, key):
                return loss_fn(model, batch, key)

            with Trainer(config=config.trainer, optimizer=optimizer, loss_fn=_loss_function) as trainer:
                if debug_checkpointer:
                    install_tensorstore_metrics_hook(trainer, every=1)
                _, training_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 2)
                state = trainer.initial_state(training_key, model=self.initial_model)
                self._drop_bootstrap_model_references()
                startup_rollout_state = _initial_rollout_state(int(state.step))
                logger.info(
                    "Trainer recovered state.step=%d; startup rollout weight_step=%d",
                    int(state.step),
                    startup_rollout_state.weight_step,
                )
                self._seed_initial_rollout_state(startup_rollout_state)

                if debug_weight_transfer:
                    logger.info(
                        "Weight transfer debug enabled: sync_interval_steps=%d, "
                        "mesh_shape=%s, parameter_axis_mapping=%s",
                        self.config.weight_transfer.sync_interval_steps,
                        config.trainer.device_mesh.devices.shape,
                        config.trainer.parameter_axis_mapping,
                    )

                with self.replay_loader:
                    # Always transfer startup weights to rollout workers before we attempt to train.
                    self.transfer_server.serve_weights(startup_rollout_state.weight_step, state.model)

                    # Wait for startup rollouts so both fresh runs and resumed runs begin with
                    # rollouts that match the currently served weights.
                    if not self._wait_for_initial_rollouts(weight_step=startup_rollout_state.weight_step):
                        raise RuntimeError("Timed out waiting for initial rollouts; aborting training.")

                    self._configure_training_hooks(trainer)
                    try:
                        trainer.train(state, self.data_loader)
                    except StopTrainerException:
                        pass
        except StopTrainerException:
            pass
        except Exception:
            logger.exception("TRAIN WORKER CRASHED")
            raise
        finally:
            if debug_checkpointer:
                unregister_debug_checkpointer_state_provider(checkpoint_debug_provider_name)
            try:
                self.stop()
            except Exception:
                logger.exception("Failed to stop train worker during cleanup")

    def _configure_training_hooks(self, trainer):
        def _weight_transfer_hook(info: levanter.callbacks.StepInfo):
            self.weight_transfer_hook(trainer, info)

        trainer.add_hook(
            _weight_transfer_hook,
            every=self.config.weight_transfer.sync_interval_steps,
        )

        def _update_current_step(info: levanter.callbacks.StepInfo):
            self._record_train_step(info.step)

        trainer.add_hook(_update_current_step, every=1)

        def _stop_on_signal(info: levanter.callbacks.StepInfo):
            if self._should_stop:
                raise StopTrainerException()

        trainer.add_hook(_stop_on_signal, every=1)

        # Log training step timing for RL analysis
        def _log_step_timing(info: levanter.callbacks.StepInfo):
            batch_prep_timing = self.data_loader._last_batch_prep_timing
            metrics = _training_step_timing_metrics(info.step_duration, batch_prep_timing)
            metrics["train/loss"] = float(info.loss)
            trainer.tracker.log(metrics, step=info.step)
            logger.info(
                "Training step %d completed: train_step=%.2fs, rollout_wait=%.2fs, "
                "batch_create=%.2fs, batch_shard=%.2fs, iteration=%.2fs, loss=%.4f",
                info.step,
                metrics["throughput/train_step_duration_seconds"],
                metrics["throughput/rollout_wait_duration_seconds"],
                metrics["throughput/batch_create_duration_seconds"],
                metrics["throughput/batch_shard_duration_seconds"],
                metrics["throughput/iteration_duration_seconds"],
                info.loss,
            )

        trainer.add_hook(_log_step_timing, every=1)

        def _log_samples_hook(info: levanter.callbacks.StepInfo):
            trajectories = self.data_loader._last_trajectories
            if trajectories is not None:
                self._log_samples(trainer, info.step, trajectories)

        trainer.add_hook(_log_samples_hook, every=1)

        # Add MFU (Model FLOPs Utilization) logging
        vocab_size = self.config.vocab_size if self.config.vocab_size is not None else self.tokenizer.vocab_size
        tokens_per_example = self.config.curriculum_config.max_seq_len
        flops_per_token = self.config.model.flops_per_token(vocab_size, tokens_per_example)
        flops_per_example = 3 * flops_per_token * tokens_per_example if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(
                tokens_per_example=tokens_per_example,
                batch_schedule=self.config.trainer.train_batch_size,
                flops_per_example=flops_per_example,
                prefix="throughput",
            ),
            every=1,
        )

        def _curriculum_checkpoint_hook(info: levanter.callbacks.StepInfo):
            checkpoint_dir = self.config.trainer.checkpointer.expanded_path(self.config.run_id)
            try:
                self._curriculum_actor.save_checkpoint.remote(checkpoint_dir).result()
            except Exception as e:
                logger.error(f"Failed to save curriculum checkpoint: {e}")

        trainer.add_hook(_curriculum_checkpoint_hook, every=self.config.curriculum_config.checkpoint_steps)

    def _record_train_step(self, step: int) -> None:
        """Publish the latest completed trainer step to local and shared state."""
        self.replay_buffer.set_current_step(step)
        self._runtime.run_state.update_train_step.remote(step)

    def weight_transfer_hook(self, trainer: Trainer, info: levanter.callbacks.StepInfo):
        step = info.step
        state = info.state

        logger.info(
            "Transferring weights at step %d, loss=%s",
            step,
            info.loss,
        )

        model_params = state.model

        # Measure weight transfer time
        transfer_start = time.time()
        self.transfer_server.serve_weights(step, model_params)
        transfer_time = time.time() - transfer_start

        attempt_metrics = dataclasses.asdict(self.transfer_server.get_metrics())
        metrics = {f"weight_transfer/attempt_{k}": v for k, v in attempt_metrics.items()}
        metrics.update(
            {
                f"weight_transfer/{k}": v
                for k, v in _resume_safe_weight_transfer_metrics(
                    step=step,
                    sync_interval_steps=self.config.weight_transfer.sync_interval_steps,
                ).items()
            }
        )
        metrics["weight_transfer/serve_time_seconds"] = transfer_time

        trainer.tracker.log(metrics, step=step)
        logger.info("Successfully transferred weights with ID %d (transfer_time=%.2fs)", step, transfer_time)

    def _log_samples(self, trainer, step, trajectories: list[StoredTrajectory]):
        """Log trainer samples for the first 5 prompts to wandb table."""
        # group by prompt
        prompts = {}
        for stored_trajectory in trajectories:
            trajectory = stored_trajectory.trajectory
            pid = trajectory.env_example_id
            if pid not in prompts:
                prompts[pid] = []
            prompts[pid].append(trajectory)

        # take first 5 prompts
        first_5_pids = list(prompts.keys())[:5]

        columns = ["step", "prompt_id", "prompt", "response", "reward"]
        data = []
        for pid in first_5_pids:
            prompt_trajectories = prompts[pid]
            prompt_text = self.tokenizer.decode(prompt_trajectories[0].prompt_tokens, skip_special_tokens=False)
            for trajectory in prompt_trajectories:
                response_text = self.tokenizer.decode(trajectory.response_tokens, skip_special_tokens=False)
                data.append([step, pid, prompt_text, response_text, float(trajectory.episode_reward)])

        table = wandb.Table(columns=columns, data=data)
        trainer.tracker.log({"train/samples": table}, step=step)

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True
        self.transfer_server.cleanup()
