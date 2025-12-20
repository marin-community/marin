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
Training worker for RL/post-training tasks.

This worker reads rollout information from a queue which is populated by the
rollout workers, and periodically dumps new checkpoints to disk. These
checkpoints are read by the rollout workers to update their models.
"""

import dataclasses
import logging
import time
from dataclasses import dataclass

import haliax as hax
import jax
import jax.random as jrandom
import levanter
from levanter import callbacks
from levanter.layers.attention import AttentionBackend
from levanter.models.flash_attention import BLOCK_SIZE as DEFAULT_FLASH_BLOCK_SIZE
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from transformers import PreTrainedTokenizer

from marin.rl import weight_transfer
from marin.rl.curriculum import CurriculumConfig, get_or_create_curriculum_actor
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.weight_transfer import WeightTransferConfig

from .replay_buffer import ReplayBuffer, ReplayBufferConfig, ReplayDataLoader
from .rl_losses import RLLossModule
from .rollout_storage import RolloutStorageConfig
from .train_batch import create_training_batch_from_rollouts

logger = logging.getLogger(__name__)


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
    loss: RLLossModule
    tokenizer: PreTrainedTokenizer
    run_id: str

    initial_checkpoint: str | None = None
    """Initial checkpoint for the reference model (auto-detects HF repo vs local path)."""

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
    ):
        """Initialize the streaming rollout loader.

        Args:
            data_loader: The replay data loader to get rollouts from
            config: Train worker config with tokenizer and curriculum information
        """
        self.data_loader = data_loader
        self.config = config
        self.timeout = 60.0

        # Get max_seq_len from curriculum (total sequence length for prompt + response)
        self.max_tokens = self.config.curriculum_config.max_seq_len

        flash_block_size = getattr(self.config.model, "flash_attention_block_size", None)
        if flash_block_size is None:
            flash_block_size = DEFAULT_FLASH_BLOCK_SIZE

        # Splash attention requires multiples of 128. Flash attention usually uses 1024 or 512.
        # If we're using splash, we should pad to at least 128 to satisfy the TPU kernel requirements.
        is_splash = getattr(self.config.model, "attn_backend", None) == AttentionBackend.SPLASH
        if is_splash:
            self.pad_to_multiple = 128
        else:
            self.pad_to_multiple = flash_block_size

        self.pad_token_id = self.config.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.config.tokenizer.eos_token_id

        # Track batch prep time for forward/backward calculation
        self._last_batch_prep_time: float = 0.0

    def __iter__(self):
        """Yield batches continuously from the replay buffer."""
        while True:
            # Measure time to get rollouts from replay buffer
            fetch_start = time.time()
            rollouts = self.data_loader.get_rollouts(timeout=self.timeout)
            fetch_time = time.time() - fetch_start

            if not rollouts:
                logger.warning("No rollouts received from data loader within timeout, retrying...")
                continue

            # Measure batch creation time
            batch_start = time.time()
            batch = create_training_batch_from_rollouts(
                rollouts, self.max_tokens, self.pad_token_id, self.pad_to_multiple
            )
            batch_time = time.time() - batch_start

            # Measure sharding time
            shard_start = time.time()
            with hax.set_mesh(self.config.trainer.device_mesh):
                sharded_batch = hax.shard(batch, self.config.trainer.compute_axis_mapping)
            shard_time = time.time() - shard_start

            total_time = fetch_time + batch_time + shard_time
            self._last_batch_prep_time = total_time
            logger.info(
                "Batch prep: fetch=%.3fs, create=%.3fs, shard=%.3fs, total=%.3fs, rollouts=%d",
                fetch_time,
                batch_time,
                shard_time,
                total_time,
                len(rollouts),
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
    tokenizer: PreTrainedTokenizer
    loss_module: RLLossModule

    def __init__(
        self,
        config: TrainWorkerConfig,
    ):
        """Initialize training worker.

        Args:
            config: Training worker configuration with Levanter components.
        """

        print("Run id: ", config.run_id)

        config.trainer.id = f"{config.run_id}-train"
        levanter.initialize(config.trainer)

        self.config = config
        self._should_stop = False
        self.tokenizer = config.tokenizer
        self.loss_module = config.loss

        self.rollout_reader = config.rollout_storage.create_reader()

        self.replay_buffer = ReplayBuffer.from_config(
            config=config.replay_buffer,
            local_batch_size=config.trainer.train_batch_size,
            total_processes=jax.process_count(),
            loss_module=self.loss_module,
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
        )
        self.transfer_server = weight_transfer.create_weight_transfer_server(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

        # Create curriculum actor with auto-restore from checkpoint
        checkpoint_dir = config.trainer.checkpointer.expanded_path(config.run_id)
        self._curriculum_actor = get_or_create_curriculum_actor(
            self.config.curriculum_config, checkpoint_path=checkpoint_dir
        )

        logger.info("Connected to curriculum actor: %s", config.curriculum_config.actor_name)

        self._build_models()

    def _build_models(self):
        """Build reference and initial policy models."""
        config = self.config
        model_key = jrandom.PRNGKey(config.seed)
        Vocab = hax.Axis("vocab", self.tokenizer.vocab_size)

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

        self.reference_model = _load_model()

    def _wait_for_initial_rollouts(self, max_wait_time: float = 1200.0, poll_interval: float = 5.0) -> bool:
        """Wait for initial rollouts from step -1 to be received.

        Args:
            max_wait_time: Maximum time to wait in seconds (default: 20 minutes)
            poll_interval: How often to check for rollouts in seconds (default: 5 seconds)

        Returns:
            True if initial rollouts were received, False if timeout
        """
        logger.info("Waiting for initial rollouts from step -1...")
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            buffer_size = self.replay_buffer.size()
            if buffer_size > 0:
                elapsed = time.time() - start_time
                logger.info(f"Received initial rollouts! Buffer size: {buffer_size} (waited {elapsed:.1f}s)")
                return True

            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed > 0:  # Log every 10 seconds
                logger.info(f"Still waiting for initial rollouts (elapsed: {elapsed:.0f}s, buffer size: {buffer_size})")

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for initial rollouts after {max_wait_time}s")
        return False

    def train(self):
        """Main training method using Levanter's standard train_lm infrastructure."""
        logger.info("Starting RLOO training with Levanter...")

        config = self.config
        optimizer = config.optimizer.build(config.trainer.num_train_steps)
        loss_fn = self.loss_module.create_loss_fn(self.reference_model, None)

        @jax.jit
        def _loss_function(model, batch, key):
            return loss_fn(model, batch, key)

        with (
            Trainer(config=config.trainer, optimizer=optimizer, loss_fn=_loss_function) as trainer,
            self.replay_loader,
        ):
            _, training_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 2)
            state = trainer.initial_state(training_key, model=self.reference_model)

            # Always transfer initial weights to rollout workers before we attempt to start training
            self.transfer_server.serve_weights(-1, state.model)
            self.replay_buffer.set_current_step(-1)

            # Wait for initial rollouts to ensure we have baseline measurements
            self._wait_for_initial_rollouts()

            self._configure_training_hooks(trainer)

            try:
                trainer.train(state, self.data_loader)
            except StopTrainerException:
                pass

    def _configure_training_hooks(self, trainer):
        def _weight_transfer_hook(info: levanter.callbacks.StepInfo):
            self.weight_transfer_hook(trainer, info)

        trainer.add_hook(
            _weight_transfer_hook,
            every=self.config.weight_transfer.sync_interval_steps,
        )

        def _update_current_step(info: levanter.callbacks.StepInfo):
            self.replay_buffer.set_current_step(info.step)

        trainer.add_hook(_update_current_step, every=1)

        def _stop_on_signal(info: levanter.callbacks.StepInfo):
            if self._should_stop:
                raise StopTrainerException()

        trainer.add_hook(_stop_on_signal, every=1)

        # Log training step timing for RL analysis
        def _log_step_timing(info: levanter.callbacks.StepInfo):
            # Get batch prep time from the data loader
            batch_prep_time = self.data_loader._last_batch_prep_time

            # Forward/backward = total step duration - batch prep time
            forward_backward_duration = max(0.0, info.step_duration - batch_prep_time)

            metrics = {
                "throughput/step_duration_seconds": info.step_duration,
                "throughput/batch_prep_duration_seconds": batch_prep_time,
                "throughput/forward_backward_duration_seconds": forward_backward_duration,
                "train/loss": float(info.loss),
            }
            trainer.tracker.log(metrics, step=info.step)
            logger.info(
                "Training step %d completed: duration=%.2fs (batch_prep=%.2fs, fwd_bwd=%.2fs), loss=%.4f",
                info.step,
                info.step_duration,
                batch_prep_time,
                forward_backward_duration,
                info.loss,
            )

        trainer.add_hook(_log_step_timing, every=1)

        # Add MFU (Model FLOPs Utilization) logging
        vocab_size = len(self.tokenizer)
        flops_per_token = self.config.model.flops_per_token(vocab_size)
        tokens_per_example = self.config.curriculum_config.max_seq_len
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
                self._curriculum_actor.save_checkpoint.call(checkpoint_dir)
            except Exception as e:
                logger.error(f"Failed to save curriculum checkpoint: {e}")

        trainer.add_hook(_curriculum_checkpoint_hook, every=self.config.curriculum_config.checkpoint_steps)

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

        metrics = {f"weight_transfer/{k}": v for k, v in dataclasses.asdict(self.transfer_server.get_metrics()).items()}
        metrics["weight_transfer/serve_time_seconds"] = transfer_time

        trainer.tracker.log(metrics, step=step)
        logger.info("Successfully transferred weights with ID %d (transfer_time=%.2fs)", step, transfer_time)

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True
        self.transfer_server.cleanup()
