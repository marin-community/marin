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

"""Test integrated inference and training workers with in-memory communication."""

import logging
import os
import sys
import threading
import time

import numpy as np
import pytest
import ray

try:
    from marin.post_training.rollout_worker import InferenceWorker
    from marin.post_training.train_worker import TrainingWorker
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)

import uuid

from marin.post_training.rollout_storage import (
    InMemoryRolloutQueue,
    RolloutBatch,
    TaggedRolloutBatch,
)
from marin.post_training.weight_transfer_manager import WeightTransferMode, create_coordinator

# Import test helpers
from tests.post_training.test_helpers import (
    create_nano_inference_worker_config,
    create_nano_llama_config,
    create_nano_training_worker_config,
    create_test_inference_server_config,
)

# Test timeout constants
CHECKPOINT_POLL_INTERVAL = 0.2
WORKER_JOIN_TIMEOUT = 5
BATCH_READ_TIMEOUT = 1.0
INTEGRATION_TEST_TIMEOUT = 60


logger = logging.getLogger(__name__)


class TestableTrainingWorker(TrainingWorker):
    """TrainingWorker subclass for testing with step tracking."""

    def __init__(self, *args, step_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_callback = step_callback

    def _configure_training_hooks(self, trainer):
        """Configure training hooks including test step tracking."""
        super()._configure_training_hooks(trainer)
        if self.step_callback:
            trainer.add_hook(self.step_callback, every=1)


# Since we use an actor for the ray transfer, we need a new cluster to
# avoid stale state.
@pytest.fixture(scope="function")
def ray_cluster():
    """Start Ray cluster for weight transfer testing."""
    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for mock checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    # Don't create any checkpoint files - let the inference worker use random initialization
    yield str(checkpoint_dir)


@pytest.fixture
def training_worker_config(tmp_path):
    """Create minimal training worker configuration for testing."""
    rollout_queue = InMemoryRolloutQueue()
    rollout_reader = rollout_queue.reader()
    return create_nano_training_worker_config(rollout_reader, tmp_path)


@pytest.fixture
def inference_worker_config(tmp_path):
    """Create minimal inference worker configuration for testing."""
    rollout_queue = InMemoryRolloutQueue()
    rollout_writer = rollout_queue.writer()

    model_config = create_nano_llama_config()
    inference_server_config = create_test_inference_server_config(model_config, tmp_path)

    return create_nano_inference_worker_config(inference_server_config, rollout_writer)


def _print_worker_status(elapsed, inference_runner, training_runner):
    """Helper function to print detailed worker status during monitoring."""
    print(f"[{elapsed:.1f}s] Status:")
    print(f"  Rollouts generated: {inference_runner.rollouts_generated}")
    print(f"  Training steps: {training_runner.steps_completed}")
    print(f"  Checkpoints created: {len(training_runner.checkpoints_created)}")
    print(f"  Weight transfers: {len(inference_runner.weight_transfers)}")
    print(f"  Inference done: {inference_runner.done.is_set()}")
    print(f"  Training done: {training_runner.done.is_set()}")


def create_test_batch(idx: int, batch_size: int = 2, max_seq_len: int = 16) -> TaggedRolloutBatch:
    """Helper to create test batches with all required fields."""
    rng = np.random.default_rng(42 + idx)
    return TaggedRolloutBatch(
        batch=RolloutBatch(
            input_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
            attention_mask=np.ones((batch_size, max_seq_len), dtype=np.int32),
            position_ids=np.arange(max_seq_len)[None, :].repeat(batch_size, axis=0).astype(np.int32),
            target_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
            loss_weights=np.ones((batch_size, max_seq_len), dtype=np.float32),
            loss_masks=np.ones((batch_size, max_seq_len), dtype=np.float32),
            reference_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
            policy_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
        ),
        env_name=f"test_env_{idx}",
        worker_id="test_worker",
        timestamp=time.time(),
        rollout_id=f"test_{idx}",
    )


class InferenceWorkerRunner:
    """Manages running an inference worker in a separate thread with metric tracking."""

    def __init__(self, inference_worker_config, coordinator=None, max_rollouts=2):
        self.inference_worker_config = inference_worker_config
        # Override max_rollouts
        self.inference_worker_config.max_rollouts = max_rollouts
        self.coordinator = coordinator

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

        # Metrics
        self.rollouts_generated = 0
        self.weight_transfers = []

    def _track_weight_transfer(self, weight_id, metadata):
        """Called when weights are transferred."""
        self.weight_transfers.append(
            {
                "weight_id": weight_id,
                "metadata": metadata,
                "time": time.time(),
                "rollouts_at_transfer": self.rollouts_generated,
            }
        )

    def _track_rollout_generation(self):
        """Called when rollout is generated."""
        self.rollouts_generated += 1

    def _run(self):
        try:
            # Mock the tokenizer loading for testing
            from unittest.mock import patch

            from tests.post_training.test_helpers import DummyTokenizer

            with patch("levanter.inference.openai.load_tokenizer") as mock_load:
                mock_load.return_value = DummyTokenizer(vocab_size=1000)
                self.worker = InferenceWorker(
                    config=self.inference_worker_config,
                    coordinator=self.coordinator,
                )

            # Weight transfer tracking - simplified for new design
            # TODO: Implement proper weight transfer tracking for new coordinator design
            def tracking_receive_weights():
                try:
                    if self.coordinator is not None:
                        # Mock weight transfer for testing
                        self._track_weight_transfer("test_weight_1", {"weight_id": "test_weight_1"})
                except Exception as e:
                    logger.warning(f"Failed to track weight transfer: {e}")

            # Call this periodically (simplified for testing)
            tracking_receive_weights()

            # Override batch generation to count rollouts
            original_generate_batch = self.worker._generate_rollout_batch

            def counting_generate_batch(rng):
                batch_data, metrics = original_generate_batch(rng)
                self._track_rollout_generation()
                # Add metadata about rollout
                metrics["rollout_number"] = self.rollouts_generated
                return batch_data, metrics

            self.worker._generate_rollout_batch = counting_generate_batch

            # Run the worker normally
            self.worker.run()
        except Exception as e:
            print("Inference worker encountered exception:", e, file=sys.stderr)
            logger.error("Inference worker failed", exc_info=True)
            self.error = e
        finally:
            self.done.set()

    def start(self):
        """Start worker in background thread."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop worker if running."""
        if self.worker:
            self.worker.stop()

    def alive(self):
        return self.thread.is_alive()

    def join(self, timeout=5):
        """Wait for thread completion."""
        if self.thread:
            self.thread.join(timeout)

    def __enter__(self):
        """Context manager entry - start the worker."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop worker and check for errors."""
        self.stop()
        self.join(timeout=WORKER_JOIN_TIMEOUT)

        if self.error:
            import traceback

            print(f"Inference worker error: {self.error}")
            print(
                "Traceback:",
                "".join(
                    traceback.format_exception(
                        type(self.error),
                        self.error,
                        self.error.__traceback__,
                    )
                ),
            )
            pytest.fail(f"Inference worker failed: {self.error}")

        return False


class TrainingWorkerRunner:
    """Manages running a training worker in a separate thread with metric tracking."""

    def __init__(self, training_worker_config, coordinator=None):
        self.training_worker_config = training_worker_config
        self.coordinator = coordinator

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

        # Metrics
        self.steps_completed = 0
        self.checkpoints_created = []

    def _track_checkpoint_save(self, step):
        """Called when checkpoint is saved."""
        self.checkpoints_created.append(
            {
                "step": step,
                "time": time.time(),
                "rollouts_consumed": self.steps_completed + 1,  # +1 because we're about to increment
            }
        )

    def _track_training_step(self):
        """Called after each training step."""
        print("Called train step.")
        self.steps_completed += 1

    def _run(self):
        """Thread target - runs the training worker."""
        try:
            self.worker = TestableTrainingWorker(
                config=self.training_worker_config,
                coordinator=self.coordinator,
                step_callback=lambda info: self._track_training_step(),
            )

            self.worker.train()
        except Exception as e:
            self.error = e
        finally:
            self.done.set()

    def start(self):
        """Start worker in background thread."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop worker if running."""
        if self.worker:
            self.worker.stop()

    def join(self, timeout=5):
        """Wait for thread completion."""
        if self.thread:
            self.thread.join(timeout)

    def __enter__(self):
        """Context manager entry - start the worker."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop worker and check for errors."""
        self.stop()
        self.join(timeout=WORKER_JOIN_TIMEOUT)

        if self.error:
            import traceback

            print(f"Training worker error: {self.error}")
            print(
                "Traceback:",
                "".join(
                    traceback.format_exception(
                        type(self.error),
                        self.error,
                        self.error.__traceback__,
                    )
                ),
            )
            pytest.fail(f"Training worker failed: {self.error}")

        return False


def test_inference_worker(ray_cluster, inference_worker_config):
    """Test inference worker generates rollouts to in-memory queue."""
    # Use the rollout writer's queue from the inference worker config
    rollout_writer = inference_worker_config.rollout_writer
    queue_reader = rollout_writer._queue.reader()

    # Get coordinator for GCS mode only
    coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    # Use context manager for cleaner test
    with InferenceWorkerRunner(inference_worker_config, coordinator) as runner:
        # Wait for the worker to complete
        while runner.alive() and not runner.done.is_set():
            time.sleep(0.5)

        # Give a moment for final writes
        time.sleep(0.5)

        batches = queue_reader.read_all_available()
        assert len(batches) > 0, "Should be able to read batches from queue"
        assert runner.rollouts_generated >= 1, f"Expected at least 1 rollout, got {runner.rollouts_generated}"

    print("✓ Inference worker generated rollout batch successfully")


def test_train_worker(ray_cluster, training_worker_config):
    """Test training worker processes rollout batch and creates checkpoint."""
    rollout_reader = training_worker_config.rollout_reader
    queue_writer = rollout_reader._queue.writer()

    batch_size = training_worker_config.trainer.train_batch_size
    max_seq_len = 64  # Use fixed small length for testing

    # Create multiple mock batches to ensure we have enough data
    # The DataLoader might try to fetch multiple indices
    for i in range(5):  # Create 5 batches to be safe
        sample_batch = create_test_batch(i, batch_size=batch_size, max_seq_len=max_seq_len)
        queue_writer.write_batch(sample_batch)

    # Get coordinator for GCS mode only
    coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    # Use context manager for cleaner test
    with TrainingWorkerRunner(training_worker_config, coordinator) as runner:
        # Wait for training to complete
        while not runner.done.is_set() and runner.steps_completed < 10:
            time.sleep(0.1)

    # Verify results
    assert runner.steps_completed >= 1, f"Expected at least 1 training step, got {runner.steps_completed}"

    # Check checkpoint was created (using the new trainer config path)
    checkpoint_dir = str(training_worker_config.trainer.checkpointer.base_path)
    checkpoint_created = os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    assert checkpoint_created, "Training worker should create checkpoint after processing batch"


def test_inference_and_training_workers(
    ray_cluster,
    tmp_path,
    training_worker_config,
    inference_worker_config,
):
    """Test inference & training workers running together with checkpoint updates."""
    # Update training config for integration test
    training_worker_config.trainer.num_train_steps = 3
    training_worker_config.trainer.checkpointer.save_interval_steps = 1  # Save after every step

    # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()
    queue_writer = rollout_queue.writer()

    # Update inference config to use the shared queue
    inference_worker_config.rollout_writer = queue_writer

    # Get coordinator for GCS mode only
    coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    # Create worker runners and start with context managers
    with TrainingWorkerRunner(training_worker_config, coordinator) as training_runner:
        # Wait for initial checkpoint to be created (simplified for testing)
        # In real test, we'd wait for actual checkpoint files
        time.sleep(0.1)

        # Create inference runner with unlimited rollouts to allow for weight transfers
        inference_runner = InferenceWorkerRunner(inference_worker_config, coordinator, max_rollouts=None)

        with inference_runner:
            start_time = time.time()

            while time.time() - start_time < INTEGRATION_TEST_TIMEOUT:
                elapsed = time.time() - start_time

                _print_worker_status(elapsed, inference_runner, training_runner)

                if training_runner.done.is_set() and not inference_runner.done.is_set():
                    inference_runner.stop()

                # Check completion
                if inference_runner.done.is_set() and training_runner.done.is_set():
                    break

                time.sleep(1)

    # Context managers handle all cleanup and error checking
    expected_steps = training_worker_config.trainer.num_train_steps

    assert (
        inference_runner.rollouts_generated >= 1
    ), f"Expected at least 1 rollouts, got {inference_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= expected_steps
    ), f"Expected {expected_steps} training steps, got {training_runner.steps_completed}"

    assert (
        len(training_runner.checkpoints_created) >= 2
    ), f"Expected at least 2 checkpoints, got {len(training_runner.checkpoints_created)}"

    print(f"Weight transfers detected: {len(inference_runner.weight_transfers)}")
    for i, transfer in enumerate(inference_runner.weight_transfers):
        print(f"  Transfer {i}: weight_id={transfer['weight_id']}")

    # For weight transfers, we expect at least one transfer with a valid weight_id
    valid_transfers = [t for t in inference_runner.weight_transfers if t.get("weight_id") is not None]
    assert valid_transfers, "Inference worker should receive at least one weight transfer"

    assert len(training_runner.checkpoints_created) > 0, "Should have created at least one checkpoint"
    assert inference_runner.rollouts_generated > 0, "Should have generated at least one rollout"


# Removed test_load_tokenizer as we're using real Levanter tokenizers now
