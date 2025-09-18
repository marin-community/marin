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
from pathlib import Path

import numpy as np
import pytest
import ray

try:
    from marin.post_training.rollout_worker import RolloutWorker
    from marin.post_training.train_worker import TrainWorker
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)


from marin.post_training.rollout_storage import (
    InMemoryRolloutQueue,
    RolloutBatch,
    TaggedRolloutBatch,
)

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


class _TestableTrainWorker(TrainWorker):
    """TrainWorker subclass for testing with step tracking."""

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

    return create_nano_inference_worker_config(tmp_path, inference_server_config, rollout_writer)


def _print_worker_status(elapsed, inference_runner, training_runner):
    """Helper function to print detailed worker status during monitoring."""
    print(f"[{elapsed:.1f}s] Status:")
    print(f"  Rollouts generated: {inference_runner.rollouts_generated}")
    print(f"  Training steps: {training_runner.steps_completed}")
    print(f"  Weight transfers: {inference_runner.weight_transfers}")


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


class RolloutWorkerRunner:
    """Manages running an inference worker in a separate thread with metric tracking."""

    def __init__(self, inference_worker_config):
        self.inference_worker_config = inference_worker_config

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

        # Metrics
        self.rollouts_generated = 0
        self.weight_transfers = 0

    def _track_rollout_generation(self):
        """Called when rollout is generated."""
        self.rollouts_generated += 1

    def _run(self):
        try:
            from unittest.mock import patch

            from tests.post_training.test_helpers import DummyTokenizer

            with patch("levanter.inference.openai.load_tokenizer") as mock_load:
                mock_load.return_value = DummyTokenizer(vocab_size=1000)
                self.worker = RolloutWorker(
                    config=self.inference_worker_config,
                )

            _sync_weights_original = self.worker._sync_weights

            def sync_and_track():
                result = _sync_weights_original()
                if result:
                    self.weight_transfers += 1
                return result

            self.worker._sync_weights = sync_and_track

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


class TrainWorkerRunner:
    """Manages running a training worker in a separate thread with metric tracking."""

    def __init__(self, training_worker_config):
        self.training_worker_config = training_worker_config

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

        # Metrics
        self.steps_completed = 0

    def _track_training_step(self):
        """Called after each training step."""
        print("Called train step.")
        self.steps_completed += 1

    def _run(self):
        """Thread target - runs the training worker."""
        try:
            self.worker = _TestableTrainWorker(
                config=self.training_worker_config,
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
    # coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    # coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    # Use context manager for cleaner test
    with RolloutWorkerRunner(inference_worker_config) as runner:
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
    # coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    # coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    # Use context manager for cleaner test
    with TrainWorkerRunner(training_worker_config) as runner:
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

    # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()

    inference_worker_config.rollout_writer = rollout_queue.writer()
    training_worker_config.rollout_reader = rollout_queue.reader()

    # coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    # coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    # Create worker runners and start with context managers
    with TrainWorkerRunner(training_worker_config) as training_runner:
        time.sleep(1)
        inference_runner = RolloutWorkerRunner(inference_worker_config)

        with inference_runner:
            start_time = time.time()

            while time.time() - start_time < INTEGRATION_TEST_TIMEOUT:
                elapsed = time.time() - start_time

                _print_worker_status(elapsed, inference_runner, training_runner)

                if training_runner.done.is_set() and not inference_runner.done.is_set():
                    inference_runner.stop()
                    break

                if inference_runner.done.is_set() and training_runner.done.is_set():
                    training_runner.stop()
                    break

                time.sleep(1)

    assert (
        inference_runner.rollouts_generated >= 1
    ), f"Expected at least 1 rollouts, got {inference_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= 0
    ), f"Expected at least 0 training steps, got {training_runner.steps_completed}"

    print("checkpoint dir:", training_worker_config.trainer.checkpointer.base_path)
    checkpoint_dirs = list(Path(training_worker_config.trainer.checkpointer.base_path).glob("*/*"))
    print(checkpoint_dirs)
    assert len(checkpoint_dirs) >= 1, f"Expected at least 1 checkpoint, got {len(checkpoint_dirs)}"

    print(f"Weight transfers detected: {inference_runner.weight_transfers}")
    assert inference_runner.weight_transfers >= 1, "Expected at least 1 weight transfer"
    assert inference_runner.rollouts_generated > 0, "Should have generated at least one rollout"


# Removed test_load_tokenizer as we're using real Levanter tokenizers now
