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
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import numpy as np
import pytest

from marin.rl.rollout_storage import (
    RolloutStorageConfig,
    StorageType,
)
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
from tests.rl.config_helpers import (
    DummyTokenizer,
    create_nano_rollout_worker_config,
    create_nano_training_worker_config,
    create_rollout_batch,
    run_inference_with_engine,
)

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for mock checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    # Don't create any checkpoint files - let the inference worker use random initialization
    yield str(checkpoint_dir)


@pytest.fixture
def rollout_storage_config():
    """Create in-memory storage config for testing."""
    test_id = uuid.uuid4().hex[:8]
    return RolloutStorageConfig(storage_type=StorageType.IN_MEMORY, queue_name=f"test_{test_id}")


@pytest.fixture
def training_worker_config(tmp_path, rollout_storage_config):
    """Create minimal training worker configuration for testing."""
    return create_nano_training_worker_config(rollout_storage_config, tmp_path)


@pytest.fixture
def rollout_worker_config(tmp_path, rollout_storage_config):
    """Create minimal inference worker configuration for testing."""
    return create_nano_rollout_worker_config(tmp_path, rollout_storage_config)


def _print_worker_status(elapsed, inference_runner, training_runner):
    """Helper function to print detailed worker status during monitoring."""
    print(f"[{elapsed:.1f}s] Status:")
    print(f"  Rollouts generated: {inference_runner.rollouts_generated}")
    print(f"  Training steps: {training_runner.steps_completed}")
    print(f"  Weight transfers: {inference_runner.weight_transfers}")


class ThreadedWorkerRunner(ABC):
    """Base class for managing workers in separate threads with error handling."""

    def __init__(self, config):
        self.config = config

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

    @abstractmethod
    def _create_and_run_worker(self):
        """Create and run the worker. Must be implemented by subclasses."""
        pass

    def _run(self):
        """Thread target - runs the worker with error handling."""
        try:
            self._create_and_run_worker()
        except Exception as e:
            print(f"{self.__class__.__name__} encountered exception:", e, file=sys.stderr)
            logger.error(f"{self.__class__.__name__} failed", exc_info=True)
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
        return self.thread.is_alive() if self.thread else False

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
        self.join(timeout=5)

        if self.error:
            import traceback

            print(f"{self.__class__.__name__} error: {self.error}")
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
            pytest.fail(f"{self.__class__.__name__} failed: {self.error}")

        return False


class RolloutWorkerRunner(ThreadedWorkerRunner):
    """Manages running an inference worker in a separate thread with metric tracking."""

    def __init__(self, rollout_worker_config):
        super().__init__(rollout_worker_config)
        self.rollout_worker_config = rollout_worker_config

        # Metrics
        self.rollouts_generated = 0
        self.weight_transfers = 0

    def _track_rollout_generation(self):
        """Called when rollout is generated."""
        self.rollouts_generated += 1

    def _create_and_run_worker(self):
        """Create and run the rollout worker with tracking hooks."""
        self.worker = RolloutWorker(
            config=self.rollout_worker_config,
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
            if batch_data is None:
                return None, None
            self._track_rollout_generation()
            # Add metadata about rollout
            metrics["rollout_number"] = self.rollouts_generated
            return batch_data, metrics

        self.worker._generate_rollout_batch = counting_generate_batch

        # Run the worker normally
        self.worker.run()


class TrainWorkerRunner(ThreadedWorkerRunner):
    """Manages running a training worker in a separate thread with metric tracking."""

    def __init__(self, training_worker_config):
        super().__init__(training_worker_config)
        self.training_worker_config = training_worker_config

        self.steps_completed = 0
        self.losses = []
        self.trained_model = None
        self.reference_model = None
        self.all_steps_seen = []

    def _track_training_step(self):
        """Called after each training step."""
        self.steps_completed += 1

    def _create_and_run_worker(self):
        """Create and run the training worker with tracking hooks."""
        self.worker = TrainWorker(config=self.training_worker_config)

        self.reference_model = self.trained_model = jax.device_get(self.worker.reference_model)

        # Override _configure_training_hooks to inject our tracking hooks
        original_configure_hooks = self.worker._configure_training_hooks

        def patched_configure_hooks(trainer):
            original_configure_hooks(trainer)

            def step_tracking_hook(info):
                current_step = int(info.step)
                self.all_steps_seen.append(current_step)
                self._track_training_step()
                current_loss = float(info.loss)
                self.losses.append(current_loss)

            def model_capture_hook(info):
                # Make a copy of the model on the CPU.
                # For whatever reason, the model state is donated if we don't do this.
                self.trained_model = jax.device_get(info.state.model)

            trainer.add_hook(step_tracking_hook, every=1)
            trainer.add_hook(model_capture_hook, every=1)

        self.worker._configure_training_hooks = patched_configure_hooks
        self.worker.train()


@pytest.mark.slow("Integration test.")
def test_rollout_worker(rollout_worker_config: RolloutWorkerConfig):
    """Test inference worker generates rollouts to in-memory queue."""
    # Use the rollout storage config to create reader for checking results
    queue_reader = rollout_worker_config.rollout_storage.create_reader()

    # Get coordinator for GCS mode only
    # coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    # coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    rollout_worker_config.max_rollouts = 10
    rollout_worker_config.n_generations = 1
    rollout_worker_config.n_prompts_per_step = 1
    rollout_worker_config.max_input_length = 8
    rollout_worker_config.max_output_length = 8

    # Use context manager for cleaner test
    with RolloutWorkerRunner(rollout_worker_config) as runner:
        while runner.alive() and not runner.done.is_set():
            time.sleep(0.5)

        # Give a moment for final writes
        time.sleep(0.5)

        batches = queue_reader.read_all_available()
        assert len(batches) > 0, "Should be able to read batches from queue"
        assert runner.rollouts_generated >= 1, f"Expected at least 1 rollout, got {runner.rollouts_generated}"

    print("Rollout worker generated rollout batch successfully")


@pytest.mark.slow("Integration test.")
def test_train_worker(ray_tpu_cluster, training_worker_config: TrainWorkerConfig):
    """Test training worker processes rollout batch and creates checkpoint."""
    # Use the rollout storage config to create writer for sending test data
    queue_writer = training_worker_config.rollout_storage.create_writer()

    batch_size = training_worker_config.trainer.train_batch_size
    tokenizer = DummyTokenizer()

    with TrainWorkerRunner(training_worker_config) as runner:
        # Wait for worker to initialize and models to be available
        while not runner.worker:
            time.sleep(0.1)

        for _ in range(5):
            batch = create_rollout_batch(
                policy_model=runner.reference_model,
                batch_size=batch_size,
                tokenizer=tokenizer,
            )
            queue_writer.write_batch(batch)

        # Wait for training to complete
        while not runner.done.is_set() and runner.steps_completed < 10:
            time.sleep(0.1)

    # Verify results
    assert runner.steps_completed >= 1, f"Expected at least 1 training step, got {runner.steps_completed}"


@pytest.mark.slow("Integration test.")
def test_inference_and_training_workers(
    ray_tpu_cluster,
    training_worker_config,
    rollout_worker_config,
):
    """Test inference & training workers running together with checkpoint updates."""

    # The workers already use the same storage config from the fixtures, so they'll automatically share data

    rollout_worker_config.max_rollouts = 10
    training_worker_config.trainer.num_train_steps = 10

    # coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    # coordinator = create_coordinator(WeightTransferMode.GCS_CHECKPOINT, name=coordinator_name)

    with TrainWorkerRunner(training_worker_config) as training_runner:
        time.sleep(1)
        inference_runner = RolloutWorkerRunner(rollout_worker_config)

        with inference_runner:
            start_time = time.time()

            while time.time() - start_time < 60:
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


def validate_model(model, tokenizer) -> dict[str, str]:
    print("\n" + "=" * 60)
    print("Testing trained model responses:")
    print("=" * 60)

    test_prompts = [
        # prompts from our training data (for train only test)
        "i like cats, give me moar cats",
        "do you like cats?",
        "cats",
        "moar cats",
        # novel prompts
        "moar moar",
        "i love i love",
        "  ",
    ]

    tokenizer = DummyTokenizer()

    _, texts = run_inference_with_engine(
        model=model,
        prompts=test_prompts,
        tokenizer=tokenizer,
        max_tokens=16,
        temperature=0.8,
    )

    for i, (prompt, response) in enumerate(zip(test_prompts, texts, strict=True)):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"Response: {response}")

        # Check if response contains cats
        cat_count = response.lower().count("cat")
        if cat_count > 0:
            print(f"  ✓ Contains {cat_count} cat references!")
        else:
            print("  - No cat references found")

    # at least responses should have cats, we should have at least 10 total
    cat_count = 0
    cat_response_count = 0
    for response in texts:
        cat_count += response.lower().count("cat")
        if response.lower().count("cat") > 0:
            cat_response_count += 1

    assert cat_response_count >= 3, f"Expected at least 3 cat responses, got {cat_response_count}"
    assert cat_count >= 10, f"Expected at least 10 cat references, got {cat_count}"

    return {prompt: response for prompt, response in zip(test_prompts, texts, strict=True)}


@pytest.mark.slow("Integration test with training loop")
def test_train_worker_with_manual_cats_rollout(ray_tpu_cluster, training_worker_config):
    """Test training worker with manually constructed cat-themed rollout batches.

    This test validates that the training worker can process rollout batches
    with varying rewards and learn to prefer high-reward (cat-heavy) responses.
    """
    target_steps = 200
    queue_writer = training_worker_config.rollout_storage.create_writer()
    training_worker_config.trainer.num_train_steps = target_steps
    batch_size = training_worker_config.trainer.train_batch_size
    tokenizer = DummyTokenizer()

    with TrainWorkerRunner(training_worker_config) as runner:
        # Wait for worker to initialize and models to be available
        while not runner.worker:
            time.sleep(0.1)

        # create an initial batch to prime the trainer
        batch = create_rollout_batch(
            policy_model=runner.reference_model,
            batch_size=batch_size,
            tokenizer=tokenizer,
        )
        queue_writer.write_batch(batch)

        while not runner.done.is_set():
            if not runner.trained_model:
                logger.warning("Waiting for trained model to be available...")
            else:
                batch = create_rollout_batch(
                    policy_model=runner.trained_model,
                    batch_size=batch_size,
                    tokenizer=tokenizer,
                )
                queue_writer.write_batch(batch)
            time.sleep(1)

    assert all(not np.isnan(loss) for loss in runner.losses), "Loss should not be NaN"
    assert all(loss < 10.0 for loss in runner.losses), f"Loss should be reasonable, got {runner.losses}"
    #
    checkpoint_base = Path(training_worker_config.trainer.checkpointer.base_path)
    checkpoint_dirs = list(checkpoint_base.glob("*/*"))
    assert len(checkpoint_dirs) >= 1, f"Expected at least 1 checkpoint, got {len(checkpoint_dirs)}"

    print(f"  - Steps completed: {runner.steps_completed}")
    print(f"  - Loss progression: {runner.losses}")
    print(f"  - Initial loss: {runner.losses[0]:.4f}")
    print(f"  - Final loss: {runner.losses[-1]:.4f}")

    # Test the trained model with example prompts
    validate_model(runner.trained_model, DummyTokenizer())


@pytest.mark.slow("Long-running integration test.")
def test_full_integration_moar_cats(
    ray_tpu_cluster,
    training_worker_config,
    rollout_worker_config,
):
    """Long-running test to validate environment objective improves over time."""
    # The workers already use the same storage config from the fixtures, so they'll automatically share data

    target_steps = 100
    training_worker_config.trainer.num_train_steps = target_steps
    metrics_history = []
    with TrainWorkerRunner(training_worker_config) as training_runner:
        time.sleep(1)

        with RolloutWorkerRunner(rollout_worker_config) as inference_runner:
            while True:
                metrics_history.append(
                    {
                        "rollouts_generated": inference_runner.rollouts_generated,
                        "steps_completed": training_runner.steps_completed,
                        "weight_transfers": inference_runner.weight_transfers,
                    }
                )

                if training_runner.done.is_set() and not inference_runner.done.is_set():
                    inference_runner.stop()
                    break

                if inference_runner.done.is_set() and training_runner.done.is_set():
                    training_runner.stop()
                    break

                time.sleep(1)

    # Validate we ran for sufficient time and generated data
    assert len(metrics_history) >= 2, "Test should run long enough to collect multiple metric snapshots"
    assert (
        inference_runner.rollouts_generated >= 5
    ), f"Expected at least 5 rollouts, got {inference_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= 2
    ), f"Expected at least 2 training steps, got {training_runner.steps_completed}"

    # Validate objective improvement - rollout generation should increase over time
    initial_rollouts = metrics_history[0]["rollouts_generated"]
    final_rollouts = metrics_history[-1]["rollouts_generated"]
    assert (
        final_rollouts > initial_rollouts
    ), f"Rollout generation should improve: {initial_rollouts} -> {final_rollouts}"

    # Validate training progresses
    initial_steps = metrics_history[0]["steps_completed"]
    final_steps = metrics_history[-1]["steps_completed"]
    assert final_steps >= initial_steps, f"Training should progress: {initial_steps} -> {final_steps}"

    # Validate weight transfers occur
    assert inference_runner.weight_transfers >= 1, "Should have at least one weight transfer during long run"

    validate_model(training_runner.trained_model, DummyTokenizer())


@pytest.mark.slow("Integration test with checkpoint restart")
def test_train_worker_checkpoint_restart(ray_tpu_cluster, training_worker_config):
    """Test that training worker correctly restarts from checkpoint without repeating steps."""
    from pathlib import Path

    # Phase 1: Initial training run - small number of steps
    initial_target_steps = 5
    training_worker_config.trainer.num_train_steps = initial_target_steps

    queue_writer = training_worker_config.rollout_storage.create_writer()
    tokenizer = DummyTokenizer()
    batch_size = training_worker_config.trainer.train_batch_size

    with TrainWorkerRunner(training_worker_config) as runner:
        # Wait for worker to initialize
        while not runner.worker:
            time.sleep(0.1)

        # Add some training data
        for _ in range(5):
            batch = create_rollout_batch(
                policy_model=runner.reference_model,
                batch_size=batch_size,
                tokenizer=tokenizer,
            )
            queue_writer.write_batch(batch)

        # Wait for completion or timeout
        start_time = time.time()
        while runner.alive() and not runner.done.is_set() and time.time() - start_time < 30:
            time.sleep(0.5)

    first_run_steps = runner.all_steps_seen.copy()
    last_step_first_run = runner.steps_completed

    # Verify we trained and created checkpoint
    assert (
        last_step_first_run >= initial_target_steps
    ), f"Expected >= {initial_target_steps} steps, got {last_step_first_run}"
    checkpoint_dir = Path(training_worker_config.trainer.checkpointer.expanded_path("test-0-train"))
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoints = list(checkpoint_dir.glob("*"))
    assert len(checkpoints) > 0, f"No checkpoints found in {checkpoint_dir}"

    print(f"First run completed {last_step_first_run} steps, found {len(checkpoints)} checkpoints")

    # Phase 2: Restart training - should auto-load checkpoint
    training_worker_config.trainer.num_train_steps = 10  # Continue to step 10

    with TrainWorkerRunner(training_worker_config) as runner:
        # Wait for worker to initialize
        while not runner.worker:
            time.sleep(0.1)

        # Add more training data
        for _ in range(5):
            batch = create_rollout_batch(
                policy_model=runner.reference_model,
                batch_size=batch_size,
                tokenizer=tokenizer,
            )
            queue_writer.write_batch(batch)

        # Wait for completion or timeout
        start_time = time.time()
        while runner.alive() and not runner.done.is_set() and time.time() - start_time < 30:
            time.sleep(0.5)

    second_run_steps = runner.all_steps_seen

    # We should never see step 0 in the second run
    assert 0 not in second_run_steps, f"Step 0 seen in second run! Steps: {second_run_steps}"

    # Second run should start from a checkpoint (step > 1)
    min_step_second_run = min(second_run_steps)
    assert min_step_second_run > 1, f"Second run should restart from checkpoint (step > 1), got {min_step_second_run}"

    # Some overlap is expected when resuming from checkpoint, but verify proper restart
    # The key is that the second run continues beyond where the first run got to
    max_step_second_run = max(second_run_steps)
    max_step_first_run = max(first_run_steps) if first_run_steps else 0
    assert (
        max_step_second_run > max_step_first_run
    ), f"Second run should progress beyond first run: first max={max_step_first_run}, second max={max_step_second_run}"
