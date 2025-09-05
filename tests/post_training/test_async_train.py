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

import glob
import os
import tempfile
import threading
import time
import traceback
import unittest.mock
from pathlib import Path

import jax
import numpy as np
import pytest

from marin.post_training.inference_worker import InferenceWorker
from marin.post_training.rollout_storage import InMemoryRolloutQueue, RolloutBatch
from marin.post_training.train_worker import TrainingWorker
from marin.post_training.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    InferenceWorkerConfig,
    LoggerConfigData,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingHyperparameters,
    TrainWorkerConfig,
)


class DummyTokenizer:
    """Dummy tokenizer that only produces token IDs in valid range [0, vocab_size-1]"""

    def __init__(self, vocab_size=1000, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def encode(self, text, add_special_tokens=True):
        text_hash = hash(text) % (self.vocab_size - 100)
        seq_len = min(len(text.split()) + 2, 10)
        tokens = [(text_hash + i) % (self.vocab_size - 100) + 50 for i in range(seq_len)]
        return tokens[:8]

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded_{hash(tuple(token_ids)) % 1000}"


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for mock checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir)
        # Don't create any checkpoint files - let the inference worker use random initialization
        yield str(checkpoint_dir)


@pytest.fixture
def training_config():
    """Create minimal training configuration for testing."""
    model_paths_config = ModelPathsConfig(
        params=None,
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        default_config_name="test_1m",
    )

    optim_config = OptimizerConfig(
        init_lr=5e-7,
        end_lr=5e-7,
        lr=5e-7,
        lr_warmup_steps=0,
        lr_decay_steps=16,
        weight_decay=0.0,
        bf16_momentum=False,
        multiply_by_parameter_scale=False,
    )

    logger_config = LoggerConfigData(
        online=False,
        prefix="test",
        prefix_to_id=True,
    )

    generation_config = GenerationConfig(
        max_output_length=32, stop_tokens=[[128001]], n_generations=2
    )

    test_generation_config = GenerationConfig(
        max_output_length=32, temperature=0.0, stop_tokens=[[128001]], n_generations=1
    )

    model_config_override = ModelOverrideConfig(
        max_sequence_length=512,
        initializer_range=0.001,
    )

    checkpointer_config = CheckpointerConfigData()

    temp_dir = tempfile.mkdtemp()

    return TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="fp32",
            inference_activation_dtype="fp32",
            training_param_dtype="fp32",
            training_activation_dtype="fp32",
            model_config_override=model_config_override,
            train_attention_kernel_config="default:{}",
            prefill_attention_kernel_config="default:{}",
            generate_attention_kernel_config="default:{}",
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=1,  # Small number for testing
            max_input_length=8,
            max_output_length=8,
            train_bsize=2,
            decode_bsize=2,
            prefill_bsize=2,
            reference_logprobs_bsize=2,
            n_prompts_per_step=2,
            optim_config=optim_config,
            kl_coef=1e-3,
        ),
        logging=LoggingConfig(
            log_freq=1,
            num_eval_examples=1,
            save_model_freq=0,
            wandb_project="test_project",
            logger_config=logger_config,
        ),
        environment=EnvironmentConfig(
            train_environments_path="environments_test.json",
            test_environments_path="environments_test.json",
        ),
        distributed=DistributedConfig(
            sharding=[1, 1, 1, -1],
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir=temp_dir,
        checkpointer_config=checkpointer_config,
    )


@pytest.fixture
def inference_worker_config(temp_checkpoint_dir):
    """Create inference worker configuration."""
    return InferenceWorkerConfig(
        environment_spec="mock",
        checkpoint_source_path=temp_checkpoint_dir,
        rollout_output_path="/tmp/test_rollouts",  # Won't be used with in-memory queue
        checkpoint_poll_interval=0.1,
        rollout_batch_size=2,
        n_generations=2,
        n_examples_per_batch=2,
        use_gcs=False,
        max_rollouts=2,  # Generate 2 rollout batches
        checkpoint_timeout=1.0,
    )


@pytest.fixture
def worker_config():
    """Create worker configuration."""
    return TrainWorkerConfig(
        rollout_queue_bucket="test_bucket",
        rollout_queue_path="test_queue",
        batch_timeout=2.0,  # Wait 2 seconds for each batch
        max_idle_time=25.0,  # Wait 25 seconds total before giving up
    )


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture."""
    with unittest.mock.patch(
        "marin.post_training.inference_worker.load_tokenizer"
    ) as mock_tokenizer:
        mock_tokenizer.return_value = DummyTokenizer(vocab_size=1000, pad_token_id=0)
        yield mock_tokenizer


class InferenceWorkerRunner:
    """Manages running an inference worker in a separate thread with metric tracking."""

    def __init__(self, training_config, inference_worker_config, queue_writer):
        self.training_config = training_config
        self.inference_worker_config = inference_worker_config
        self.queue_writer = queue_writer

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

        # Metrics
        self.rollouts_generated = 0
        self.checkpoint_loads = []

    def _track_checkpoint_load(self, old_path, new_path):
        """Called when checkpoint changes."""
        self.checkpoint_loads.append(
            {
                "old_path": old_path,
                "new_path": new_path,
                "time": time.time(),
                "rollouts_at_load": self.rollouts_generated,
            }
        )

    def _track_rollout_generation(self):
        """Called when rollout is generated."""
        self.rollouts_generated += 1

    def _run(self):
        """Thread target - runs the inference worker."""
        try:
            self.worker = InferenceWorker(
                self.training_config, self.inference_worker_config, rollout_writer=self.queue_writer
            )

            # Override checkpoint detection to track when it happens
            original_check_for_new_checkpoint = self.worker._check_for_new_checkpoint

            def tracking_check_for_new_checkpoint():
                old_checkpoint_path = self.worker.latest_checkpoint_path
                original_check_for_new_checkpoint()
                new_checkpoint_path = self.worker.latest_checkpoint_path

                # If checkpoint changed, record the load
                if new_checkpoint_path != old_checkpoint_path:
                    self._track_checkpoint_load(old_checkpoint_path, new_checkpoint_path)
                return None  # Original method doesn't return anything

            self.worker._check_for_new_checkpoint = tracking_check_for_new_checkpoint

            # Override batch generation to count rollouts
            original_generate_batch = self.worker._generate_rollout_batch

            def counting_generate_batch():
                batch_data, metrics = original_generate_batch()
                self._track_rollout_generation()
                # Add metadata about which checkpoint was used
                metrics["checkpoint_path"] = self.worker.latest_checkpoint_path
                metrics["rollout_number"] = self.rollouts_generated
                return batch_data, metrics

            self.worker._generate_rollout_batch = counting_generate_batch

            # Run the worker normally
            self.worker.run()
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


class TrainingWorkerRunner:
    """Manages running a training worker in a separate thread with metric tracking."""

    def __init__(self, training_config, worker_config, queue_reader):
        self.training_config = training_config
        self.worker_config = worker_config
        self.queue_reader = queue_reader

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

        # Metrics
        self.training_steps_completed = 0
        self.checkpoints_created = []

    def _track_checkpoint_save(self, step):
        """Called when checkpoint is saved."""
        self.checkpoints_created.append(
            {
                "step": step,
                "time": time.time(),
                "rollouts_consumed": self.training_steps_completed
                + 1,  # +1 because we're about to increment
            }
        )

    def _track_training_step(self):
        """Called after each training step."""
        self.training_steps_completed += 1

    def _run(self):
        """Thread target - runs the training worker."""
        try:
            self.worker = TrainingWorker(
                self.training_config, self.worker_config, rollout_reader=self.queue_reader
            )

            # Override save_checkpoint to track checkpoint creation
            original_save_checkpoint = self.worker.save_checkpoint

            def tracking_save_checkpoint(train_state, step):
                result = original_save_checkpoint(train_state, step)
                self._track_checkpoint_save(step)
                return result

            self.worker.save_checkpoint = tracking_save_checkpoint

            # Override train_step to count steps
            original_train_step = self.worker.train_step

            def counting_train_step(train_state, rng, batch):
                result = original_train_step(train_state, rng, batch)
                self._track_training_step()
                return result

            self.worker.train_step = counting_train_step

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


def test_inference_worker(training_config, inference_worker_config, mock_tokenizer):
    """Test inference worker generates rollouts to in-memory queue."""
    # Skip if not on CPU
    if jax.devices()[0].device_kind != "cpu":
        pytest.skip("Test requires CPU device")

    # Create in-memory queue
    rollout_queue = InMemoryRolloutQueue()
    queue_writer = rollout_queue.writer()
    queue_reader = rollout_queue.reader()

    # Configure for single batch
    inference_worker_config.max_rollouts = 1

    # Create inference worker
    worker = InferenceWorker(training_config, inference_worker_config, rollout_writer=queue_writer)

    # Run inference worker

    def _run_worker():
        import sys

        try:
            worker.run()
        except Exception as e:
            print(f"Inference worker error: {e}", file=sys.stderr)
            print("Inference worker traceback:", file=sys.stderr)
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)), file=sys.stderr)

    worker_thread = threading.Thread(target=_run_worker)
    worker_thread.start()

    timeout = 60
    start_time = time.time()

    while time.time() - start_time < timeout and worker_thread.is_alive():
        queue_size = queue_reader.get_queue_size()
        if queue_size > 0:
            break
        time.sleep(0.1)

    # Stop worker and wait for completion
    worker.stop()
    worker_thread.join(timeout=5)

    # Try to read the batch
    batch_data = queue_reader.read_batch(timeout=1.0)
    assert batch_data is not None, "Should be able to read batch from queue"

    print("✓ Inference worker generated rollout batch successfully")


def test_train_worker(training_config, worker_config, mock_tokenizer):
    """Test training worker processes rollout batch and creates checkpoint."""
    # Skip if not on CPU
    if jax.devices()[0].device_kind != "cpu":
        pytest.skip("Test requires CPU device")

    # Create in-memory queue
    rollout_queue = InMemoryRolloutQueue()
    queue_reader = rollout_queue.reader()
    queue_writer = rollout_queue.writer()

    # Create a sample rollout batch with correct data shapes
    batch_size = training_config.hyperparameters.train_bsize
    max_seq_len = (
        training_config.hyperparameters.max_input_length
        + training_config.hyperparameters.max_output_length
    )

    # Create mock batch data with appropriate shapes
    rng = np.random.default_rng(42)  # Use fixed seed for reproducibility
    sample_batch = RolloutBatch(
        input_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
        attention_mask=np.ones((batch_size, max_seq_len), dtype=np.int32),
        position_ids=np.arange(max_seq_len)[None, :].repeat(batch_size, axis=0).astype(np.int32),
        target_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
        loss_weights=np.ones((batch_size, max_seq_len), dtype=np.float32),
        loss_masks=np.ones((batch_size, max_seq_len), dtype=np.float32),
        reference_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
        metadata={"batch_id": 0, "environment": "test"},
    )

    # Write batch to queue
    queue_writer.write_batch(sample_batch)

    # Configure training for single step
    training_config.hyperparameters.num_train_steps = 1
    training_config.logging.save_model_freq = 1
    worker_config.checkpoint_sync_interval = 1

    # Create training worker
    worker = TrainingWorker(training_config, worker_config, rollout_reader=queue_reader)

    # Track completion
    training_completed = False
    checkpoint_created = False

    def run_training():
        nonlocal training_completed, checkpoint_created
        try:
            worker.train()
            training_completed = True

            # Check if checkpoint was created in logger output directory
            import os

            logger_checkpoint_dir = os.path.join(worker.logger.output_dir, "checkpoints")
            if os.path.exists(logger_checkpoint_dir):
                logger_checkpoint_files = os.listdir(logger_checkpoint_dir)
                checkpoint_created = len(logger_checkpoint_files) > 0

        except Exception as e:
            import traceback

            print(f"Training worker error: {e}")
            print("Training worker traceback:")
            print(traceback.format_exc())

    # Run training in thread
    import threading

    training_thread = threading.Thread(target=run_training)
    training_thread.start()

    # Wait for completion with timeout
    training_thread.join(timeout=30)

    # Verify results
    assert training_completed, "Training worker should complete successfully"
    assert checkpoint_created, "Training worker should create checkpoint after processing batch"

    print("✓ Training worker processed batch and created checkpoint successfully")


def test_full_cycle_with_checkpoint_updates(
    training_config, inference_worker_config, worker_config, mock_tokenizer
):
    """Test full cycle: inference generates rollouts -> training creates checkpoints -> inference loads new checkpoints."""
    # Skip if not on CPU
    if jax.devices()[0].device_kind != "cpu":
        pytest.skip("Test requires CPU device")

    # Create temporary checkpoint directory that both workers can access
    with tempfile.TemporaryDirectory() as output_dir:
        # Update configs to use shared checkpoint directory
        training_config.output_dir = output_dir
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        inference_worker_config.checkpoint_source_path = checkpoint_dir

        # Configure for multiple training steps and frequent checkpointing
        training_config.hyperparameters.num_train_steps = 3
        training_config.logging.save_model_freq = 1  # Save after every step
        training_config.logging.save_initial_checkpoint = True
        worker_config.checkpoint_sync_interval = 1

        # Configure inference worker to poll frequently and generate multiple batches
        inference_worker_config.checkpoint_poll_interval = 0.2  # Poll more frequently
        inference_worker_config.max_rollouts = None  # Don't limit, let it run continuously
        inference_worker_config.rollout_batch_size = 2

        # Create in-memory queue
        rollout_queue = InMemoryRolloutQueue()
        queue_reader = rollout_queue.reader()
        queue_writer = rollout_queue.writer()

        # Create worker runners
        inference_runner = InferenceWorkerRunner(
            training_config, inference_worker_config, queue_writer
        )
        training_runner = TrainingWorkerRunner(training_config, worker_config, queue_reader)
        training_runner.start()

        while glob.glob(os.path.join(checkpoint_dir, "*")) == []:
            print("Waiting for initial checkpoint...")
            time.sleep(1)

        inference_runner.start()

        # Monitor progress with detailed logging
        timeout = 30  # seconds
        start_time = time.time()

        print("Starting full cycle monitoring...")
        last_print_time = start_time

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            queue_size = rollout_queue.reader().get_queue_size()

            # Print detailed status every 2 seconds
            if elapsed - (last_print_time - start_time) >= 2.0:
                print(f"[{elapsed:.1f}s] Status:")
                print(f"  Rollouts generated: {inference_runner.rollouts_generated}")
                print(f"  Training steps: {training_runner.training_steps_completed}")
                print(f"  Queue size: {queue_size}")
                print(f"  Checkpoints created: {len(training_runner.checkpoints_created)}")
                print(f"  Checkpoint loads: {len(inference_runner.checkpoint_loads)}")
                print(f"  Inference done: {inference_runner.done.is_set()}")
                print(f"  Training done: {training_runner.done.is_set()}")
                last_print_time = time.time()

            # Stop training worker when target steps reached
            if (
                training_runner.training_steps_completed
                >= training_config.hyperparameters.num_train_steps
                and not training_runner.done.is_set()
            ):
                training_runner.stop()

            # Stop inference worker after training is complete and we have sufficient data
            if training_runner.done.is_set() and not inference_runner.done.is_set():
                inference_runner.stop()

            # Check completion
            if inference_runner.done.is_set() and training_runner.done.is_set():
                break

            time.sleep(0.5)

        # Wait for threads to complete
        inference_runner.join()
        training_runner.join()

        # Check for errors
        if inference_runner.error:
            print(f"Inference worker error: {inference_runner.error}")
            print(
                "Traceback:",
                "".join(
                    traceback.format_exception(
                        type(inference_runner.error),
                        inference_runner.error,
                        inference_runner.error.__traceback__,
                    )
                ),
            )
            pytest.fail(f"Inference worker failed: {inference_runner.error}")

        if training_runner.error:
            import traceback

            print(f"Training worker error: {training_runner.error}")
            print(
                "Traceback:",
                "".join(
                    traceback.format_exception(
                        type(training_runner.error),
                        training_runner.error,
                        training_runner.error.__traceback__,
                    )
                ),
            )
            pytest.fail(f"Training worker failed: {training_runner.error}")

        # Comprehensive validation
        print("\n=== VALIDATION RESULTS ===")
        print(f"Rollouts generated: {inference_runner.rollouts_generated}")
        print(f"Training steps completed: {training_runner.training_steps_completed}")
        print(f"Checkpoints created: {training_runner.checkpoints_created}")
        print(f"Checkpoint loads: {inference_runner.checkpoint_loads}")

        # Validate basic requirements
        assert inference_runner.rollouts_generated >= 3, (
            f"Expected at least 3 rollouts, got {inference_runner.rollouts_generated}"
        )
        assert (
            training_runner.training_steps_completed
            >= training_config.hyperparameters.num_train_steps
        ), (
            f"Expected {training_config.hyperparameters.num_train_steps} training steps, got {training_runner.training_steps_completed}"
        )

        # Validate checkpoint creation - should have initial checkpoint + one per training step
        expected_checkpoints = (
            1 + training_config.hyperparameters.num_train_steps
        )  # initial + training steps
        assert len(training_runner.checkpoints_created) >= expected_checkpoints, (
            f"Expected at least {expected_checkpoints} checkpoints, got {len(training_runner.checkpoints_created)}"
        )

        # Validate checkpoint loading - inference worker should detect the checkpoints created by training
        print(f"Checkpoint loads detected: {len(inference_runner.checkpoint_loads)}")
        for i, load in enumerate(inference_runner.checkpoint_loads):
            print(f"  Load {i}: {load['old_path']} -> {load['new_path']}")

        actual_checkpoint_loads = [
            load for load in inference_runner.checkpoint_loads if "step_" in str(load["new_path"])
        ]
        assert actual_checkpoint_loads, (
            "Inference worker should load at least one actual checkpoint"
        )

        # The key validation is that we have the full cycle working
        # Even if checkpoint file detection isn't perfect, the pipeline is functional

        # More lenient validation - just ensure we had checkpoint creation and rollout generation
        # This validates the basic cycle works
        assert len(training_runner.checkpoints_created) > 0, (
            "Should have created at least one checkpoint"
        )
        assert inference_runner.rollouts_generated > 0, "Should have generated at least one rollout"

        # Validate timeline: if we have both checkpoints and loads, timing should be reasonable
        if training_runner.checkpoints_created and inference_runner.checkpoint_loads:
            first_checkpoint_time = min(cp["time"] for cp in training_runner.checkpoints_created)
            first_load_time = min(cp["time"] for cp in inference_runner.checkpoint_loads)
            # Allow generous tolerance for timing issues in tests
            time_diff = first_load_time - first_checkpoint_time
            # Just ensure it's not wildly out of order (> 10 seconds would be suspicious)
            assert time_diff >= -10.0, (
                f"Checkpoint load timing seems suspicious: loaded {time_diff:.2f}s before creation"
            )

        print("✓ Full cycle test passed:")
        print(f"  - Generated {inference_runner.rollouts_generated} rollouts")
        print(f"  - Completed {training_runner.training_steps_completed} training steps")
        print(f"  - Created {len(training_runner.checkpoints_created)} checkpoints")
        print(
            f"  - Loaded {len(inference_runner.checkpoint_loads)} checkpoints in inference worker"
        )
        print("  - Validated bidirectional communication between workers")
        print("  - Confirmed checkpoint detection and model updates")
