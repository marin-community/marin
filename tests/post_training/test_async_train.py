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
import logging
import os
import sys
import tempfile
import threading
import time
import unittest.mock
from pathlib import Path

import numpy as np
import pytest
import ray

try:
    from marin.post_training.inference_worker import InferenceWorker
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)

from marin.post_training.model_helpers import load_tokenizer
from marin.post_training.rollout_storage import (
    FileRolloutReader,
    FileRolloutWriter,
    InMemoryRolloutQueue,
    RolloutBatch,
)
from marin.post_training.train_worker import TrainingWorker
from marin.post_training.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    InferenceWorkerConfig,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TokenizerOverrideConfig,
    TrainingConfig,
    TrainingHyperparameters,
    TrainWorkerConfig,
    WeightTransferConfig,
    WeightTransferMode,
)

# Test timeout constants
CHECKPOINT_POLL_INTERVAL = 0.2
WORKER_JOIN_TIMEOUT = 5
BATCH_READ_TIMEOUT = 1.0
INTEGRATION_TEST_TIMEOUT = 60


logger = logging.getLogger(__name__)


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


# Since we use an actor for the ray transfer, we need a new cluster to
# avoid stale state.
@pytest.fixture(scope="function")
def ray_cluster():
    """Start Ray cluster for weight transfer testing."""
    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


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

    generation_config = GenerationConfig(max_output_length=32, stop_tokens=[[128001]], n_generations=2)

    test_generation_config = GenerationConfig(
        max_output_length=32, temperature=0.0, stop_tokens=[[128001]], n_generations=1
    )

    model_config_override = ModelOverrideConfig(
        max_sequence_length=512,
        initializer_range=0.001,
    )

    checkpointer_config = CheckpointerConfigData(
        save_model_freq=100,  # Save checkpoints infrequently
    )

    temp_dir = tempfile.mkdtemp()

    weight_transfer_config = WeightTransferConfig(
        mode=WeightTransferMode.RAY_REMOTING,
        sync_interval_steps=1,  # Transfer weights frequently
        poll_interval_seconds=1.0,
        coordinator_name="test_coordinator",
        checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
    )

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
            # paged attention doesn't work on CPU
            # generate_attention_kernel_config=(
            #     'paged:{"page_size": 1, "pages_per_compute_block": 1, "inline_seq_dim": true, "use_int8": false}'
            # ),
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
            wandb_project="test_project",
            enable=False,
            online=False,
            prefix="test",
            prefix_to_id=True,
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
        checkpoint=checkpointer_config,
        weight_transfer=weight_transfer_config,
    )


@pytest.fixture
def inference_worker_config(training_config):
    """Create inference worker configuration."""
    return InferenceWorkerConfig(
        training_config=training_config,
        environment_spec="mock",
        rollout_output_path="/tmp/test_rollouts",  # Won't be used with in-memory queue
        rollout_batch_size=2,
        max_rollouts=2,  # Generate 2 rollout batches
    )


@pytest.fixture
def worker_config(training_config):
    """Create worker configuration."""
    return TrainWorkerConfig(
        training_config=training_config,
        rollout_queue_path="test_queue",
    )


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture."""
    with unittest.mock.patch("marin.post_training.inference_worker.load_tokenizer") as mock_tokenizer:
        mock_tokenizer.return_value = DummyTokenizer(vocab_size=1000, pad_token_id=0)
        yield mock_tokenizer


def _print_worker_status(elapsed, inference_runner, training_runner):
    """Helper function to print detailed worker status during monitoring."""
    print(f"[{elapsed:.1f}s] Status:")
    print(f"  Rollouts generated: {inference_runner.rollouts_generated}")
    print(f"  Training steps: {training_runner.steps_completed}")
    print(f"  Checkpoints created: {len(training_runner.checkpoints_created)}")
    print(f"  Weight transfers: {len(inference_runner.weight_transfers)}")
    print(f"  Inference done: {inference_runner.done.is_set()}")
    print(f"  Training done: {training_runner.done.is_set()}")


@pytest.fixture
def rollout_queue(request, tmp_path):
    """Create rollout queue based on parameter, with cleanup."""
    queue_type = getattr(request, "param", "memory")

    if queue_type == "memory":
        queue = InMemoryRolloutQueue()
        reader = queue.reader()
        writer = queue.writer()
        yield reader, writer
        # No cleanup needed for in-memory

    elif queue_type == "file":
        queue_path = str(tmp_path / "rollout_queue")
        reader = FileRolloutReader(queue_path)
        writer = FileRolloutWriter(queue_path)
        yield reader, writer
        # Cleanup happens automatically with tmp_path


@pytest.mark.parametrize("rollout_queue", ["memory", "file"], indirect=True)
def test_rollout_queue(rollout_queue):
    """Test in-memory rollout queue operations."""
    reader, writer = rollout_queue

    batch_size = 2
    max_seq_len = 16
    rng = np.random.default_rng(42)

    batch1 = RolloutBatch(
        input_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
        attention_mask=np.ones((batch_size, max_seq_len), dtype=np.int32),
        position_ids=np.arange(max_seq_len)[None, :].repeat(batch_size, axis=0).astype(np.int32),
        target_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
        loss_weights=np.ones((batch_size, max_seq_len), dtype=np.float32),
        loss_masks=np.ones((batch_size, max_seq_len), dtype=np.float32),
        reference_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
    )

    batch2 = RolloutBatch(
        input_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
        attention_mask=np.ones((batch_size, max_seq_len), dtype=np.int32),
        position_ids=np.arange(max_seq_len)[None, :].repeat(batch_size, axis=0).astype(np.int32),
        target_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
        loss_weights=np.ones((batch_size, max_seq_len), dtype=np.float32),
        loss_masks=np.ones((batch_size, max_seq_len), dtype=np.float32),
        reference_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
    )

    # Test timeout on empty queue
    empty_batch = reader.read_batch(timeout=0.1)
    assert empty_batch is None

    # Test writing
    writer.write_batch(batch1)
    writer.write_batch(batch2)

    # Test reading (FIFO order)
    read_batch1 = reader.read_batch(timeout=BATCH_READ_TIMEOUT)
    assert read_batch1 is not None

    read_batch2 = reader.read_batch(timeout=BATCH_READ_TIMEOUT)
    assert read_batch2 is not None


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
            self.worker = InferenceWorker(self.inference_worker_config, rollout_writer=self.queue_writer)

            def tracking_check_for_new_weights():
                if self.worker.weight_transfer_client is not None:
                    try:
                        params, metadata = self.worker.weight_transfer_client.receive_weights()
                        if params is not None:
                            self.worker.current_params = params
                            self._track_weight_transfer(metadata.get("weight_id"), metadata)
                            logger.info(f"Received new weights: {metadata}")
                    except Exception as e:
                        logger.warning(f"Failed to receive weights: {e}")
                else:
                    logger.debug("Weight transfer manager not initialized, skipping weight check")

            self.worker._check_for_new_weights = tracking_check_for_new_weights

            # Override batch generation to count rollouts
            original_generate_batch = self.worker._generate_rollout_batch

            def counting_generate_batch():
                batch_data, metrics = original_generate_batch()
                self._track_rollout_generation()
                # Add metadata about rollout
                metrics["rollout_number"] = self.rollouts_generated
                return batch_data, metrics

            self.worker._generate_rollout_batch = counting_generate_batch

            # Run the worker normally
            self.worker.run()
        except Exception as e:
            print("Inference worker encountered exception:", e, file=sys.stderr)
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
            self.worker = TrainingWorker(self.worker_config, rollout_reader=self.queue_reader)

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


def test_inference_worker(ray_cluster, training_config, inference_worker_config, mock_tokenizer):
    """Test inference worker generates rollouts to in-memory queue."""
    rollout_queue = InMemoryRolloutQueue()
    queue_writer = rollout_queue.writer()
    queue_reader = rollout_queue.reader()

    # Configure for single batch
    inference_worker_config.max_rollouts = 1

    # Use context manager for cleaner test
    with InferenceWorkerRunner(training_config, inference_worker_config, queue_writer) as runner:
        batch_data = queue_reader.read_batch()
        assert batch_data is not None, "Should be able to read batch from queue"
        assert runner.rollouts_generated >= 1, f"Expected at least 1 rollout, got {runner.rollouts_generated}"

    print("✓ Inference worker generated rollout batch successfully")


def test_train_worker(ray_cluster, training_config, worker_config, mock_tokenizer):
    """Test training worker processes rollout batch and creates checkpoint."""
    rollout_queue = InMemoryRolloutQueue()
    queue_reader = rollout_queue.reader()
    queue_writer = rollout_queue.writer()

    # Create a sample rollout batch with correct data shapes
    batch_size = training_config.hyperparameters.train_bsize
    max_seq_len = training_config.hyperparameters.max_input_length + training_config.hyperparameters.max_output_length

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
    )

    # Write batch to queue
    queue_writer.write_batch(sample_batch)

    # Configure training for single step
    training_config.hyperparameters.num_train_steps = 1
    training_config.checkpoint.save_model_freq = 1

    # Use context manager for cleaner test
    with TrainingWorkerRunner(training_config, worker_config, queue_reader) as runner:
        # Wait for training to complete
        while not runner.done.is_set() and runner.steps_completed < 1:
            time.sleep(0.1)

    # Verify results
    assert runner.steps_completed >= 1, f"Expected at least 1 training step, got {runner.steps_completed}"

    # Check checkpoint was created
    checkpoint_dir = os.path.join(training_config.output_dir, "checkpoints")
    checkpoint_created = os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    assert checkpoint_created, "Training worker should create checkpoint after processing batch"


@pytest.mark.parametrize("rollout_queue", ["memory", "file"], indirect=True)
def test_inference_and_training_workers(
    ray_cluster,
    tmp_path,
    training_config,
    inference_worker_config,
    worker_config,
    mock_tokenizer,
    rollout_queue,
):
    """Test inference & training workers running together with checkpoint updates."""
    # Update configs to use shared checkpoint directory
    training_config.output_dir = str(tmp_path)
    checkpoint_dir = str(tmp_path / "checkpoints")
    training_config.weight_transfer.checkpoint_dir = checkpoint_dir

    training_config.hyperparameters.num_train_steps = 3
    training_config.checkpoint.save_model_freq = 1  # Save after every step
    training_config.logging.save_initial_checkpoint = True

    # Configure inference worker to poll frequently and generate multiple batches
    training_config.weight_transfer.poll_interval_seconds = CHECKPOINT_POLL_INTERVAL
    inference_worker_config.max_rollouts = None  # Don't limit, let it run continuously
    inference_worker_config.rollout_batch_size = 2

    # Use provided rollout queue from fixture
    queue_reader, queue_writer = rollout_queue

    # Create worker runners and start with context managers
    with TrainingWorkerRunner(training_config, worker_config, queue_reader) as training_runner:
        # Wait for initial checkpoint to be created
        while glob.glob(os.path.join(checkpoint_dir, "*")) == []:
            time.sleep(1)

        with InferenceWorkerRunner(training_config, inference_worker_config, queue_writer) as inference_runner:
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
    hyperparams = training_config.hyperparameters

    assert (
        inference_runner.rollouts_generated >= 1
    ), f"Expected at least 1 rollouts, got {inference_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= hyperparams.num_train_steps
    ), f"Expected {hyperparams.num_train_steps} training steps, got {training_runner.steps_completed}"

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


def test_load_tokenizer():
    """Test load_tokenizer function with standard configuration."""
    model_paths = ModelPathsConfig(
        params=None,
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        config=None,
    )

    tokenizer_override = TokenizerOverrideConfig()

    with unittest.mock.patch("marin.post_training.model_helpers.AutoTokenizer") as mock_auto_tokenizer:
        mock_tokenizer = unittest.mock.MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer(model_paths, tokenizer_override)

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            truncation_side="right",
            padding_side="right",
            pad_token="<|reserved_special_token_0|>",
        )

        assert tokenizer is mock_tokenizer
