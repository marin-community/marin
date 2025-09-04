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

import tempfile
import threading
import time
import unittest.mock
from pathlib import Path

import jax
import pytest

from marin.post_training.inference_worker import InferenceWorker
from marin.post_training.rollout_storage import InMemoryRolloutQueue
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
        checkpoint_sync_interval=10,
        batch_timeout=2.0,  # Wait 2 seconds for each batch
        max_idle_time=25.0,  # Wait 25 seconds total before giving up
        checkpoint_bucket=None,
        checkpoint_path="checkpoints",
    )


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture."""
    with unittest.mock.patch(
        "marin.post_training.inference_worker.load_tokenizer"
    ) as mock_tokenizer:
        mock_tokenizer.return_value = DummyTokenizer(vocab_size=1000, pad_token_id=0)
        yield mock_tokenizer


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
        import traceback
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


def test_workers_with_in_memory_queue(
    training_config, inference_worker_config, worker_config, mock_tokenizer
):
    """Test inference and training workers communicating through in-memory queue."""
    # Skip if not on CPU
    if jax.devices()[0].device_kind != "cpu":
        pytest.skip("Test requires CPU device")

    # Create in-memory queue
    rollout_queue = InMemoryRolloutQueue()
    queue_reader = rollout_queue.reader()
    queue_writer = rollout_queue.writer()

    # Track worker states
    inference_worker_done = threading.Event()
    training_worker_done = threading.Event()
    inference_error = None
    training_error = None
    training_worker_instance = None

    # Track metrics
    rollouts_generated = 0
    training_steps_completed = 0

    def run_inference_worker():
        """Run inference worker in thread."""
        nonlocal inference_error, rollouts_generated
        try:
            worker = InferenceWorker(
                training_config, inference_worker_config, rollout_writer=queue_writer
            )

            # Override the run method to count generated rollouts
            original_generate_batch = worker._generate_rollout_batch

            def counting_generate_batch():
                nonlocal rollouts_generated
                batch_data, metrics = original_generate_batch()
                rollouts_generated += 1
                return batch_data, metrics

            worker._generate_rollout_batch = counting_generate_batch
            worker.run()
        except Exception as e:
            inference_error = e
        finally:
            inference_worker_done.set()

    def run_training_worker():
        """Run training worker in thread."""
        nonlocal training_error, training_steps_completed, training_worker_instance
        try:
            worker = TrainingWorker(training_config, worker_config, rollout_reader=queue_reader)
            training_worker_instance = worker

            # Override train method to count steps
            original_train_step = worker.train_step

            def counting_train_step(train_state, rng, batch):
                nonlocal training_steps_completed
                result = original_train_step(train_state, rng, batch)
                training_steps_completed += 1
                return result

            worker.train_step = counting_train_step
            worker.train()
        except Exception as e:
            training_error = e
        finally:
            training_worker_done.set()

    # Start workers in separate threads
    inference_thread = threading.Thread(target=run_inference_worker, daemon=True)
    training_thread = threading.Thread(target=run_training_worker, daemon=True)

    inference_thread.start()
    time.sleep(0.5)  # Let inference worker start first
    training_thread.start()

    # Wait for both workers to complete with timeout
    timeout = 10  # seconds
    start_time = time.time()

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        queue_size = rollout_queue.reader().get_queue_size()
        print(
            f"Time: {elapsed:.1f}s, Inference done: {inference_worker_done.is_set()}, "
            f"Training done: {training_worker_done.is_set()}, Queue size: {queue_size}, "
            f"Rollouts: {rollouts_generated}, Training steps: {training_steps_completed}"
        )

        # Stop training worker if we've completed the expected steps
        if (
            training_steps_completed >= training_config.hyperparameters.num_train_steps
            and training_worker_instance is not None
            and not training_worker_done.is_set()
        ):
            training_worker_instance.stop()

        if inference_worker_done.is_set() and training_worker_done.is_set():
            break
        time.sleep(1.0)

    # Check for timeouts
    if not inference_worker_done.is_set():
        pytest.fail("Inference worker did not complete within timeout")

    if not training_worker_done.is_set():
        pytest.fail("Training worker did not complete within timeout")

    # Wait for threads to fully finish
    inference_thread.join(timeout=5)
    training_thread.join(timeout=5)

    # Check for errors
    if inference_error:
        import traceback

        print(f"Inference worker error: {inference_error}")
        print("Inference worker traceback:")
        print(
            traceback.format_exception(
                type(inference_error), inference_error, inference_error.__traceback__
            )
        )
        pytest.fail(f"Inference worker failed: {inference_error}")

    if training_error:
        import traceback

        print(f"Training worker error: {training_error}")
        print("Training worker traceback:")
        print(
            traceback.format_exception(
                type(training_error), training_error, training_error.__traceback__
            )
        )
        pytest.fail(f"Training worker failed: {training_error}")

    # Verify expected steps completed
    expected_rollouts = inference_worker_config.max_rollouts
    expected_training_steps = training_config.hyperparameters.num_train_steps

    assert rollouts_generated >= expected_rollouts, (
        f"Expected {expected_rollouts} rollouts, got {rollouts_generated}"
    )
    assert training_steps_completed >= expected_training_steps, (
        f"Expected {expected_training_steps} training steps, got {training_steps_completed}"
    )

    print(f"✓ Generated {rollouts_generated} rollouts")
    print(f"✓ Completed {training_steps_completed} training steps")
    print("✓ Workers communicated successfully through in-memory queue")
