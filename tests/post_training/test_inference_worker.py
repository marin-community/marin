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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from marin.post_training.inference_worker import InferenceWorker
from marin.post_training.rollout_storage import FileRolloutWriter, RolloutBatch
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
)


@pytest.fixture
def training_config():
    """Create a minimal training configuration for testing."""
    model_paths_config = ModelPathsConfig(
        params=None,
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
    )

    optim_config = OptimizerConfig(
        init_lr=5e-7,
        end_lr=5e-7,
        lr=5e-7,
        lr_warmup_steps=0,
        lr_decay_steps=16,
        b1=0.9,
        b2=0.95,
        clip_gradient=1.0,
        weight_decay=0.0,
        bf16_momentum=False,
        multiply_by_parameter_scale=False,
        weight_decay_exclusions=(),
        schedule="cos",
        grad_accum_steps=16,
    )

    logger_config = LoggerConfigData(
        online=False,
        prefix="test",
        prefix_to_id=True,
    )

    generation_config = GenerationConfig(
        max_output_length=256, temperature=1.0, stop_tokens=[[128001]], n_generations=4
    )

    test_generation_config = GenerationConfig(
        max_output_length=256, temperature=0.0, stop_tokens=[[128001]], n_generations=1
    )

    model_config_override = ModelOverrideConfig(
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128002,
        max_sequence_length=512,
        remat_block="nothing_saveable",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    checkpointer_config = CheckpointerConfigData(save_optimizer_state=False, save_float_dtype="bf16")

    return TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="bf16",
            inference_activation_dtype="bf16",
            training_param_dtype="bf16",
            training_activation_dtype="bf16",
            model_config_override=model_config_override,
            train_attention_kernel_config="default:{}",
            prefill_attention_kernel_config="default:{}",
            generate_attention_kernel_config="default:{}",
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=0,
            max_input_length=128,
            max_output_length=256,
            train_bsize=4,
            decode_bsize=2,
            prefill_bsize=2,
            reference_logprobs_bsize=2,
            n_prompts_per_step=4,
            optim_config=optim_config,
            pad_token_id=128002,
            kl_coef=1e-3,
        ),
        logging=LoggingConfig(
            log_freq=8,
            num_eval_examples=8,
            save_model_freq=0,
            wandb_project="test_project",
            logger_config=logger_config,
            save_initial_checkpoint=False,
            log_initial_step=True,
            max_checkpoints=None,
        ),
        environment=EnvironmentConfig(
            train_environments_path="environments_test.json",
            test_environments_path="environments_test.json",
        ),
        distributed=DistributedConfig(
            sharding=[1, 1, 1, -1],
            physical_axis_splitting=False,
            jax_distributed_initalize_config={},
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir="/tmp/test_output",
        checkpointer_config=checkpointer_config,
    )


@pytest.fixture
def inference_config():
    """Create an inference worker configuration for testing."""
    return InferenceWorkerConfig(
        environment_spec="math:difficulty=easy",
        checkpoint_source_path="/tmp/test_checkpoints",
        rollout_output_path="/tmp/test_rollouts",
        checkpoint_poll_interval=1.0,
        rollout_batch_size=4,
        n_generations=2,
        n_examples_per_batch=2,
        use_gcs=False,
        max_rollouts=2,
        checkpoint_timeout=10.0,
    )


def test_inference_worker_config():
    """Test InferenceWorkerConfig creation and validation."""
    config = InferenceWorkerConfig(
        environment_spec="math:difficulty=easy",
        checkpoint_source_path="/path/to/checkpoints",
        rollout_output_path="/path/to/rollouts",
    )

    assert config.environment_spec == "math:difficulty=easy"
    assert config.checkpoint_source_path == "/path/to/checkpoints"
    assert config.rollout_output_path == "/path/to/rollouts"
    assert config.checkpoint_poll_interval == 30.0  # default value
    assert config.n_generations == 64  # default value


def test_rollout_writer():
    """Test FileRolloutWriter functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = FileRolloutWriter(temp_dir)

        # Create mock rollout batch
        import numpy as np

        mock_batch = RolloutBatch(
            input_ids=np.array([[1, 2, 3]]),
            attention_mask=np.array([[1, 1, 1]]),
            position_ids=np.array([[0, 1, 2]]),
            target_ids=np.array([[2, 3, 4]]),
            loss_weights=np.array([[1.0, 1.0, 1.0]]),
            loss_masks=np.array([[1.0, 1.0, 1.0]]),
            reference_logprobs=np.array([[0.1, 0.2, 0.3]]),
            metadata={"test_metric": 0.5},
        )

        # Write batch
        writer.write_batch(mock_batch)

        # Check file was created (using fsspec methods)
        batch_path = f"{temp_dir}/batch_0000000000.pkl"
        assert writer.fs.exists(batch_path)

        # Test reading the batch back by reading directly
        import pickle
        with writer.fs.open(batch_path, "rb") as f:
            read_batch = pickle.load(f)
        assert read_batch.metadata["test_metric"] == 0.5

        # Write metadata
        writer.write_metadata({"worker_id": "test", "status": "running"})

        # Check metadata file exists
        metadata_path = f"{temp_dir}/metadata.json"
        assert writer.fs.exists(metadata_path)


def test_create_rollout_writer():
    """Test rollout writer creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        FileRolloutWriter(temp_dir)
        FileRolloutWriter(f"file://{temp_dir}")


def test_inference_worker_config_assignment():
    """Test InferenceWorker config assignment without full JAX initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        inference_config = InferenceWorkerConfig(
            environment_spec="math:difficulty=easy",
            checkpoint_source_path=temp_dir,
            rollout_output_path=temp_dir,
        )

        # Test that we can create the config objects properly
        assert inference_config.environment_spec == "math:difficulty=easy"
        assert inference_config.checkpoint_source_path == temp_dir
        assert inference_config.rollout_output_path == temp_dir


def test_find_latest_checkpoint():
    """Test checkpoint discovery functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir)

        # Create mock checkpoint directories
        (checkpoint_dir / "step_100").mkdir()
        (checkpoint_dir / "step_200").mkdir()
        (checkpoint_dir / "step_150").mkdir()

        # Create params files
        (checkpoint_dir / "step_100" / "params.msgpack").touch()
        (checkpoint_dir / "step_200" / "params.msgpack").touch()
        (checkpoint_dir / "step_150" / "params.msgpack").touch()

        worker = InferenceWorker.__new__(InferenceWorker)
        worker.inference_config = Mock()
        worker.inference_config.checkpoint_source_path = str(checkpoint_dir)

        latest = worker._find_latest_checkpoint()
        assert latest is not None
        assert "step_200" in latest
        assert latest.endswith("params.msgpack")


def test_find_latest_checkpoint_no_checkpoints():
    """Test checkpoint discovery when no checkpoints exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(InferenceWorker, "_setup_components"), patch.object(
            InferenceWorker, "__init__", return_value=None
        ):
            worker = InferenceWorker.__new__(InferenceWorker)
            worker.inference_config = Mock()
            worker.inference_config.checkpoint_source_path = str(temp_dir)

            latest = worker._find_latest_checkpoint()
            assert latest is None


def test_find_latest_checkpoint_nonexistent_dir():
    """Test checkpoint discovery when checkpoint directory doesn't exist."""
    with patch.object(InferenceWorker, "_setup_components"), patch.object(
        InferenceWorker, "__init__", return_value=None
    ):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.inference_config = Mock()
        worker.inference_config.checkpoint_source_path = "/nonexistent/path"

        latest = worker._find_latest_checkpoint()
        assert latest is None


def test_inference_worker_main(training_config):
    """Test InferenceWorker main loop with mocked dependencies."""

    inference_config = InferenceWorkerConfig(
        environment_spec="math:difficulty=easy",
        checkpoint_source_path="/tmp/test_checkpoints",
        rollout_output_path="/tmp/test_rollouts",
        checkpoint_poll_interval=0.1,
        n_generations=2,
        n_examples_per_batch=2,
        use_gcs=False,
        max_rollouts=2,
        checkpoint_timeout=1.0,
    )

    _ = InferenceWorker(training_config, inference_config)
