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
Test RolloutContext and its utility functions.

This file contains:
1. Unit tests for stateless utility functions (compute_batch_metrics, etc.)
2. Mock-based tests for RolloutContext API (verifying method signatures and behavior)
3. Integration test placeholders (full infrastructure tests in tests/rl/integration/)

The RolloutContext refactor split responsibilities:
- RolloutContext: Manages inference, curriculum, rollout generation
- RolloutWorker: Handles I/O (weight transfer, storage, logging)
"""

import time
from unittest.mock import MagicMock, patch, Mock

import numpy as np

from marin.rl.rollout_context import (
    RolloutContext,
    compute_batch_metrics,
    build_eval_metrics,
    format_sample_for_logging,
    compute_rollout_stats,
)
from marin.rl.types import Rollout, RolloutGroup, RolloutBatch, RolloutMetadata


class SimpleTokenizer:
    """Minimal tokenizer for testing."""

    vocab_size = 50000

    def decode(self, tokens, skip_special_tokens=True):
        return f"Decoded text with {len(tokens)} tokens"

    def convert_tokens_to_ids(self, token):
        return hash(token) % self.vocab_size


def create_test_rollout(reward: float, example_id: str, n_tokens: int = 10) -> Rollout:
    """Helper to create a test rollout."""
    return Rollout(
        env_name="TestEnv",
        env_example_id=example_id,
        prompt_tokens=np.array([1, 2, 3], dtype=np.int32),
        response_tokens=np.array(list(range(n_tokens)), dtype=np.int32),
        response_logprobs=np.array([0.0] * n_tokens, dtype=np.float32),
        token_rewards=np.array([reward / n_tokens] * n_tokens, dtype=np.float32),
        episode_reward=reward,
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )


def test_compute_rollout_stats():
    """Test compute_rollout_stats utility function."""
    rollout1 = create_test_rollout(reward=1.0, example_id="ex1")
    rollout2 = create_test_rollout(reward=0.0, example_id="ex2")
    rollout3 = create_test_rollout(reward=0.5, example_id="ex3")

    batch = RolloutBatch(
        groups=[
            RolloutGroup(rollouts=[rollout1, rollout2]),
            RolloutGroup(rollouts=[rollout3]),
        ],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    stats = compute_rollout_stats(batch, "test_lesson")

    assert len(stats) == 3
    assert stats[0].lesson_id == "test_lesson"
    assert stats[0].episode_reward == 1.0
    assert stats[0].env_example_id == "ex1"
    assert stats[1].episode_reward == 0.0
    assert stats[2].episode_reward == 0.5


def test_compute_batch_metrics():
    """Test compute_batch_metrics utility function."""
    rollout1 = create_test_rollout(reward=1.0, example_id="ex1")
    rollout2 = create_test_rollout(reward=0.0, example_id="ex2")
    rollout3 = create_test_rollout(reward=1.0, example_id="ex3")
    rollout4 = create_test_rollout(reward=0.5, example_id="ex4")

    batch = RolloutBatch(
        groups=[
            RolloutGroup(rollouts=[rollout1, rollout2]),
            RolloutGroup(rollouts=[rollout3, rollout4]),
        ],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    metrics = compute_batch_metrics(batch, "test_lesson")

    assert metrics.total_count == 4
    assert metrics.success_count == 3  # rewards > 0 (three rollouts: 1.0, 1.0, 0.5)
    assert metrics.avg_reward == 0.625  # (1.0 + 0.0 + 1.0 + 0.5) / 4
    assert len(metrics.rollout_stats) == 4


def test_compute_batch_metrics_empty():
    """Test compute_batch_metrics with empty batch."""
    batch = RolloutBatch(
        groups=[],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    metrics = compute_batch_metrics(batch, "test_lesson")

    assert metrics.total_count == 0
    assert metrics.success_count == 0
    assert metrics.avg_reward == 0.0
    assert len(metrics.rollout_stats) == 0


def test_build_eval_metrics():
    """Test build_eval_metrics utility function."""
    rollout1 = create_test_rollout(reward=1.0, example_id="ex1")
    rollout2 = create_test_rollout(reward=0.0, example_id="ex2")
    rollout3 = create_test_rollout(reward=1.0, example_id="ex3")
    rollout4 = create_test_rollout(reward=0.0, example_id="ex4")

    batch = RolloutBatch(
        groups=[
            RolloutGroup(rollouts=[rollout1, rollout2]),
            RolloutGroup(rollouts=[rollout3, rollout4]),
        ],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    batch_metrics = compute_batch_metrics(batch, "math_lesson")
    metrics = build_eval_metrics("inference.eval", "math_lesson", batch_metrics)

    assert "inference.eval/math_lesson/success_rate" in metrics
    assert "inference.eval/math_lesson/avg_reward" in metrics
    assert "inference.eval/math_lesson/total_count" in metrics

    assert metrics["inference.eval/math_lesson/success_rate"] == 0.5
    assert metrics["inference.eval/math_lesson/avg_reward"] == 0.5
    assert metrics["inference.eval/math_lesson/total_count"] == 4


def test_build_eval_metrics_empty():
    """Test build_eval_metrics with empty batch."""
    batch = RolloutBatch(
        groups=[],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    batch_metrics = compute_batch_metrics(batch, "test_lesson")
    metrics = build_eval_metrics("rollout", "test_lesson", batch_metrics)

    assert metrics == {}


def test_format_sample_for_logging():
    """Test format_sample_for_logging utility function."""
    tokenizer = SimpleTokenizer()

    rollout1 = create_test_rollout(reward=1.0, example_id="example_123", n_tokens=15)
    rollout2 = create_test_rollout(reward=0.0, example_id="example_456", n_tokens=20)

    batch = RolloutBatch(
        groups=[
            RolloutGroup(rollouts=[rollout1]),
            RolloutGroup(rollouts=[rollout2]),
        ],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    sample = format_sample_for_logging(batch, tokenizer)

    assert "sample_prompt" in sample
    assert "sample_response" in sample
    assert "sample_example_id" in sample

    assert sample["sample_example_id"] == "example_123"
    assert "3 tokens" in sample["sample_prompt"]  # 3 prompt tokens
    assert "15 tokens" in sample["sample_response"]  # 15 response tokens


def test_format_sample_for_logging_empty():
    """Test format_sample_for_logging with empty batch."""
    tokenizer = SimpleTokenizer()

    batch = RolloutBatch(
        groups=[],
        metadata=RolloutMetadata(
            worker_id="test_worker",
            timestamp=time.time(),
            weight_step=100,
        ),
    )

    sample = format_sample_for_logging(batch, tokenizer)

    assert sample == {}


# === RolloutContext API Tests (using mocks) ===


@patch("marin.rl.rollout_context.InferenceServer")
@patch("marin.rl.rollout_context.get_or_create_curriculum_actor")
@patch("marin.rl.rollout_context.threading.Thread")
@patch("marin.rl.rollout_context.time.sleep")
def test_rollout_context_initialization(mock_sleep, mock_thread, mock_curriculum, mock_server):
    """Test RolloutContext initializes with required components."""
    from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
    from marin.rl.environments.base import EnvConfig

    # Mock the required configs
    mock_inference_config = Mock()
    mock_model_config = Mock()
    mock_trainer_config = Mock()
    mock_trainer_config.device_mesh = MagicMock()  # Support context manager
    mock_trainer_config.device_mesh.__enter__ = Mock(return_value=None)
    mock_trainer_config.device_mesh.__exit__ = Mock(return_value=None)
    mock_trainer_config.compute_axis_mapping = {}

    mock_tokenizer = SimpleTokenizer()
    mock_model = Mock()

    # Mock curriculum config
    curriculum_config = CurriculumConfig(
        lessons={
            "test_lesson": LessonConfig(
                lesson_id="test_lesson",
                env_config=EnvConfig(
                    env_class="test.DummyEnv",
                    env_args={},
                ),
                sampling_params=SamplingParams(
                    temperature=0.7,
                    n_prompts=10,
                    n_generations_per_prompt=2,
                ),
            )
        },
        eval_frequency=100,
        eval_n_examples=10,
    )

    # Mock the inference server creation
    mock_server_instance = Mock()
    mock_server.create.return_value = mock_server_instance

    # Mock curriculum actor
    mock_curriculum_actor = Mock()
    mock_curriculum.return_value = mock_curriculum_actor

    # Create RolloutContext
    ctx = RolloutContext(
        inference_config=mock_inference_config,
        model_config=mock_model_config,
        trainer_config=mock_trainer_config,
        curriculum_config=curriculum_config,
        tokenizer=mock_tokenizer,
        initial_model=mock_model,
        worker_id="test_worker",
    )

    # Verify initialization
    assert ctx.tokenizer == mock_tokenizer
    assert ctx.curriculum_config == curriculum_config
    assert ctx.worker_id == "test_worker"
    assert ctx._policy_model == mock_model

    # Verify inference server was created
    mock_server.create.assert_called_once()

    # Verify curriculum actor was created
    mock_curriculum.assert_called_once_with(curriculum_config)
    assert ctx._curriculum_actor == mock_curriculum_actor


@patch("marin.rl.rollout_context.InferenceServer")
@patch("marin.rl.rollout_context.get_or_create_curriculum_actor")
@patch("marin.rl.rollout_context.threading.Thread")
@patch("marin.rl.rollout_context.time.sleep")
def test_rollout_context_update_model(mock_sleep, mock_thread, mock_curriculum, mock_server):
    """Test RolloutContext.update_model() updates the model and reloads inference server."""
    from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
    from marin.rl.environments.base import EnvConfig

    # Setup mocks
    mock_inference_config = Mock()
    mock_model_config = Mock()
    mock_trainer_config = Mock()
    mock_trainer_config.device_mesh = MagicMock()  # Support context manager
    mock_trainer_config.device_mesh.__enter__ = Mock(return_value=None)
    mock_trainer_config.device_mesh.__exit__ = Mock(return_value=None)
    mock_trainer_config.compute_axis_mapping = {}

    mock_tokenizer = SimpleTokenizer()
    mock_initial_model = Mock(name="initial_model")
    mock_new_model = Mock(name="new_model")

    curriculum_config = CurriculumConfig(
        lessons={
            "test": LessonConfig(
                lesson_id="test",
                env_config=EnvConfig(env_class="test.Env", env_args={}),
                sampling_params=SamplingParams(temperature=1.0, n_prompts=1, n_generations_per_prompt=1),
            )
        },
        eval_frequency=100,
        eval_n_examples=10,
    )

    mock_server_instance = Mock()
    mock_server.create.return_value = mock_server_instance
    mock_curriculum.return_value = Mock()

    # Create context
    ctx = RolloutContext(
        inference_config=mock_inference_config,
        model_config=mock_model_config,
        trainer_config=mock_trainer_config,
        curriculum_config=curriculum_config,
        tokenizer=mock_tokenizer,
        initial_model=mock_initial_model,
    )

    # Verify initial model
    assert ctx._policy_model == mock_initial_model

    # Update model
    ctx.update_model(mock_new_model)

    # Verify model was updated
    assert ctx._policy_model == mock_new_model

    # Verify inference server reload was called
    mock_server_instance.reload.assert_called_once()


@patch("marin.rl.rollout_context.InferenceServer")
@patch("marin.rl.rollout_context.get_or_create_curriculum_actor")
@patch("marin.rl.rollout_context.threading.Thread")
@patch("marin.rl.rollout_context.time.sleep")
def test_rollout_context_shutdown(mock_sleep, mock_thread, mock_curriculum, mock_server):
    """Test RolloutContext.shutdown() cleans up resources."""
    from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
    from marin.rl.environments.base import EnvConfig

    # Setup mocks
    mock_inference_config = Mock()
    mock_model_config = Mock()
    mock_trainer_config = Mock()
    mock_trainer_config.device_mesh = MagicMock()  # Support context manager
    mock_trainer_config.device_mesh.__enter__ = Mock(return_value=None)
    mock_trainer_config.device_mesh.__exit__ = Mock(return_value=None)
    mock_trainer_config.compute_axis_mapping = {}

    curriculum_config = CurriculumConfig(
        lessons={
            "test": LessonConfig(
                lesson_id="test",
                env_config=EnvConfig(env_class="test.Env", env_args={}),
                sampling_params=SamplingParams(temperature=1.0, n_prompts=1, n_generations_per_prompt=1),
            )
        },
        eval_frequency=100,
        eval_n_examples=10,
    )

    mock_server_instance = Mock()
    mock_server.create.return_value = mock_server_instance
    mock_curriculum.return_value = Mock()

    # Create context
    ctx = RolloutContext(
        inference_config=mock_inference_config,
        model_config=mock_model_config,
        trainer_config=mock_trainer_config,
        curriculum_config=curriculum_config,
        tokenizer=SimpleTokenizer(),
        initial_model=Mock(),
    )

    # Shutdown
    ctx.shutdown()

    # Verify inference server shutdown was called
    mock_server_instance.shutdown.assert_called_once()


def test_rollout_context_set_weight_step():
    """Test RolloutContext.set_weight_step() updates the weight step."""
    with (
        patch("marin.rl.rollout_context.InferenceServer"),
        patch("marin.rl.rollout_context.get_or_create_curriculum_actor"),
        patch("marin.rl.rollout_context.threading.Thread"),
        patch("marin.rl.rollout_context.time.sleep"),
    ):

        from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
        from marin.rl.environments.base import EnvConfig

        mock_trainer_config = Mock()
        mock_trainer_config.device_mesh = MagicMock()  # Support context manager
        mock_trainer_config.device_mesh.__enter__ = Mock(return_value=None)
        mock_trainer_config.device_mesh.__exit__ = Mock(return_value=None)
        mock_trainer_config.compute_axis_mapping = {}

        curriculum_config = CurriculumConfig(
            lessons={
                "test": LessonConfig(
                    lesson_id="test",
                    env_config=EnvConfig(env_class="test.Env", env_args={}),
                    sampling_params=SamplingParams(temperature=1.0, n_prompts=1, n_generations_per_prompt=1),
                )
            },
            eval_frequency=100,
            eval_n_examples=10,
        )

        ctx = RolloutContext(
            inference_config=Mock(),
            model_config=Mock(),
            trainer_config=mock_trainer_config,
            curriculum_config=curriculum_config,
            tokenizer=SimpleTokenizer(),
            initial_model=Mock(),
        )

        # Initial weight step should be 0
        assert ctx._current_weight_step == 0

        # Update weight step
        ctx.set_weight_step(42)
        assert ctx._current_weight_step == 42

        ctx.set_weight_step(100)
        assert ctx._current_weight_step == 100
