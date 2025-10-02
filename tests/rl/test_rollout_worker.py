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

"""Tests for rollout worker curriculum evaluation."""

from unittest.mock import MagicMock

import jax.random as jrandom
import numpy as np
import pytest
import ray

from marin.rl.curriculum import Curriculum, CurriculumConfig, LessonConfig, LessonDependency
from marin.rl.environments.base import EnvConfig
from marin.rl.rollout_worker import LevanterInferenceContext, RolloutWorker
from marin.rl.types import InferenceChoice, InferenceResponse


class MockTokenizer:
    """Mock tokenizer for testing."""

    vocab_size = 1000
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        """Simple hash-based encoding."""
        text_hash = hash(text) % (self.vocab_size - 100)
        seq_len = min(len(text.split()) + 2, 10)
        return [(text_hash + i) % (self.vocab_size - 100) + 50 for i in range(seq_len)]

    def decode(self, token_ids, skip_special_tokens=True):
        """Simple dummy decode."""
        return f"decoded_{hash(tuple(token_ids)) % 1000}"


class MockInferenceServer:
    """Mock inference server that generates deterministic responses."""

    def __init__(self):
        self.config = MagicMock()
        self.config.host = "localhost"
        self.config.port = 8000


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def curriculum_config():
    """Create a simple curriculum with two lessons."""
    return CurriculumConfig(
        lessons={
            "basic": LessonConfig(
                lesson_id="basic",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
            ),
            "advanced": LessonConfig(
                lesson_id="advanced",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition", "seed": 43},
                ),
                dependencies=[LessonDependency(dependency_id="basic", reward_threshold=0.5)],
            ),
        },
        eval_frequency=10,
        eval_n_examples=4,
        eval_n_generations=1,
    )


def test_evaluate_curriculum(curriculum_config, mock_tokenizer):
    """Test that _evaluate_curriculum evaluates all lessons and updates curriculum."""
    # Initialize Ray for curriculum actor
    if not ray.is_initialized():
        ray.init(num_cpus=1, ignore_reinit_error=True)

    try:
        # Create curriculum actor
        curriculum_actor = ray.remote(Curriculum).remote(curriculum_config)

        # Create mock inference context that returns predictable responses
        class TestInferenceContext:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            @property
            def tokenizer(self):
                return self._tokenizer

            def generate(self, prompts, temperature, n_generations):
                """Generate mock responses with high reward."""
                responses = []
                for prompt in prompts:
                    # For cats task, respond with "cats" (high reward)
                    # For addition task, respond with correct answer (high reward)
                    if "cats" in prompt.lower():
                        response_text = "cats cats cats"
                    else:
                        # Extract numbers from addition prompt
                        import re

                        numbers = re.findall(r"\d+", prompt)
                        if len(numbers) >= 2:
                            result = int(numbers[0]) + int(numbers[1])
                            response_text = str(result)
                        else:
                            response_text = "42"

                    prompt_tokens = self._tokenizer.encode(prompt)
                    response_tokens = self._tokenizer.encode(response_text)
                    logprobs = np.full(len(response_tokens), -0.1, dtype=np.float32)

                    choices = []
                    for _ in range(n_generations):
                        choices.append(
                            InferenceChoice(
                                response_text=response_text,
                                response_tokens=np.array(response_tokens, dtype=np.int32),
                                logprobs=logprobs,
                            )
                        )

                    responses.append(
                        InferenceResponse(
                            prompt=prompt,
                            prompt_tokens=np.array(prompt_tokens, dtype=np.int32),
                            choices=choices,
                        )
                    )
                return responses

        # Create a minimal RolloutWorker just to test _evaluate_curriculum
        # We'll mock most of the initialization to avoid needing a full setup
        worker = MagicMock()
        worker.config = MagicMock()
        worker.config.curriculum_config = curriculum_config
        worker.config.max_input_length = 128
        worker.config.max_output_length = 128
        worker.config.temperature = 0.7
        worker.config.stop_tokens = None
        worker.config.trainer = MagicMock()
        worker.config.trainer.device_mesh = MagicMock()
        worker.config.trainer.compute_axis_mapping = {}
        worker.curriculum_actor = curriculum_actor
        worker._tokenizer = mock_tokenizer
        worker._environments = {}

        # Mock the _load_environment method to create and cache environments
        def load_environment(lesson_id):
            if lesson_id not in worker._environments:
                from marin.rl.environments.base import load_environment_from_spec

                lesson_config = curriculum_config.lessons[lesson_id]
                worker._environments[lesson_id] = load_environment_from_spec(lesson_config.env_config)
            return worker._environments[lesson_id]

        worker._load_environment = load_environment

        # Bind the _evaluate_curriculum method to our worker
        from types import MethodType

        worker._evaluate_curriculum = MethodType(RolloutWorker._evaluate_curriculum, worker)

        # Create test inference context
        test_ctx = TestInferenceContext(mock_tokenizer)

        # Monkeypatch LevanterInferenceContext to return our test context
        original_init = LevanterInferenceContext.__init__

        def mock_init(self, **kwargs):
            # Manually set the private attribute and bind the methods
            self._tokenizer = test_ctx._tokenizer
            self._stop_tokens = kwargs.get("stop_tokens")
            self.generate = test_ctx.generate
            # Don't set the inference_server or max_tokens since we're mocking generate

        LevanterInferenceContext.__init__ = mock_init

        try:
            # Run evaluation
            rng = jrandom.PRNGKey(42)
            eval_metrics = worker._evaluate_curriculum(rng, step=100)

            # Verify metrics were returned
            assert eval_metrics is not None
            assert isinstance(eval_metrics, dict)

            # Check that both lessons were evaluated
            assert "eval/basic/success_rate" in eval_metrics
            assert "eval/advanced/success_rate" in eval_metrics

            # Verify curriculum was updated by checking if curriculum actor received the stats
            # Get curriculum state
            curriculum_state = ray.get(curriculum_actor.get_metrics.remote())

            # Both lessons should have been evaluated (step=100)
            # The curriculum should have been updated with eval stats
            assert curriculum_state["step"] >= 100

        finally:
            # Restore original init
            LevanterInferenceContext.__init__ = original_init

    finally:
        ray.shutdown()


def test_evaluate_curriculum_updates_lesson_unlock(curriculum_config, mock_tokenizer):
    """Test that evaluation can trigger lesson unlocking via curriculum updates."""
    # Initialize Ray for curriculum actor
    if not ray.is_initialized():
        ray.init(num_cpus=1, ignore_reinit_error=True)

    try:
        # Create curriculum actor
        curriculum_actor = ray.remote(Curriculum).remote(curriculum_config)

        # Initially only basic should be unlocked
        metrics = ray.get(curriculum_actor.get_metrics.remote())
        assert metrics["unlocked_lessons"] == 1
        assert "basic" in metrics["sampling_weights"]
        assert "advanced" not in metrics["sampling_weights"]

        # Simulate training on basic lesson to build up stats
        from marin.rl.types import RolloutStats

        # Add training stats with plateau (constant high rewards)
        training_stats = []
        for _ in range(50):
            training_stats.append(RolloutStats(lesson_id="basic", episode_reward=0.8, env_example_id="example_1"))

        ray.get(curriculum_actor.update_lesson_stats.remote(training_stats, mode="training", current_step=50))

        # Now add eval stats (this is what _evaluate_curriculum would do)
        eval_stats = []
        for _ in range(10):
            eval_stats.append(RolloutStats(lesson_id="basic", episode_reward=0.8, env_example_id="eval_example_1"))

        ray.get(curriculum_actor.update_lesson_stats.remote(eval_stats, mode="eval", current_step=100))

        # Check if advanced lesson unlocked
        metrics = ray.get(curriculum_actor.get_metrics.remote())

        # Advanced should now be unlocked since basic has plateaued above threshold
        assert metrics["unlocked_lessons"] == 2, f"Expected 2 unlocked lessons, got {metrics['unlocked_lessons']}"
        assert "advanced" in metrics["sampling_weights"], "Advanced lesson should be unlocked"

    finally:
        ray.shutdown()
