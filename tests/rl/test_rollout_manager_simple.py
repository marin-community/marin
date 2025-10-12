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
Simple test demonstrating RolloutManager usage without infrastructure overhead.

The core rollout generation logic can be implemented in under 20 lines, compared to 100+ lines with full infrastructure.
"""

import numpy as np
import jax.numpy as jnp

from marin.rl.environments.base import MarinEnv
from marin.rl.inference_ctx import InferenceContext
from marin.rl.rollout_worker import RolloutManager
from marin.rl.types import Rollout, RolloutGroup


class MockInferenceContext(InferenceContext):
    """Minimal mock context for testing."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = 128
        self._stop_tokens = None

    def batch_completions(self, prompts, temperature, n, max_tokens=None, stop=None):
        """Return fake completions for testing.

        We create mock objects that have the necessary fields
        without importing OpenAI types directly.
        """
        completions = []
        for i, _prompt in enumerate(prompts):
            choices = []
            for j in range(n):
                # Create mock choice object with required fields
                class MockChoice:
                    def __init__(self, idx=j):
                        self.index = idx
                        self.message = type("obj", (), {"content": f"Response {idx}", "role": "assistant"})()

                        # Mock logprobs structure
                        class LogprobToken:
                            def __init__(self, token, logprob):
                                self.token = token
                                self.logprob = logprob
                                self.top_logprobs = []
                                self.bytes = None

                        self.logprobs = type(
                            "obj", (), {"content": [LogprobToken(f"tok_{k}", -0.5) for k in range(5)]}
                        )()
                        self.finish_reason = "stop"

                choices.append(MockChoice())

            # Create mock completion
            class MockCompletion:
                def __init__(self, comp_id=i, comp_choices=choices):
                    self.id = f"test-{comp_id}"
                    self.choices = comp_choices
                    self.usage = type("obj", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})()

            completions.append(MockCompletion())
        return completions

    def tokenize_prompt(self, prompt):
        return np.array([1, 2, 3], dtype=np.int32)

    def response_tokens_from_choice(self, choice):
        return np.array([10, 11, 12, 13, 14], dtype=np.int32)

    def logprobs_from_choice(self, choice):
        return np.array([-0.5] * 5, dtype=np.float32)


class SimpleTestEnv(MarinEnv):
    """Minimal test environment."""

    def sample(self, inference_ctx, n_examples, n_generations, temperature, prng_key, mode="train"):
        # Generate fake prompts
        prompts = [f"Test prompt {i}" for i in range(n_examples)]

        # Get completions
        completions = inference_ctx.batch_completions(prompts, temperature, n_generations)

        # Create rollouts
        groups = []
        for prompt, completion in zip(prompts, completions, strict=False):
            rollouts = []
            for choice in completion.choices:
                rollout = Rollout(
                    env_name="SimpleTestEnv",
                    env_example_id=f"test_{hash(prompt)}",
                    prompt_tokens=inference_ctx.tokenize_prompt(prompt),
                    response_tokens=inference_ctx.response_tokens_from_choice(choice),
                    response_logprobs=inference_ctx.logprobs_from_choice(choice),
                    token_rewards=jnp.ones(5) * 0.5,
                    episode_reward=0.5,
                )
                rollouts.append(rollout)
            groups.append(RolloutGroup(rollouts=rollouts))

        metrics = {"test_metric": 1.0}
        return groups, metrics


def test_simple_rollout_generation():
    """Test rollout generation without any infrastructure setup."""

    # Create a simple tokenizer mock
    class SimpleTokenizer:
        vocab_size = 50000

    # Create manager - no JAX mesh, no threads, no servers!
    manager = RolloutManager(
        env=SimpleTestEnv(),
        tokenizer=SimpleTokenizer(),
        inference_ctx=MockInferenceContext(SimpleTokenizer()),
    )

    # Generate rollouts synchronously
    batch, metrics = manager.sample_rollout_batch(
        n_examples=4,
        n_generations=2,
        temperature=0.7,
    )

    # Verify results
    assert batch is not None
    assert len(batch.groups) == 4
    assert batch.metadata.weight_step == 0

    for group in batch.groups:
        assert len(group.rollouts) == 2  # n_generations
        for rollout in group.rollouts:
            assert rollout.env_name == "SimpleTestEnv"
            assert len(rollout.response_tokens) == 5
            assert rollout.metadata.weight_step == 0

    print("✅ Rollout generation test passed!")
    print(f"Generated {len(batch.groups)} rollout groups")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    test_simple_rollout_generation()
