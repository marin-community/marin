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
Test RolloutContext - demonstrates lightweight testing without infrastructure.

Compare this to test_rollout_worker.py which requires full infrastructure setup.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any

from marin.rl.environments.base import MarinEnv, EnvConfig
from marin.rl.inference_ctx import InferenceContext
from marin.rl.rollout_context import (
    RolloutContext,
    compute_batch_metrics,
    build_eval_metrics,
    format_sample_for_logging,
)
from marin.rl.types import Rollout, RolloutGroup
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams


class MockInferenceContext(InferenceContext):
    """Mock inference context for testing - no server required!"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = 128
        self._stop_tokens = None
        
    def batch_completions(self, prompts, temperature, n, max_tokens=None, stop=None):
        """Return fake completions."""
        completions = []
        for prompt in prompts:
            choices = []
            for i in range(n):
                class MockChoice:
                    def __init__(self):
                        self.message = type("obj", (), {"content": f"Response {i}", "role": "assistant"})()
                        class LogprobToken:
                            def __init__(self, token, logprob):
                                self.token = token
                                self.logprob = logprob
                        self.logprobs = type(
                            "obj", (), {"content": [LogprobToken(f"tok_{j}", -0.5) for j in range(5)]}
                        )()
                choices.append(MockChoice())
                
            class MockCompletion:
                def __init__(self):
                    self.choices = choices
            completions.append(MockCompletion())
        return completions
    
    def tokenize_prompt(self, prompt):
        return np.array([1, 2, 3], dtype=np.int32)
    
    def response_tokens_from_choice(self, choice):
        return np.array([10, 11, 12, 13, 14], dtype=np.int32)
    
    def logprobs_from_choice(self, choice):
        return np.array([-0.5] * 5, dtype=np.float32)


class SimpleTestEnv(MarinEnv):
    """Test environment that returns controllable rewards."""
    
    def __init__(self, success_rate: float = 0.5, env_name: str = "SimpleTestEnv"):
        self.success_rate = success_rate
        self.env_name = env_name
        self._call_count = 0
        
    def sample(self, inference_ctx, n_examples, n_generations, temperature, prng_key, mode="train"):
        self._call_count += 1
        prompts = [f"Test prompt {i}" for i in range(n_examples)]
        completions = inference_ctx.batch_completions(prompts, temperature, n_generations)
        
        groups = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            rollouts = []
            for j, choice in enumerate(completion.choices):
                # Deterministic reward based on index for testing
                reward = 1.0 if (i + j) % int(1/self.success_rate) == 0 else 0.0
                
                rollout = Rollout(
                    env_name=self.env_name,
                    env_example_id=f"test_{i}",
                    prompt_tokens=inference_ctx.tokenize_prompt(prompt),
                    response_tokens=inference_ctx.response_tokens_from_choice(choice),
                    response_logprobs=inference_ctx.logprobs_from_choice(choice),
                    token_rewards=jnp.ones(5) * reward,
                    episode_reward=reward,
                )
                rollouts.append(rollout)
            groups.append(RolloutGroup(rollouts=rollouts))
            
        metrics = {"test_metric": 1.0, "call_count": self._call_count}
        return groups, metrics


class SimpleTokenizer:
    """Minimal tokenizer for testing."""
    vocab_size = 50000
    
    def decode(self, tokens, skip_special_tokens=True):
        return f"Decoded text with {len(tokens)} tokens"
    
    def convert_tokens_to_ids(self, token):
        return hash(token) % self.vocab_size


def test_rollout_context_basic():
    """Test basic rollout generation with RolloutContext."""
    tokenizer = SimpleTokenizer()
    ctx = RolloutContext(tokenizer=tokenizer, seed=42)
    
    # Create test environment
    env = SimpleTestEnv(success_rate=0.5)
    inference_ctx = MockInferenceContext(tokenizer)
    
    # Generate rollouts - no JAX setup needed!
    batch, metrics = ctx.sample_rollouts(
        inference_ctx=inference_ctx,
        env_or_lesson_id=env,
        n_examples=4,
        n_generations=2,
        temperature=0.7,
    )
    
    assert batch is not None
    assert len(batch.groups) == 4  # n_examples
    assert all(len(group.rollouts) == 2 for group in batch.groups)  # n_generations
    
    # Check metrics
    batch_metrics = compute_batch_metrics(batch, "test_env")
    assert batch_metrics.total_count == 8  # 4 examples * 2 generations
    assert batch_metrics.success_count == 4  # 50% success rate
    assert batch_metrics.avg_reward == 0.5
    
    # Check environment was called
    assert metrics["call_count"] == 1


def test_rollout_context_with_curriculum():
    """Test RolloutContext with curriculum configuration."""
    tokenizer = SimpleTokenizer()
    
    # Create curriculum config
    curriculum_config = CurriculumConfig(
        lessons={
            "easy": LessonConfig(
                lesson_id="easy",
                env_config=EnvConfig(
                    env_class="tests.rl.test_rollout_context.SimpleTestEnv",
                    env_args={"success_rate": 0.8, "env_name": "EasyEnv"},
                ),
                sampling_params=SamplingParams(
                    temperature=0.5,
                    n_prompts=10,
                    n_generations_per_prompt=2,
                    max_tokens=100,
                ),
            ),
            "hard": LessonConfig(
                lesson_id="hard",
                env_config=EnvConfig(
                    env_class="tests.rl.test_rollout_context.SimpleTestEnv",
                    env_args={"success_rate": 0.2, "env_name": "HardEnv"},
                ),
                sampling_params=SamplingParams(
                    temperature=1.0,
                    n_prompts=5,
                    n_generations_per_prompt=3,
                    max_tokens=200,
                ),
            ),
        },
        eval_frequency=100,
        eval_n_examples=10,
    )
    
    ctx = RolloutContext(
        tokenizer=tokenizer,
        curriculum_config=curriculum_config,
        seed=42,
    )
    
    inference_ctx = MockInferenceContext(tokenizer)
    
    # Test loading environment by lesson ID
    easy_batch, _ = ctx.sample_rollouts(
        inference_ctx=inference_ctx,
        env_or_lesson_id="easy",
        n_examples=4,
        n_generations=2,
        temperature=0.7,  # Should be overridden by lesson config
    )
    
    assert easy_batch is not None
    # Check that environment was loaded correctly
    assert easy_batch.groups[0].rollouts[0].env_name == "EasyEnv"
    
    # Test evaluate_all_lessons
    results = ctx.evaluate_all_lessons(inference_ctx, n_examples_per_lesson=5)
    
    assert "easy" in results
    assert "hard" in results
    
    easy_metrics, _ = results["easy"]
    hard_metrics, _ = results["hard"]
    
    # Easy env should have higher success rate
    assert easy_metrics.avg_reward > hard_metrics.avg_reward


def test_utility_functions():
    """Test the stateless utility functions."""
    tokenizer = SimpleTokenizer()
    ctx = RolloutContext(tokenizer=tokenizer)
    
    env = SimpleTestEnv(success_rate=0.5)
    inference_ctx = MockInferenceContext(tokenizer)
    
    batch, _ = ctx.sample_rollouts(
        inference_ctx=inference_ctx,
        env_or_lesson_id=env,
        n_examples=2,
        n_generations=2,
        temperature=0.7,
    )
    
    # Test compute_batch_metrics
    metrics = compute_batch_metrics(batch, "test")
    assert metrics.total_count == 4
    assert metrics.success_count == 2
    assert len(metrics.rollout_stats) == 4
    
    # Test build_eval_metrics
    eval_metrics = build_eval_metrics("eval", "test", metrics)
    assert "eval/test/success_rate" in eval_metrics
    assert eval_metrics["eval/test/success_rate"] == 0.5
    assert eval_metrics["eval/test/avg_reward"] == 0.5
    
    # Test format_sample_for_logging
    sample_data = format_sample_for_logging(batch, tokenizer)
    assert "sample_prompt" in sample_data
    assert "sample_response" in sample_data
    assert "sample_example_id" in sample_data


def test_rollout_context_hooks_usage():
    """Demonstrate how hooks can use RolloutContext for evaluation."""
    
    class EvalHook:
        """Example evaluation hook that uses RolloutContext."""
        
        def __init__(self, rollout_context: RolloutContext, inference_ctx: InferenceContext):
            self.context = rollout_context
            self.inference_ctx = inference_ctx
            
        def evaluate_environments(self):
            """Evaluate all loaded environments."""
            results = {}
            for env_id, env in self.context.get_loaded_environments().items():
                batch, env_metrics = self.context.sample_rollouts(
                    self.inference_ctx,
                    env_or_lesson_id=env,
                    n_examples=10,
                    n_generations=1,
                    temperature=1.0,
                    mode="eval",
                )
                if batch:
                    metrics = compute_batch_metrics(batch, env_id)
                    results[env_id] = {
                        "success_rate": metrics.success_count / metrics.total_count if metrics.total_count > 0 else 0,
                        "avg_reward": metrics.avg_reward,
                        "env_metrics": env_metrics,
                    }
            return results
    
    # Setup
    tokenizer = SimpleTokenizer()
    ctx = RolloutContext(tokenizer=tokenizer)
    inference_ctx = MockInferenceContext(tokenizer)
    
    # Load some environments
    env1 = SimpleTestEnv(success_rate=0.7, env_name="Env1")
    env2 = SimpleTestEnv(success_rate=0.3, env_name="Env2")
    
    # Sample from them to get them loaded
    ctx.sample_rollouts(inference_ctx, env1, 1, 1, 0.7)
    ctx.sample_rollouts(inference_ctx, env2, 1, 1, 0.7)
    
    # Create hook and evaluate
    hook = EvalHook(ctx, inference_ctx)
    results = hook.evaluate_environments()
    
    # Check results
    assert len(results) == 2  # Should have evaluated both environments
    
    # Get results values (order might vary)
    results_list = list(results.values())
    # Sort by success rate to identify which is which
    results_list.sort(key=lambda x: x["success_rate"])
    
    # Lower success rate should be env2 (0.3), higher should be env1 (0.7)
    assert results_list[0]["success_rate"] < 0.5  # env2
    assert results_list[1]["success_rate"] > 0.5  # env1


if __name__ == "__main__":
    test_rollout_context_basic()
    test_rollout_context_with_curriculum()
    test_utility_functions()
    test_rollout_context_hooks_usage()
    print("✅ All tests passed!")
