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

"""Math-focused RL environment mirroring Tinker's math environment."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
import jax

from marin.post_training.tinker_environments.math_env import (
    MathEnv as TinkerMathEnvBase,
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
    parse_response_for_stop_token,
)
from marin.post_training.tinker_environments.math_grading import extract_boxed, grade_answer, normalize_answer
from marin.rl.inference_ctx import InferenceContext
from marin.rl.math_utils import last_boxed_only_string
from marin.rl.types import Rollout, RolloutGroup

from .base import MarinEnv

logger = logging.getLogger(__name__)


@dataclass
class MathEnvExample:
    """Single math example with cleaned prompt/answer."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TinkerMathEnv(MarinEnv):
    """Math environment using Tinker's grading and prompt format."""

    def __init__(
        self,
        tokenizer=None,
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        seed: int = 0,
        format_coef: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self.format_coef = format_coef
        self.seed = seed
        self._rng = jax.random.PRNGKey(seed)
        
        # Get few-shot prefix from TinkerMathEnvBase
        self.fewshot_prefix = TinkerMathEnvBase.standard_fewshot_prefix()

        self.train_examples = list(self.training_data())
        if max_train_examples is not None:
            logger.info("Limiting train examples to %d.", max_train_examples)
            self.train_examples = self.train_examples[:max_train_examples]

        self.eval_examples = list(self.eval_data())
        if max_eval_examples is not None:
            logger.info("Limiting eval examples to %d.", max_eval_examples)
            self.eval_examples = self.eval_examples[:max_eval_examples]

        logger.info(
            "Initialized TinkerMathEnv with %d train examples and %d eval examples.",
            len(self.train_examples),
            len(self.eval_examples),
        )
    
    def clean_example(self, raw_prompt: str, raw_answer: str, example_id: str) -> MathEnvExample:
        """Clean and process a single example."""
        # Show the transformation pipeline
        boxed_answer = last_boxed_only_string(raw_answer)
        # Use Tinker's normalize_answer instead of our own
        cleaned_answer = normalize_answer(boxed_answer) if boxed_answer else normalize_answer(raw_answer)
        
        # Build chat messages with few-shot examples
        question_text = self.add_instruction(raw_prompt)
        messages = self.fewshot_prefix + [
            {"role": "user", "content": question_text}
        ]
        
        # Apply chat template to get the formatted prompt
        if self.tokenizer:
            processed_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback if tokenizer not available during init
            processed_prompt = question_text

        return MathEnvExample(
            raw_prompt=raw_prompt,
            raw_answer=raw_answer,
            processed_prompt=processed_prompt,
            processed_answer=cleaned_answer,
            example_id=example_id,
        )

    def _prepare_split(
        self,
        dataset: list[dict[str, Any]],
        split_name: str,
        limit: int | None,
    ) -> list[MathEnvExample]:
        """Process a dataset split into MathEnvExample objects."""
        cleaned_examples: list[MathEnvExample] = []
        for idx, item in enumerate(dataset):
            example_id = f"{split_name}_{idx}"
            example = self.clean_example(item["problem"], item["solution"], example_id)
            assert example is not None, f"Failed to clean example {example_id}"

            example.metadata.update({"split": split_name, "source_index": idx})
            cleaned_examples.append(example)
            if limit is not None and len(cleaned_examples) >= limit:
                break

        return cleaned_examples

    @classmethod
    def question_suffix(cls) -> str:
        """Use Tinker's question suffix format."""
        return TinkerMathEnvBase.question_suffix()

    def add_instruction(self, math_problem: str) -> str:
        """Add the standard instruction to a math problem."""
        return f"{math_problem}{self.question_suffix()}"

    def check_format(self, sample_str: str) -> bool:
        """Check if the response follows the boxed format."""
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str, ground_truth: str) -> bool:
        """Grade the answer using Tinker's grading."""
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return grade_answer(answer, ground_truth)

    def sample(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        step: int = 0,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample prompts, evaluate responses, and create rollouts."""

        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode}")

        available_examples = self.train_examples if mode == "train" else self.eval_examples
        if not available_examples:
            raise ValueError(f"No examples available for mode '{mode}'")

        n_to_sample = min(n_examples, len(available_examples))
        
        if mode == "train":
            # Sequential sampling for training
            total_n = len(available_examples)
            start_idx = (step * n_examples) % total_n
            indices = [(start_idx + i) % total_n for i in range(n_to_sample)]
        else:
            # Random sampling for evaluation
            indices = jax.random.choice(prng_key, len(available_examples), shape=(n_to_sample,), replace=False)

        sampled_examples = [available_examples[int(idx)] for idx in indices]

        prompts = [example.processed_prompt for example in sampled_examples]
        completions = inference_ctx.batch_completions(prompts=prompts, temperature=temperature, n=n_generations)

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        format_sum = 0.0
        correct_sum = 0.0
        response_token_count = 0
        prompt_token_count = 0
        truncated_count = 0
        
        # Group stats tracking
        n_mixed = 0
        n_all_good = 0
        n_all_bad = 0
        n_groups = 0

        for example, completion in zip(sampled_examples, completions, strict=True):
            group_rollouts: list[Rollout] = []
            group_correct_scores = []

            for choice in completion.choices:
                # finish_reason mapping might be needed depending on Levanter's output
                finish_reason = getattr(choice, "finish_reason", "unknown")
                
                reward, fmt_score, correct_score = self._score_choice(
                    example=example,
                    response_text=choice.message.content or "",
                    finish_reason=finish_reason,
                    tokenizer=inference_ctx.tokenizer,
                )

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.processed_prompt,
                    choice=choice,
                    env_name="math",
                    env_example_id=example.example_id,
                    reward=reward,
                )
                
                group_rollouts.append(rollout)
                group_correct_scores.append(correct_score)
                
                total_choices += 1
                reward_sum += reward
                format_sum += fmt_score
                correct_sum += correct_score
                response_token_count += rollout.response_tokens.size
                prompt_token_count += rollout.prompt_tokens.size

                if finish_reason == "length":
                    truncated_count += 1

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))
                n_groups += 1
                
                # Calculate group stats
                avg_correct = sum(group_correct_scores) / len(group_correct_scores)
                if avg_correct == 1.0:
                    n_all_good += 1
                elif avg_correct == 0.0:
                    n_all_bad += 1
                else:
                    n_mixed += 1

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        # Metrics matching user request
        # Logged under env/all/xxx for training and test/env/all/xxx for testing
        # We return keys starting with "all/" and let the logger handle the prefix
        
        metrics = {
            "all/ac_tokens_per_turn": response_token_count / total_choices,
            "all/ob_tokens_per_turn": prompt_token_count / total_choices,
            "all/turns_per_episode": 1.0,
            "all/total_episodes": float(total_choices),
            "all/total_turns": float(total_choices),
            "all/total_ac_tokens": float(response_token_count),
            "all/total_ob_tokens": float(prompt_token_count),
            
            "all/by_group/frac_mixed": n_mixed / n_groups if n_groups > 0 else 0.0,
            "all/by_group/frac_all_good": n_all_good / n_groups if n_groups > 0 else 0.0,
            "all/by_group/frac_all_bad": n_all_bad / n_groups if n_groups > 0 else 0.0,
            
            "all/format": format_sum / total_choices,
            "all/correct": correct_sum / total_choices,
            "all/reward/total": reward_sum / total_choices,
            
            # Legacy/Debug metrics
            f"math.{mode}_mean_reward": reward_sum / total_choices,
            f"math.{mode}_truncated_percentage": float(truncated_count) / total_choices,
        }

        return rollout_groups, metrics

    def _score_choice(
        self, example: MathEnvExample, response_text: str, finish_reason: str, tokenizer
    ) -> tuple[float, float, float]:
        """Score a single generated response text using MathEnv logic."""

        decoded_response = response_text.strip()
        
        # Penalize truncated responses
        parse_success = finish_reason != "length"
        
        # Check format
        format_valid = float(parse_success and self.check_format(decoded_response))
        
        true_answer = example.processed_answer.strip()
        
        # Grade using Tinker's grading
        correct_answer = float(self.check_answer(decoded_response, true_answer))
        
        reward = self.format_coef * (format_valid - 1) + correct_answer
        
        return reward, format_valid, correct_answer

    # ------------------------------------------------------------------
    # Dataset inspection helpers
    # ------------------------------------------------------------------
    def training_data(self) -> Iterator[MathEnvExample]:
        train_dataset = _get_hendrycks_math_train().shuffle(seed=self.seed)

        for i, item in enumerate(train_dataset):
            raw_prompt = item["problem"]
            raw_answer = item["solution"]
            example_id = str(i)

            yield self.clean_example(raw_prompt, raw_answer, example_id)

    def eval_data(self) -> Iterator[MathEnvExample]:
        test_dataset = _get_hendrycks_math_test()

        for i, item in enumerate(test_dataset):
            raw_prompt = item["problem"]
            raw_answer = item["solution"]
            example_id = str(i)
            yield self.clean_example(raw_prompt, raw_answer, example_id)

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Sample evaluation examples deterministically."""

        if not self.eval_examples:
            return []

        eval_key = jax.random.PRNGKey(42)
        n_to_sample = min(n_examples, len(self.eval_examples))
        indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
        return [
            {
                "prompt": self.eval_examples[int(idx)].processed_prompt,
                "answer": self.eval_examples[int(idx)].processed_answer,
                "example_id": self.eval_examples[int(idx)].example_id,
            }
            for idx in indices
        ]
