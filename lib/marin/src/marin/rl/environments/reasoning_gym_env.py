# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup

from .base import MarinEnv, extract_seed

logger = logging.getLogger(__name__)

MODE_TRAIN = "train"
MODE_EVAL = "eval"
COMPOSITE_DATASET_NAME = "composite"
REQUIRED_DATASET_ARG_KEYS = ("seed", "size")
QUESTION_KEY = "question"
METADATA_KEY = "metadata"
ENTRY_ID_KEY = "entry_id"
SOURCE_DATASET_KEY = "source_dataset"
SOURCE_INDEX_KEY = "source_index"
DATASETS_KEY = "datasets"
ENV_NAME_PREFIX = "reasoning_gym"
DEFAULT_SUCCESS_THRESHOLD = 1.0
QUESTION_TEMPLATE_FIELD = "{question}"
NON_ALNUM_UNDERSCORE_PATTERN = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class ReasoningGymExample:
    """Normalized view of a single Reasoning Gym example."""

    prompt: str
    example_id: str
    raw_entry: dict[str, Any]
    source_dataset: str | None


class ReasoningGymEnv(MarinEnv):
    """Marin RL environment backed by the Reasoning Gym Python API."""

    def __init__(
        self,
        dataset_name: str,
        train_dataset_args: dict[str, Any],
        eval_dataset_args: dict[str, Any],
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
        sample_with_replacement: bool = False,
        prompt_template: str = QUESTION_TEMPLATE_FIELD,
    ) -> None:
        if QUESTION_TEMPLATE_FIELD not in prompt_template:
            raise ValueError("prompt_template must include '{question}'")
        if not math.isfinite(success_threshold):
            raise ValueError("success_threshold must be finite")

        self.dataset_name = dataset_name
        self.env_name = f"{ENV_NAME_PREFIX}:{dataset_name}"
        self.success_threshold = success_threshold
        self.sample_with_replacement = sample_with_replacement
        self.prompt_template = prompt_template

        reasoning_gym = self._ensure_reasoning_gym_installed()
        self._train_dataset_args = self._normalize_dataset_args(dataset_name, train_dataset_args)
        self._eval_dataset_args = self._normalize_dataset_args(dataset_name, eval_dataset_args)
        self._validate_dataset_args(MODE_TRAIN, self._train_dataset_args)
        self._validate_dataset_args(MODE_EVAL, self._eval_dataset_args)
        self._train_dataset = reasoning_gym.create_dataset(dataset_name, **self._train_dataset_args)
        self._eval_dataset = reasoning_gym.create_dataset(dataset_name, **self._eval_dataset_args)

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = MODE_TRAIN,
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample prompts from Reasoning Gym, score completions, and build rollouts."""
        if n_examples <= 0:
            raise ValueError("n_examples must be positive")
        if n_generations <= 0:
            raise ValueError("n_generations must be positive")

        dataset = self._dataset_for_mode(mode)
        rng = np.random.default_rng(extract_seed(prng_key))
        indices = self._sample_indices(dataset, n_examples, rng)
        sampled_examples = [self._build_example(dataset, mode, int(idx)) for idx in indices]
        prompts = [example.prompt for example in sampled_examples]

        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=system_prompt,
        )

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        solve_sum = 0.0
        response_token_count = 0
        truncated_count = 0
        source_counts: dict[str, int] = {}

        for example, completion in zip(sampled_examples, completions, strict=True):
            group_rollouts: list[Rollout] = []
            source_name = example.source_dataset or self.dataset_name
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

            for choice in completion.choices:
                response_text = choice.message.content or ""
                reward = self._score_choice(dataset, example.raw_entry, response_text)
                solved = float(reward >= self.success_threshold)

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.prompt,
                    choice=choice,
                    env_name=self.env_name,
                    env_example_id=example.example_id,
                    reward=reward,
                    correctness_reward=solved,
                    temperature=temperature,
                    top_k=top_k,
                    system_prompt=system_prompt,
                )

                group_rollouts.append(rollout)
                total_choices += 1
                reward_sum += reward
                solve_sum += solved
                response_token_count += rollout.response_tokens.size

                if choice.finish_reason == "length":
                    truncated_count += 1

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        prefix = self._metrics_prefix(mode)
        metrics: dict[str, float] = {
            f"{prefix}_mean_reward": reward_sum / total_choices,
            f"{prefix}_solve_rate": solve_sum / total_choices,
            f"{prefix}_mean_response_tokens": response_token_count / total_choices,
            f"{prefix}_total_responses": float(total_choices),
            f"{prefix}_sampled_examples": float(len(sampled_examples)),
            f"{prefix}_truncated_percentage": float(truncated_count) / total_choices,
        }
        for source_name, count in sorted(source_counts.items()):
            metrics[f"{prefix}_source_{self._metric_name_fragment(source_name)}_count"] = float(count)

        return rollout_groups, metrics

    def _dataset_for_mode(self, mode: str):
        if mode == MODE_TRAIN:
            dataset = self._train_dataset
        elif mode == MODE_EVAL:
            dataset = self._eval_dataset
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if len(dataset) == 0:
            raise ValueError(f"No examples available for mode '{mode}'")

        return dataset

    def _sample_indices(self, dataset, n_examples: int, rng: np.random.Generator) -> np.ndarray:
        dataset_size = len(dataset)
        if self.sample_with_replacement:
            return rng.choice(dataset_size, size=n_examples, replace=True)

        n_to_sample = min(n_examples, dataset_size)
        return rng.choice(dataset_size, size=n_to_sample, replace=False)

    def _build_example(self, dataset, mode: str, idx: int) -> ReasoningGymExample:
        entry = dataset[idx]
        question = entry.get(QUESTION_KEY)
        if not isinstance(question, str):
            raise ValueError(f"Reasoning Gym entry at index {idx} is missing a string '{QUESTION_KEY}' field")

        metadata = entry.get(METADATA_KEY, {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Reasoning Gym entry at index {idx} has non-dict metadata: {type(metadata)!r}")

        source_dataset = metadata.get(SOURCE_DATASET_KEY)
        if source_dataset is not None and not isinstance(source_dataset, str):
            raise ValueError(f"Reasoning Gym metadata field '{SOURCE_DATASET_KEY}' must be a string")

        example_id = self._build_example_id(dataset, mode, idx, metadata)
        prompt = self.prompt_template.format(question=question)
        return ReasoningGymExample(
            prompt=prompt,
            example_id=example_id,
            raw_entry=entry,
            source_dataset=source_dataset,
        )

    def _build_example_id(self, dataset, mode: str, idx: int, metadata: dict[str, Any]) -> str:
        if isinstance(metadata.get(ENTRY_ID_KEY), str):
            return f"{self.env_name}:{mode}:{metadata[ENTRY_ID_KEY]}"

        source_dataset = metadata.get(SOURCE_DATASET_KEY)
        source_index = metadata.get(SOURCE_INDEX_KEY)
        if isinstance(source_dataset, str):
            source_index_fragment = source_index if source_index is not None else idx
            return f"{self.env_name}:{mode}:{source_dataset}:{source_index_fragment}"

        dataset_seed = getattr(dataset, "seed", "unknown")
        if source_index is not None:
            return f"{self.env_name}:{mode}:{dataset_seed}:{source_index}"
        return f"{self.env_name}:{mode}:{dataset_seed}:{idx}"

    def _score_choice(self, dataset, entry: dict[str, Any], response_text: str) -> float:
        score = float(dataset.score_answer(response_text, entry))
        if not math.isfinite(score):
            raise ValueError(f"Reasoning Gym returned a non-finite score for dataset '{self.dataset_name}': {score}")
        if score < 0.0 or score > 1.0:
            logger.warning(
                "Reasoning Gym score for dataset '%s' fell outside [0, 1]: %f",
                self.dataset_name,
                score,
            )
        return score

    def _normalize_dataset_args(self, dataset_name: str, dataset_args: dict[str, Any]) -> dict[str, Any]:
        normalized_args = copy.deepcopy(dataset_args)
        if dataset_name == COMPOSITE_DATASET_NAME:
            return self._normalize_composite_specs(normalized_args)
        return normalized_args

    def _normalize_composite_specs(self, dataset_args: dict[str, Any]) -> dict[str, Any]:
        datasets = dataset_args.get(DATASETS_KEY)
        if datasets is None:
            return dataset_args
        if not isinstance(datasets, list):
            raise ValueError(f"Composite dataset args field '{DATASETS_KEY}' must be a list")

        composite_module = importlib.import_module("reasoning_gym.composite")
        dataset_spec_cls = composite_module.DatasetSpec
        normalized_specs = []
        for dataset_spec in datasets:
            if isinstance(dataset_spec, dict):
                normalized_specs.append(dataset_spec_cls(**dataset_spec))
            else:
                normalized_specs.append(dataset_spec)
        dataset_args[DATASETS_KEY] = normalized_specs
        return dataset_args

    def _validate_dataset_args(self, mode: str, dataset_args: dict[str, Any]) -> None:
        missing_keys = [key for key in REQUIRED_DATASET_ARG_KEYS if key not in dataset_args]
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{mode}_dataset_args must include explicit {missing}")
        if self.dataset_name == COMPOSITE_DATASET_NAME and not dataset_args.get(DATASETS_KEY):
            raise ValueError(f"{mode}_dataset_args must include a non-empty '{DATASETS_KEY}' list for composite")

    def _metrics_prefix(self, mode: str) -> str:
        return f"{ENV_NAME_PREFIX}.{self._metric_name_fragment(self.dataset_name)}.{mode}"

    def _ensure_reasoning_gym_installed(self):
        try:
            return importlib.import_module("reasoning_gym")
        except ModuleNotFoundError as exc:
            if exc.name != "reasoning_gym":
                raise
            raise ImportError(
                "The 'reasoning_gym' package is required to use ReasoningGymEnv. "
                "Install it with: uv sync --extra reasoning-gym"
            ) from exc

    @staticmethod
    def _metric_name_fragment(name: str) -> str:
        return NON_ALNUM_UNDERSCORE_PATTERN.sub("_", name)
