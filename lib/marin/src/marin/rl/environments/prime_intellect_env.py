# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment wrapper for Prime Intellect verifier environments."""

import logging
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from marin.rl.environments import MarinEnv
from marin.rl.environments.inference_ctx import BaseInferenceContext, PromptLike
from marin.rl.types import RolloutGroup

logger = logging.getLogger(__name__)

_SUPPORTED_OWNER = "primeintellect"
_ENV_NAME_PREFIX = "prime_intellect:"


def _freeze_cache_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, Mapping):
        frozen_items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"PrimeIntellectEnv env_args keys must be strings, got {type(key).__name__}")
            frozen_items.append((key, _freeze_cache_value(item)))
        return tuple(sorted(frozen_items))
    raise TypeError(
        "PrimeIntellectEnv env_args must contain only JSON-like values, " f"got unsupported type {type(value).__name__}"
    )


def _scalarize_metric(metric_name: str, values: object) -> float:
    if isinstance(values, (int, float, np.number, bool)):
        return float(values)

    if not isinstance(values, list):
        raise TypeError(f"Metric {metric_name!r} must be numeric or a list of numeric values")
    if not values:
        raise ValueError(f"Metric {metric_name!r} cannot be an empty list")

    try:
        return float(np.mean(np.asarray(values, dtype=np.float32)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Metric {metric_name!r} must contain only numeric values") from exc


class PrimeIntellectEnv(MarinEnv):
    """Adapter for Phase 1 Prime Intellect verifier environments."""

    INSTALLED_ENV_IDS: ClassVar[set[str]] = set()
    LOADED_ENVIRONMENTS: ClassVar[dict[tuple[str, object], Any]] = {}

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, object] | None = None,
        max_tokens: int = 1024,
        max_concurrent: int = 32,
    ):
        self.env_id = env_id
        self.env_args = dict(env_args or {})
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self._normalized_env_args = _freeze_cache_value(self.env_args)
        self._is_prepared = False

    def _verifiers_module(self) -> Any:
        try:
            import verifiers as vf
        except ImportError as exc:
            raise ImportError(
                "The 'verifiers' package is required to use PrimeIntellectEnv. "
                "Please install it with: uv pip install 'marin[rl]' or uv pip install verifiers"
            ) from exc

        return vf

    def _short_env_id(self) -> str:
        owner, separator, slug = self.env_id.partition("/")
        if owner != _SUPPORTED_OWNER or separator == "" or not slug:
            raise ValueError(f"PrimeIntellectEnv Phase 1 only supports '{_SUPPORTED_OWNER}/*' IDs, got {self.env_id!r}")
        return slug

    def _verifier_cache_key(self) -> tuple[str, object]:
        return self.env_id, self._normalized_env_args

    def prepare(self) -> None:
        self._verifiers_module()
        self._short_env_id()

        prime_executable = shutil.which("prime")
        if prime_executable is None:
            raise RuntimeError(
                "PrimeIntellectEnv requires the 'prime' executable on PATH. "
                "Install the Prime CLI before running Prime verifier environments."
            )

        if self.env_id not in self.INSTALLED_ENV_IDS:
            subprocess.run([prime_executable, "env", "install", self.env_id], check=True)
            self.INSTALLED_ENV_IDS.add(self.env_id)

        self._is_prepared = True

    def _load_verifier_env(self) -> Any:
        cache_key = self._verifier_cache_key()
        if cache_key in self.LOADED_ENVIRONMENTS:
            return self.LOADED_ENVIRONMENTS[cache_key]

        vf = self._verifiers_module()
        short_env_id = self._short_env_id()
        verifier_env = vf.load_environment(env_id=short_env_id, **self.env_args)
        self.LOADED_ENVIRONMENTS[cache_key] = verifier_env
        return verifier_env

    def _validate_sample_request(self, mode: str, system_prompt: str | None) -> None:
        if not self._is_prepared:
            raise RuntimeError("PrimeIntellectEnv.prepare() must be called before sample()")
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode!r}")
        if system_prompt is not None:
            raise ValueError("PrimeIntellectEnv Phase 1 does not support Marin-level system prompts")

    def _validate_verifier_env(self, verifier_env: Any) -> None:
        message_type = getattr(verifier_env, "message_type", None)
        if message_type != "chat":
            raise ValueError(
                f"PrimeIntellectEnv Phase 1 only supports chat-format verifier environments, got {message_type!r}"
            )

        if getattr(verifier_env, "oai_tools", None):
            raise ValueError("PrimeIntellectEnv Phase 1 does not support tool-enabled verifier environments")

    def _select_inputs(self, verifier_env: Any, mode: str, n_examples: int) -> Any:
        if mode == "train":
            return verifier_env.get_dataset(n=n_examples)
        return verifier_env.get_eval_dataset(n=n_examples)

    def _repeat_inputs(self, inputs: Any, n_generations: int) -> Any:
        if n_generations == 1:
            return inputs
        if hasattr(inputs, "repeat"):
            return inputs.repeat(n_generations)
        raise TypeError("PrimeIntellectEnv expects verifier datasets to expose repeat()")

    def _extract_example_ids(self, inputs: Any) -> list[str]:
        if hasattr(inputs, "column_names") and "id" in inputs.column_names:
            return [str(example_id) for example_id in inputs["id"]]
        return [f"example_{index}" for index in range(len(inputs))]

    def _validate_generate_outputs(self, result: Any, expected_rollouts: int) -> None:
        for field_name in ("prompt", "completion", "state", "reward"):
            field_value = getattr(result, field_name, None)
            if not isinstance(field_value, list):
                raise ValueError(f"PrimeIntellectEnv expected result.{field_name} to be a list")
            if len(field_value) != expected_rollouts:
                raise ValueError(
                    f"PrimeIntellectEnv expected {expected_rollouts} {field_name} entries, got {len(field_value)}"
                )

        metrics = getattr(result, "metrics", {})
        if metrics is None:
            return
        if not isinstance(metrics, Mapping):
            raise ValueError("PrimeIntellectEnv expected result.metrics to be a mapping")

    def _scalarize_metrics(self, raw_metrics: Mapping[str, object]) -> dict[str, float]:
        metrics = {}
        for metric_name, values in raw_metrics.items():
            metrics[f"{self.env_id}.{metric_name}"] = _scalarize_metric(metric_name, values)
        return metrics

    def _extract_phase1_rollout(
        self,
        rollout_index: int,
        prompt: PromptLike,
        completion: object,
        state: object,
    ) -> tuple[list[dict[str, object]], Any]:
        if not isinstance(prompt, list):
            raise ValueError(
                f"PrimeIntellectEnv Phase 1 only supports chat prompts, got {type(prompt).__name__} "
                f"for rollout {rollout_index}"
            )
        prompt_messages = [dict(message) for message in prompt]

        if not isinstance(completion, list):
            raise ValueError(
                f"PrimeIntellectEnv Phase 1 only supports chat completions, got {type(completion).__name__} "
                f"for rollout {rollout_index}"
            )

        completion_messages: list[dict[str, object]] = []
        for message in completion:
            if not isinstance(message, Mapping):
                raise TypeError(
                    f"PrimeIntellectEnv expected completion messages to be mappings, got {type(message).__name__}"
                )
            completion_messages.append(dict(message))

        if any(message.get("role") != "assistant" for message in completion_messages):
            raise ValueError("PrimeIntellectEnv Phase 1 does not support non-assistant turns in completions")
        if len(completion_messages) != 1:
            raise ValueError("PrimeIntellectEnv Phase 1 requires exactly one assistant completion turn")

        assistant_content = completion_messages[0].get("content")
        if not isinstance(assistant_content, str):
            raise ValueError("PrimeIntellectEnv Phase 1 expects assistant completion content to be a string")

        if not isinstance(state, Mapping):
            raise TypeError(f"PrimeIntellectEnv expected rollout state to be a mapping, got {type(state).__name__}")

        responses = state.get("responses")
        if not isinstance(responses, list):
            raise ValueError("PrimeIntellectEnv Phase 1 expected state['responses'] to be a list")
        if len(responses) != 1:
            raise ValueError("PrimeIntellectEnv Phase 1 requires exactly one response object per rollout")

        response = responses[0]
        if not hasattr(response, "choices"):
            raise ValueError("PrimeIntellectEnv Phase 1 expected state['responses'] entries to be ChatCompletion-like")
        if len(response.choices) != 1:
            raise ValueError("PrimeIntellectEnv Phase 1 requires exactly one assistant choice per rollout")

        choice = response.choices[0]
        if choice.message.role != "assistant":
            raise ValueError("PrimeIntellectEnv Phase 1 only supports assistant response choices")
        if choice.message.content is None:
            raise ValueError("PrimeIntellectEnv Phase 1 requires assistant responses with text content")
        if choice.message.content != assistant_content:
            raise ValueError("PrimeIntellectEnv Phase 1 requires completion messages to match response choices")

        return prompt_messages, choice

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        del prng_key

        self._validate_sample_request(mode, system_prompt)
        verifier_env = self._load_verifier_env()
        self._validate_verifier_env(verifier_env)

        base_inputs = self._select_inputs(verifier_env, mode, n_examples)
        if base_inputs is None:
            raise ValueError(f"PrimeIntellectEnv could not load any inputs for mode {mode!r}")

        example_ids = self._extract_example_ids(base_inputs)
        repeated_inputs = self._repeat_inputs(base_inputs, n_generations)
        expected_rollouts = len(example_ids) * n_generations

        sampling_args = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "logprobs": True,
            "stop": stop,
        }

        result = verifier_env.generate(
            inputs=repeated_inputs,
            client=inference_ctx.openai_client(),
            model="marin-model",
            sampling_args=sampling_args,
            max_concurrent=self.max_concurrent,
        )
        self._validate_generate_outputs(result, expected_rollouts)

        raw_metrics = getattr(result, "metrics", {}) or {}
        metrics = self._scalarize_metrics(raw_metrics)

        if expected_rollouts == 0:
            metrics[f"{self.env_id}.total_rollouts"] = 0.0
            return [], metrics

        rollout_groups = []
        reward_sum = 0.0

        # Dataset.repeat(n) orders rows as [ex0, ex1, ex0, ex1, ...], so regroup
        # by generation first and example second.
        n_sampled_examples = len(example_ids)
        for example_index, example_id in enumerate(example_ids):
            group_rollouts = []
            for generation_index in range(n_generations):
                rollout_index = generation_index * n_sampled_examples + example_index
                prompt_messages, choice = self._extract_phase1_rollout(
                    rollout_index=rollout_index,
                    prompt=result.prompt[rollout_index],
                    completion=result.completion[rollout_index],
                    state=result.state[rollout_index],
                )
                reward = float(result.reward[rollout_index])
                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=prompt_messages,
                    choice=choice,
                    env_name=f"{_ENV_NAME_PREFIX}{self.env_id}",
                    env_example_id=f"{self.env_id}:{example_id}",
                    reward=reward,
                    temperature=temperature,
                    top_k=top_k,
                )
                group_rollouts.append(rollout)
                reward_sum += reward

            rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        metrics[f"{self.env_id}.mean_reward"] = reward_sum / expected_rollouts
        metrics[f"{self.env_id}.total_rollouts"] = float(expected_rollouts)

        logger.info("Generated %d rollout groups for %s", len(rollout_groups), self.env_id)
        return rollout_groups, metrics
