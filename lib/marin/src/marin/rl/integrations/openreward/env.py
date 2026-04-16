# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from marin.rl.environments.base import MarinEnv, extract_seed
from marin.rl.environments.inference_ctx import BaseInferenceContext, ToolSpec
from marin.rl.integrations.openreward.client import load_openreward_client
from marin.rl.integrations.openreward.manifest import load_openreward_task_manifest
from marin.rl.integrations.openreward.models import (
    OpenRewardPromptBlockType,
    OpenRewardTaskManifest,
    OpenRewardTaskManifestEntry,
    OpenRewardToolSpec,
)
from marin.rl.types import RolloutGroup
from marin.training.run_environment import resolve_required_env_vars

logger = logging.getLogger(__name__)

_PERMISSIVE_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}


class OpenRewardToolResultLike(Protocol):
    """Minimal tool-call result returned by the OpenReward SDK."""

    reward: float | int | None
    finished: bool


class OpenRewardSessionLike(Protocol):
    """Minimal runtime session interface used by OpenRewardEnv."""

    def __enter__(self) -> OpenRewardSessionLike: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> OpenRewardToolResultLike: ...


def _load_manifest(path: str | None) -> OpenRewardTaskManifest | None:
    if path is None:
        return None
    return load_openreward_task_manifest(path)


def _validate_manifest_pair(
    train_manifest: OpenRewardTaskManifest | None,
    eval_manifest: OpenRewardTaskManifest | None,
) -> tuple[str, str]:
    first_manifest = train_manifest or eval_manifest
    if first_manifest is None:
        raise ValueError("At least one OpenReward manifest path must be provided.")

    deployment_name = first_manifest.deployment_name
    environment_name = first_manifest.environment_name
    for manifest in (train_manifest, eval_manifest):
        if manifest is None:
            continue
        if manifest.deployment_name != deployment_name:
            raise ValueError(
                "OpenRewardEnv requires train/eval manifests from the same deployment, "
                f"got {deployment_name!r} and {manifest.deployment_name!r}."
            )
        if manifest.environment_name != environment_name:
            raise ValueError(
                "OpenRewardEnv requires train/eval manifests from the same environment, "
                f"got {environment_name!r} and {manifest.environment_name!r}."
            )

    return deployment_name, environment_name


def _manifest_for_mode(
    mode: str,
    train_manifest: OpenRewardTaskManifest | None,
    eval_manifest: OpenRewardTaskManifest | None,
) -> OpenRewardTaskManifest:
    if mode == "train":
        if train_manifest is None:
            raise ValueError("OpenRewardEnv has no train manifest configured.")
        return train_manifest
    if mode == "eval":
        if eval_manifest is None:
            raise ValueError("OpenRewardEnv has no eval manifest configured.")
        return eval_manifest
    raise ValueError(f"Unsupported mode: {mode}")


def _prompt_text(entry: OpenRewardTaskManifestEntry) -> str:
    text_blocks: list[str] = []
    for block in entry.prompt_blocks:
        if block.type != OpenRewardPromptBlockType.TEXT:
            raise ValueError(
                "OpenRewardEnv currently supports only text prompt blocks; "
                f"task {entry.task_index} contains {block.type.value!r}."
            )
        assert block.text is not None
        text_blocks.append(block.text)

    if not text_blocks:
        raise ValueError(f"OpenReward task {entry.task_index} produced an empty prompt.")
    return "\n\n".join(text_blocks)


def _tool_specs(tools: list[OpenRewardToolSpec]) -> list[ToolSpec]:
    rendered_tools: list[ToolSpec] = []
    for tool in tools:
        parameters = tool.input_schema or _PERMISSIVE_OBJECT_SCHEMA
        rendered_tools.append(
            ToolSpec(
                function=ToolSpec.FunctionBody(
                    name=tool.name,
                    description=tool.description,
                    parameters=parameters,
                )
            )
        )
    return rendered_tools


def _merge_stop_sequences(stop: list[str] | None, tool_call_stop: str) -> list[str]:
    merged = list(stop) if stop is not None else []
    if tool_call_stop not in merged:
        merged.append(tool_call_stop)
    return merged


def _resolve_optional_env_var(env_var_name: str | None) -> str | None:
    if env_var_name is None:
        return None
    return resolve_required_env_vars([env_var_name])[env_var_name]


def _resolve_optional_secrets(secret_env_vars: list[str] | None) -> dict[str, str] | None:
    if not secret_env_vars:
        return None
    return resolve_required_env_vars(secret_env_vars)


class OpenRewardEnv(MarinEnv):
    """Manifest-backed single-turn OpenReward environment adapter."""

    def __init__(
        self,
        *,
        train_manifest_path: str | None = None,
        eval_manifest_path: str | None = None,
        base_url: str | None = None,
        api_key_env_var: str | None = None,
        variant: str | None = None,
        secret_env_vars: list[str] | None = None,
        invalid_tool_call_reward: float = 0.0,
        tool_call_stop: str = "</tool_call>",
    ) -> None:
        self.train_manifest = _load_manifest(train_manifest_path)
        self.eval_manifest = _load_manifest(eval_manifest_path)
        self.deployment_name, self.environment_name = _validate_manifest_pair(self.train_manifest, self.eval_manifest)

        self.base_url = base_url
        self.api_key_env_var = api_key_env_var
        self.variant = variant
        self.secret_env_vars = list(secret_env_vars) if secret_env_vars is not None else None
        self.invalid_tool_call_reward = invalid_tool_call_reward
        self.tool_call_stop = tool_call_stop

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
        manifest = _manifest_for_mode(mode, self.train_manifest, self.eval_manifest)
        if not manifest.tasks:
            raise ValueError(f"OpenReward manifest for mode {mode!r} contains no tasks.")

        n_to_sample = min(n_examples, len(manifest.tasks))
        rng = np.random.default_rng(extract_seed(prng_key))
        sampled_indices = rng.choice(len(manifest.tasks), size=n_to_sample, replace=False)
        sampled_entries = [manifest.tasks[int(index)] for index in sampled_indices]

        reward_sum = 0.0
        response_token_count = 0
        total_choices = 0
        parse_failures = 0
        invalid_tool_call_count = 0
        tool_execution_failures = 0
        finished_count = 0
        rollout_groups: list[RolloutGroup] = []
        api_key = _resolve_optional_env_var(self.api_key_env_var)
        secrets = _resolve_optional_secrets(self.secret_env_vars)

        openreward_client = load_openreward_client()
        with openreward_client(api_key=api_key, base_url=self.base_url) as client:
            environment = client.environments.get(self.deployment_name, variant=self.variant)

            for entry in sampled_entries:
                prompt = _prompt_text(entry)
                tools = _tool_specs(entry.tools)
                completion = inference_ctx.batch_completions(
                    prompts=[prompt],
                    temperature=temperature,
                    n=n_generations,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    stop=_merge_stop_sequences(stop, self.tool_call_stop),
                    system_prompt=system_prompt,
                    tools=tools,
                )[0]

                group_rollouts = []
                task = environment.get_task(manifest.split, entry.task_index)
                valid_tool_names = {tool.name for tool in entry.tools}

                for choice in completion.choices:
                    reward = self.invalid_tool_call_reward
                    parse_result = inference_ctx.assistant_turn_from_choice(choice)
                    parsed_tool_calls = parse_result.assistant_turn.tool_calls

                    if not parse_result.parse_success:
                        parse_failures += 1
                    elif len(parsed_tool_calls) != 1:
                        invalid_tool_call_count += 1
                    else:
                        tool_call = parsed_tool_calls[0]
                        if tool_call.function.name not in valid_tool_names:
                            invalid_tool_call_count += 1
                        else:
                            arguments = json.loads(tool_call.function.arguments)
                            with environment.session(task=task, secrets=secrets) as session:
                                try:
                                    tool_result = session.call_tool(
                                        tool_call.function.name,
                                        arguments,
                                    )
                                except ValueError:
                                    invalid_tool_call_count += 1
                                    logger.info(
                                        "OpenReward tool call was rejected for deployment=%s task=%s tool=%s",
                                        self.deployment_name,
                                        entry.task_index,
                                        tool_call.function.name,
                                    )
                                except Exception as exc:
                                    raise RuntimeError(
                                        "OpenReward tool execution failed for "
                                        f"deployment={self.deployment_name} "
                                        f"task={entry.task_index} "
                                        f"tool={tool_call.function.name}"
                                    ) from exc
                                else:
                                    if tool_result.finished:
                                        finished_count += 1
                                        reward = float(tool_result.reward or 0.0)
                                    else:
                                        invalid_tool_call_count += 1

                    rollout = inference_ctx.create_rollout_from_choice(
                        prompt=prompt,
                        choice=choice,
                        env_name=f"openreward:{self.deployment_name}",
                        env_example_id=f"{manifest.split}:{entry.task_index}",
                        reward=reward,
                        temperature=temperature,
                        top_k=top_k,
                        system_prompt=system_prompt,
                    )
                    group_rollouts.append(rollout)
                    reward_sum += reward
                    response_token_count += rollout.response_tokens.size
                    total_choices += 1

                if group_rollouts:
                    rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics.")

        metric_prefix = f"openreward.{self.environment_name}.{mode}"
        metrics = {
            f"{metric_prefix}.mean_reward": reward_sum / total_choices,
            f"{metric_prefix}.mean_response_tokens": response_token_count / total_choices,
            f"{metric_prefix}.total_responses": float(total_choices),
            f"{metric_prefix}.sampled_examples": float(len(sampled_entries)),
            f"{metric_prefix}.parse_failure_rate": parse_failures / total_choices,
            f"{metric_prefix}.invalid_tool_call_rate": invalid_tool_call_count / total_choices,
            f"{metric_prefix}.tool_execution_failure_rate": tool_execution_failures / total_choices,
            f"{metric_prefix}.finished_rate": finished_count / total_choices,
        }
        return rollout_groups, metrics
