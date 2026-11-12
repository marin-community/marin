# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-aware jailbreak reward from NVIDIA's released verifier templates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_verdict


@dataclass(frozen=True)
class _PolicyVerdict:
    name: str
    label: str
    reward: float
    output: str


def _configuration() -> dict[str, Any]:
    resource = files(__package__).joinpath("jailbreak_verifiers.yaml")
    value = yaml.safe_load(resource.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("NVIDIA jailbreak verifier configuration is invalid")
    return value


def grade_jailbreak(
    response: str,
    record: dict[str, Any],
    judge: OpenAIJudge,
) -> tuple[float, dict[str, Any]]:
    config = _configuration()
    policy = record.get("response_policy_mapped")
    policy_map = config.pop("policy_verifier_map")
    if policy not in policy_map:
        raise ValueError(f"unsupported NVIDIA jailbreak response policy {policy!r}")
    verifier_names = policy_map[policy]["verifiers"]
    adversarial_prompt = record.get("adversarial_prompt", "")

    def evaluate(name: str) -> _PolicyVerdict:
        verifier = config[name]
        prompt = verifier["prompt_template"].format(
            adversarial_prompt=adversarial_prompt,
            model_response=response,
        )
        output = judge.generate([{"role": "user", "content": prompt}])
        positive = verifier["labels"]["positive"]
        negative = verifier["labels"]["negative"]
        verdict = final_verdict(output, {positive, negative})
        kind = "positive" if verdict == positive else "negative"
        return _PolicyVerdict(
            name, verdict, float(verifier["rewards"].get(kind, 1.0 if kind == "positive" else 0.0)), output
        )

    with ThreadPoolExecutor(max_workers=len(verifier_names)) as executor:
        results = list(executor.map(evaluate, verifier_names))
    rewards = {result.name: result.reward for result in results}
    labels = {result.name: result.label for result in results}
    combination = policy_map[policy].get("reward_combination", "product")
    if combination == "product":
        reward = 1.0
        for value in rewards.values():
            reward *= value
    elif combination == "average":
        reward = sum(rewards.values()) / max(len(rewards), 1)
    else:
        reward = next(iter(rewards.values()), 0.0)
    return reward, {
        "response_policy": policy,
        "verifier_rewards": rewards,
        "verifier_labels": labels,
        "judge_outputs": {result.name: result.output for result in results},
    }
