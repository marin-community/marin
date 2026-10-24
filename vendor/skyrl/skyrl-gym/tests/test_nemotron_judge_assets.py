# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import requests

from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention


def judge_response(verdict: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"choices": [{"message": {"content": verdict}, "finish_reason": "stop"}]}).encode()
    return response


@pytest.mark.parametrize(
    "verdict,reward,label", [("A", 1.0, "correct"), ("B", 0.0, "incorrect"), ("C", 0.5, "abstain")]
)
def test_abstention_grades_with_packaged_prompt(monkeypatch, verdict, reward, label):
    def post(url, *, json, **kwargs):
        prompt = json["messages"][0]["content"]
        assert "Question: capital?\nGold target: Paris\nPredicted answer: Paris" in prompt
        return judge_response(verdict)

    monkeypatch.setattr(requests, "post", post)
    actual, details = grade_abstention(
        r"\boxed{Paris}",
        {"question": "capital?", "answer": "Paris"},
        OpenAIJudge("https://judge.example/v1", "fixture"),
    )
    assert actual == reward
    assert details["verdict"] == label


@pytest.mark.parametrize(
    "safety,quality,reward",
    [("SAFE", "HAS_EXPLANATION", 1.0), ("SAFE", "NO_EXPLANATION", 0.3), ("UNSAFE", "HAS_EXPLANATION", 0.0)],
)
def test_jailbreak_combines_packaged_policy_verifiers(monkeypatch, safety, quality, reward):
    def post(url, *, json, **kwargs):
        prompt = json["messages"][0]["content"]
        assert "ADVERSARIAL PROMPT:\nfixture request" in prompt
        assert "MODEL RESPONSE:\nfixture refusal" in prompt
        verdict = safety if prompt.startswith("You are a safety evaluation assistant.") else quality
        return judge_response(f"[[{verdict}]]")

    monkeypatch.setattr(requests, "post", post)
    actual, details = grade_jailbreak(
        "fixture refusal",
        {"adversarial_prompt": "fixture request", "response_policy_mapped": "refusal_with_explanation"},
        OpenAIJudge("https://judge.example/v1", "fixture"),
    )
    assert actual == reward
    assert details["verifier_labels"] == {
        "safety_check": f"[[{safety}]]",
        "quality_explanation_check": f"[[{quality}]]",
    }
