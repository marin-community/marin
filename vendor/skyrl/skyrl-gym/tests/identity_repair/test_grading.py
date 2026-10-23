import json

import pytest
import requests

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.identity_repair.facts import FactManifest
from skyrl_gym.identity_repair.grading import ProductionIdentityJudge
from skyrl_gym.identity_repair.reward import IdentityFact, RewardWeights, TurnContext


@pytest.mark.parametrize("completed,score", [(True, 1.25), (False, 0.5)])
def test_production_judge_uses_requested_facts_and_task_evidence(monkeypatch, completed, score):
    requests_seen = []

    def respond(url, *, headers, json: dict, timeout):
        requests_seen.append(json)
        response = requests.Response()
        response.status_code = 200
        response._content = encode_response({
            "task_completed": True, "fact_correctness": {"name": 1.0}, "unsupported_claims": 0,
            "inappropriate_refusal": False, "unsolicited_identity": False, "explanation": "Name is correct.",
        })
        return response

    monkeypatch.setattr(requests, "post", respond)
    judge = ProductionIdentityJudge(OpenAIJudge("http://judge/v1", "judge-model", strict_completion=True), "pin", 512)
    manifest = FactManifest("v1", "Marin Snowball", (), "OpenAthena", (), "OpenAthena", (), (), {})
    context = TurnContext("What is your name?", True, (IdentityFact.NAME,), completed)
    result = judge.grade_answer("I am Marin Snowball.", context, manifest, RewardWeights())
    assert result.reward.total == score
    assert result.reward.judge_version == "judge-model@pin/identity-repair-v2-request-specific"
    payload = json.loads(requests_seen[0]["messages"][-1]["content"])
    assert payload["requested_facts"] == ["name"]
    assert payload["answer"] == "I am Marin Snowball."


def encode_response(assessment):
    return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(assessment)}}]}).encode()
