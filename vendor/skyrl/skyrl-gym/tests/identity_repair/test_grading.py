import json

import pytest
import requests
from pydantic import ValidationError

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.identity_repair.facts import FactManifest
from skyrl_gym.identity_repair.grading import ProductionIdentityJudge
from skyrl_gym.identity_repair.judge import AttributionClaim, ClaimKind
from skyrl_gym.identity_repair.reward import IdentityFact, RewardWeights, TurnContext


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("completed,score", [(True, 1.25), (False, 0.5)])
def test_production_judge_uses_requested_facts_and_task_evidence(monkeypatch, completed, score, fenced):
    requests_seen = []

    def respond(url, *, headers, json: dict, timeout):
        requests_seen.append(json)
        response = requests.Response()
        response.status_code = 200
        response._content = encode_response({
            "task_completed": True, "fact_correctness": {"name": 1.0}, "unsupported_claims": 0,
            "inappropriate_refusal": False, "unsolicited_identity": False, "explanation": "Name is correct.",
        }, fenced=fenced)
        return response

    monkeypatch.setattr(requests, "post", respond)
    judge = ProductionIdentityJudge(OpenAIJudge("http://judge/v1", "judge-model", strict_completion=True), "pin", 512)
    manifest = FactManifest("v1", "Marin Snowball", (), "OpenAthena", (), "OpenAthena", (), (), {})
    context = TurnContext("What is your name?", True, (IdentityFact.NAME,), completed)
    result = judge.grade_answer("I am Marin Snowball.", context, manifest, RewardWeights())
    assert result.reward.total == score
    assert result.reward.judge_version == "judge-model@pin/identity-repair-v4-task-criteria"
    payload = json.loads(requests_seen[0]["messages"][-1]["content"])
    assert payload["requested_facts"] == ["name"]
    assert payload["answer"] == "I am Marin Snowball."
    schema = requests_seen[0]["response_format"]["json_schema"]["schema"]
    assert "inappropriate_refusal" in schema["required"]
    assert "appropriate_refusal" not in schema["properties"]
    assert schema["additionalProperties"] is False
    facts = schema["properties"]["fact_correctness"]
    assert facts["required"] == ["name"]
    assert set(facts["properties"]) == {"name"}
    assert facts["additionalProperties"] is False


def encode_response(assessment, *, fenced=False):
    content = json.dumps(assessment)
    if fenced:
        content = "```json\n" + content + "\n```"
    return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()


@pytest.mark.parametrize("prefix", ["", "Here is my judgment: "])
def test_claim_judge_accepts_complete_json_fence_but_rejects_surrounding_prose(monkeypatch, prefix):
    def respond(url, *, headers, json, timeout):
        response = requests.Response()
        response.status_code = 200
        content = prefix + '```json\n{"incorrect": true, "ambiguous": false, "explanation": "Wrong name."}\n```'
        response._content = encode_wire_content(content)
        return response

    monkeypatch.setattr(requests, "post", respond)
    judge = ProductionIdentityJudge(OpenAIJudge("http://judge/v1", "judge-model", strict_completion=True), "pin", 512)
    manifest = FactManifest("v1", "Marin Snowball", (), "OpenAthena", (), "OpenAthena", (), (), {})
    claim = AttributionClaim(ClaimKind.NAME, "ChatGPT", 5, 12, "chatgpt")
    if prefix:
        with pytest.raises(ValidationError):
            judge.verify_claim("I am ChatGPT.", claim, manifest)
    else:
        verdict = judge.verify_claim("I am ChatGPT.", claim, manifest)
        assert verdict.incorrect and not verdict.ambiguous


def encode_wire_content(content):
    return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
