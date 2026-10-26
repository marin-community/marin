"""Post-trial RLAIF transport has its own identity and never reads verifier scores."""

import json

import httpx
import pytest
from pydantic import ValidationError

from skyrl_train.rollouts.pi_judging import ExternalPiJudge, PiJudgeSettings
from skyrl_train.rollouts.pi_scenario_etl import HydratedPiTrial
from tests.cpu.rollouts.pi_trial_fixtures import trial


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("PI_JUDGE_UNIT_TOKEN", "unit-token")
    return PiJudgeSettings(
        endpoint="https://judge.example/v1",
        model="independent-critic",
        revision="a" * 40,
        auth="env:PI_JUDGE_UNIT_TOKEN",
        principle="The file exists and matches the task",
        prompt_revision="rlaif_judge_v1",
        chat_template_kwargs={"enable_thinking": False},
    )


def response(*, content="1", model="independent-critic", request_id="judge-response-1", finish_reason="stop"):
    return {
        "id": request_id,
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
    }


@pytest.mark.asyncio
async def test_judge_scores_only_executed_student_trajectory_not_verifier(settings):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response())

    raw = trial()
    raw.update(verifier_reward=0.0, verifier_passed=False)
    hydrated = HydratedPiTrial.model_validate(raw)
    service = ExternalPiJudge(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        evidence = await service.score(hydrated)
    finally:
        await service.close()
    assert evidence.trial_id == "trial-1"
    assert evidence.student_request_ids == ("student-0", "student-1")
    assert evidence.score == 1
    assert evidence.request_id == "judge-response-1"
    assert requests[0].url == "https://judge.example/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer unit-token"
    payload = json.loads(requests[0].content)
    assert payload["model"] == settings.model and payload["temperature"] == 0
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][0]["role"] == "system"
    assert json.loads(payload["messages"][1]["content"])[-1] == {"role": "assistant", "content": "Done"}
    assert all(secret not in requests[0].content for secret in (b"verifier_reward", b"verifier_passed", b"unit-token"))


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    response(content="1 because correct"),
    response(content="true"),
    response(content="1", finish_reason="length"),
    response(model="student"),
    response(request_id=""),
    {**response(), "choices": []},
])
async def test_judge_rejects_incomplete_or_nonbinary_replies(settings, reply):
    service = ExternalPiJudge(
        settings,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply))),
    )
    try:
        with pytest.raises(ValueError):
            await service.score(HydratedPiTrial.model_validate(trial()))
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_judge_transport_error_does_not_leak_capability_url(settings):
    protected = settings.model_copy(update={"endpoint": "https://judge.example/proxy/t/private-capability/v1"})
    service = ExternalPiJudge(
        protected,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))),
    )
    try:
        with pytest.raises(RuntimeError, match="Pi judge HTTP request failed") as error:
            await service.score(HydratedPiTrial.model_validate(trial()))
    finally:
        await service.close()
    assert "private-capability" not in str(error.value)


@pytest.mark.asyncio
async def test_judge_rejects_teacher_evidence_before_external_request(settings):
    raw = trial(dagger=True)
    calls = []
    service = ExternalPiJudge(
        settings,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request))),
    )
    try:
        with pytest.raises(ValueError, match="student-only"):
            await service.score(HydratedPiTrial.model_validate(raw))
    finally:
        await service.close()
    assert calls == []


@pytest.mark.parametrize("change", [
    {"revision": "main"},
    {"auth": "plain-secret"},
    {"endpoint": "http://judge.example/v1"},
    {"endpoint": "https://token@judge.example/v1"},
    {"endpoint": "https://judge.example/v1?key=token"},
    {"model": ""},
    {"chat_template_kwargs": {"enable_thinking": True}},
    {"chat_template_kwargs": {"arbitrary": False}},
])
def test_judge_settings_fail_closed(settings, change):
    with pytest.raises(ValidationError):
        PiJudgeSettings.model_validate(settings.model_dump() | change)
