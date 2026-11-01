import pytest
import requests


@pytest.mark.parametrize("stop_reason,expected", [("stop", 0.5), ("length", 0.0)])
@pytest.mark.asyncio
async def test_empty_abstention_retains_native_not_attempted_reward(
    stop_reason, expected, nemotron_session, model_turn, monkeypatch
):
    completion = requests.Response()
    completion.status_code = 200
    completion._content = b'{"choices":[{"finish_reason":"stop","message":{"content":"C"}}]}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: completion)
    session = await nemotron_session(
        "abstention_simple_agent",
        {"question": "What is two plus two?", "answer": "4"},
        {
            "verifyit_enabled": False,
            "judges": {"general": {"base_url": "http://127.0.0.1:9/v1", "model": "unused"}},
        },
    )
    result = await session.advance(model_turn("", stop_reason=stop_reason))
    assert result.done and result.reward == result.grade.reward == expected
