"""Negative protocol checks for the authenticated external Pi teacher transport.

These responses test rejection only; the opt-in Qwen/Docker test samples real
teacher tokens through the same transport and a separate HTTP endpoint.
"""

import pytest
from aiohttp import web

from skyrl_train.inference_engines.pi_external_vllm import ExternalPiTeacherEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["bad_model", "missing_ids", "missing_logprobs", "partial_logprobs", "bad_tokenize"])
async def test_external_teacher_rejects_missing_or_wrong_provenance(monkeypatch, damage):
    monkeypatch.setenv("PI_EXTERNAL_TEST_KEY", "unit-only-key")
    calls = []
    app = web.Application()

    async def respond(request):
        assert request.headers.get("Authorization") == "Bearer unit-only-key"
        calls.append(request.path)
        if request.path == "/v1/tokenize":
            return web.json_response({"tokens": [1, "not-an-id"] if damage == "bad_tokenize" else [1, 2]})
        result = {
            "model": "wrong" if damage == "bad_model" else "teacher",
            "choices": [
                {
                    "token_ids": None if damage == "missing_ids" else [17],
                    "logprobs": (
                        None
                        if damage == "missing_logprobs"
                        else {"content": [] if damage == "partial_logprobs" else [{"logprob": -0.25}]}
                    ),
                    "message": {"role": "assistant", "content": "17"},
                }
            ],
        }
        return web.json_response(result)

    app.router.add_post("/v1/tokenize", respond)
    app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    endpoint = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    teacher = ExternalPiTeacherEngine(
        endpoint=endpoint, model="teacher", auth="env:PI_EXTERNAL_TEST_KEY", max_model_len=128
    )
    try:
        if damage == "bad_tokenize":
            with pytest.raises(ValueError, match="exact integer token IDs"):
                await teacher.tokenize({"json": {"model": "teacher", "messages": []}})
        else:
            assert (await teacher.tokenize({"json": {"model": "teacher", "messages": []}}))["tokens"] == [1, 2]
            if damage == "missing_logprobs":
                result = await teacher.chat_completion({"json": {"model": "teacher", "messages": []}})
                assert result["choices"][0]["token_ids"] == [17]
                assert result["choices"][0]["logprobs"] is None
            else:
                with pytest.raises(ValueError, match="wrong model|sampled token IDs|incomplete sampled logprobs"):
                    await teacher.chat_completion({"json": {"model": "teacher", "messages": []}})
        assert calls
    finally:
        await teacher.close()
        await runner.cleanup()
