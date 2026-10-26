"""The vLLM engine logs the message of an ErrorResponse before it returns it as a server error."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from loguru import logger  # noqa: E402
from skyrl_train.inference_engines.vllm.vllm_engine import (  # noqa: E402
    AsyncVLLMInferenceEngine,
    ErrorInfo,
    ErrorResponse,
)

AUTO_TOOL_ERROR = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'


class ServingChat:
    async def create_chat_completion(self, request, raw_request):
        return ErrorResponse(error=ErrorInfo(message=AUTO_TOOL_ERROR, type="BadRequestError", code=400))


@pytest.mark.asyncio
async def test_error_response_is_logged_with_its_message():
    engine = SimpleNamespace(
        openai_serving_chat=ServingChat(), _openai_sampling_params={}, _validate_rollout_logprob_sampling=False
    )
    lines = []
    sink = logger.add(lines.append, format="{level} {message}")
    try:
        response = await AsyncVLLMInferenceEngine._handle_openai_request(
            engine,
            {
                "json": {"model": "student", "messages": [{"role": "user", "content": "hi"}]},
                "headers": {"x-request-id": "req-1"},
            },
            "/chat/completions",
        )
    finally:
        logger.remove(sink)

    assert response["error_category"] == "server_error"
    assert response["request_id"] == "req-1"
    assert response["error"]["message"] == AUTO_TOOL_ERROR
    (line,) = [line for line in lines if "req-1" in line]
    assert line.startswith("WARNING")
    assert "/chat/completions" in line and "code=400" in line and "type=BadRequestError" in line
    assert AUTO_TOOL_ERROR in line
