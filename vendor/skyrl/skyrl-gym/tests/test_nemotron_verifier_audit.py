import json

import pytest


@pytest.mark.asyncio
async def test_structured_tool_call_answer_is_graded(nemotron_session, model_turn):
    schema = {"type": "object", "properties": {"required": {"type": "integer"}}, "required": ["required"]}
    record = {
        "schema_type": "json",
        "schema_str": json.dumps(schema),
        "response_mode": "tool_call",
        "tool_name": "submit",
        "tool_payload_key": "payload",
    }
    call = {"type": "function", "function": {"name": "submit", "arguments": '{"payload":{"required":7}}'}}
    session = await nemotron_session("structured_outputs_simple_agent", record)
    result = await session.advance(model_turn("", message={"role": "assistant", "content": "", "tool_calls": [call]}))
    assert result.reward == 1.0
