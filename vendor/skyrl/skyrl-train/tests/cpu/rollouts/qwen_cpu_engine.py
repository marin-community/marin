"""CPU Qwen tool-call adapter shared by opt-in physical Pi trials."""

import json
import re
from uuid import uuid4

from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine

QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TOOL_CALL = re.compile(r"(?s)(.*?)<tool_call>\s*(\{.*?\})\s*</tool_call>\s*")


class QwenToolCPUInferenceEngine(CPUInferenceEngine):
    """Render Pi text blocks and parse sampled Qwen tool markup into OpenAI calls."""

    async def tokenize(self, request_payload):
        body = request_payload["json"]
        messages = []
        for message in body["messages"]:
            message = dict(message)
            if isinstance(message.get("content"), list):
                if any(block.get("type") != "text" for block in message["content"]):
                    raise ValueError("Qwen CPU test only supports text content blocks")
                message["content"] = "\n".join(block["text"] for block in message["content"])
            if message.get("tool_calls"):
                # The Qwen template renders argument objects, but the OpenAI
                # wire format carries a JSON string. Decode *only for prompt
                # rendering*, preserving Pi's attributed wire evidence.
                message["tool_calls"] = [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": json.loads(call["function"]["arguments"]),
                        },
                    }
                    for call in message["tool_calls"]
                ]
            messages.append(message)
        return await super().tokenize({**request_payload, "json": {**body, "messages": messages}})

    async def chat_completion(self, request_payload):
        result = await super().chat_completion(request_payload)
        choice = result["choices"][0]
        generated = choice["message"]["content"]
        match = TOOL_CALL.fullmatch(generated)
        if not match:
            return result
        tool = json.loads(match.group(2))
        if tool["name"] != "bash" or not isinstance(tool["arguments"], dict):
            raise ValueError(f"Unexpected model-generated tool request: {tool}")
        choice["message"] = {
            "role": "assistant",
            "content": match.group(1).strip() or None,
            "tool_calls": [
                {
                    "id": f"call-{uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": json.dumps(tool["arguments"])},
                }
            ],
        }
        choice["finish_reason"] = "tool_calls"
        return result
