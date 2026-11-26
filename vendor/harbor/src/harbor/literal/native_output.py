"""Translate recorded chat output into Messages and Responses wire formats."""

import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from harbor.literal.chat_stream import merge_chat_delta


def native_output(chat: dict[str, Any], protocol: str, model: str) -> dict[str, Any]:
    choice = chat["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    usage = chat.get("usage") or {}
    content = []
    if protocol == "anthropic":
        if reasoning:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning,
                    "signature": "",
                }
            )
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "input": json.loads(call["function"]["arguments"]),
                }
            )
        return {
            "id": chat["id"],
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_sequence": None,
            "stop_reason": {"tool_calls": "tool_use", "length": "max_tokens"}.get(
                choice["finish_reason"], "end_turn"
            ),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }
    if reasoning:
        content.append(
            {
                "type": "reasoning",
                "id": f"rs_{chat['id']}",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    if message.get("content"):
        content.append(
            {
                "type": "message",
                "id": f"msg_{chat['id']}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": message["content"],
                        "annotations": [],
                    }
                ],
            }
        )
    for call in message.get("tool_calls") or []:
        content.append(
            {
                "type": "function_call",
                "id": f"fc_{call['id']}",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "status": "completed",
            }
        )
    return {
        "id": f"resp_{chat['id']}",
        "object": "response",
        "created_at": chat.get("created", 0),
        "model": model,
        "status": "incomplete" if choice["finish_reason"] == "length" else "completed",
        "incomplete_details": (
            {"reason": "max_output_tokens"}
            if choice["finish_reason"] == "length"
            else None
        ),
        "error": None,
        "output": content,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def native_events(
    chunks: AsyncStream[ChatCompletionChunk],
    resources: AsyncExitStack,
    protocol: str,
    model: str,
    prompt_tokens: int,
) -> AsyncIterator[bytes]:
    """Relay text and tool deltas immediately, then seal the native response."""
    sequence = 0

    def event(kind: str, **fields: Any) -> bytes:
        nonlocal sequence
        body = {"type": kind, **fields}
        if protocol == "responses":
            body["sequence_number"] = sequence
            sequence += 1
        return f"event: {kind}\ndata: {json.dumps(body)}\n\n".encode()

    message: dict[str, Any] = {"role": "assistant", "content": None}
    chat = {
        "id": "",
        "created": 0,
        "choices": [{"message": message, "finish_reason": None}],
        "usage": {"prompt_tokens": prompt_tokens},
    }
    blocks: dict[tuple[str, int], int] = {}
    started = False
    async with resources:
        async for chunk in chunks:
            data = chunk.model_dump(exclude_none=True)
            chat["id"] = data["id"]
            chat["created"] = data["created"]
            if data.get("usage"):
                chat["usage"] = data["usage"]
            if not started:
                output = native_output(chat, protocol, model)
                if protocol == "anthropic":
                    yield event(
                        "message_start",
                        message={**output, "content": [], "stop_reason": None},
                    )
                else:
                    yield event(
                        "response.created", response={**output, "status": "in_progress"}
                    )
                    yield event(
                        "response.in_progress",
                        response={**output, "status": "in_progress"},
                    )
                started = True
            if not data.get("choices"):
                continue
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            if not delta.get("reasoning_content") and delta.get("reasoning"):
                delta["reasoning_content"] = delta["reasoning"]
            merge_chat_delta(message, delta)
            if choice.get("finish_reason"):
                chat["choices"][0]["finish_reason"] = choice["finish_reason"]
            for field, kind in (("reasoning_content", "thinking"), ("content", "text")):
                text = delta.get(field)
                if not text:
                    continue
                key = (kind, 0)
                if key not in blocks:
                    index = blocks[key] = len(blocks)
                    if protocol == "anthropic":
                        block = {
                            "type": kind,
                            "thinking" if kind == "thinking" else "text": "",
                        }
                        yield event(
                            "content_block_start", index=index, content_block=block
                        )
                    elif kind == "thinking":
                        item = {
                            "type": "reasoning",
                            "id": f"rs_{chat['id']}",
                            "status": "in_progress",
                            "summary": [],
                        }
                        yield event(
                            "response.output_item.added", output_index=index, item=item
                        )
                        yield event(
                            "response.reasoning_summary_part.added",
                            output_index=index,
                            item_id=item["id"],
                            summary_index=0,
                            part={"type": "summary_text", "text": ""},
                        )
                    else:
                        item = {
                            "type": "message",
                            "id": f"msg_{chat['id']}",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        }
                        yield event(
                            "response.output_item.added", output_index=index, item=item
                        )
                        yield event(
                            "response.content_part.added",
                            output_index=index,
                            item_id=item["id"],
                            content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []},
                        )
                index = blocks[key]
                if protocol == "anthropic":
                    yield event(
                        "content_block_delta",
                        index=index,
                        delta={
                            "type": f"{kind}_delta",
                            "thinking" if kind == "thinking" else "text": text,
                        },
                    )
                elif kind == "thinking":
                    yield event(
                        "response.reasoning_summary_text.delta",
                        output_index=index,
                        item_id=f"rs_{chat['id']}",
                        summary_index=0,
                        delta=text,
                    )
                else:
                    yield event(
                        "response.output_text.delta",
                        output_index=index,
                        item_id=f"msg_{chat['id']}",
                        content_index=0,
                        delta=text,
                        logprobs=[],
                    )
            for part in delta.get("tool_calls", []):
                tool_index = part["index"]
                call = message["tool_calls"][tool_index]
                key = ("tool", tool_index)
                if key not in blocks:
                    index = blocks[key] = len(blocks)
                    if protocol == "anthropic":
                        yield event(
                            "content_block_start",
                            index=index,
                            content_block={
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["function"]["name"],
                                "input": {},
                            },
                        )
                    else:
                        yield event(
                            "response.output_item.added",
                            output_index=index,
                            item={
                                "type": "function_call",
                                "id": f"fc_{call['id']}",
                                "call_id": call["id"],
                                "name": call["function"]["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                        )
                arguments = part.get("function", {}).get("arguments")
                if arguments:
                    if protocol == "anthropic":
                        yield event(
                            "content_block_delta",
                            index=blocks[key],
                            delta={
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        )
                    else:
                        yield event(
                            "response.function_call_arguments.delta",
                            output_index=blocks[key],
                            item_id=f"fc_{call['id']}",
                            delta=arguments,
                        )
        output = native_output(chat, protocol, model)
        if protocol == "anthropic":
            for index in blocks.values():
                yield event("content_block_stop", index=index)
            yield event(
                "message_delta",
                delta={"stop_reason": output["stop_reason"], "stop_sequence": None},
                usage=output["usage"],
            )
            yield event("message_stop")
        else:
            item_indices = {
                f"rs_{chat['id']}": blocks.get(("thinking", 0)),
                f"msg_{chat['id']}": blocks.get(("text", 0)),
                **{
                    f"fc_{call['id']}": blocks[("tool", index)]
                    for index, call in enumerate(message.get("tool_calls") or [])
                },
            }
            output["output"].sort(key=lambda item: item_indices[item["id"]])
            for index, item in enumerate(output["output"]):
                if item["type"] == "message":
                    part = item["content"][0]
                    yield event(
                        "response.output_text.done",
                        output_index=index,
                        item_id=item["id"],
                        content_index=0,
                        text=part["text"],
                        logprobs=[],
                    )
                    yield event(
                        "response.content_part.done",
                        output_index=index,
                        item_id=item["id"],
                        content_index=0,
                        part=part,
                    )
                elif item["type"] == "reasoning":
                    part = item["summary"][0]
                    yield event(
                        "response.reasoning_summary_text.done",
                        output_index=index,
                        item_id=item["id"],
                        summary_index=0,
                        text=part["text"],
                    )
                    yield event(
                        "response.reasoning_summary_part.done",
                        output_index=index,
                        item_id=item["id"],
                        summary_index=0,
                        part=part,
                    )
                else:
                    yield event(
                        "response.function_call_arguments.done",
                        output_index=index,
                        item_id=item["id"],
                        name=item["name"],
                        arguments=item["arguments"],
                    )
                yield event("response.output_item.done", output_index=index, item=item)
            yield event(f"response.{output['status']}", response=output)
