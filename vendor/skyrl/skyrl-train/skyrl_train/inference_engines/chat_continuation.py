"""Exact sampled-token continuation for chat templates that rewrite assistant history."""

from copy import deepcopy
from typing import Any, Awaitable, Callable

EXACT_PROMPT_TOKEN_IDS_KEY = "_skyrl_exact_prompt_token_ids"
CHAT_TOKENIZE_FIELDS = {
    "model",
    "messages",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "chat_template",
    "chat_template_kwargs",
    "media_io_kwargs",
    "mm_processor_kwargs",
    "tools",
    "tool_choice",
}


def _tokenize_body(body: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    request = {key: deepcopy(value) for key, value in body.items() if key in CHAT_TOKENIZE_FIELDS}
    request["messages"] = deepcopy(messages)
    return request


async def render_exact_chat_continuation(
    tokenize: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    request_payload: dict[str, Any],
    *,
    assistant_message_index: int,
    served_prefix_token_ids: list[int],
) -> list[int] | None:
    """Append the renderer's boundary without re-tokenizing sampled content.

    Return None when the renderer does not preserve the structural prefix needed
    to splice the sampled tokens safely.
    """
    body = request_payload["json"]
    messages = body["messages"]
    if not 0 <= assistant_message_index < len(messages):
        raise ValueError("continuation assistant index is outside the message history")
    if messages[assistant_message_index].get("role") != "assistant":
        raise ValueError("continuation message must be an assistant turn")

    history_messages = messages[:assistant_message_index]
    prefix_messages = messages[: assistant_message_index + 1]
    full_request = _tokenize_body(body, messages)
    prefix_request = _tokenize_body(body, prefix_messages)
    prefix_request["add_generation_prompt"] = False
    prefix_request["continue_final_message"] = False
    boundary_request = _tokenize_body(body, [*history_messages, {"role": "assistant", "content": "boundary"}])
    boundary_request["add_generation_prompt"] = False
    boundary_request["continue_final_message"] = False
    continued_request = {**boundary_request, "continue_final_message": True}

    headers = request_payload.get("headers", {})
    renders = []
    for tokenize_request in (full_request, prefix_request, continued_request, boundary_request):
        result = await tokenize({"json": tokenize_request, "headers": headers})
        renders.append(result.get("tokens") if isinstance(result, dict) else None)
    full_ids, prefix_ids, continued_ids, boundary_ids = renders
    if (
        not all(isinstance(ids, list) and all(isinstance(token, int) for token in ids) for ids in renders)
        or boundary_ids[: len(continued_ids)] != continued_ids
    ):
        return None

    assistant_boundary = boundary_ids[len(continued_ids) :]
    if assistant_boundary and prefix_ids[-len(assistant_boundary) :] != assistant_boundary:
        return None
    suffix_ids = full_ids[len(prefix_ids) :]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        # Some templates remove reasoning from earlier assistant turns. Render the new
        # observations after the initial user/system prefix, then verify the same suffix
        # in the full rendering. The served assistant tokens stay unchanged.
        first_assistant = next(i for i, message in enumerate(messages) if message.get("role") == "assistant")
        base_messages = messages[:first_assistant]
        base_request = _tokenize_body(body, base_messages)
        base_request["add_generation_prompt"] = False
        base_request["continue_final_message"] = False
        suffix_request = _tokenize_body(body, [*base_messages, *messages[assistant_message_index + 1 :]])
        base_ids = (await tokenize({"json": base_request, "headers": headers})).get("tokens")
        tail_ids = (await tokenize({"json": suffix_request, "headers": headers})).get("tokens")
        if not isinstance(base_ids, list) or not isinstance(tail_ids, list) or tail_ids[: len(base_ids)] != base_ids:
            return None
        suffix_ids = tail_ids[len(base_ids) :]
        if suffix_ids and full_ids[-len(suffix_ids) :] != suffix_ids:
            return None
    overlap = max(
        (
            count
            for count in range(1, min(len(served_prefix_token_ids), len(assistant_boundary)) + 1)
            if served_prefix_token_ids[-count:] == assistant_boundary[:count]
        ),
        default=0,
    )
    return [*served_prefix_token_ids, *assistant_boundary[overlap:], *suffix_ids]
