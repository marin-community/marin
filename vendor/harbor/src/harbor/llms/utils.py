import copy
import re
from typing import Any, Dict, List, Tuple

import harbor.llms._litellm_setup  # noqa: F401 - configures litellm
from litellm import Message


def add_vllm_reasoning(
    messages: List[Dict[str, Any] | Message],
) -> List[Dict[str, Any] | Message]:
    """Copy assistant ``reasoning_content`` into ``reasoning`` for vLLM.

    vLLM's chat endpoint reads replayed reasoning only from the ``reasoning``
    field of an assistant message and ignores ``reasoning_content``, so without
    this copy interleaved thinking never reaches the chat template.
    """
    return [
        {**msg, "reasoning": msg["reasoning_content"]}
        if isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and msg.get("reasoning_content") is not None
        else msg
        for msg in messages
    ]


def add_anthropic_caching(
    messages: List[Dict[str, Any] | Message], model_name: str
) -> List[Dict[str, Any] | Message]:
    """
    Add ephemeral caching to the most recent messages for Anthropic models.

    Args:
        messages: List of message dictionaries
        model_name: The model name to check if it's an Anthropic model

    Returns:
        List of messages with caching added to the most recent 3 messages
    """
    # Only apply caching for Anthropic models
    if not ("anthropic" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    # Create a deep copy to avoid modifying the original messages
    cached_messages = copy.deepcopy(messages)

    # Add cache_control to the most recent 3 messages
    for n in range(len(cached_messages)):
        if n >= len(cached_messages) - 3:
            msg = cached_messages[n]

            # Handle both dict and Message-like objects
            if isinstance(msg, dict):
                # Ensure content is in the expected format
                if isinstance(msg.get("content"), str):
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(msg.get("content"), list):
                    # Add cache_control to each content item
                    for content_item in msg["content"]:
                        if isinstance(content_item, dict) and "type" in content_item:
                            content_item["cache_control"] = {"type": "ephemeral"}
            elif hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    msg.content = [  # type: ignore
                        {
                            "type": "text",
                            "text": msg.content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(msg.content, list):
                    for content_item in msg.content:
                        if isinstance(content_item, dict) and "type" in content_item:
                            content_item["cache_control"] = {"type": "ephemeral"}

    return cached_messages


_HOSTED_VLLM_PREFIX = "hosted_vllm/"
# Per-segment length bound is intentionally generous (128, not 64). For
# self-hosted vLLM the `<org>/<model>` name is an ARBITRARY local identifier
# litellm dispatches verbatim to our OpenAI-compatible endpoint — there is no
# downstream HF/Docker-tag or hosted-API 64-char limit that bites here. Long RL
# checkpoint names are run-descriptor concatenations (e.g.
# `rl__24GPU_qwen3_coder_30b_a3b__exp_rpt_pymethods2test-large__Qwen3-Coder-30B-A3B-40-30B`,
# 87 chars) and were spuriously rejected at 64. The pattern's real job is
# character-class + structure validation; keep that, just raise the bound.
_HOSTED_VLLM_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_HOSTED_VLLM_REQUIRED_INT_FIELDS = ("max_input_tokens", "max_output_tokens")
_HOSTED_VLLM_REQUIRED_FLOAT_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
)


def validate_hosted_vllm_model_config(
    full_model_name: str, model_info: dict[str, Any] | None
) -> Tuple[str, dict[str, Any]]:
    """
    Validate hosted_vllm model configuration.

    Args:
        full_model_name: The user-provided model name (e.g., hosted_vllm/llama)
        model_info: Optional metadata dictionary that must include token limits + cost info

    Returns:
        (canonical_model_name, normalized_model_info)

    Raises:
        ValueError: If validation fails
    """

    if not full_model_name.startswith(_HOSTED_VLLM_PREFIX):
        raise ValueError(
            "hosted_vllm models must start with 'hosted_vllm/'. "
            f"Got '{full_model_name}'."
        )

    # Strip the provider prefix and validate the remainder. Both the bare
    # `hosted_vllm/<model>` and the org-qualified `hosted_vllm/<org>/<model>`
    # forms are valid: vLLM serves an HF model under its full repo id
    # (`<org>/<model>`) by default, and litellm routes `hosted_vllm/<org>/<model>`
    # correctly (it strips only the `hosted_vllm/` provider prefix before
    # dispatching `<org>/<model>` to the OpenAI-compatible vLLM endpoint). We
    # therefore allow 1 or 2 remaining path segments and validate each segment
    # individually against the canonical-name pattern, preserving rejection of
    # genuinely-malformed names (empty/odd segments, bad characters, deep paths).
    remainder = full_model_name[len(_HOSTED_VLLM_PREFIX) :]
    segments = remainder.split("/")
    if len(segments) not in (1, 2) or not all(
        _HOSTED_VLLM_MODEL_PATTERN.fullmatch(segment) for segment in segments
    ):
        raise ValueError(
            "hosted_vllm model names must be 'hosted_vllm/<model>' or "
            "'hosted_vllm/<org>/<model>', where each segment contains only "
            "letters, numbers, '.', '-', '_', is 1-64 characters, and has no "
            f"spaces. Got '{full_model_name}'."
        )

    # canonical mirrors the served name litellm dispatches to vLLM (the full
    # org-qualified id when present), used for register_model / param lookups.
    canonical = remainder

    if not model_info:
        raise ValueError(
            "hosted_vllm models require `model_info` specifying token limits and costs. "
            "Please provide max_input_tokens, max_output_tokens, "
            "input_cost_per_token, and output_cost_per_token."
        )

    normalized_info = dict(model_info)

    for field in _HOSTED_VLLM_REQUIRED_INT_FIELDS:
        value = model_info.get(field)
        if value is None:
            raise ValueError(f"hosted_vllm model_info missing '{field}'.")
        try:
            normalized_info[field] = int(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"hosted_vllm model_info field '{field}' must be a number. "
                f"Got '{value}'."
            )

    for field in _HOSTED_VLLM_REQUIRED_FLOAT_FIELDS:
        value = model_info.get(field)
        if value is None:
            raise ValueError(f"hosted_vllm model_info missing '{field}'.")
        try:
            normalized_info[field] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"hosted_vllm model_info field '{field}' must be a float. "
                f"Got '{value}'."
            )

    return canonical, normalized_info
