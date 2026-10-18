# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render sweep trajectories in a Qwen3 chat template with a loss mask on what the model emits (task #230).

The served prompt is the model's own chat template, applied by vLLM 0.25.1 (``DEFAULT_CUDA_VLLM_VERSION``; the serve
config sets only ``apply_chat_template: true``). Before rendering, vLLM parses assistant tool-call arguments from JSON
strings into objects (``chat_utils._postprocess_messages``), so the template prints them with ``tojson``; this module
does the same.

Reasoning (decided for #230, option B): Qwen3's thinking template renders ``<think>`` blocks only for assistant turns
after the last real user message and drops them from earlier turns. Rows here keep a think block on every assistant
turn, as the v0 rows keep reasoning on every turn. The template is patched so every assistant turn renders as

    <|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n{content}{tool calls}<|im_end|>\\n

The first three tokens are the generation prompt vLLM appends (``<|im_start|>assistant\\n<think>\\n``) and are context;
the trained span runs from the reasoning through ``<|im_end|>``, so each trained turn is exactly what the model emits
at inference after its generation prompt. A row whose only user message is the first one (every turn after the last
query) renders byte-identically to the native template when every assistant turn has reasoning. The history mismatch
(earlier reasoning stays in context, while the served prompt drops it) is the one the v0 rows already carry.
"""

import json
from dataclasses import dataclass

QWEN_MODELS = {
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json")
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"
END_OF_TURN = "<|im_end|>"
# Native: a think block only after the last real user message, and only for the last turn or one with reasoning.
NATIVE_THINK_CONDITIONS = (
    "{%- if loop.index0 > ns.last_query_index %}",
    "{%- if loop.last or (not loop.last and reasoning_content) %}",
)


@dataclass(frozen=True)
class Rendered:
    text: str
    ids: list[int]
    mask: list[int]
    turn_spans: list[tuple[int, int]]  # [start, end) token range of each assistant turn's trained span

    @property
    def supervised(self) -> int:
        return sum(self.mask[1:])


def every_turn_thinks(native: str) -> str:
    """The native template with both think-block conditions made unconditional (option B)."""
    patched = native
    for condition in NATIVE_THINK_CONDITIONS:
        if patched.count(condition) != 1:
            raise ValueError(f"Template does not contain exactly one {condition!r}; review the patch for this model")
        patched = patched.replace(condition, "{%- if true %}")
    return patched


def served_messages(messages: list[dict]) -> list[dict]:
    """Messages as vLLM 0.25.1 hands them to a Qwen3 template.

    Tool-call arguments are parsed to objects and empty tool_calls dropped (chat_utils._postprocess_messages). The
    template has no developer role, so developer messages become system messages and every system message is merged
    into one at position 0, joined by a blank line (renderers/hf.py _convert_developer_to_system and
    _consolidate_system_messages, applied only when a developer message is present).
    """
    if any(message["role"] == "developer" for message in messages):
        system, rest = [], []
        for message in messages:
            if message["role"] in ("developer", "system"):
                if message.get("content"):
                    system.append(message["content"])
            else:
                rest.append(message)
        messages = [{"role": "system", "content": "\n\n".join(system)}, *rest]
    converted = []
    for message in messages:
        message = dict(message)
        if message["role"] == "assistant" and "tool_calls" in message:
            calls = []
            for call in message["tool_calls"] or []:
                call = json.loads(json.dumps(call))
                arguments = call["function"].get("arguments")
                if not arguments:
                    call["function"]["arguments"] = {}
                elif not isinstance(arguments, dict | list):
                    parsed = json.loads(arguments)
                    call["function"]["arguments"] = parsed if parsed is not None else {}
                calls.append(call)
            if calls:
                message["tool_calls"] = calls
            else:
                del message["tool_calls"]
        converted.append(message)
    return converted


def render(tokenizer, template: str, messages: list[dict], tools: list[dict] | None = None) -> Rendered:
    """Render with `template` (every_turn_thinks of the native one) and mask everything but each turn's output."""
    messages = served_messages(messages)
    text = tokenizer.apply_chat_template(messages, tools=tools or None, chat_template=template, tokenize=False)
    assistants = sum(message["role"] == "assistant" for message in messages)
    spans, position = [], 0
    while (header := text.find(GENERATION_PROMPT, position)) >= 0:
        start = header + len(GENERATION_PROMPT)
        end = text.find(END_OF_TURN, start)
        if end < 0:
            raise ValueError("Assistant turn without an end-of-turn token")
        spans.append((start, end + len(END_OF_TURN)))
        position = end
    if len(spans) != assistants:
        raise ValueError(f"{len(spans)} assistant headers for {assistants} assistant turns")
    # Context and trained segments are tokenized separately, as at inference: the prompt (ending in the generation
    # prompt) is tokenized on its own and the turn is the model's own output. Tokenizing the whole text instead would
    # merge an empty reasoning's "\n" with the prompt's final "\n" into one token across the boundary.
    ids, mask, turn_spans, position = [], [], [], 0
    for start, end in [*spans, (len(text), len(text))]:
        for segment, trained in ((text[position:start], 0), (text[start:end], 1)):
            if segment:
                pieces = tokenizer(segment, add_special_tokens=False)["input_ids"]
                if trained:
                    turn_spans.append((len(ids), len(ids) + len(pieces)))
                ids += pieces
                mask += [trained] * len(pieces)
        position = end
    return Rendered(text=text, ids=ids, mask=mask, turn_spans=turn_spans)


def mask_turns(rendered: Rendered, trained: list[bool]) -> Rendered:
    """Zero the loss on assistant turns the source marks untrained (replay train_turns)."""
    if len(trained) != len(rendered.turn_spans):
        raise ValueError(f"{len(trained)} train flags for {len(rendered.turn_spans)} assistant turns")
    mask = list(rendered.mask)
    for (start, end), keep in zip(rendered.turn_spans, trained, strict=True):
        if not keep:
            mask[start:end] = [0] * (end - start)
    return Rendered(text=rendered.text, ids=rendered.ids, mask=mask, turn_spans=rendered.turn_spans)
