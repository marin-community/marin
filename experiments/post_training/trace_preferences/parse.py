# Copyright 2026 Baixin Guo (Max)
# SPDX-License-Identifier: Apache-2.0
"""Loader, span proof and syntax parser for Qwen native tool-call markup.  A parser
pass is evidence about FORMAT ONLY; contract and limits: README.md.

Native markup is the literal token stream -- NOT the JSON tool-call envelope of
the exported `conversations` column (same action, different offsets)::

    <function=NAME>\\n<parameter=P>\\nvalue\\n</parameter>\\n</function>
"""

import json
import re
from dataclasses import dataclass

# Pinned by the dataset's tokenizer_provenance.json; the id is part of the contract
# (a same-family tokenizer decodes word tokens to garbage).
PINNED_TOKENIZER_ID = "Qwen/Qwen3.5-122B-A10B-FP8"
PINNED_TOKENIZER_REVISION = "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"

FUNCTION_BLOCK = re.compile(r"<function=[^>]*>.*?</function>", re.DOTALL)
INLINE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
OPENER = re.compile(r"<function=([^>\s]*)>")
PARAM_OPENER = re.compile(r"<parameter=([^>\s]*)>")
TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
CLOSER = "</function>"
PARAM_CLOSER = "</parameter>"

REASON_ROUNDTRIP = "DECODE_ENCODE_ROUNDTRIP_MISMATCH"
REASON_NO_BLOCK = "NO_NATIVE_FUNCTION_BLOCK"
REASON_MULTIPLE_BLOCKS = "MULTIPLE_NATIVE_FUNCTION_BLOCKS"
REASON_SEGMENTATION = "SEGMENTATION_MISMATCH_CROSS_BOUNDARY_TOKEN"

STATUS_VALID = "valid"
STATUS_MISSING_OPENER = "missing_opener"
STATUS_MISSING_CLOSER = "missing_closer"
STATUS_REPEATED_OPENER = "repeated_opener"
STATUS_UNCLOSED_PARAMETER = "unclosed_parameter"
STATUS_UNKNOWN_FUNCTION = "unknown_function"
STATUS_UNDECLARED_PARAMETER = "undeclared_parameter"
# Format-checkable but the served schema needed to judge function/parameter
# declarations was not available in that turn's prefix. NOT a pass, NOT malformed.
STATUS_SCHEMA_UNAVAILABLE = "schema_unavailable"


class NativeTokenizer:
    """Pinned tokenizer loaded from a local tokenizer.json (never downloads)."""

    def __init__(self, tokenizer_json_path: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(tokenizer_json_path)

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)


@dataclass(frozen=True)
class SpanProof:
    """The verdict plus the gate that produced it (``skip_reason``)."""

    span: tuple[int, int] | None
    span_text: str | None
    roundtrip: bool
    verified: bool
    skip_reason: str | None


def prove_action_span(tokenizer: NativeTokenizer, response_ids: tuple[int, ...]) -> SpanProof:
    """Prove which token offsets the action occupies.

    gate 1  round trip      encode(decode(ids)) == ids
    gate 2  single action   exactly one <function=...> block
    gate 3  segmentation    encode(pre) ++ encode(action) ++ encode(post) == ids

    Gate 3 IS the cross-boundary check -- "decode the neighbours and see if they
    concatenate" is vacuous (decode is plain concatenation).  One token covering
    either boundary breaks it, after which a character-offset span is a guess.
    """
    if not response_ids:
        return SpanProof(None, None, False, False, REASON_NO_BLOCK)

    decoded = tokenizer.decode(response_ids)
    if tokenizer.encode(decoded) != tuple(response_ids):
        return SpanProof(None, None, False, False, REASON_ROUNDTRIP)

    matches = list(FUNCTION_BLOCK.finditer(decoded))
    if not matches:
        return SpanProof(None, None, True, False, REASON_NO_BLOCK)
    if len(matches) > 1:
        return SpanProof(None, None, True, False, REASON_MULTIPLE_BLOCKS)

    match = matches[0]
    span_text = match.group(0)
    pre_ids = tokenizer.encode(decoded[: match.start()])
    span_ids = tokenizer.encode(span_text)
    post_ids = tokenizer.encode(decoded[match.end() :])
    start, end = len(pre_ids), len(pre_ids) + len(span_ids)

    if not (tuple(response_ids[start:end]) == span_ids
            and tuple(response_ids[:start]) == pre_ids
            and tuple(response_ids[end:]) == post_ids):
        return SpanProof(None, span_text, True, False, REASON_SEGMENTATION)

    return SpanProof((start, end), span_text, True, True, None)


def extract_served_tools(prompt_text: str) -> list[dict] | None:
    """Tool definitions the engine put in this prompt, or None if it had no ``<tools>`` block."""
    match = TOOLS_BLOCK.search(prompt_text)
    if match is None:
        return None
    tools = [json.loads(s) for s in (x.strip() for x in match.group(1).splitlines()) if s]
    return tools or None


def function_block_status(text: str, served_tools: list[dict] | None) -> str:
    """One status for one native function block.

    CHECKS (literal text): one ``<function=`` opener; ``</function>`` present; every
    ``<parameter=NAME>`` closed; (only if a served schema is available) the function
    served in that prompt; (only if available) every parameter declared for it.
    NOT checked: execution, success, parameter values, framing, environment state,
    action quality.

    When ``served_tools`` is ``None`` the schema-dependent checks cannot run. That is
    reported as ``STATUS_SCHEMA_UNAVAILABLE`` -- it is NOT silently treated as valid.
    """
    openers = OPENER.findall(text)
    if len(openers) > 1:
        return STATUS_REPEATED_OPENER
    if not openers:
        return STATUS_MISSING_OPENER
    if CLOSER not in text:
        return STATUS_MISSING_CLOSER

    body = text[: text.index(CLOSER)]
    param_names = PARAM_OPENER.findall(body)
    if body.count(PARAM_CLOSER) != len(param_names):
        return STATUS_UNCLOSED_PARAMETER

    if served_tools is None:
        return STATUS_SCHEMA_UNAVAILABLE

    declared: dict[str, set[str]] = {}
    for tool in served_tools:
        fn = tool.get("function", tool)
        if fn.get("name"):
            declared[fn["name"]] = set(((fn.get("parameters") or {}).get("properties")) or {})
    if openers[0] not in declared:
        return STATUS_UNKNOWN_FUNCTION
    if any(name not in declared[openers[0]] for name in param_names):
        return STATUS_UNDECLARED_PARAMETER
    return STATUS_VALID


def native_call_names(text: str) -> tuple[str, ...]:
    """Function names the native token stream encodes for one turn."""
    return tuple(OPENER.findall(text))


def json_call_names(content: str) -> tuple[str, ...]:
    """Function names in the JSON tool-call envelopes of one message."""
    names = []
    for block in INLINE_TOOL_CALL.findall(content):
        try:
            names.append(str(json.loads(block).get("name")))
        except (json.JSONDecodeError, AttributeError):
            names.append("<undecodable>")
    return tuple(names)
