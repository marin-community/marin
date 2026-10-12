# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from rolloutengine.closure_repair import ClosureTargetKind, verified_closure_candidates, verified_closure_target
from rolloutengine.closure_scoring import ClosureFormat, closure_state, score_closure
from rolloutengine.structure_closure import json_closure_state


def codepoints(text: str) -> list[int]:
    return list(map(ord, text))


def text(tokens: list[int]) -> str:
    return "".join(map(chr, tokens))


@pytest.mark.parametrize(
    "structure_format,prefix,suffix",
    [
        (ClosureFormat.JSON, '{"x":1', "}"),
        (ClosureFormat.JSON, '["x"', "]"),
        (ClosureFormat.TOOL, '<tool_call>{"x":1}', "</tool_call>"),
        (ClosureFormat.REASONING, "<|start_think|>done", "<|end_think|>"),
        (ClosureFormat.CODE_FENCE, "```json\n{}\n", "```"),
    ],
)
def test_token_completing_outer_structure_and_adding_text_is_not_a_closure_error(structure_format, prefix, suffix):
    sampled_token = 1_000_000

    def decode(tokens):
        return "".join(suffix + "\nextra" if token == sampled_token else chr(token) for token in tokens)

    target = verified_closure_target(
        codepoints(prefix),
        [sampled_token],
        structure_prefix=prefix,
        inspect=lambda value: closure_state(value, structure_format),
        verify_complete=lambda value: score_closure(value, prefix + suffix, structure_format).success,
        encode=codepoints,
        decode=decode,
    )
    assert target is None


@pytest.mark.parametrize("response", ["", " ", "]", " ]discarded later text"])
def test_json_repair_keeps_sampled_history_and_masks_only_closure(response):
    prefix = '{"items":[{"value":3'
    prompt = codepoints("user context\nassistant\n" + prefix)
    sampled = codepoints(response)
    target = verified_closure_target(
        prompt,
        sampled,
        structure_prefix=prefix,
        inspect=json_closure_state,
        verify_complete=lambda value: json.loads(value) == {"items": [{"value": 3}]},
        encode=codepoints,
        decode=text,
    )
    assert target is not None
    assert text(list(target.repair_token_ids)) == "}]}"
    position = response.index("]") if "]" in response else len(response)
    assert target.response_position == position
    assert target.kind == (ClosureTargetKind.WRONG_TOKEN if "]" in response else ClosureTargetKind.MISSING_SUFFIX)
    assert text(prompt + sampled[:position] + list(target.repair_token_ids)) == (
        "user context\nassistant\n" + prefix + response[:position] + "}]}"
    )
    candidates = verified_closure_candidates(
        target,
        prompt,
        sampled,
        codepoints("]}x"),
        structure_prefix=prefix,
        verify_complete=lambda value: score_closure(value, '{"items":[{"value":3}]}', ClosureFormat.JSON).success,
        decode=text,
    )
    assert candidates == ((ord("}"),) if "]" in response else ())


def test_array_candidates_accept_complete_whitespace_variants_without_content_changes():
    prefix = '["x"'
    pieces = {
        1000000: "]",
        1000001: "]\n",
        1000002: " ]",
        1000003: "]\nextra",
        1000004: ',"other"]',
        1000005: "}\n",
        1000006: " ",
    }

    def decode(tokens):
        return "".join(pieces[token] if token in pieces else chr(token) for token in tokens)

    def verify(value):
        return score_closure(value, '["x"]', ClosureFormat.JSON).success

    prompt = codepoints("instruction\n" + prefix)
    response = codepoints("?")
    target = verified_closure_target(
        prompt,
        response,
        structure_prefix=prefix,
        inspect=json_closure_state,
        verify_complete=verify,
        encode=codepoints,
        decode=decode,
    )
    assert target is not None
    candidates = verified_closure_candidates(
        target,
        prompt,
        response,
        list(pieces),
        structure_prefix=prefix,
        verify_complete=verify,
        decode=decode,
    )
    assert candidates == (1000000, 1000001, 1000002)


@pytest.mark.parametrize("prefix", ['{"x":', '["unfinished', '["escape\\', "[1e", "[tru"])
def test_unfinished_content_does_not_become_closing_supervision(prefix):
    target = verified_closure_target(
        codepoints(prefix),
        [],
        structure_prefix=prefix,
        inspect=json_closure_state,
        verify_complete=lambda value: json.loads(value) == ["unfinished content"],
        encode=codepoints,
        decode=text,
    )
    assert target is None


def test_premature_close_is_not_reinforced_as_a_valid_repair():
    target = verified_closure_target(
        codepoints('["unfinished'),
        codepoints('"]'),
        structure_prefix='["unfinished',
        inspect=json_closure_state,
        verify_complete=lambda value: json.loads(value) == ["unfinished content"],
        encode=codepoints,
        decode=text,
    )
    assert target is None


@pytest.mark.parametrize("ending", ['"', '"]', '"}'])
def test_split_utf8_character_preserves_exact_repair_history(ending):
    prefix = '["'
    response = list(("笹" + ending).encode())
    target = verified_closure_target(
        list(prefix.encode()),
        response,
        structure_prefix=prefix,
        inspect=json_closure_state,
        verify_complete=lambda value: json.loads(value) == ["笹"],
        encode=lambda value: list(value.encode()),
        decode=lambda tokens: bytes(tokens).decode(errors="replace"),
    )
    if ending == '"]':
        assert target is None
    else:
        assert target is not None
        assert target.response_position == 4
        assert target.repair_token_ids == (ord("]"),)
        assert target.kind == (
            ClosureTargetKind.WRONG_TOKEN if ending.endswith("}") else ClosureTargetKind.MISSING_SUFFIX
        )


def test_truncated_utf8_does_not_create_repair_at_unstable_boundary():
    prefix = '["笹'
    target = verified_closure_target(
        list(prefix.encode()),
        [0xE7],
        structure_prefix=prefix,
        inspect=json_closure_state,
        verify_complete=lambda value: json.loads(value) == ["笹"],
        encode=lambda value: list(value.encode()),
        decode=lambda tokens: bytes(tokens).decode(errors="replace"),
    )
    assert target is None


def test_rewrite_of_stable_text_still_rejects_supervision():
    def decode(tokens):
        return '["x' if tokens[-1] == 1000 else text(tokens)

    with pytest.raises(ValueError, match="rewrites its conditioning prefix"):
        verified_closure_target(
            codepoints('["a'),
            [1000],
            structure_prefix='["a',
            inspect=json_closure_state,
            verify_complete=lambda value: json.loads(value) == ["a"],
            encode=codepoints,
            decode=decode,
        )


@pytest.mark.parametrize("response", ["```", "```\nextra", "```<tool_call>"])
def test_fence_closing_line_must_stay_valid_when_generation_continues(response):
    prefix = "```json\n{}\n"
    target = verified_closure_target(
        codepoints(prefix),
        codepoints(response),
        structure_prefix=prefix,
        inspect=lambda value: closure_state(value, ClosureFormat.CODE_FENCE),
        verify_complete=lambda value: score_closure(value, prefix + "```", ClosureFormat.CODE_FENCE).success,
        encode=codepoints,
        decode=text,
    )
    if response == "```<tool_call>":
        assert target is not None
        assert target.kind == ClosureTargetKind.WRONG_TOKEN
        assert target.response_position == 3
        assert text(list(target.repair_token_ids)) == "\n"
    else:
        assert target is None
