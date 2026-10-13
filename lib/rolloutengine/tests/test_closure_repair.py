# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from rolloutengine.closure_repair import (
    ClosureTargetKind,
    verified_closure_candidates,
    verified_closure_target,
    verified_continuation_boundary,
)
from rolloutengine.closure_scoring import ClosureFormat, closure_state, score_closure
from rolloutengine.structure_closure import json_closure_state


def codepoints(text: str) -> list[int]:
    return list(map(ord, text))


def text(tokens: list[int]) -> str:
    return "".join(map(chr, tokens))


@pytest.mark.parametrize(
    "structure_format,prefix,expected,closer,continuation",
    [
        (ClosureFormat.JSON, '["unfin', '["unfinished"]', '"]', "ished"),
        (ClosureFormat.JSON, '"unfin', '"unfinished"', '"', "ished"),
        (
            ClosureFormat.TOOL,
            '<tool_call>{"x":"unfin',
            '<tool_call>{"x":"unfinished"}</tool_call>',
            '"}</tool_call>',
            "ished",
        ),
        (
            ClosureFormat.REASONING,
            "<|start_think|>first",
            "<|start_think|>first second<|end_think|>",
            "<|end_think|>",
            " second",
        ),
        (ClosureFormat.CODE_FENCE, "```text\nfirst\n", "```text\nfirst\nsecond\n```", "```", "second\n"),
    ],
)
def test_premature_closure_selects_only_reference_extending_recorded_tokens(
    structure_format, prefix, expected, closer, continuation
):
    pieces = {1000000: closer, 1000001: continuation, 1000002: "unrelated"}

    def decode(tokens):
        return "".join(pieces[token] if token in pieces else chr(token) for token in tokens)

    prompt = codepoints("context\n" + prefix)
    boundary = verified_continuation_boundary(
        prompt,
        [1000000],
        [list(pieces)],
        structure_prefix=prefix,
        expected=expected,
        structure_format=structure_format,
        decode=decode,
    )
    assert boundary is not None
    assert boundary.response_position == 0
    assert boundary.candidate_token_ids == (1000001,)
    assert decode([*prompt, 1000001]) == "context\n" + prefix + continuation


@pytest.mark.parametrize("response", ['changed"]', 'ished"]', "still open"])
def test_continuation_target_excludes_content_drift_correct_completion_and_unclosed_response(response):
    assert (
        verified_continuation_boundary(
            codepoints('["unfin'),
            codepoints(response),
            [codepoints("ix")] * len(response),
            structure_prefix='["unfin',
            expected='["unfinished"]',
            structure_format=ClosureFormat.JSON,
            decode=text,
        )
        is None
    )


def test_continuation_target_does_not_move_back_before_already_diverged_closing_token():
    # Separate quote and bracket tokens have already lost the exact reference
    # prefix by the first outer-completion token.
    assert (
        verified_continuation_boundary(
            codepoints('["unfin'),
            codepoints('"]'),
            [codepoints("ix")] * 2,
            structure_prefix='["unfin',
            expected='["unfinished"]',
            structure_format=ClosureFormat.JSON,
            decode=text,
        )
        is None
    )


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


@pytest.mark.parametrize(
    "structure_format,prefix,required_response,drift,suffix",
    [
        (ClosureFormat.JSON, '"slash=\\\\', "", "n\\\\n", '"'),
        (ClosureFormat.JSON, '"unfin', "ished", " extra", '"'),
        (ClosureFormat.REASONING, "<|start_think|>Done.", "", " More thoughts", "<|end_think|>"),
        (ClosureFormat.CODE_FENCE, "~~~~json\n{}\n", "", "```\n", "~~~~"),
    ],
)
def test_unclosed_content_drift_repairs_latest_verified_reference_boundary(
    structure_format, prefix, required_response, drift, suffix
):
    expected = prefix + required_response + suffix
    prompt = codepoints("instruction\n" + prefix)
    response = codepoints(required_response + drift)
    target = verified_closure_target(
        prompt,
        response,
        structure_prefix=prefix,
        inspect=lambda value: closure_state(value, structure_format),
        verify_complete=lambda value: score_closure(value, expected, structure_format).success,
        encode=codepoints,
        decode=text,
    )
    assert target is not None
    assert target.kind == ClosureTargetKind.MISSING_SUFFIX
    assert target.response_position == len(required_response)
    assert text(list(target.repair_token_ids)) == suffix
    repaired = text(prompt + response[: target.response_position] + list(target.repair_token_ids))
    assert repaired == "instruction\n" + expected


@pytest.mark.parametrize(
    "prefix,response,expected",
    [
        ('"unfinished', " other words", '"unfinished content"'),
        ('{"value":', " ", '{"value":1}'),
        ('"done', ' extra"', '"done"'),
    ],
)
def test_missing_closure_fallback_neither_invents_content_nor_changes_completed_structure(prefix, response, expected):
    target = verified_closure_target(
        codepoints(prefix),
        codepoints(response),
        structure_prefix=prefix,
        inspect=json_closure_state,
        verify_complete=lambda value: score_closure(value, expected, ClosureFormat.JSON).success,
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


@pytest.mark.parametrize(
    "response", ["```", "```\nextra", "```<tool_call>", "```<tool_call>\n```", "```<tool_call>\n```\nextra"]
)
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
