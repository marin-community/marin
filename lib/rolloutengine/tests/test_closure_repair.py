# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from rolloutengine.closure_repair import ClosureTargetKind, verified_closure_candidates, verified_closure_target
from rolloutengine.structure_closure import json_closure_state


def codepoints(text: str) -> list[int]:
    return list(map(ord, text))


def text(tokens: list[int]) -> str:
    return "".join(map(chr, tokens))


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
    candidates = verified_closure_candidates(target, prompt, sampled, codepoints("]}x"), decode=text)
    assert candidates == ((ord("}"),) if "]" in response else ())


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
