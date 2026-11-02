# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest
from marin.datakit.chat_render import render_marin_chat
from openai_harmony import Author, Message, Role
from transformers import AutoTokenizer

from experiments.grug_sft.reasoning_span_batches import (
    PAD_SEGMENT,
    Mode,
    SpanTokens,
    TurnShape,
    conversation_mode,
    find_pattern,
    patch_batch,
    patch_document,
    patch_row,
)
from experiments.grug_sft.replay_data import StepBatch
from experiments.marin_tokenizer import MARIN_CUSTOM_SPECIAL_TOKENS, create_marin_tokenizer

_RESERVED_SPECIAL_TOKENS = ("<|reserved_special_token_0|>", "<|reserved_special_token_1|>")
EMPTY_SPAN_RE = re.compile(r"(assistant<\|end_header_id\|>\n)(?!<\|start_think\|>)")


@pytest.fixture(scope="module")
def tokenizer(gpt2_tokenizer_path, tmp_path_factory):
    base = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    base.add_special_tokens({"additional_special_tokens": list(_RESERVED_SPECIAL_TOKENS)})
    reserved_ids = base.convert_tokens_to_ids(list(_RESERVED_SPECIAL_TOKENS))
    renames = dict(zip(reserved_ids, MARIN_CUSTOM_SPECIAL_TOKENS.values(), strict=True))
    path = tmp_path_factory.mktemp("marin_tokenizer")
    create_marin_tokenizer(base, renames).save_pretrained(path)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.fixture(scope="module")
def spans(tokenizer) -> SpanTokens:
    return SpanTokens.from_tokenizer(tokenizer)


def tool_call(name: str, arguments: str) -> Message:
    return (
        Message.from_role_and_content(Role.ASSISTANT, arguments)
        .with_channel("commentary")
        .with_recipient(f"functions.{name}")
    )


def tool_reply(name: str, text: str) -> Message:
    return (
        Message.from_author_and_content(Author(role=Role.TOOL, name=f"functions.{name}"), text)
        .with_channel("commentary")
        .with_recipient("assistant")
    )


def conversation() -> list[Message]:
    """One turn per shape: adjacent spans before a call, a call-only turn, one span, a final turn without reasoning."""
    return [
        Message.from_role_and_content(Role.USER, "List the files, then summarize."),
        Message.from_role_and_content(Role.ASSISTANT, "Need the file list.").with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "Run ls.").with_channel("analysis"),
        tool_call("ls", '{"path": "."}'),
        tool_reply("ls", "a.py b.py"),
        tool_call("cat", '{"path": "a.py"}'),
        tool_reply("cat", "print(1)"),
        Message.from_role_and_content(Role.ASSISTANT, "Two files; one prints.").with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "a.py prints 1 and b.py is empty.").with_channel("final"),
        Message.from_role_and_content(Role.USER, "Thanks."),
        Message.from_role_and_content(Role.ASSISTANT, "You're welcome.").with_channel("final"),
    ]


def render(tokenizer, *, enable_thinking: bool) -> tuple[str, np.ndarray]:
    text = render_marin_chat(conversation(), bos_token="", enable_thinking=enable_thinking)
    return text, np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.uint32)


def expected_fixed_text(rendered: str) -> str:
    """The #10519 target format applied to the rendered text instead of its tokens."""
    joined = rendered.replace("<|end_think|><|start_think|>", "\n")
    return EMPTY_SPAN_RE.sub(r"\1<|start_think|><|end_think|>", joined)


def test_patched_tokens_match_retokenized_target_format(tokenizer, spans):
    rendered, tokens = render(tokenizer, enable_thinking=True)
    weight = np.ones(len(tokens), dtype=np.uint8)
    assert conversation_mode(tokens, spans) == Mode.THINK

    patch = patch_document(tokens, weight, spans, repair=True)

    expected = tokenizer.encode(expected_fixed_text(rendered), add_special_tokens=False)
    assert patch.tokens.tolist() == expected
    assert patch.shapes == {
        TurnShape.ADJACENT_SPANS: 1,
        TurnShape.ZERO_SPAN_TOOL_CALL: 1,
        TurnShape.ONE_SPAN: 1,
        TurnShape.ZERO_SPAN_TEXT: 1,
    }
    assert (patch.inserted_spans, patch.merged_pairs) == (2, 1)
    assert len(patch.loss_weight) == len(patch.tokens)


def test_new_tokens_take_the_weight_of_the_predicting_position(tokenizer, spans):
    _, tokens = render(tokenizer, enable_thinking=True)
    # Weight 1 only on the positions that predict assistant content, as a completions-masked SFT row has.
    weight = np.zeros(len(tokens), dtype=np.uint8)
    for header in np.flatnonzero(tokens == spans.assistant_header[0]):
        if tuple(tokens[header : header + len(spans.assistant_header)]) != spans.assistant_header:
            continue
        end = (
            header
            + len(spans.assistant_header)
            + int(np.argmax(tokens[header + len(spans.assistant_header) :] == spans.eot[0]))
        )
        weight[header + len(spans.assistant_header) - 1 : end] = 1

    patch = patch_document(tokens, weight, spans, repair=True)

    # Two inserted empty spans add four weighted positions; the merged pair removes one.
    assert int(patch.loss_weight.sum()) == int(weight.sum()) + 4 - 1
    assert patch.loss_weight[patch.tokens == spans.start_think].tolist() == [1, 1, 1, 1]


def test_nothink_and_sliced_documents_are_counted_but_unchanged(tokenizer, spans):
    _, nothink = render(tokenizer, enable_thinking=False)
    assert conversation_mode(nothink, spans) == Mode.NOTHINK
    patch = patch_document(nothink, np.ones(len(nothink), dtype=np.uint8), spans, repair=True)
    assert patch.tokens.tolist() == nothink.tolist()
    # The marin-v3 adapter wraps analysis in think tokens whatever the mode, so the shapes still count.
    assert patch.shapes == {
        TurnShape.ADJACENT_SPANS: 1,
        TurnShape.ZERO_SPAN_TOOL_CALL: 1,
        TurnShape.ONE_SPAN: 1,
        TurnShape.ZERO_SPAN_TEXT: 1,
    }

    _, think = render(tokenizer, enable_thinking=True)
    first_user = int(find_pattern(think, spans.start_header)[1])
    sliced = think[first_user:]
    assert conversation_mode(sliced, spans) == Mode.UNDEFINED
    assert conversation_mode(sliced[1:], spans) == Mode.UNKNOWN
    patch = patch_document(sliced[1:], np.ones(len(sliced) - 1, dtype=np.uint8), spans, repair=True)
    assert patch.tokens.tolist() == sliced[1:].tolist()


def packed_row(tokens: np.ndarray, copies: int, padding: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_tokens = np.concatenate([tokens] * copies + [np.full(padding, 7, dtype=np.uint32)])
    weight = np.concatenate([np.ones(len(tokens) * copies, dtype=np.uint8), np.zeros(padding, dtype=np.uint8)])
    segments = np.concatenate(
        [np.full(len(tokens), i, dtype=np.int32) for i in range(copies)]
        + [np.full(padding, PAD_SEGMENT, dtype=np.int32)]
    )
    return row_tokens, weight, segments


def test_rows_keep_their_length_through_padding_and_trimming(tokenizer, spans):
    _, tokens = render(tokenizer, enable_thinking=True)
    growth = 2 * 2 - 1  # two empty spans in, one end/start pair out, per document

    row, weight, segments = packed_row(tokens, copies=2, padding=10)
    patched_row = patch_row(row, weight, segments, spans, repair=True, pad_token=7)
    new_tokens, new_weight, new_segments, report = (
        patched_row.tokens,
        patched_row.loss_weight,
        patched_row.segment_ids,
        patched_row.report,
    )
    assert len(new_tokens) == len(row)
    assert report.length_delta == 2 * growth
    assert int(np.sum(new_segments == PAD_SEGMENT)) == 10 - 2 * growth
    assert new_tokens[new_segments == PAD_SEGMENT].tolist() == [7] * (10 - 2 * growth)
    assert new_weight[new_segments == PAD_SEGMENT].sum() == 0
    assert new_weight[np.flatnonzero(new_segments == PAD_SEGMENT)[0] - 1] == 0
    assert (new_segments[: len(tokens) + growth] == 0).all()

    row, weight, segments = packed_row(tokens, copies=1, padding=0)
    batch = StepBatch(row[None], weight[None], segments[None], sliding_window=None)
    patched, summary = patch_batch(batch, spans, repair=True)
    assert patched.tokens.shape == batch.tokens.shape
    assert summary.rows_trimmed == 1 and summary.rows_padded == 0
    assert summary.max_length_delta == growth
    assert summary.think_turns == {
        TurnShape.ADJACENT_SPANS: 1,
        TurnShape.ZERO_SPAN_TOOL_CALL: 1,
        TurnShape.ONE_SPAN: 1,
        TurnShape.ZERO_SPAN_TEXT: 1,
    }
    unpatched, scan = patch_batch(batch, spans, repair=False)
    assert unpatched.tokens.tolist() == batch.tokens.tolist()
    assert scan.inserted_spans == 0 and scan.think_turns == summary.think_turns
