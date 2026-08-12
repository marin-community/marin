# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the bme window cutter and the scale-up verdict parsing."""

import pytest

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    GEOMETRY_512,
    GEOMETRY_2048,
    WindowGeometry,
    doc_windows,
)
from experiments.datakit.cluster.quality.fast_transformer.data import load_tokenizer
from experiments.datakit.cluster.quality.fast_transformer.label_windows_openrouter import window_user_content
from experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 import _parse_verdict
from experiments.datakit.cluster.quality.fast_transformer.sample_labels import EXCERPT_NOTICE


def test_parse_verdict_keeps_the_why_field():
    reply = '</think>\n{"idx": 0, "content_type": "code", "valid": true, "quality": 4, "why": "clean helper"}'
    verdict = _parse_verdict(reply)
    assert verdict == {"quality": 4, "content_type": "code", "valid": True, "why": "clean helper"}


def test_overlapping_geometry_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        WindowGeometry(window_tokens=2048, long_doc_tokens=4096)


def test_middle_and_end_windows_carry_a_position_notice_and_begin_does_not():
    whole_doc = {"window": "begin", "text": "doc text", "token_end": 12, "doc_tokens": 12}
    assert window_user_content(whole_doc) == '<document index="0">\ndoc text\n</document>'
    for position, phrase in (("middle", "MIDDLE"), ("end", "END")):
        content = window_user_content({"window": position, "text": "doc text", "token_end": 12, "doc_tokens": 99})
        assert content.startswith(f"[This is a window from the {phrase}")
        # The notice frames the document rather than contaminating it: the text
        # inside the tag stays the window's own.
        assert content.endswith('<document index="0">\ndoc text\n</document>')


def test_begin_window_of_a_continuing_document_carries_the_excerpt_marker():
    """The fix for the scale-up's 36.5% invalid begin windows: a begin window that
    stops before the document's end is marked as an excerpt, which the rubric
    tells the grader never to penalize."""
    cut = window_user_content({"window": "begin", "text": "doc text", "token_end": 2048, "doc_tokens": 9000})
    assert cut == f'<document index="0">\ndoc text{EXCERPT_NOTICE}\n</document>'


@pytest.mark.data_integration
def test_short_doc_yields_one_begin_window_covering_the_whole_doc():
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "A short document about nothing in particular."
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = doc_windows(ids, GEOMETRY_512)
    assert [w.position for w in windows] == ["begin"]
    assert (windows[0].token_start, windows[0].token_end) == (0, len(ids))
    assert windows[0].text == text


@pytest.mark.data_integration
def test_mid_doc_yields_one_begin_window_of_one_geometry_width():
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "word " * 1000  # ~1000 tokens: over one 512 window, under three
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert GEOMETRY_512.window_tokens < len(ids) < GEOMETRY_512.long_doc_tokens
    windows = doc_windows(ids, GEOMETRY_512)
    assert [w.position for w in windows] == ["begin"]
    assert (windows[0].token_start, windows[0].token_end) == (0, GEOMETRY_512.window_tokens)


@pytest.mark.data_integration
@pytest.mark.parametrize("geometry", [GEOMETRY_512, GEOMETRY_2048])
def test_long_doc_yields_three_disjoint_windows_with_exact_offsets(geometry):
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "word " * (4 * geometry.long_doc_tokens)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > geometry.long_doc_tokens
    windows = doc_windows(ids, geometry)
    assert [w.position for w in windows] == ["begin", "middle", "end"]
    for w in windows:
        assert w.token_end - w.token_start == geometry.window_tokens
        assert w.text == tokenizer.decode(ids[w.token_start : w.token_end], clean_up_tokenization_spaces=False)
    begin, middle, end = windows
    assert begin.token_end <= middle.token_start < middle.token_end <= end.token_start
    assert end.token_end == len(ids)


@pytest.mark.data_integration
def test_a_document_just_under_the_2048_threshold_is_graded_on_its_begin_window_alone():
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "word " * 4000  # over 2048 tokens, under 8192
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert GEOMETRY_2048.window_tokens < len(ids) < GEOMETRY_2048.long_doc_tokens
    windows = doc_windows(ids, GEOMETRY_2048)
    assert [w.position for w in windows] == ["begin"]
    assert windows[0].token_end == GEOMETRY_2048.window_tokens
