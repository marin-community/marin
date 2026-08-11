# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the bme window cutter and the scale-up verdict parsing."""

import pytest

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    LONG_DOC_TOKENS,
    WINDOW_TOKENS,
    doc_windows,
)
from experiments.datakit.cluster.quality.fast_transformer.data import load_tokenizer
from experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 import _parse_verdict


def test_parse_verdict_keeps_the_why_field():
    reply = '</think>\n{"idx": 0, "content_type": "code", "valid": true, "quality": 4, "why": "clean helper"}'
    verdict = _parse_verdict(reply)
    assert verdict == {"quality": 4, "content_type": "code", "valid": True, "why": "clean helper"}


@pytest.mark.data_integration
def test_short_doc_yields_one_begin_window_covering_the_whole_doc():
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "A short document about nothing in particular."
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = doc_windows(ids)
    assert [w.position for w in windows] == ["begin"]
    assert (windows[0].token_start, windows[0].token_end) == (0, len(ids))
    assert windows[0].text == text


@pytest.mark.data_integration
def test_mid_doc_yields_one_begin_window_of_512_tokens():
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "word " * 1000  # ~1000 tokens: over one window, under three
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert WINDOW_TOKENS < len(ids) <= LONG_DOC_TOKENS
    windows = doc_windows(ids)
    assert [w.position for w in windows] == ["begin"]
    assert (windows[0].token_start, windows[0].token_end) == (0, WINDOW_TOKENS)


@pytest.mark.data_integration
def test_long_doc_yields_three_disjoint_512_token_windows_with_exact_offsets():
    tokenizer = load_tokenizer("unsloth/gemma-3-270m-it")
    text = "word " * 4000
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > LONG_DOC_TOKENS
    windows = doc_windows(ids)
    assert [w.position for w in windows] == ["begin", "middle", "end"]
    for w in windows:
        assert w.token_end - w.token_start == WINDOW_TOKENS
        assert w.text == tokenizer.decode(ids[w.token_start : w.token_end], clean_up_tokenization_spaces=False)
    begin, middle, end = windows
    assert begin.token_end <= middle.token_start < middle.token_end <= end.token_start
    assert end.token_end == len(ids)
