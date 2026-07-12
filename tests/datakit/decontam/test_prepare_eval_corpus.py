# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the decontam eval-corpus text building (marin#6852 cluster B).

Passage-bearing reading-comprehension / QA eval docs must index question+answer
but not the public passage, so a corpus doc that merely quotes the passage is
not falsely flagged. Non-passage docs are unchanged.
"""

import pytest

from experiments.datakit.decontam.prepare_eval_corpus import _PASSAGE_FIELDS, _lmh_doc_text

# A rendered prompt embeds the passage (as lm-eval-harness doc_to_text does).
_PASSAGE = "The rain had continued for a week and a flood created a big river by the farm."


def _prompt(doc: dict) -> str:
    body = " ".join(str(doc.get(k, "")) for k in ("article", "passage", "context", "premise", "story"))
    return f"Read: {body}\nQuestion: {doc.get('question', '')}"


def _target(doc: dict) -> str:
    return str(doc.get("answer", ""))


def test_lmh_passage_doc_drops_passage_keeps_qa():
    """RC doc: passage dropped (raw field + doc_to_text), question/answer/options kept."""
    doc = {
        "article": _PASSAGE,
        "question": "What did Nancy do when the flood came",
        "answer": "C",
        "options": ["ran away", "hid inside", "gathered her cows", "slept through it"],
    }
    text = _lmh_doc_text(doc, _prompt, _target)
    assert _PASSAGE not in text, "public passage must not be indexed"
    assert "What did Nancy do when the flood came" in text  # question kept
    assert "gathered her cows" in text  # options kept
    assert "C" in text  # answer kept


def test_lmh_non_passage_doc_unchanged():
    """No passage field → doc_to_text (the question) is kept as before."""
    doc = {"question": "What is the sum of two and two", "answer": "4"}
    text = _lmh_doc_text(doc, _prompt, _target)
    assert "What is the sum of two and two" in text
    assert "4" in text


@pytest.mark.parametrize("field", sorted(_PASSAGE_FIELDS))
def test_lmh_every_passage_field_suppresses_passage(field: str):
    """Each passage-like field name triggers suppression; the question survives."""
    doc = {
        field: "UNIQUE_PUBLIC_PASSAGE_MARKER spanning several ordinary words here",
        "question": "Q_MARKER here",
        "answer": "yes",
    }
    text = _lmh_doc_text(doc, lambda d: f"renders {d.get(field, '')} then {d['question']}", _target)
    assert "UNIQUE_PUBLIC_PASSAGE_MARKER" not in text, field
    assert "Q_MARKER" in text, field
    assert "yes" in text, field


def test_lmh_doc_to_text_exception_is_tolerated():
    """A doc_to_text that raises still yields the answer + raw fields (no crash)."""

    def boom(_doc):
        raise RuntimeError("no template")

    doc = {"question": "Q here", "answer": "42"}
    text = _lmh_doc_text(doc, boom, _target)
    assert "Q here" in text  # from raw fields
    assert "42" in text  # answer
