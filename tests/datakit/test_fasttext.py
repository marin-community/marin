# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiments.datakit.fasttext."""

from __future__ import annotations

from typing import Any

import pytest
from fray import LocalClient, set_current_client

from experiments.datakit.fasttext import (
    _value_from_prediction,
    get_fasttext_batch_predict,
)


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    # zephyr.counters.increment requires a client context; LocalClient is
    # the lightweight in-process backend used by the rest of the test suite.
    with set_current_client(LocalClient()):
        yield


class _FakeModel:
    """Deterministic stand-in for the fasttext model object.

    ``predict(texts, k, threshold)`` returns ``(labels_list, probs_list)``
    parallel to *texts*. The caller hands in fixed per-text ``labels`` /
    ``probs`` lookups so tests can choose what the model "sees".
    """

    def __init__(
        self,
        *,
        labels: list[str] | None = None,
        probs: list[float] | None = None,
    ) -> None:
        # Default: a binary classifier-shaped output with class "1" winning.
        self._labels = labels if labels is not None else ["__label__1", "__label__0"]
        self._probs = probs if probs is not None else [0.7, 0.3]

    def predict(
        self, texts: list[str], k: int = -1, threshold: float = 0.0
    ) -> tuple[list[list[str]], list[list[float]]]:
        return [list(self._labels) for _ in texts], [list(self._probs) for _ in texts]


def _fake_loader(model: _FakeModel) -> Any:
    """Return a model_load_fn closure that hands back *model* regardless of path."""
    return lambda _model_path_str: model


# ---------- _value_from_prediction ----------


def test_value_from_prediction_returns_float_for_matching_label():
    assert _value_from_prediction(stripped=["1", "0"], probs=[0.9, 0.1], score_target_label="1") == pytest.approx(0.9)


def test_value_from_prediction_raises_when_target_label_missing():
    with pytest.raises(ValueError, match=r"score_target_label='2' not in predicted labels"):
        _value_from_prediction(stripped=["1", "0"], probs=[0.9, 0.1], score_target_label="2")


def test_value_from_prediction_returns_full_struct_when_no_target():
    result = _value_from_prediction(
        stripped=["topic_a", "topic_b"],
        probs=[0.7, 0.3],
        score_target_label=None,
    )
    assert result == {
        "top_label": "topic_a",
        "top_score": pytest.approx(0.7),
        "labels": ["topic_a", "topic_b"],
        "scores": [pytest.approx(0.7), pytest.approx(0.3)],
    }


# ---------- get_fasttext_batch_predict (via the returned callable) ----------


def test_annotates_with_default_field_name():
    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        model_load_fn=_fake_loader(_FakeModel()),
    )
    [record] = list(fn([{"id": "a", "text": "hello"}]))
    assert record == {"id": "a", "text": "hello", "fasttext_result": pytest.approx(0.7)}


def test_annotation_preserves_all_input_fields():
    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        output_field_name="score",
        model_load_fn=_fake_loader(_FakeModel()),
    )
    [record] = list(fn([{"id": "a", "text": "hi", "extra": "x", "_source_path": "p"}]))
    assert record == {"id": "a", "text": "hi", "extra": "x", "_source_path": "p", "score": pytest.approx(0.7)}


def test_custom_output_field_name():
    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        output_field_name="quality",
        model_load_fn=_fake_loader(_FakeModel()),
    )
    [record] = list(fn([{"id": "a", "text": "hi"}]))
    assert "quality" in record and "fasttext_result" not in record


def test_full_distribution_annotation_is_nested_struct():
    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label=None,
        output_field_name="topic",
        model_load_fn=_fake_loader(_FakeModel(labels=["__label__art", "__label__sci"], probs=[0.8, 0.2])),
    )
    [record] = list(fn([{"id": "a", "text": "painting"}]))
    assert record["topic"] == {
        "top_label": "art",
        "top_score": pytest.approx(0.8),
        "labels": ["art", "sci"],
        "scores": [pytest.approx(0.8), pytest.approx(0.2)],
    }


def test_skips_empty_text_records():
    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        output_field_name="score",
        model_load_fn=_fake_loader(_FakeModel()),
    )
    results = list(fn([{"id": "a", "text": "hi"}, {"id": "b", "text": ""}, {"id": "c", "text": "yo"}]))
    assert [r["id"] for r in results] == ["a", "c"]


def test_skips_whitespace_only_after_normalization():
    # ``_normalize_for_fasttext`` replaces newlines with spaces; "\n\n\n" -> "   " -> .strip() empty.
    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        output_field_name="score",
        model_load_fn=_fake_loader(_FakeModel()),
    )
    results = list(fn([{"id": "a", "text": "hi"}, {"id": "b", "text": "\n\n\n"}, {"id": "c", "text": "  \t  "}]))
    assert [r["id"] for r in results] == ["a"]


def test_all_empty_batch_does_not_call_model():
    """When no records survive the empty-text filter, ``model.predict`` is skipped entirely."""

    class _FailModel:
        def predict(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("model.predict should not be called when no records survive the empty-text filter")

    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        model_load_fn=_fake_loader(_FailModel()),
    )
    assert list(fn([{"id": "a", "text": ""}, {"id": "b", "text": "\n"}])) == []


def test_max_text_chars_truncates_input_passed_to_model():
    captured: dict[str, list[str]] = {}

    class _CaptureModel:
        def predict(self, texts: list[str], k: int = -1, threshold: float = 0.0) -> Any:
            captured["texts"] = list(texts)
            return [["__label__1", "__label__0"] for _ in texts], [[0.7, 0.3] for _ in texts]

    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=5,
        score_target_label="1",
        model_load_fn=_fake_loader(_CaptureModel()),
    )
    list(fn([{"id": "a", "text": "abcdefghij"}]))
    assert captured["texts"] == ["abcde"]
