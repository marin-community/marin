# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiments.datakit.fasttext."""

from typing import Any

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient

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

    ``predict(text, k, threshold)`` returns ``(labels, probs)`` for the single
    input text -- mirrors fasttext-wheel's single-string ``predict`` contract.
    The wrapper calls this once per text (not in batch) because batch
    ``predict(list, k=-1)`` returns duplicated probs in fasttext-wheel 0.9.2;
    see ``_predict_batch`` and ``test_predict_batch_is_per_text``.
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

    def predict(self, text: str, k: int = -1, threshold: float = 0.0) -> tuple[list[str], list[float]]:
        return list(self._labels), list(self._probs)


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
    captured: dict[str, list[str]] = {"texts": []}

    class _CaptureModel:
        def predict(self, text: str, k: int = -1, threshold: float = 0.0) -> Any:
            captured["texts"].append(text)
            return ["__label__1", "__label__0"], [0.7, 0.3]

    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=5,
        score_target_label="1",
        model_load_fn=_fake_loader(_CaptureModel()),
    )
    list(fn([{"id": "a", "text": "abcdefghij"}]))
    assert captured["texts"] == ["abcde"]


# ---------- per-text predict (fasttext-wheel 0.9.2 batch quirk regression) ----------


def test_predict_batch_is_per_text():
    """``_predict_batch`` must call ``model.predict`` once per text, not in batch.

    fasttext-wheel 0.9.2 returns duplicate probs across labels when handed a
    list (e.g. ``predict([t1, t2], k=-1)`` -> ``[[0.97, 0.97], [0.97, 0.97]]``
    instead of the per-label softmax), so the only correct invocation is one
    text at a time. This test wires up a model that returns DIFFERENT probs
    per call, then asserts every record in the input batch gets its own
    distinct prediction -- impossible if the wrapper had reverted to
    ``predict(list)``.
    """
    call_log: list[str] = []

    class _PerTextModel:
        """Returns a different ``P(label="1")`` for each text based on length."""

        def predict(self, text: str, k: int = -1, threshold: float = 0.0) -> Any:
            call_log.append(text)
            # Make the prob depend on the text -- one-shot batch predict would
            # collapse them to a single repeated value.
            p_one = (len(text) % 7) / 10.0  # 0.0, 0.1, ... 0.6 -- deterministic per text
            return ["__label__1", "__label__0"], [p_one, 1.0 - p_one]

    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        output_field_name="score",
        model_load_fn=_fake_loader(_PerTextModel()),
    )
    inputs = [
        {"id": "a", "text": "x"},  # len 1 -> p=0.1
        {"id": "b", "text": "xxx"},  # len 3 -> p=0.3
        {"id": "c", "text": "xxxxx"},  # len 5 -> p=0.5
    ]
    out = list(fn(inputs))

    # One predict call per surviving text, in the same order.
    assert call_log == ["x", "xxx", "xxxxx"]
    # Each record gets its own distinct score derived from its own text.
    assert [r["score"] for r in out] == [pytest.approx(0.1), pytest.approx(0.3), pytest.approx(0.5)]


def test_predict_batch_rejects_batch_predict_call():
    """If the wrapper ever regresses to ``model.predict(list_of_texts, ...)`` again,
    we want the test suite to scream. ``_PoisonBatchModel.predict`` raises on
    list inputs but accepts strings, so any list-mode call kills the test.
    """

    class _PoisonBatchModel:
        def predict(self, text: str, k: int = -1, threshold: float = 0.0) -> Any:
            assert isinstance(
                text, str
            ), f"_predict_batch must call model.predict with a single str; got {type(text).__name__}: {text!r}"
            return ["__label__1", "__label__0"], [0.6, 0.4]

    fn = get_fasttext_batch_predict(
        model_path="ignored",
        max_text_chars=None,
        score_target_label="1",
        output_field_name="score",
        model_load_fn=_fake_loader(_PoisonBatchModel()),
    )
    out = list(fn([{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]))
    assert [r["score"] for r in out] == [pytest.approx(0.6), pytest.approx(0.6)]
