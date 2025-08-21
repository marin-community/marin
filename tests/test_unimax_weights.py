import pytest

from marin.utilities.unimax_weights import unimax_weights


def test_negative_budget():
    with pytest.raises(ValueError):
        unimax_weights({"a": 1}, budget=-1)


def test_nonpositive_max_epochs():
    with pytest.raises(ValueError):
        unimax_weights({"a": 1}, budget=1, max_epochs=0)

    with pytest.raises(ValueError):
        unimax_weights({"a": 1}, budget=1, max_epochs=-1)


def test_negative_corpus_size():
    with pytest.raises(ValueError):
        unimax_weights({"a": -1}, budget=1)


def test_empty_corpus_nonzero_budget():
    with pytest.raises(ValueError):
        unimax_weights({"a": 0, "b": 0}, budget=1)
