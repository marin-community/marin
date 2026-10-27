# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""label_check recomputes the answer of a synthesized instance instead of trusting its own label."""

import pytest

pytest.importorskip("sympy")

from experiments.post_training.rsi.label_check import label_correct

SORT_QUESTION = (
    "Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: "
    "banana, Apple, cherry"
)
SORT_METADATA = {"original_words": ["banana", "Apple", "cherry"], "direction": "ascending"}


def test_word_sorting_accepts_code_point_order_and_rejects_dictionary_order():
    assert label_correct("word_sorting", SORT_QUESTION, {"answer": "Apple, banana, cherry", "metadata": SORT_METADATA})
    # Dictionary order agrees here, so swap to a case where code points differ.
    question = SORT_QUESTION.replace("Apple", "apple").replace("cherry", "Cherry")
    metadata = {"original_words": ["banana", "apple", "Cherry"], "direction": "ascending"}
    assert label_correct("word_sorting", question, {"answer": "Cherry, apple, banana", "metadata": metadata})
    assert not label_correct("word_sorting", question, {"answer": "apple, banana, Cherry", "metadata": metadata})


def test_word_sorting_rejects_metadata_that_disagrees_with_the_question():
    metadata = {"original_words": ["banana", "Apple", "plum"], "direction": "ascending"}
    assert not label_correct("word_sorting", SORT_QUESTION, {"answer": "Apple, banana, plum", "metadata": metadata})


def test_simple_equations_solves_the_equation():
    entry = {"metadata": {"equation": "3*k + 4 - 1 = 18", "variable": "k"}}
    question = "Find the value of k in the equation: 3*k + 4 - 1 = 18"
    assert label_correct("simple_equations", question, {**entry, "answer": "5"})
    assert not label_correct("simple_equations", question, {**entry, "answer": "6"})


def test_rotate_matrix_reduces_the_angle_modulo_360():
    question = "Rotate the matrix below by 450 degrees clockwise:\n1 2\n3 4\n"
    metadata = {"matrix": [[1, 2], [3, 4]]}
    assert label_correct("rotate_matrix", question, {"answer": "3 1\n4 2", "metadata": metadata})
    assert not label_correct("rotate_matrix", question, {"answer": "1 2\n3 4", "metadata": metadata})


def test_acre_answers_undetermined_without_evidence():
    question = (
        "Previous experimental results:\nred cube → on\nblue sphere, red cube → on\ngreen cylinder → off\n\n"
        "New test case:\nblue sphere\n\nWhat is the detector light status?"
    )
    assert label_correct("acre", question, {"answer": "undetermined", "metadata": {}})
    assert not label_correct("acre", question, {"answer": "on", "metadata": {}})


def test_a_type_without_a_check_is_unchecked():
    assert label_correct("countdown", "Make 10 from 2 and 5.", {"answer": "2*5", "metadata": {}}) is None
