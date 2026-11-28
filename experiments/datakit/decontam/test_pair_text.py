# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datakit.decontam.pair_text import pair_left


def test_missing_answer_words_stay_out_of_the_question_hint():
    question = "def common(l1: list, l2: list):\n    Return sorted unique common elements."
    right = "```py\ndef common(l1: list, l2: list):\n    Return sorted unique common elements.```"
    left = pair_left(question, [], "def check(candidate): assert candidate", "", right)
    assert left.endswith("\n[missing] none\n[answer missing] check candidate assert")
    stem_only = pair_left("What is the GDP deflator?", ["A price index", "A tax"], "", "", "What is the GDP deflator?")
    assert "[choices] A price index | A tax" in stem_only and "[missing] a price index tax" in stem_only


def test_an_item_with_a_passage_shows_the_passage_and_its_missing_words_apart():
    passage = "The Bears beat the Lions 24-10 in Chicago."
    right = "Question: How many yards was the longest field goal? It was 52 yards in another game."
    left = pair_left("How many yards was the longest field goal?", [], "52", passage, right)
    assert left.startswith("[passage] The Bears beat the Lions")
    assert "\n[question] How many yards" in left
    # "the" and "in" occur in the record, so only the passage's own words are missing.
    assert "\n[passage missing] bears beat lions 24 10 chicago\n[missing] none\n" in left
