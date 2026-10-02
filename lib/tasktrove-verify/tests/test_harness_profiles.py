# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.adapters.harness_profiles import span_equal, token_accuracy, token_f1
from tasktrove_verify.grade import InvalidTask


def test_counter_f1_counts_repeated_tokens_instead_of_set_overlap():
    assert token_f1(["fox", "fox", "red"], ["fox", "red"]).reward == pytest.approx(0.8)
    assert token_f1(["fox", "fox"], ["fox", "fox"]).reward == 1
    assert token_f1([], []).reward == 0
    assert token_f1(["Fox"], ["fox"]).reward == 0


def test_pos_accuracy_preserves_source_shorter_length_policy():
    result = token_accuracy(["NOUN", "VERB", "ADP"], ["NOUN", "ADP"])
    assert result.reward == 0.5
    assert result.detail["compared"] == 2
    with pytest.raises(InvalidTask, match="no comparable"):
        token_accuracy([], ["NOUN"])


def test_span_matching_preserves_tag_and_full_entity_identity():
    assert span_equal(("person", "mary jane"), ("person", "mary jane")).reward == 1
    assert span_equal(("person", "mary jane"), ("location", "mary jane")).reward == 0
    assert span_equal(("person", "maryjane"), ("person", "mary jane")).reward == 0
