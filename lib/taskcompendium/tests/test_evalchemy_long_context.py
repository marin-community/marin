# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.importers.evalchemy.scorers.long_context import grade_nonce_similarity


@pytest.mark.parametrize("prefix", ["", "\n", "\n\n  "])
def test_mrcr_reply_is_graded_from_its_first_non_whitespace_character(prefix):
    reward = grade_nonce_similarity("NONCE a poem about tapirs", "NONCE", prefix + "NONCE a poem about tapirs")

    assert reward.detail["metrics"]["prefix_hit_rate"] == 1.0
    assert reward.detail["metrics"]["accuracy"] == 1.0


def test_mrcr_reply_without_the_nonce_scores_zero():
    reward = grade_nonce_similarity("NONCE a poem about tapirs", "NONCE", "\na poem about tapirs")

    assert reward.detail["metrics"] == {"accuracy": 0.0, "prefix_hit_rate": 0.0}
