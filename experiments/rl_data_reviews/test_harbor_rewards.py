# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.rl_data_reviews.review_runtime.harbor_rewards import skill2env_verification


@pytest.mark.parametrize(
    "components,score,passed",
    [
        ((1.0, 1.0, 1.0, 1.0), 1.0, True),
        ((1.0, 0.5, 0.0, 1.0), 0.625, False),
        ((0.0, 0.0, 0.0, 0.0), 0.0, False),
    ],
)
def test_skill2env_component_rewards_preserve_native_scores_and_full_pass(components, score, passed):
    rewards = dict(
        zip(("review_accuracy", "property_quality", "mutation_threshold", "behavior_families"), components, strict=True)
    )

    verification = skill2env_verification(rewards)

    assert verification["status"] == "verified"
    assert verification["score"] == score
    assert verification["passed"] is passed
    assert verification["diagnostics"]["native_rewards"] == rewards
