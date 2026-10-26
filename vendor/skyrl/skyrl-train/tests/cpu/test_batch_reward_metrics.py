import pytest
from skyrl_gym.verification import VerificationResult
from skyrl_train.batch_metrics import reward_metrics
from skyrl_train.rollouts.buffer import RowFacts


@pytest.mark.parametrize(
    ("rewards", "responses", "expected", "domain_mean"),
    [
        ([1e16, 1.0, -1e16], [[1], [1], [1]], 1.0 / 3.0, 0.0),
        ([[1e16, 1.0, -1e16], [2e16, 3.0, -2e16]], [[1, 2, 3], [1, 2, 3]], 2.0, 2.0),
    ],
    ids=["scalar", "token"],
)
def test_reward_mean_uses_compensated_totals_without_changing_domain_arithmetic(
    rewards, responses, expected, domain_mean
):
    count = len(responses)
    batch = {
        "prompt_token_ids": [[0]] * count,
        "response_ids": responses,
        "rewards": rewards,
        "loss_masks": [[1] * len(row) for row in responses],
        "data_sources": ["alpha"] * count,
    }
    metrics = reward_metrics(
        (RowFacts.from_batch(batch),), [str(index) for index in range(count)], n_samples_per_prompt=1
    )
    assert metrics["reward/avg_raw_reward"] == expected
    assert metrics["reward/domain/alpha/avg_raw_reward"] == domain_mean


def test_compensated_reward_mean_ignores_ungraded_token_rows():
    batch = {
        "prompt_token_ids": [[0], [0]],
        "response_ids": [[1, 2], [1]],
        "loss_masks": [[0, 0], [1]],
        "rewards": [[float("inf"), -float("inf")], [1.0]],
        "verification_results": [VerificationResult.skipped("row did not grade"), VerificationResult.verified(1.0)],
    }
    metrics = reward_metrics((RowFacts.from_batch(batch),), ["a", "b"], n_samples_per_prompt=1)
    assert metrics["reward/avg_raw_reward"] == 1.0
    assert metrics["reward/avg_pass_at_1"] == 1.0
