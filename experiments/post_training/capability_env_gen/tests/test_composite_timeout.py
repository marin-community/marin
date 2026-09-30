from capability_pipeline.composite_timeout import (
    calibration_wall_timeout,
    composite_verifier_timeout,
)


def test_composite_trial_budget_covers_machine_checks_and_consensus_calls():
    config = {
        "steps": [{
            "step_index": 0,
            "machine_checks": [{"timeout": 60}] * 3,
            "judge": {
                "criterion_weights": [1.0] * 12,
                "consensus": {"initial_samples": 2},
            },
        }],
    }
    # Three isolated checks, two complete scoring samples, and a possible
    # third sample must fit under the Harbor whole-verifier deadline.
    expected = 3 * (300 + 60) + 12 * 3 * 120 + 60
    assert composite_verifier_timeout(config, 0) == expected
    assert calibration_wall_timeout(
        config, cases=81, repeats=3, concurrency=16,
    ) == 16 * expected + 120
