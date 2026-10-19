# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.grug.fast_track.quality_judge import GROUP_TASKS, Decision, RunScores, judge

EXAMPLES = 400


def _run(rng: np.random.Generator, level: dict[str, float], long_context_loss: float) -> RunScores:
    gold_bpb = {}
    for group, tasks in GROUP_TASKS.items():
        for task in tasks:
            gold_bpb[task] = level[group] + 0.3 * rng.standard_normal(EXAMPLES) + 0.01 * rng.standard_normal()
    return RunScores(gold_bpb, long_context_loss + 0.002 * rng.standard_normal())


def _runs(level: dict[str, float], long_context_loss: float, *, seed: int) -> list[RunScores]:
    rng = np.random.default_rng(seed)
    return [_run(rng, level, long_context_loss) for _ in range(3)]


CONTROL = {"E": 1.8, "C": 1.2, "M": 1.5}


@pytest.mark.parametrize(
    ("level", "long_context_loss", "expected"),
    [
        # Every group improves by about 3%.
        ({"E": 1.75, "C": 1.16, "M": 1.45}, 3.0, Decision.BETTER),
        # English improves, but code loses more than its 2% harm limit: the web-only failure.
        ({"E": 1.65, "C": 1.5, "M": 1.45}, 3.0, Decision.WORSE),
        # The groups improve, but long-context use loses 10%: the short-document failure.
        ({"E": 1.7, "C": 1.1, "M": 1.45}, 3.3, Decision.WORSE),
        # The changes are smaller than the example noise.
        ({"E": 1.8, "C": 1.2, "M": 1.5}, 3.0, Decision.INCONCLUSIVE),
    ],
)
def test_judge_decision_uses_weighted_gain_and_group_harm_limits(level, long_context_loss, expected):
    control = _runs(CONTROL, 3.0, seed=0)
    candidate = _runs(level, long_context_loss, seed=1)

    judgement = judge(candidate, control, samples=200)

    assert judgement.decision is expected
    assert judgement.lower["G"] <= judgement.gains["G"] <= judgement.upper["G"]


def test_judge_rejects_a_single_run_per_arm():
    runs = _runs(CONTROL, 3.0, seed=0)
    with pytest.raises(ValueError, match="at least two"):
        judge(runs[:1], runs)
