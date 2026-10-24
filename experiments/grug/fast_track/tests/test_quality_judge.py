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
    example_ids = {task: tuple(str(i) for i in range(EXAMPLES)) for task in gold_bpb}
    return RunScores(gold_bpb, example_ids, long_context_loss + 0.002 * rng.standard_normal(), "tasks/scorer")


def _runs(level: dict[str, float], long_context_loss: float, *, seed: int) -> list[RunScores]:
    rng = np.random.default_rng(seed)
    return [_run(rng, level, long_context_loss) for _ in range(3)]


CONTROL = {"K": 1.8, "S": 1.8, "C": 1.2, "M": 1.5}


@pytest.mark.parametrize(
    ("level", "long_context_loss", "expected"),
    [
        # Every group improves by about 3%.
        ({"K": 1.75, "S": 1.75, "C": 1.16, "M": 1.45}, 3.0, Decision.BETTER),
        # English improves, but code loses more than its 2% harm limit: the web-only failure.
        ({"K": 1.65, "S": 1.65, "C": 1.5, "M": 1.45}, 3.0, Decision.WORSE),
        # Knowledge gains hide a commonsense loss: the quality-classifier trade-off.
        ({"K": 1.6, "S": 1.95, "C": 1.16, "M": 1.45}, 3.0, Decision.WORSE),
        # The groups improve, but long-context use loses 10%: the short-document failure.
        ({"K": 1.7, "S": 1.7, "C": 1.1, "M": 1.45}, 3.3, Decision.WORSE),
        # The changes are smaller than the example noise.
        ({"K": 1.8, "S": 1.8, "C": 1.2, "M": 1.5}, 3.0, Decision.INCONCLUSIVE),
    ],
)
def test_judge_decision_uses_weighted_gain_and_group_harm_limits(level, long_context_loss, expected):
    control = _runs(CONTROL, 3.0, seed=0)
    candidate = _runs(level, long_context_loss, seed=1)

    judgement = judge(candidate, control, samples=200)

    assert judgement.decision is expected
    assert judgement.lower["G"] <= judgement.gains["G"] <= judgement.upper["G"]
