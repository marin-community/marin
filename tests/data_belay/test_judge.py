# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.data_belay.judge import GROUP_METRICS, Verdict, judge

CONTROL = [1.000, 1.002, 0.998, 1.001]
LIMITS = {group: 0.01 for group in GROUP_METRICS}


def _arms(changes: dict[str, float]) -> tuple[dict, dict]:
    control = {group: CONTROL for group in GROUP_METRICS}
    treatment = {group: [v * (1 + changes.get(group, 0.0)) for v in CONTROL[:3]] for group in GROUP_METRICS}
    return control, treatment


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"math": -0.05}, Verdict.HELPS),
        # A gain in one group does not hide a harm beyond the limit in another.
        ({"math": -0.05, "paloma": 0.03}, Verdict.HURTS),
        ({}, Verdict.NEUTRAL),
        # A harm inside the limit keeps a gain a gain.
        ({"code": -0.04, "paloma": 0.004}, Verdict.HELPS),
    ],
)
def test_verdict(changes, expected):
    control, treatment = _arms(changes)
    verdict, deltas = judge(control, treatment, LIMITS)
    assert verdict is expected
    for delta in deltas:
        assert delta.low <= changes.get(delta.group, 0.0) + 1e-9 <= delta.high + 2e-9


def test_noisy_arms_are_inconclusive():
    control = {group: CONTROL for group in GROUP_METRICS}
    treatment = {group: [0.97, 1.03, 1.01] for group in GROUP_METRICS}
    verdict, _ = judge(control, treatment, LIMITS)
    assert verdict is Verdict.INCONCLUSIVE
