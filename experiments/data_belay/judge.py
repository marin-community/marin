# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Data Belay verdict: per-group BPB change of a treatment against the shared control, over seeds.

For each metric group, ``Δ`` is the treatment's mean BPB over the control's mean BPB, minus 1 (negative is better),
with a Welch t interval over the seeds of both arms. Each group gets a call:

* ``better``: the interval lies wholly below 0.
* ``worse``: the interval lies wholly above the group's harm limit.
* ``neutral``: the interval reaches neither below 0 wholly nor up to the harm limit: no gain shown, and no harm
  beyond the limit.
* ``unresolved``: the interval reaches the harm limit but does not lie above it; more seeds narrow it.

The verdict:

* ``hurts``: some group is worse.
* ``helps``: some group is better, and every other group is better or neutral.
* ``trade-off``: some group is better, and some other group is unresolved: it may be harmed beyond its limit.
* ``no gain``: no group is better and none is worse.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from scipy import stats

from experiments.data_belay.answer_bpb import TASKS, TaskGroup, task_component_name

# BoolQ is logged but left out of the verdict: its one-token yes/no answer measures the model's prior for "yes", whose
# BPB varied 10.7% between control seeds at d512 (PIQA and WinoGrande: 0.6%).
EXCLUDED_TASKS = frozenset({"boolq"})
# Each group's BPB is the mean of these W&B summary keys.
GROUP_METRICS = {
    "paloma": ("eval/paloma/macro_bpb",),
    "uncheatable": ("eval/uncheatable_eval/macro_bpb",),
    **{
        str(group): tuple(
            f"eval/{task_component_name(task)}/bpb"
            for task in TASKS
            if task.group is group and task.name not in EXCLUDED_TASKS
        )
        for group in TaskGroup
    },
}
CONFIDENCE = 0.95
# Relative BPB increase past which a group counts as harmed: 1%, or 2.5 seed SDs where the seeds vary more, so that
# 3 seeds per arm can call a group neutral. The SDs are from the first four d512 control seeds
# (dbelay-d512-control-s0..s3): Paloma 0.45%, Uncheatable 0.54%, knowledge 0.92%, commonsense 0.40%, math 1.06%,
# code 1.22%.
HARM_LIMITS = {
    "paloma": 0.011,
    "uncheatable": 0.013,
    "knowledge": 0.023,
    "commonsense": 0.010,
    "math": 0.027,
    "code": 0.030,
}


class GroupCall(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    NEUTRAL = "neutral"
    UNRESOLVED = "unresolved"


class Verdict(StrEnum):
    HELPS = "helps"
    HURTS = "hurts"
    TRADE_OFF = "trade-off"
    NO_GAIN = "no gain"


@dataclasses.dataclass(frozen=True)
class GroupDelta:
    group: str
    delta: float
    low: float
    high: float
    harm_limit: float

    @property
    def call(self) -> GroupCall:
        if self.high < 0:
            return GroupCall.BETTER
        if self.low > self.harm_limit:
            return GroupCall.WORSE
        if self.high < self.harm_limit:
            return GroupCall.NEUTRAL
        return GroupCall.UNRESOLVED


def group_delta(group: str, control: Sequence[float], treatment: Sequence[float], harm_limit: float) -> GroupDelta:
    """Relative change of the treatment's mean over the control's mean, with a Welch t interval.

    The interval is on the difference of means, divided by the control mean.
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError(f"{group}: each arm needs at least 2 seeds, got {len(control)} and {len(treatment)}")
    control_mean = sum(control) / len(control)
    treatment_mean = sum(treatment) / len(treatment)
    var_c = stats.tvar(control) / len(control)
    var_t = stats.tvar(treatment) / len(treatment)
    se = math.sqrt(var_c + var_t)
    # Welch-Satterthwaite degrees of freedom.
    dof = (var_c + var_t) ** 2 / (var_c**2 / (len(control) - 1) + var_t**2 / (len(treatment) - 1)) if se > 0 else 1
    half = stats.t.ppf(0.5 + CONFIDENCE / 2, dof) * se
    diff = treatment_mean - control_mean
    return GroupDelta(group, diff / control_mean, (diff - half) / control_mean, (diff + half) / control_mean, harm_limit)


def verdict(deltas: Sequence[GroupDelta]) -> Verdict:
    calls = [d.call for d in deltas]
    if GroupCall.WORSE in calls:
        return Verdict.HURTS
    if GroupCall.BETTER not in calls:
        return Verdict.NO_GAIN
    if GroupCall.UNRESOLVED in calls:
        return Verdict.TRADE_OFF
    return Verdict.HELPS


def judge(
    control: Mapping[str, Sequence[float]], treatment: Mapping[str, Sequence[float]], harm_limits: Mapping[str, float]
) -> tuple[Verdict, list[GroupDelta]]:
    """Verdict and group deltas from each arm's per-seed BPB, keyed by group."""
    deltas = [group_delta(g, control[g], treatment[g], harm_limits[g]) for g in GROUP_METRICS]
    return verdict(deltas), deltas
