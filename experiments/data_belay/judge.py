# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Data Belay verdict: per-group BPB change of a treatment against the shared control, over seeds.

For each metric group, ``Δ`` is the treatment's mean BPB over the control's mean BPB, minus 1 (negative is better),
with a Welch t interval over the seeds of both arms. The verdict:

* ``hurts``: some group's interval lies wholly above its harm limit.
* ``helps``: some group's interval lies wholly below 0, and no group's interval reaches its harm limit.
* ``neutral``: every group's interval lies within plus or minus its harm limit, and none lies wholly below 0.
* ``inconclusive``: anything else; more seeds narrow the intervals.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from scipy import stats

# W&B summary key of each metric group's macro BPB.
GROUP_METRICS = {
    "paloma": "eval/paloma/macro_bpb",
    "uncheatable": "eval/uncheatable_eval/macro_bpb",
    "knowledge": "eval/eval_policy/knowledge/macro_bpb",
    "commonsense": "eval/eval_policy/commonsense/macro_bpb",
    "math": "eval/eval_policy/math/macro_bpb",
    "code": "eval/eval_policy/code/macro_bpb",
}
CONFIDENCE = 0.95
# Relative BPB increase past which a group counts as harmed. Set from the control's seed spread (gate 5).
HARM_LIMITS = {group: 0.01 for group in GROUP_METRICS}


class Verdict(StrEnum):
    HELPS = "helps"
    HURTS = "hurts"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"


@dataclasses.dataclass(frozen=True)
class GroupDelta:
    group: str
    delta: float
    low: float
    high: float
    harm_limit: float

    @property
    def improves(self) -> bool:
        return self.high < 0

    @property
    def harms(self) -> bool:
        return self.low > self.harm_limit


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
    if any(d.harms for d in deltas):
        return Verdict.HURTS
    within_limits = all(d.high < d.harm_limit for d in deltas)
    if within_limits and any(d.improves for d in deltas):
        return Verdict.HELPS
    if all(-d.harm_limit < d.low and d.high < d.harm_limit for d in deltas):
        return Verdict.NEUTRAL
    return Verdict.INCONCLUSIVE


def judge(
    control: Mapping[str, Sequence[float]], treatment: Mapping[str, Sequence[float]], harm_limits: Mapping[str, float]
) -> tuple[Verdict, list[GroupDelta]]:
    """Verdict and group deltas from each arm's per-seed BPB, keyed by group."""
    deltas = [group_delta(g, control[g], treatment[g], harm_limits[g]) for g in GROUP_METRICS]
    return verdict(deltas), deltas
