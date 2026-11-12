# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Data Belay verdict: per-group change in answer loss of a treatment, over seeds.

For each metric group, ``Δ`` is the treatment's mean over a baseline's mean, minus 1 (negative is better), with a
Welch t interval over the values of both arms. Each group has two:

* the gain, against the shared control, which trains on the base mix alone;
* the harm, against the dilution references at the treatment's share, which give that share to text no eval measures.
  Giving up the share of the base mix costs Uncheatable about 2% on its own at a 20% share, past its harm limit, so
  the harm is what the treatment costs beyond that. Without references at the share, the harm is against the control.

Against another candidate (``report --against``), both are against that candidate: the two give up the same share.

Each group gets a call:

* ``better``: the gain's upper end is below 0, and its ``Δ`` is at least MIN_GAIN in size.
* ``worse``: the harm's lower end is above the group's harm limit.
* ``neutral``: not better, and the harm's upper end is below the harm limit.
* ``unresolved``: not better or worse, and the harm's upper end is at or above the harm limit; more seeds narrow it.

The verdict:

* ``hurts``: some group is worse.
* ``helps``: some group is better, and every other group is better or neutral.
* ``trade-off``: some group is better, and some other group is unresolved.
* ``no gain``: no group is better and none is worse.

A candidate climbs the ladder of rungs until two consecutive rungs say no (``hurts`` or ``no gain``), unless a rung is
left and some group's gain ``Δ`` at the second of them is below 0. A yes is ``helps`` or ``trade-off``: a gain on one of
the evals. At the top rung the answer is yes when it and the rung below both say yes, and mixed when they disagree.
"""

import dataclasses
import math
import statistics
from collections.abc import Mapping, Sequence
from enum import StrEnum

from scipy import stats

from experiments.datasets.eval_policy_answer_bpb import TASKS, AnswerTask, TaskGroup, task_component_name

# BoolQ is logged but left out of the verdict: its one-token yes/no answer measures the model's prior for "yes", whose
# BPB varied 10.7% between control seeds at d512 (PIQA and WinoGrande: 0.6%).
EXCLUDED_TASKS = frozenset({"boolq"})
# Multiple-choice tasks judged by choice loss instead of answer BPB: -log2 of the gold option's softmax probability
# over the options' summed log-likelihoods, which experiments.data_belay.accuracy scores after training. Answer BPB
# misses a gain in the gold answer's margin over the distractors, and choice loss sees it. PIQA and WinoGrande stay on
# answer BPB: in the pilot that scored them with the Eval Policy prompts, their choice loss varied about 4 times as much
# between control seeds.
CHOICE_TASKS = frozenset({"mmlu_pro", "truthful_qa"})
CHOICE_METRIC_PREFIX = "choice/"


def _task_metric(task: AnswerTask) -> str:
    if task.name in CHOICE_TASKS:
        return f"{CHOICE_METRIC_PREFIX}{task.name}"
    return f"eval/{task_component_name(task)}/bpb"


# Each group's metrics: W&B summary keys, and ``choice/<task>`` for the tasks in CHOICE_TASKS.
GROUP_METRICS = {
    "paloma": ("eval/paloma/macro_bpb",),
    "uncheatable": ("eval/uncheatable_eval/macro_bpb",),
    **{
        str(group): tuple(
            _task_metric(task) for task in TASKS if task.group is group and task.name not in EXCLUDED_TASKS
        )
        for group in TaskGroup
    },
}
# Groups whose metrics mix units: knowledge's choice loss is in bits per question, TriviaQA's answer BPB in bits per
# byte. Their value averages each metric over its mean on the rung's control seeds; the other groups average BPB.
SCALED_GROUPS = frozenset({"knowledge"})
CONFIDENCE = 0.95
# Smallest gain that counts, the harm limits' floor: a change under 1% is neither a gain nor a harm. With the interval
# alone, a simulated null d512 rung read yes 13.8% of the time, whatever the seeds' spread; with the floor,
# 5.7% at the vanilla Track's seed SDs and 1.2% at the fast-track dense model's. The fast-track model's null arm
# (control seeds 6-8) came out math -0.56% [-1.09%, -0.03%] at d512.
MIN_GAIN = 0.01
# Relative increase past which a group counts as harmed: 1%, or 2.5 seed SDs where the seeds vary more, so that
# 3 seeds per arm can call a group neutral. The SDs are from the vanilla Track's first four d512 control seeds
# (dbelay-d512-control-s0..s3): Paloma 0.45%, Uncheatable 0.54%, knowledge 0.83%, commonsense 0.40%, math 1.06%,
# code 1.22%.
HARM_LIMITS = {
    "paloma": 0.011,
    "uncheatable": 0.013,
    "knowledge": 0.0206,  # 2.5 x 0.826%, unrounded
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

    @property
    def is_yes(self) -> bool:
        return self in (Verdict.HELPS, Verdict.TRADE_OFF)


class LadderStep(StrEnum):
    NEXT_RUNG = "run the next rung"
    STOP = "stop: no at two consecutive rungs"
    YES = "yes at the top two rungs"
    MIXED = "mixed: the top two rungs disagree"


@dataclasses.dataclass(frozen=True)
class Change:
    """Relative change of the treatment's mean over a baseline's mean, and its interval."""

    delta: float
    low: float
    high: float


@dataclasses.dataclass(frozen=True)
class GroupDelta:
    group: str
    gain: Change
    """Against the control, or the other candidate."""
    harm: Change
    """Against the dilution references, or the gain's baseline when there are none."""
    harm_limit: float

    @property
    def call(self) -> GroupCall:
        if self.gain.high < 0 and self.gain.delta <= -MIN_GAIN:
            return GroupCall.BETTER
        if self.harm.low > self.harm_limit:
            return GroupCall.WORSE
        if self.harm.high < self.harm_limit:
            return GroupCall.NEUTRAL
        return GroupCall.UNRESOLVED


@dataclasses.dataclass(frozen=True)
class RungResult:
    verdict: Verdict
    groups: tuple[GroupDelta, ...]


def change(group: str, baseline: Sequence[float], treatment: Sequence[float]) -> Change:
    """Relative change of the treatment's mean over the baseline's mean, with a Welch t interval.

    The interval is on the difference of means, divided by the baseline mean.
    """
    if len(baseline) < 2 or len(treatment) < 2:
        raise ValueError(f"{group}: each arm needs at least 2 values, got {len(baseline)} and {len(treatment)}")
    baseline_mean = sum(baseline) / len(baseline)
    treatment_mean = sum(treatment) / len(treatment)
    var_b = stats.tvar(baseline) / len(baseline)
    var_t = stats.tvar(treatment) / len(treatment)
    se = math.sqrt(var_b + var_t)
    # Welch-Satterthwaite degrees of freedom.
    dof = (var_b + var_t) ** 2 / (var_b**2 / (len(baseline) - 1) + var_t**2 / (len(treatment) - 1)) if se > 0 else 1
    half = stats.t.ppf(0.5 + CONFIDENCE / 2, dof) * se
    diff = treatment_mean - baseline_mean
    return Change(diff / baseline_mean, (diff - half) / baseline_mean, (diff + half) / baseline_mean)


def group_values(metrics: Mapping[str, float], control_means: Mapping[str, float]) -> dict[str, float]:
    """One run's value of each group, from its metrics and each metric's mean over the rung's control seeds."""
    return {
        group: statistics.fmean(metrics[k] / (control_means[k] if group in SCALED_GROUPS else 1.0) for k in keys)
        for group, keys in GROUP_METRICS.items()
    }


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
    control: Mapping[str, Sequence[float]],
    treatment: Mapping[str, Sequence[float]],
    harm_limits: Mapping[str, float],
    dilution: Mapping[str, Sequence[float]] | None,
) -> RungResult:
    """Verdict and group deltas from each arm's group values, keyed by group.

    ``control`` and ``treatment`` hold one value per seed; ``control`` may be another candidate's runs. ``dilution``
    holds one value per dilution reference at the treatment's share (results.dilution_arm), or is None, and the harm
    is then against ``control`` too.
    """
    harm_baseline = control if dilution is None else dilution
    groups = tuple(
        GroupDelta(g, change(g, control[g], treatment[g]), change(g, harm_baseline[g], treatment[g]), harm_limits[g])
        for g in GROUP_METRICS
    )
    return RungResult(verdict(groups), groups)


def ladder_step(results: Sequence[RungResult], rungs: int) -> LadderStep:
    """The next step after ``results``, one per rung from the bottom, on a ladder of ``rungs`` rungs."""
    if len(results) >= 2 and not results[-1].verdict.is_yes and not results[-2].verdict.is_yes:
        # A gain at one rung can clear its interval at the next: of 49 Datakit sources stopped at d768 that ran on to
        # d1024, 6 were trade-offs there, and each had its gaining group below 0 at d768. Any group counts, including a
        # better one in a rung that hurts. The run-on costs runs only: a yes still needs both top rungs.
        if len(results) < rungs and any(g.gain.delta < 0 for g in results[-1].groups):
            return LadderStep.NEXT_RUNG
        return LadderStep.STOP
    if len(results) < rungs:
        return LadderStep.NEXT_RUNG
    return LadderStep.YES if results[-1].verdict.is_yes and results[-2].verdict.is_yes else LadderStep.MIXED
