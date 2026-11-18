# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp4 long-loop RL curriculum: the .55 mix with 35% agentic groups.

Exp4 keeps the six nonagentic .55 source shares, cuts the agentic route from
50% to 35%, and assigns the freed 15% to the Nemotron instruction-following
source (``MarinSkyRL:nemotron_if``). Its entry pins the upstream dataset revision.
The six carry-over shares come from the Exp3 definition.

On the .55 batch of 64 groups, the 35% agentic target is 22.4 groups. The
trainer's largest-remainder apportionment (vendored ``weighted_quotas``) assigns
22 agentic and 42 nonagentic groups in every window, or 34.375% agentic. Five
other sources have larger fractional remainders than the agentic source and
receive the remaining seats. Uniform draws from a fixed nonagentic route pool
with nemotron_if's 54,953 rows would give it about 35% of that pool. Its target
is 15% of all groups. The declared domain-weighted sampler requires per-source
``teacher_route`` labels in the input pool.

Seed policy: unchanged from .55. The trainer seed 20261020 seeds the sampler, as
in the .55 run whose ``data.sampling.seed`` fell back to the trainer seed; this
curriculum inherits the value and the policy from the Exp3 definition.
"""

import math
from types import MappingProxyType

from marin.rl.curriculum import RlCurriculum

from experiments.post_training.task_curation.curriculum import weighted_curriculum
from experiments.post_training.task_curation.datasets.long_loop_rl import AGENTIC_TRAIN_SOURCE
from experiments.post_training.task_curation.exp3_curriculum import (
    AGENTIC_RELEASE,
    NONAGENTIC_RELEASE,
    SEED,
)
from experiments.post_training.task_curation.exp3_curriculum import (
    weights as exp3_weights,
)
from experiments.post_training.task_curation.sources import runnable_sources

CURRICULUM_NAME = "long-loop/exp4-rl"
CURRICULUM_VERSION = "2026.11.18"
NEMOTRON_IF_LABEL = "nemotron_if"

AGENTIC_SHARE = 0.35
NEMOTRON_IF_SHARE = 0.15

CARRY_OVER_SHARES = MappingProxyType(
    {label: share for label, share in exp3_weights().items() if label != AGENTIC_TRAIN_SOURCE}
)

TARGET_SHARES = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: AGENTIC_SHARE,
        NEMOTRON_IF_LABEL: NEMOTRON_IF_SHARE,
        **CARRY_OVER_SHARES,
    }
)
if not math.isclose(sum(TARGET_SHARES.values()), 1.0, abs_tol=1e-9):
    raise ValueError("the Exp4 target shares must sum to one")


def exp4_rl_curriculum() -> RlCurriculum:
    """The Exp4 task mix as a versioned weighted curriculum over the eight sources."""
    catalog = {
        AGENTIC_TRAIN_SOURCE: AGENTIC_RELEASE,
        NEMOTRON_IF_LABEL: runnable_sources()[NEMOTRON_IF_LABEL],
        **dict.fromkeys(CARRY_OVER_SHARES, NONAGENTIC_RELEASE),
    }
    return weighted_curriculum(
        CURRICULUM_NAME,
        CURRICULUM_VERSION,
        weights=TARGET_SHARES,
        seed=SEED,
        catalog=catalog,
    )
