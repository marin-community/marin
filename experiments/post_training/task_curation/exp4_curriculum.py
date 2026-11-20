# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp4 long-loop RL curriculum: the .55 mix with 35% agentic groups.

Exp4 keeps the six nonagentic .55 source shares, cuts the agentic route from
50% to 35%, and assigns the freed 15% to the Nemotron instruction-following
source (``MarinSkyRL:nemotron_if``). Its entry pins the upstream dataset revision.
The six carry-over shares derive from the adopted releases' pinned per-source row
counts under the .55 route quotas.

This curriculum's entries are keyed by per-source ``teacher_route`` labels,
because the policy is per-source: uniform draws within a route pool cannot
realize it. On the .55 batch of 64 groups, the 35% agentic target is 22.4 groups.
The trainer's largest-remainder apportionment (vendored ``weighted_quotas``)
assigns 22 agentic and 42 nonagentic groups in every window, or 34.375% agentic.
Five other sources have larger fractional remainders than the agentic source and
receive the remaining seats. Uniform draws from a fixed nonagentic route pool
with nemotron_if's 54,953 rows would give it about 35% of that pool. Its target
is 15% of all groups.

Sampling contract: the domain-weighted sampler requires per-source
``teacher_route`` labels in the input pool. No pool in this stack carries them
yet (the recorded .55 parquets carry ``data_source`` provenance and route through
``source_routes``), and the ``nemotron_if`` entry pins the upstream dataset
revision rather than a curated pool, so a launch needs the stamped-pool and
curated-``nemotron_if``-pool artifacts before it can sample this curriculum.

Seed policy: unchanged from .55. The trainer seed 20261020 seeds the sampler, as
in the .55 run whose ``data.sampling.seed`` fell back to the trainer seed; this
curriculum inherits the value and the policy from the Exp3 definition.
"""

import math
from types import MappingProxyType

from marin.rl.curriculum import RlCurriculum

from experiments.post_training.task_curation.curriculum import weighted_curriculum
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE,
    AGENTIC_TRAIN_SOURCE,
    NONAGENTIC_RELEASE,
    RELEASE_SOURCE_ROUTES,
)
from experiments.post_training.task_curation.exp3_curriculum import SEED
from experiments.post_training.task_curation.sources import runnable_sources

CURRICULUM_NAME = "long-loop/exp4-rl"
CURRICULUM_VERSION = "2026.11.18"
NEMOTRON_IF_LABEL = "nemotron_if"

AGENTIC_SHARE = 0.35
NEMOTRON_IF_SHARE = 0.15
NONAGENTIC_ROUTE = "nonagentic"

SOURCE_ROUTES = MappingProxyType({**RELEASE_SOURCE_ROUTES, NEMOTRON_IF_LABEL: NONAGENTIC_ROUTE})


def train_source_counts() -> MappingProxyType:
    counts = {}
    for release in (AGENTIC_RELEASE, NONAGENTIC_RELEASE):
        config = release.config
        if config is None:
            raise ValueError(f"adopted release {release.name} has no configuration")
        for member in config.members:
            if member.source_counts:
                counts.update(member.source_counts)
    return MappingProxyType(counts)


_TRAIN_SOURCE_COUNTS = train_source_counts()

_NONAGENTIC_ROWS = sum(
    count for label, count in _TRAIN_SOURCE_COUNTS.items() if RELEASE_SOURCE_ROUTES[label] == NONAGENTIC_ROUTE
)

CARRY_OVER_SHARES = MappingProxyType(
    {
        label: 0.5 * count / _NONAGENTIC_ROWS
        for label, count in _TRAIN_SOURCE_COUNTS.items()
        if RELEASE_SOURCE_ROUTES[label] == NONAGENTIC_ROUTE
    }
)
"""The six nonagentic sources keep their exact .55 consumed shares: the .55 route's
batch share (32 of 64 groups) times each source's row share of the route pool."""

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
