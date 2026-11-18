# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp4 long-loop RL curriculum: the .55 mix rebalanced toward agentic tasks.

Exp4 keeps every .55 source and its exact consumed share, cuts the agentic route
from 50% to 35% of groups, and spends the freed 15% on the newly admitted
Nemotron instruction-following source (``MarinSkyRL:nemotron_if``), pinned to its
current admitted revision with all its admitted tasks. The composition is policy:
the eight weights below are the expected draw shares, not row-share derivations.

Route geometry on the .55 batch (64 groups per update). The targets put 35% of
groups on the agentic route and 65% on the nonagentic route: exactly 22.4 and
41.6 groups per update. Fixed integer buffer quotas cannot hit that split; the
nearest fixed split is 22/42 (34.375% agentic), and the curriculum's weighted
windows realize 35% exactly by rotating the remainder (agentic windows alternate
22 and 23 groups). Fixed route quotas also cannot realize the per-source shares:
uniform draws within a nonagentic route pool that appends nemotron_if's 54,953
rows would give it about 35% of that pool's rows, not the 15-point share the
targets state. Exp4 therefore samples through the curriculum's domain-weighted
windows, which resolve rows by per-source ``teacher_route`` labels; the route
quotas bound only the route masses.

Seed policy: unchanged from .55. The trainer seed 20261020 seeds the sampler, as
in the .55 run whose ``data.sampling.seed`` fell back to the trainer seed.
"""

from types import MappingProxyType

from marin.rl.curriculum import RlCurriculum, SamplingMode

from experiments.post_training.task_curation.curriculum import curriculum_entry
from experiments.post_training.task_curation.datasets.long_loop_rl import REASONING_GYM_SOURCE
from experiments.post_training.task_curation.datasets.skyrl.ifeval import sources as ifeval_sources
from experiments.post_training.task_curation.exp3_curriculum import AGENTIC_RELEASE, NONAGENTIC_RELEASE
from experiments.post_training.task_curation.source import RlDataSource

CURRICULUM_NAME = "long-loop/exp4-rl"
CURRICULUM_VERSION = "2026.11.18"
SEED = 20261020
BATCH_GROUPS = 64

AGENTIC_TRAIN_SOURCE = "long-loop-smartie-swesmith"
AGENTIC_ROUTE = "agentic"
NONAGENTIC_ROUTE = "nonagentic"
NEMOTRON_IF_LABEL = "nemotron_if"
NEMOTRON_IF_SOURCE_ID = "MarinSkyRL:nemotron_if"
NEMOTRON_IF_ADMITTED_TASKS = 54_953
"""The source's admitted population as the Atlas overlap receipt measured it."""

AGENTIC_SHARE = 0.35
NEMOTRON_IF_SHARE = 0.15

SOURCE_ROUTES = MappingProxyType({
    AGENTIC_TRAIN_SOURCE: AGENTIC_ROUTE,
    NEMOTRON_IF_LABEL: NONAGENTIC_ROUTE,
    "open-thoughts/TaskTrove": NONAGENTIC_ROUTE,
    REASONING_GYM_SOURCE: NONAGENTIC_ROUTE,
    "nvidia/OpenScience": NONAGENTIC_ROUTE,
    "nvidia/Nemotron-RL-Ultra-Training-Blends": NONAGENTIC_ROUTE,
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": NONAGENTIC_ROUTE,
    "PRIME-RL/Eurus-2-RL-Data": NONAGENTIC_ROUTE,
})

TARGET_SHARES = MappingProxyType({
    AGENTIC_TRAIN_SOURCE: AGENTIC_SHARE,
    NEMOTRON_IF_LABEL: NEMOTRON_IF_SHARE,
    # The six .55 sources keep their exact consumed shares, which sum to 50%.
    "open-thoughts/TaskTrove": 0.120165,
    REASONING_GYM_SOURCE: 0.104,
    "nvidia/OpenScience": 0.10395,
    "nvidia/Nemotron-RL-Ultra-Training-Blends": 0.06213,
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 0.05817,
    "PRIME-RL/Eurus-2-RL-Data": 0.051585,
})
assert sum(TARGET_SHARES.values()) == 1.0

EXACT_ROUTE_GROUPS = MappingProxyType(
    {
        route: sum(share for label, share in TARGET_SHARES.items() if SOURCE_ROUTES[label] == route)
        * BATCH_GROUPS
        for route in (AGENTIC_ROUTE, NONAGENTIC_ROUTE)
    }
)
"""Exact per-route group counts per 64-group update: 22.4 agentic, 41.6 nonagentic."""

FIXED_ROUTE_QUOTAS = MappingProxyType({AGENTIC_ROUTE: 22, NONAGENTIC_ROUTE: 42})
"""The nearest fixed integer buffer quotas; 22.4 and 41.6 are not integers, so this
split samples the agentic route at 34.375%, 0.625 points under the target. The
curriculum's weighted windows close that gap by rotating the remainder."""

def _nemotron_if() -> RlDataSource:
    """The admitted Nemotron instruction-following source, from its dataset module."""
    for source in ifeval_sources():
        if source.name == NEMOTRON_IF_LABEL:
            return source
    raise ValueError(f"the ifeval dataset module does not declare {NEMOTRON_IF_SOURCE_ID}")


def target_weights() -> dict[str, float]:
    """The declared composition as ``{label: expected draw share}``."""
    return dict(TARGET_SHARES)


def exp4_rl_curriculum() -> RlCurriculum:
    """The Exp4 task mix as a versioned weighted curriculum over the eight sources."""
    entries = []
    for label, weight in TARGET_SHARES.items():
        if label == AGENTIC_TRAIN_SOURCE:
            source = AGENTIC_RELEASE
        elif label == NEMOTRON_IF_LABEL:
            source = _nemotron_if()
        else:
            source = NONAGENTIC_RELEASE
        entries.append(curriculum_entry(label, source, weight))
    return RlCurriculum(
        name=CURRICULUM_NAME,
        version=CURRICULUM_VERSION,
        mode=SamplingMode.WEIGHTED,
        entries=tuple(entries),
        stages=(),
        seed=SEED,
    )
