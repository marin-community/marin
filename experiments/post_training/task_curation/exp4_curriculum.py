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

Validation holdout: the curriculum owns a random 200-prompt holdout drawn from
the stamped pools at these source shares, recorded in its metadata, removed
from the pools the run trains on, and materialized as the launch's validation
data. The .55 lineage's adopted 120-row validation.parquet is no longer
consumed. The trainer seed 20261020 seeds the holdout draw too.
"""

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from marin.rl.curriculum import (
    CurriculumValidation,
    PoolRow,
    RlCurriculum,
    ValidationRow,
    ValidationSplit,
    draw_validation_split,
)

from experiments.post_training.task_curation.curriculum import curriculum_with_validation
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE,
    AGENTIC_TRAIN_SOURCE,
    NONAGENTIC_RELEASE,
    RELEASE_SOURCE_ROUTES,
    TEACHER_ROUTE_LABELS,
)
from experiments.post_training.task_curation.exp3_curriculum import SEED
from experiments.post_training.task_curation.sources import runnable_sources

CURRICULUM_NAME = "long-loop/exp4-rl"
CURRICULUM_VERSION = "2026.11.18"
NEMOTRON_IF_LABEL = "nemotron_if"

AGENTIC_SHARE = 0.35
NEMOTRON_IF_SHARE = 0.15
NONAGENTIC_ROUTE = "nonagentic"

SOURCE_ROUTES = MappingProxyType(
    {
        **{TEACHER_ROUTE_LABELS[source]: route for source, route in RELEASE_SOURCE_ROUTES.items()},
        NEMOTRON_IF_LABEL: NONAGENTIC_ROUTE,
    }
)
"""Each curriculum label and the rollout route the buffer admits it through."""


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
        TEACHER_ROUTE_LABELS[label]: 0.5 * count / _NONAGENTIC_ROWS
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

VALIDATION_SIZE = 200

EXP4_CATALOG: Mapping[str, object] = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: AGENTIC_RELEASE,
        NEMOTRON_IF_LABEL: runnable_sources()[NEMOTRON_IF_LABEL],
        **dict.fromkeys(CARRY_OVER_SHARES, NONAGENTIC_RELEASE),
    }
)
"""Each Exp4 label and the catalog source its population comes from."""


def exp4_validation_split(pool_rows: Mapping[str, Sequence[PoolRow]]) -> ValidationSplit:
    """Draw the Exp4 holdout: 200 prompts apportioned over the eight source shares.

    ``pool_rows`` carries every stamped pool's row identities keyed by
    ``teacher_route``; the draw is deterministic for the seed, so record and
    data derive one split."""
    return draw_validation_split(dict(TARGET_SHARES), pool_rows, size=VALIDATION_SIZE, seed=SEED)


def exp4_validation_from_manifest(payload: Mapping) -> CurriculumValidation:
    """Rebuild the recorded holdout from a pool manifest's validation payload.

    The manifest is the record of the split the pool step drew; the curriculum
    reconstructed from it carries the same identity the manifest names."""
    rows = tuple(ValidationRow(**row) for row in payload["rows"])
    return CurriculumValidation(size=payload["size"], seed=payload["seed"], rows=rows)


def exp4_rl_curriculum(validation: CurriculumValidation) -> RlCurriculum:
    """The Exp4 task mix with its recorded holdout.

    ``validation`` is the split the pool step drew and recorded — read back with
    :func:`exp4_validation_from_manifest`, or drawn with
    :func:`exp4_validation_split` from the stamped pools' row identities."""
    return curriculum_with_validation(
        CURRICULUM_NAME,
        CURRICULUM_VERSION,
        weights=TARGET_SHARES,
        seed=SEED,
        validation=validation,
        catalog=EXP4_CATALOG,
    )
