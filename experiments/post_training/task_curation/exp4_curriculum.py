# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp4 long-loop RL curriculum: the .55 mix with 35% agentic groups.

Exp4 keeps the six nonagentic .55 source shares, cuts the agentic route from
50% to 35%, and assigns the freed 15% to the Nemotron instruction-following
source (``nemotron_if``). The six carry-over shares derive from the adopted
releases' pinned per-source row counts under the .55 route quotas.

This curriculum's entries are keyed by per-source ``teacher_route`` labels,
because the policy is per-source: uniform draws within a route pool cannot
realize it. On the .55 batch of 64 groups, the 35% agentic target is 22.4 groups.
The trainer's largest-remainder apportionment (vendored ``weighted_quotas``)
assigns 22 agentic and 42 nonagentic groups in every window, or 34.375% agentic.
Five other sources have larger fractional remainders than the agentic source and
receive the remaining seats. Uniform draws from a fixed nonagentic route pool
with nemotron_if's 54,953 rows would give it about 35% of that pool. Its target
is 15% of all groups.

Pools: the entries pin one pool per source, extracted from the two regional
stamped pools (the adopted releases plus the admitted Nemotron rows,
route-labeled by ``teacher_route``). Each pool step excludes the holdout rows
drawn for its label, so the per-source pools the run trains on carry no
validation prompts, and the step names are the labels themselves — the sampling
keys the spec's curriculum check matches against ``train_data``.

Validation holdout: the curriculum owns a random 200-prompt holdout drawn from
the stamped pools at these source shares, recorded in its metadata, removed
from the pools, and materialized as the launch's validation data. The .55
lineage's adopted 120-row validation.parquet is no longer consumed.

Seed policy: unchanged from .55. The trainer seed 20261020 seeds the sampler,
as in the .55 run whose ``data.sampling.seed`` fell back to the trainer seed;
this curriculum inherits the value and the policy from the Exp3 definition —
and seeds the holdout draw.
"""

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from marin.execution.lazy import OUT, ArtifactStep, apply, artifact_identity
from marin.rl.curriculum import (
    CurriculumEntry,
    PoolRow,
    RlCurriculum,
    SamplingMode,
    ValidationSplit,
    draw_validation_split,
)

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE,
    AGENTIC_ROUTE,
    AGENTIC_TRAIN_SOURCE,
    NONAGENTIC_RELEASE,
    NONAGENTIC_ROUTE,
    REASONING_GYM_LABEL,
    REASONING_GYM_SOURCE,
    RELEASE_SOURCE_ROUTES,
    STAMPED_POOLS,
    extract_source_pool,
    extract_validation_pool,
    stamped_pool_step,
)
from experiments.post_training.task_curation.exp3_curriculum import SEED

CURRICULUM_NAME = "long-loop/exp4-rl"
CURRICULUM_VERSION = "2026.11.18"
NEMOTRON_IF_LABEL = "nemotron_if"

AGENTIC_SHARE = 0.35
NEMOTRON_IF_SHARE = 0.15

_TRAIN_SOURCE_COUNTS = {}
for _release in (AGENTIC_RELEASE, NONAGENTIC_RELEASE):
    _config = _release.config
    if _config is None:
        raise ValueError(f"adopted release {_release.name} has no configuration")
    for _member in _config.members:
        if _member.source_counts:
            _TRAIN_SOURCE_COUNTS.update(_member.source_counts)

TEACHER_ROUTE_LABELS: Mapping[str, str] = MappingProxyType(
    {**{source: source for source in _TRAIN_SOURCE_COUNTS}, REASONING_GYM_SOURCE: REASONING_GYM_LABEL}
)
"""Each training row's ``data_source`` and the ``teacher_route`` the stamped pools
label it with — identical except the reasoning-gym wheel URL, renamed to its
source identity so every pool step name is a valid path segment."""

SOURCE_ROUTES: Mapping[str, str] = MappingProxyType(
    {
        **{TEACHER_ROUTE_LABELS[source]: route for source, route in RELEASE_SOURCE_ROUTES.items()},
        NEMOTRON_IF_LABEL: NONAGENTIC_ROUTE,
    }
)
"""Each curriculum label and the rollout route the buffer admits it through."""

_NONAGENTIC_ROWS = sum(
    count for label, count in _TRAIN_SOURCE_COUNTS.items() if RELEASE_SOURCE_ROUTES[label] == NONAGENTIC_ROUTE
)

CARRY_OVER_SHARES: Mapping[str, float] = MappingProxyType(
    {
        TEACHER_ROUTE_LABELS[label]: 0.5 * count / _NONAGENTIC_ROWS
        for label, count in _TRAIN_SOURCE_COUNTS.items()
        if RELEASE_SOURCE_ROUTES[label] == NONAGENTIC_ROUTE
    }
)
"""The six nonagentic sources keep their exact .55 consumed shares: the .55 route's
batch share (32 of 64 groups) times each source's row share of the route pool."""

TARGET_SHARES: Mapping[str, float] = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: AGENTIC_SHARE,
        NEMOTRON_IF_LABEL: NEMOTRON_IF_SHARE,
        **CARRY_OVER_SHARES,
    }
)
if not math.isclose(sum(TARGET_SHARES.values()), 1.0, abs_tol=1e-9):
    raise ValueError("the Exp4 target shares must sum to one")

VALIDATION_SIZE = 200


def exp4_validation_split(pool_rows: Mapping[str, Sequence[PoolRow]]) -> ValidationSplit:
    """Draw the Exp4 holdout: 200 prompts apportioned over the eight source shares.

    ``pool_rows`` carries every stamped pool's row identities keyed by
    ``teacher_route`` (``stamped_pool_rows`` streams them). The draw is
    deterministic for the seed, so record and data derive one split."""
    return draw_validation_split(dict(TARGET_SHARES), pool_rows, size=VALIDATION_SIZE, seed=SEED)


def _stamped_pool_for(label: str) -> str:
    """Which stamped route pool a label's population lives in."""
    return AGENTIC_ROUTE if SOURCE_ROUTES[label] == AGENTIC_ROUTE else NONAGENTIC_ROUTE


def exp4_pool_steps(version: str, split: ValidationSplit) -> dict[str, ArtifactStep]:
    """The per-source pools the run trains on, each excluding its holdout rows.

    Every source population is extracted from its stamped route pool; the step's
    config carries the excluded holdout ids, so each pool's identity covers the
    split that produced it. The step names are the labels themselves — the
    trainer's sampling keys, and the identity keys the spec's curriculum check
    matches against ``train_data``."""
    stamped = {route: stamped_pool_step(route) for route in (AGENTIC_ROUTE, NONAGENTIC_ROUTE)}
    pools = {}
    for label in sorted(TARGET_SHARES):
        route = _stamped_pool_for(label)
        pools[label] = apply(
            label,
            extract_source_pool,
            version=version,
            output=OUT,
            pool={"uri": stamped[route], "sha256": STAMPED_POOLS[route][1]},
            label=label,
            excluded=tuple(sorted(row.id for row in split.validation.rows if row.label == label)),
            expected_rows=len(split.training[label]),
        )
    return pools


def exp4_validation_step(version: str, split: ValidationSplit) -> ArtifactStep:
    """The holdout's validation.parquet, extracted and hash-verified from the stamped pools."""
    stamped = [stamped_pool_step(route) for route in (AGENTIC_ROUTE, NONAGENTIC_ROUTE)]
    return apply(
        "long-loop-exp4/validation",
        extract_validation_pool,
        version=version,
        output=OUT,
        pools=[
            {"uri": step, "sha256": STAMPED_POOLS[route][1]}
            for step, route in zip(stamped, (AGENTIC_ROUTE, NONAGENTIC_ROUTE), strict=True)
        ],
        rows=[{"label": row.label, "id": row.id, "sha256": row.sha256} for row in split.validation.rows],
        seed=SEED,
    )


def exp4_rl_curriculum(version: str, split: ValidationSplit) -> RlCurriculum:
    """The Exp4 task mix, pinned to the holdout-excluding per-source pools.

    ``split`` is the draw :func:`exp4_validation_split` returned for the stamped
    pools; the entries pin the pools built from its training remainder."""
    pools = exp4_pool_steps(version, split)
    entries = tuple(
        CurriculumEntry(label=label, identity=artifact_identity(pools[label]), weight=TARGET_SHARES[label])
        for label in sorted(TARGET_SHARES)
    )
    return RlCurriculum(
        name=CURRICULUM_NAME,
        version=CURRICULUM_VERSION,
        mode=SamplingMode.WEIGHTED,
        entries=entries,
        stages=(),
        seed=SEED,
        validation=split.validation,
    )
