# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind training pools into versioned RL curricula.

The pool steps a run trains on pin their source records through the artifact graph; this module
owns the policy binding: which of them a curriculum samples, at what weights or in what stage
order. ``weighted_curriculum`` is the RL analog of :func:`marin.experiment.data.mixture` — the
caller maps pool steps to weights, and the curriculum's entries pin the same pools by artifact
identity, so the mix record and the training data cannot drift apart. Fixing a curriculum also
draws its validation holdout: the builders require an explicit ``validation_size`` and each
pool's row identities, and the drawn split is recorded in the curriculum's metadata. The pool
builder that materializes the pools calls :func:`marin.rl.curriculum.draw_validation_split`
itself — the same pure call, so record and data derive one split — writes the training pools
from its ``ValidationSplit.training`` remainder and the validation data from its held-out rows,
so the holdout is a random subset of the training set and disjoint from it.
"""

from collections.abc import Mapping, Sequence

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.rl.curriculum import (
    CurriculumEntry,
    CurriculumStage,
    PoolRow,
    RlCurriculum,
    SamplingMode,
    draw_validation_split,
)


def _entry(label: str, pool: ArtifactStep[Artifact], weight: float | None) -> CurriculumEntry:
    return CurriculumEntry(label=label, identity=artifact_identity(pool), weight=weight)


def weighted_curriculum(
    name: str,
    version: str,
    *,
    train: Mapping[ArtifactStep[Artifact], float],
    seed: int,
    validation_size: int,
    pool_rows: Mapping[str, Sequence[PoolRow]],
) -> RlCurriculum:
    """Build a weight-random curriculum over the pool steps the run trains on.

    ``train`` maps each pool step to its mixture weight, as ``mixture()`` maps dataset handles.
    The pool step's artifact identity pins the source records behind it; two steps sharing a
    name are rejected because the label is the trainer's sampling key. Fixing the curriculum
    draws its validation holdout here: ``validation_size`` is required (there is no default),
    ``pool_rows`` carries each pool's row identities keyed by pool step name, and the drawn
    split is recorded in the curriculum's metadata. The pools the run trains on must exclude
    the drawn rows: the pool builder materializes them from
    :func:`marin.rl.curriculum.draw_validation_split`'s ``ValidationSplit.training`` — the same
    deterministic call this builder makes — so record and data cannot disagree.
    """
    pools: dict[str, ArtifactStep[Artifact]] = {}
    mixture: dict[str, float] = {}
    entries = []
    for pool, weight in train.items():
        previous = pools.setdefault(pool.name, pool)
        if previous is not pool:
            raise ValueError(f"curriculum {name}@{version} maps two pools to the label {pool.name!r}")
        mixture[pool.name] = weight
        entries.append(_entry(pool.name, pool, weight))
    split = draw_validation_split(mixture, pool_rows, size=validation_size, seed=seed)
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.WEIGHTED,
        entries=tuple(entries),
        stages=(),
        seed=seed,
        validation=split.validation,
    )


def ordered_curriculum(
    name: str,
    version: str,
    *,
    stages: Sequence[CurriculumStage],
    pools: Mapping[str, ArtifactStep[Artifact]],
    seed: int,
    validation_size: int,
    pool_rows: Mapping[str, Sequence[PoolRow]],
) -> RlCurriculum:
    """Build an ordered curriculum over stage objects and the pools they visit.

    ``stages`` declares the schedule in the order the trainer consumes it, each stage carrying its
    own draw quota (``None`` draws the pool's row count). ``pools`` maps each stage's label to the
    pool step the run trains on; every stage label must resolve, and every pool must be visited.
    Fixing the curriculum draws its validation holdout: the holdout's per-label allocation follows
    each pool's consumed prompts (its stage quotas, capped by the pool's row count), with
    ``validation_size`` required and ``pool_rows`` carrying each pool's row identities.
    """
    if not stages:
        raise ValueError(f"ordered curriculum {name}@{version} needs a non-empty stage order")
    entries = []
    seen: set[str] = set()
    consumption: dict[str, float] = {}
    for stage in stages:
        if stage.label in seen:
            continue
        seen.add(stage.label)
        pool = pools.get(stage.label)
        if pool is None:
            raise ValueError(
                f"ordered curriculum {name}@{version} stage {stage.label!r} has no pool; "
                f"the schedule knows: {sorted(pools)}"
            )
        entries.append(_entry(stage.label, pool, None))
    visited = {stage.label for stage in stages}
    stray = sorted(set(pools) - visited)
    if stray:
        raise ValueError(f"ordered curriculum {name}@{version} declares pools no stage visits: {stray}")
    for label in sorted(visited):
        rows = pool_rows.get(label, ())
        quotas = sum(stage.quota if stage.quota is not None else len(rows) for stage in stages if stage.label == label)
        consumption[label] = float(min(quotas, len(rows)))
    split = draw_validation_split(consumption, pool_rows, size=validation_size, seed=seed)
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.ORDERED,
        entries=tuple(entries),
        stages=tuple(stages),
        seed=seed,
        validation=split.validation,
    )
