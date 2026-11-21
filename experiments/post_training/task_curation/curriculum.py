# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind training pools into versioned RL curricula.

The pool steps a run trains on pin their source records through the artifact graph; this module
owns the policy binding: which of them a curriculum samples, at what weights or in what stage
order. ``weighted_curriculum`` is the RL analog of :func:`marin.experiment.data.mixture` — the
caller maps pool steps to weights, and the curriculum's entries pin the same pools by artifact
identity, so the mix record and the training data cannot drift apart.
"""

from collections.abc import Mapping, Sequence

from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.rl.curriculum import CurriculumEntry, CurriculumStage, RlCurriculum, SamplingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact


def _entry(label: str, pool: ArtifactStep[CampaignArtifact], weight: float | None) -> CurriculumEntry:
    return CurriculumEntry(label=label, identity=artifact_identity(pool), weight=weight)


def weighted_curriculum(
    name: str,
    version: str,
    *,
    train: Mapping[ArtifactStep[CampaignArtifact], float],
    seed: int,
) -> RlCurriculum:
    """Build a weight-random curriculum over the pool steps the run trains on.

    ``train`` maps each pool step to its mixture weight, as ``mixture()`` maps dataset handles.
    The pool step's artifact identity pins the source records behind it; two steps sharing a
    name are rejected because the label is the trainer's sampling key.
    """
    pools: dict[str, ArtifactStep[CampaignArtifact]] = {}
    entries = []
    for pool, weight in train.items():
        previous = pools.setdefault(pool.name, pool)
        if previous is not pool:
            raise ValueError(f"curriculum {name}@{version} maps two pools to the label {pool.name!r}")
        entries.append(_entry(pool.name, pool, weight))
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.WEIGHTED,
        entries=tuple(entries),
        stages=(),
        seed=seed,
    )


def ordered_curriculum(
    name: str,
    version: str,
    *,
    stages: Sequence[CurriculumStage],
    pools: Mapping[str, ArtifactStep[CampaignArtifact]],
    seed: int,
) -> RlCurriculum:
    """Build an ordered curriculum over stage objects and the pools they visit.

    ``stages`` declares the schedule in the order the trainer consumes it, each stage carrying its
    own draw quota (``None`` draws the pool's row count). ``pools`` maps each stage's label to the
    pool step the run trains on; every stage label must resolve, and every pool must be visited.
    """
    if not stages:
        raise ValueError(f"ordered curriculum {name}@{version} needs a non-empty stage order")
    entries = []
    seen: set[str] = set()
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
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.ORDERED,
        entries=tuple(entries),
        stages=tuple(stages),
        seed=seed,
    )
