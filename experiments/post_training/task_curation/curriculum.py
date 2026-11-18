# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind catalog sources into versioned RL curricula.

The catalog (:func:`~experiments.post_training.task_curation.sources.runnable_sources`) owns which
sources exist and what pins them; this module owns the policy binding: which of them a curriculum
samples, at what weights or in what stage order. The experiment that launches the run declares
weights, as with SFT mixture weights.
"""

from collections.abc import Mapping, Sequence

from marin.rl.curriculum import CurriculumEntry, CurriculumStage, RlCurriculum, SamplingMode, SourcePin

from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import runnable_sources


def _recipe_identity(pipeline: object) -> str:
    """The ``module.qualname`` identity of a source's curation pipeline callable."""
    qualname = getattr(pipeline, "__qualname__", None)
    if qualname is not None:
        return f"{getattr(pipeline, '__module__', '')}.{qualname}"
    kind = type(pipeline)
    return f"{kind.__module__}.{kind.__qualname__}"


def curriculum_entry(label: str, source: RlDataSource, weight: float | None) -> CurriculumEntry:
    """One curriculum entry over a catalog source: its sampling label, pin, recipe identity, and weight.

    ``label`` is the sampling key the trainer resolves rows by (a ``teacher_route``
    value); it usually equals the source's catalog name but need not.
    """
    dataset = source.dataset
    if dataset is None:
        raise ValueError(f"source {label} has no pinned dataset; curricula record pinned sources only")
    if source.pipeline is None:
        raise ValueError(f"source {label} has no curation pipeline; curricula record runnable sources only")
    return CurriculumEntry(
        label=label,
        source=source.name,
        version=source.version,
        pin=SourcePin(name=dataset.name, revision=dataset.revision, url=dataset.url),
        recipe=_recipe_identity(source.pipeline),
        weight=weight,
    )


def _resolved(name: str, catalog: Mapping[str, RlDataSource]) -> RlDataSource:
    source = catalog.get(name)
    if source is None:
        raise ValueError(f"unknown RL source {name!r}; the catalog knows: {sorted(catalog)}")
    return source


def weighted_curriculum(
    name: str,
    version: str,
    *,
    weights: Mapping[str, float],
    seed: int = 0,
    catalog: Mapping[str, RlDataSource] | None = None,
) -> RlCurriculum:
    """Build a weight-random curriculum over named catalog sources.

    ``weights`` maps a source's catalog name to its mixture weight. Every named source must exist
    in the catalog (default: :func:`runnable_sources`) and carry a pinned dataset; the source's
    current version, pin, and pipeline identity are recorded at build time.
    """
    sources = runnable_sources() if catalog is None else catalog
    entries = tuple(curriculum_entry(label, _resolved(label, sources), weight) for label, weight in weights.items())
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.WEIGHTED,
        entries=entries,
        stages=(),
        seed=seed,
    )


def ordered_curriculum(
    name: str,
    version: str,
    *,
    stages: Sequence[str],
    quotas: Mapping[str, int | None] | None = None,
    seed: int = 0,
    catalog: Mapping[str, RlDataSource] | None = None,
) -> RlCurriculum:
    """Build an ordered curriculum over named catalog sources.

    ``stages`` lists source names in the order the trainer consumes them; ``quotas`` bounds each
    stage visit in prompts (an absent quota draws the source pool's row count). Every
    named source must exist in the catalog (default: :func:`runnable_sources`) and carry a pinned
    dataset; the source's current version, pin, and pipeline identity are recorded at build time.
    """
    sources = runnable_sources() if catalog is None else catalog
    if not stages:
        raise ValueError(f"ordered curriculum {name}@{version} needs a non-empty stage order")
    quota_map = quotas or {}
    stray = sorted(set(quota_map) - set(stages))
    if stray:
        raise ValueError(f"ordered curriculum {name}@{version} sets quotas for sources no stage samples: {stray}")
    entries = tuple(curriculum_entry(label, _resolved(label, sources), None) for label in dict.fromkeys(stages))
    declared = tuple(CurriculumStage(label=label, quota=quota_map.get(label)) for label in stages)
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.ORDERED,
        entries=entries,
        stages=declared,
        seed=seed,
    )
