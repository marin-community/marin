# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind catalog sources into versioned RL curricula.

The catalog (:func:`~experiments.post_training.task_curation.sources.runnable_sources`) owns which
sources exist and what pins them; this module owns the policy binding: which of them a curriculum
samples, at what weights or in what stage order. Mixes are policy, not catalog — declare weights in
the experiment that launches the run, exactly as SFT mixture weights stay out of the dataset
catalog.
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


def _entry(label: str, source: RlDataSource, weight: float | None) -> CurriculumEntry:
    dataset = source.dataset
    if dataset is None:
        raise ValueError(f"source {label} has no pinned dataset; curricula record pinned sources only")
    if source.pipeline is None:
        raise ValueError(f"source {label} has no curation pipeline; curricula record runnable sources only")
    return CurriculumEntry(
        label=label,
        source=label,
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


def rl_curriculum(
    name: str,
    version: str,
    *,
    mode: SamplingMode,
    weights: Mapping[str, float] | None = None,
    stages: Sequence[str] = (),
    quotas: Mapping[str, int | None] | None = None,
    seed: int = 0,
    catalog: Mapping[str, RlDataSource] | None = None,
) -> RlCurriculum:
    """Build a curriculum over named catalog sources.

    ``weights`` maps a source's catalog name to its weighted-mode mixture weight. ``stages`` lists
    the ordered-mode sequence of source names, and ``quotas`` bounds each stage visit in prompts (an
    absent quota runs one full pass over that source's rows). Every named source must exist in the
    catalog (default: :func:`runnable_sources`) and carry a pinned dataset; the source's current
    version, pin, and pipeline identity are recorded at build time.
    """
    sources = runnable_sources() if catalog is None else catalog
    if mode is SamplingMode.WEIGHTED:
        if weights is None or not weights:
            raise ValueError(f"weighted curriculum {name}@{version} needs weights")
        if stages or quotas:
            raise ValueError(f"weighted curriculum {name}@{version} takes weights, not stages or quotas")
        entries = tuple(_entry(label, _resolved(label, sources), weight) for label, weight in weights.items())
        return RlCurriculum(
            name=name, version=version, mode=mode, entries=entries, stages=(), seed=seed, fingerprint=""
        )
    if not stages:
        raise ValueError(f"ordered curriculum {name}@{version} needs a non-empty stage order")
    if weights is not None:
        raise ValueError(f"ordered curriculum {name}@{version} takes stages, not weights")
    quota_map = quotas or {}
    stray = sorted(set(quota_map) - set(stages))
    if stray:
        raise ValueError(f"ordered curriculum {name}@{version} sets quotas for sources no stage samples: {stray}")
    entries = tuple(_entry(label, _resolved(label, sources), None) for label in dict.fromkeys(stages))
    declared = tuple(CurriculumStage(label=label, quota=quota_map.get(label)) for label in stages)
    return RlCurriculum(
        name=name, version=version, mode=mode, entries=entries, stages=declared, seed=seed, fingerprint=""
    )
