# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind catalog sources into versioned RL curricula.

The catalog (:func:`~experiments.post_training.task_curation.sources.runnable_sources`) owns which
sources exist and what pins them; this module owns the policy binding: which of them a curriculum
samples, at what weights or in what stage order. The experiment that launches the run declares
weights, as with SFT mixture weights. Fixing a curriculum also draws its validation holdout: the
builders require an explicit ``validation_size`` and each pool's row identities, and the drawn
split is recorded in the curriculum's metadata. The pool builder that materializes the pools
calls :func:`marin.rl.curriculum.draw_validation_split` itself — the same pure call, so record
and data derive one split — and writes the training pools from its ``ValidationSplit.training``
remainder and the validation data from its held-out rows, so the holdout is a random subset of
the training set and disjoint from it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from marin.rl.curriculum import (
    CurriculumEntry,
    CurriculumStage,
    CurriculumValidation,
    PoolRow,
    RlCurriculum,
    SamplingMode,
    SourcePin,
    draw_validation_split,
)

from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import runnable_sources


@dataclass(frozen=True)
class ServedCatalogSource:
    """One admitted pipeline binding read from an immutable served catalog load."""

    name: str
    version: str
    pin: SourcePin
    recipe: str


type CurriculumCatalog = Mapping[str, RlDataSource | ServedCatalogSource]


def served_catalog_sources(
    document: dict, *, admitted_statuses: tuple[str, ...] = ("Available",)
) -> dict[str, ServedCatalogSource]:
    """Bind admitted training sources, refusing inconsistent release/pipeline pins."""
    load_id = document["catalog_load"]["load_id"]
    bound = {}
    for label, source in document["sources"].items():
        if source["status"] not in admitted_statuses or source["intended_use"] != "train":
            continue
        release, pipeline = source["current_release"], source["pipeline"]
        if (release["repo"], release["revision"]) != (pipeline["source"], pipeline["revision"]):
            raise ValueError(f"Served catalog release and pipeline disagree: {label}")
        if (source["dataset_id"], source["dataset_revision"]) != (release["repo"], release["revision"]):
            raise ValueError(f"Served catalog dataset and release disagree: {label}")
        bound[label] = ServedCatalogSource(
            pipeline["name"],
            pipeline["version"],
            SourcePin(release["repo"], release["revision"], release["url"]),
            f"atlas:{load_id}/{pipeline['name']}@{pipeline['version']}",
        )
    return bound


def _recipe_identity(pipeline: object) -> str:
    """The ``module.qualname`` identity of a source's curation pipeline callable."""
    qualname = getattr(pipeline, "__qualname__", None)
    if qualname is not None:
        return f"{getattr(pipeline, '__module__', '')}.{qualname}"
    kind = type(pipeline)
    return f"{kind.__module__}.{kind.__qualname__}"


def _entry(label: str, source: RlDataSource | ServedCatalogSource, weight: float | None) -> CurriculumEntry:
    """One curriculum entry over a catalog source: its sampling label, pin, recipe identity, and weight.

    ``label`` is the sampling key the trainer resolves rows by: the pool row's
    ``teacher_route`` value. A curriculum samples only a pool whose rows carry its
    labels, so a label may name a route or a stamped source population rather than
    the catalog name; ``source`` is always the catalog invocation name.
    """
    if isinstance(source, ServedCatalogSource):
        pin, recipe = source.pin, source.recipe
    else:
        dataset = source.dataset
        if dataset is None:
            raise ValueError(f"source {label} has no pinned dataset; curricula record pinned sources only")
        if source.pipeline is None:
            raise ValueError(f"source {label} has no curation pipeline; curricula record runnable sources only")
        pin = SourcePin(name=dataset.name, revision=dataset.revision, url=dataset.url)
        recipe = _recipe_identity(source.pipeline)
    return CurriculumEntry(
        label=label,
        source=source.name,
        version=source.version,
        pin=pin,
        recipe=recipe,
        weight=weight,
    )


def _resolved(name: str, catalog: CurriculumCatalog) -> RlDataSource | ServedCatalogSource:
    source = catalog.get(name)
    if source is None:
        raise ValueError(f"unknown RL source {name!r}; the catalog knows: {sorted(catalog)}")
    return source


def curriculum_entries(
    name: str,
    version: str,
    *,
    weights: Mapping[str, float],
    catalog: CurriculumCatalog | None = None,
) -> tuple[CurriculumEntry, ...]:
    """The catalog-bound entries a weight-random curriculum over these sources records."""
    sources = runnable_sources() if catalog is None else catalog
    return tuple(_entry(label, _resolved(label, sources), weight) for label, weight in weights.items())


def weighted_curriculum(
    name: str,
    version: str,
    *,
    weights: Mapping[str, float],
    seed: int = 0,
    validation_size: int,
    pool_rows: Mapping[str, Sequence[PoolRow]],
    catalog: CurriculumCatalog | None = None,
) -> RlCurriculum:
    """Build a weight-random curriculum over named catalog sources.

    ``weights`` maps a source's catalog name to its mixture weight. Every named source must exist
    in the catalog (default: :func:`runnable_sources`) and carry a pinned dataset; the source's
    current version, pin, and pipeline identity are recorded at build time. Fixing the curriculum
    draws its validation holdout here: ``validation_size`` is required (there is no default), and
    ``pool_rows`` carries each label's row identities keyed by sampling label. The pools the run
    trains on must exclude the drawn rows: the pool builder materializes them from
    :func:`marin.rl.curriculum.draw_validation_split`'s ``ValidationSplit.training`` — the same
    deterministic call this builder makes — so record and data cannot disagree.
    """
    entries = curriculum_entries(name, version, weights=weights, catalog=catalog)
    split = draw_validation_split(dict(weights), pool_rows, size=validation_size, seed=seed)
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.WEIGHTED,
        entries=entries,
        stages=(),
        seed=seed,
        validation=split.validation,
    )


def curriculum_with_validation(
    name: str,
    version: str,
    *,
    weights: Mapping[str, float],
    seed: int,
    validation: CurriculumValidation,
    catalog: CurriculumCatalog | None = None,
) -> RlCurriculum:
    """Build a weight-random curriculum over named catalog sources with a drawn holdout.

    For callers that already hold the split: a pool step that drew it and
    recorded it, or a launch reading the record back. The validation's seed must
    be the curriculum's, and its rows must name declared sources — ``RlCurriculum``
    enforces both."""
    entries = curriculum_entries(name, version, weights=weights, catalog=catalog)
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.WEIGHTED,
        entries=entries,
        stages=(),
        seed=seed,
        validation=validation,
    )


def ordered_curriculum(
    name: str,
    version: str,
    *,
    stages: Sequence[str],
    quotas: Mapping[str, int | None] | None = None,
    seed: int = 0,
    validation_size: int,
    pool_rows: Mapping[str, Sequence[PoolRow]],
    catalog: CurriculumCatalog | None = None,
) -> RlCurriculum:
    """Build an ordered curriculum over named catalog sources.

    ``stages`` lists source names in the order the trainer consumes them; ``quotas`` bounds each
    stage visit in prompts (an absent quota draws the source pool's row count). Every
    named source must exist in the catalog (default: :func:`runnable_sources`) and carry a pinned
    dataset; the source's current version, pin, and pipeline identity are recorded at build time.
    Fixing the curriculum draws its validation holdout: the per-label allocation follows each
    pool's consumed prompts (its stage quotas, capped by the pool's row count), with
    ``validation_size`` required and ``pool_rows`` carrying each label's row identities.
    """
    sources = runnable_sources() if catalog is None else catalog
    if not stages:
        raise ValueError(f"ordered curriculum {name}@{version} needs a non-empty stage order")
    quota_map = quotas or {}
    stray = sorted(set(quota_map) - set(stages))
    if stray:
        raise ValueError(f"ordered curriculum {name}@{version} sets quotas for sources no stage samples: {stray}")
    entries = tuple(_entry(label, _resolved(label, sources), None) for label in dict.fromkeys(stages))
    declared = tuple(CurriculumStage(label=label, quota=quota_map.get(label)) for label in stages)
    consumption: dict[str, float] = {}
    for label in sorted(set(stages)):
        rows = pool_rows.get(label, ())
        draws = sum(
            (quota_map.get(stage) if quota_map.get(stage) is not None else len(rows))
            for stage in stages
            if stage == label
        )
        consumption[label] = float(min(draws, len(rows)))
    split = draw_validation_split(consumption, pool_rows, size=validation_size, seed=seed)
    return RlCurriculum(
        name=name,
        version=version,
        mode=SamplingMode.ORDERED,
        entries=entries,
        stages=declared,
        seed=seed,
        validation=split.validation,
    )
