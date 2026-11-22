# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the versioned RL curriculum type and its validation holdout."""

from collections import Counter
from dataclasses import replace

import pytest
from marin.rl.curriculum import (
    CurriculumEntry,
    CurriculumStage,
    PoolRow,
    RlCurriculum,
    SamplingMode,
    SourcePin,
    consumed_route_quotas,
    draw_validation_split,
)

RECIPE = "experiments.post_training.task_curation.pipeline.process_rows"


def _entry(
    label: str = "gsm8k", version: str = "1", weight: float | None = 0.7, pin: str = "e53f048"
) -> CurriculumEntry:
    return CurriculumEntry(
        label=label,
        source=label,
        version=version,
        pin=SourcePin(
            name="openai/gsm8k" if pin else "", revision=pin or None, url="https://huggingface.co/datasets/openai/gsm8k"
        ),
        recipe=RECIPE,
        weight=weight,
    )


def _pool_rows(label: str, count: int) -> tuple[PoolRow, ...]:
    return tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(count))


def _rows_by_label(counts: dict[str, int]) -> dict[str, tuple[PoolRow, ...]]:
    return {label: _pool_rows(label, count) for label, count in counts.items()}


def _weighted(entries: tuple[CurriculumEntry, ...] | None = None, seed: int = 17) -> RlCurriculum:
    validation = draw_validation_split(
        {"gsm8k": 0.7, "svamp": 0.3},
        _rows_by_label({"gsm8k": 1000, "svamp": 1000}),
        size=200,
        seed=seed,
    )
    return RlCurriculum(
        name="post-training/math-mix",
        version="2026.11.18",
        mode=SamplingMode.WEIGHTED,
        entries=entries if entries is not None else (_entry(), _entry("svamp", weight=0.3)),
        stages=(),
        seed=seed,
        validation=validation.validation,
    )


def _ordered(
    stages: tuple[CurriculumStage, ...] = (CurriculumStage("gsm8k", 4096), CurriculumStage("svamp")),
    entries: tuple[CurriculumEntry, ...] | None = None,
    seed: int = 17,
) -> RlCurriculum:
    counts = {"gsm8k": 4096, "svamp": 1000}
    mixture = {"gsm8k": float(min(4096, counts["gsm8k"])), "svamp": float(counts["svamp"])}
    validation = draw_validation_split(mixture, _rows_by_label(counts), size=200, seed=seed)
    return RlCurriculum(
        name="post-training/math-ladder",
        version="2026.11.18",
        mode=SamplingMode.ORDERED,
        entries=entries if entries is not None else (_entry(weight=None), _entry("svamp", weight=None)),
        stages=stages,
        seed=seed,
        validation=validation.validation,
    )


def test_curriculum_fingerprint_follows_the_declared_mix():
    """Source pins, seed, and stage order affect the fingerprint; curriculum version does not."""
    base = _ordered()
    bumped_source = _ordered(entries=(_entry(version="2", weight=None), _entry("svamp", weight=None)))
    assert base.fingerprint != bumped_source.fingerprint
    assert base.fingerprint != _ordered(seed=18).fingerprint
    assert base.fingerprint != _ordered(stages=(CurriculumStage("svamp"), CurriculumStage("gsm8k", 4096))).fingerprint
    rebadged = replace(base, version="2026.11.19")
    assert base.fingerprint == rebadged.fingerprint


def test_curriculum_weighted_mode_projects_the_domain_weights_contract():
    curriculum = _weighted()
    assert curriculum.weights() == {"gsm8k": 0.7, "svamp": 0.3}
    assert curriculum.sampling_config() == {
        "kind": "domain-weighted",
        "domain_weights": {"gsm8k": 0.7, "svamp": 0.3},
        "seed": 17,
    }


def test_curriculum_ordered_mode_projects_the_declared_stages():
    curriculum = _ordered()
    assert curriculum.sampling_config() == {
        "kind": "ordered",
        "stages": [{"source": "gsm8k", "quota": 4096}, {"source": "svamp", "quota": None}],
        "seed": 17,
    }


@pytest.mark.parametrize(
    "change",
    [
        # an ordered entry no stage samples
        lambda: _ordered(entries=(_entry("unused", weight=None), _entry(weight=None), _entry("svamp", weight=None))),
        lambda: _ordered(stages=(CurriculumStage("web", 16),)),  # a stage names no declared source
        lambda: _ordered(stages=()),  # empty ordering
        lambda: _ordered(entries=(_entry(pin="", weight=None), _entry("svamp", weight=None))),  # unpinned source
    ],
)
def test_curriculum_rejects_invalid_ordered_construction(change):
    with pytest.raises(ValueError):
        change()


@pytest.mark.parametrize(
    "entries",
    [
        (_entry(weight=None), _entry("svamp", weight=None)),  # weighted mode without weights
        (_entry(), _entry("svamp", weight=0.3), _entry("svamp", weight=0.3)),  # duplicate label
    ],
)
def test_curriculum_rejects_invalid_weighted_construction(entries):
    with pytest.raises(ValueError):
        _weighted(entries=entries)


def test_validation_split_is_deterministic_for_a_seed():
    """The same mixture, pools, size, and seed draw the same split — and the same identity."""
    first = _weighted()
    second = _weighted()
    assert first.validation == second.validation
    assert first.fingerprint == second.fingerprint


def test_validation_draw_is_independent_of_enumeration_order():
    """Rows arrive in whatever order the pool reader yields; the split may not depend on it."""
    pool_rows = _rows_by_label({"gsm8k": 1000, "svamp": 1000})
    straight = draw_validation_split({"gsm8k": 0.7, "svamp": 0.3}, pool_rows, size=200, seed=17)
    reversed_rows = {label: tuple(reversed(rows)) for label, rows in pool_rows.items()}
    reversed_draw = draw_validation_split({"gsm8k": 0.7, "svamp": 0.3}, reversed_rows, size=200, seed=17)
    assert straight.validation == reversed_draw.validation


def test_validation_split_partitions_the_pools():
    """Every pool row lands in exactly one side: the holdout and the training remainder are disjoint."""
    counts = {"gsm8k": 400, "svamp": 300}
    split = draw_validation_split({"gsm8k": 0.7, "svamp": 0.3}, _rows_by_label(counts), size=100, seed=17)
    held = {(row.label, row.id) for row in split.validation.rows}
    trained = {(label, row.id) for label, rows in split.training.items() for row in rows}
    assert held.isdisjoint(trained)
    assert held | trained == {(label, f"{label}-{index}") for label, count in counts.items() for index in range(count)}


def test_validation_split_follows_the_mixture():
    """Per-label holdout counts are the mixture's largest-remainder apportionment of the size."""
    split = draw_validation_split(
        {"gsm8k": 0.7, "svamp": 0.3}, _rows_by_label({"gsm8k": 1000, "svamp": 1000}), size=200, seed=17
    )
    assert Counter(row.label for row in split.validation.rows) == {"gsm8k": 140, "svamp": 60}
    thirds = draw_validation_split(
        {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, _rows_by_label({"a": 500, "b": 500, "c": 500}), size=200, seed=17
    )
    assert Counter(row.label for row in thirds.validation.rows) == {"a": 67, "b": 67, "c": 66}


def test_changing_the_curriculum_changes_the_split():
    """A different seed, mixture, or pool draws a different holdout and a different identity."""
    base = _weighted()
    assert base.validation.rows != _weighted(seed=18).validation.rows
    assert base.fingerprint != _weighted(seed=18).fingerprint
    rebadged = replace(base, version="2026.11.19")
    assert base.validation == rebadged.validation
    assert base.fingerprint == rebadged.fingerprint


def test_curriculum_fingerprint_covers_the_validation_split():
    """Two curricula identical except for one held-out row are different objects."""
    base = _weighted()
    first = base.validation.rows[0]
    altered = RlCurriculum(
        name=base.name,
        version=base.version,
        mode=base.mode,
        entries=base.entries,
        stages=base.stages,
        seed=base.seed,
        validation=replace(base.validation, rows=(replace(first, sha256="f" * 64), *base.validation.rows[1:])),
    )
    assert base.validation != altered.validation
    assert base.fingerprint != altered.fingerprint


def test_curriculum_rejects_a_validation_seed_mismatch():
    split = draw_validation_split(
        {"gsm8k": 0.7, "svamp": 0.3}, _rows_by_label({"gsm8k": 1000, "svamp": 1000}), size=200, seed=17
    )
    with pytest.raises(ValueError, match="validation seed"):
        RlCurriculum(
            name="post-training/math-mix",
            version="2026.11.18",
            mode=SamplingMode.WEIGHTED,
            entries=(_entry(), _entry("svamp", weight=0.3)),
            stages=(),
            seed=17,
            validation=replace(split.validation, seed=18),
        )


def test_curriculum_rejects_validation_rows_from_undeclared_sources():
    split = draw_validation_split(
        {"gsm8k": 0.7, "svamp": 0.3, "web": 1.0},
        _rows_by_label({"gsm8k": 100, "svamp": 100, "web": 100}),
        size=100,
        seed=17,
    )
    with pytest.raises(ValueError, match="undeclared"):
        RlCurriculum(
            name="post-training/math-mix",
            version="2026.11.18",
            mode=SamplingMode.WEIGHTED,
            entries=(_entry(), _entry("svamp", weight=0.3)),
            stages=(),
            seed=17,
            validation=split.validation,
        )


def test_validation_draw_rejects_impossible_pools():
    with pytest.raises(ValueError, match="allocates"):
        draw_validation_split({"gsm8k": 1.0}, _rows_by_label({"gsm8k": 10}), size=200, seed=1)
    with pytest.raises(ValueError, match="names pools with no rows"):
        draw_validation_split({"gsm8k": 0.5, "svamp": 0.5}, _rows_by_label({"gsm8k": 100}), size=10, seed=1)
    with pytest.raises(ValueError, match="does not declare"):
        draw_validation_split(
            {"gsm8k": 1.0}, {"gsm8k": _pool_rows("gsm8k", 10), "svamp": _pool_rows("svamp", 10)}, size=5, seed=1
        )
    duplicated = _pool_rows("gsm8k", 10) + _pool_rows("gsm8k", 1)
    with pytest.raises(ValueError, match="repeats a row identity"):
        draw_validation_split({"gsm8k": 1.0}, {"gsm8k": duplicated}, size=5, seed=1)


def test_validation_draw_rejects_incomplete_rows():
    rows = (PoolRow(id="gsm8k-0", sha256=""),)
    with pytest.raises(ValueError, match="incomplete"):
        draw_validation_split({"gsm8k": 1.0}, {"gsm8k": rows}, size=1, seed=1)


def test_ordered_curriculum_draws_from_consumed_prompts():
    """An ordered holdout allocates by each pool's consumed prompts, not its declared quotas alone."""
    curriculum = _ordered()
    counts_drawn = Counter(row.label for row in curriculum.validation.rows)
    # consumption: gsm8k quota 4096 (pool 4096), svamp None -> 1000; 200 of the 5096 consumed prompts
    assert counts_drawn == {"gsm8k": 161, "svamp": 39}
    assert curriculum.validation.size == 200


def _quota_curriculum(weights: dict[str, float]) -> RlCurriculum:
    """A small weighted curriculum over arbitrary labels, for the quota derivations."""
    entries = tuple(_entry(label, weight=weight) for label, weight in weights.items())
    validation = draw_validation_split(weights, _rows_by_label({label: 500 for label in weights}), size=50, seed=17)
    return RlCurriculum(
        name="post-training/quotas",
        version="2026.11.18",
        mode=SamplingMode.WEIGHTED,
        entries=entries,
        stages=(),
        seed=17,
        validation=validation.validation,
    )


def test_consumed_route_quotas_derive_from_the_curriculum_mixture():
    """The buffer's window quotas are the mixture's apportionment at the batch size."""
    curriculum = _quota_curriculum({"gsm8k": 0.7, "svamp": 0.3})
    assert consumed_route_quotas(curriculum, {}, 16) == {"gsm8k": 11, "svamp": 5}
    assert consumed_route_quotas(curriculum, {}, 32) == {"gsm8k": 22, "svamp": 10}
    assert consumed_route_quotas(curriculum, {}, 64) == {"gsm8k": 45, "svamp": 19}


def test_consumed_route_quotas_aggregate_over_source_routes():
    """Per-source weights aggregate over the run's route mapping before apportioning."""
    curriculum = _quota_curriculum({"src/gsm8k": 0.35, "src/svamp": 0.15, "src/web": 0.5})
    quotas = consumed_route_quotas(curriculum, {"src/gsm8k": "math", "src/svamp": "math", "src/web": "web"}, 64)
    assert quotas == {"math": 32, "web": 32}
    assert consumed_route_quotas(curriculum, {"src/gsm8k": "math", "src/svamp": "math", "src/web": "web"}, 32) == {
        "math": 16,
        "web": 16,
    }


def test_consumed_route_quotas_never_starve_an_admitted_route():
    """A route the apportionment would zero takes its seat from the largest allocation."""
    curriculum = _quota_curriculum({"big": 0.98, "tiny": 0.02})
    quotas = consumed_route_quotas(curriculum, {}, 10)
    assert quotas == {"big": 9, "tiny": 1}
    assert sum(quotas.values()) == 10


def test_consumed_route_quotas_reject_impossible_batches():
    with pytest.raises(ValueError, match="ordered curriculum"):
        consumed_route_quotas(_ordered(), {}, 32)
    with pytest.raises(ValueError, match="cannot admit"):
        consumed_route_quotas(_quota_curriculum({"gsm8k": 0.7, "svamp": 0.3}), {}, 1)
    with pytest.raises(ValueError, match="must be positive"):
        consumed_route_quotas(_quota_curriculum({"gsm8k": 0.7, "svamp": 0.3}), {}, 0)
