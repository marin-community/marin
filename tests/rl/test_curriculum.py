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
    draw_validation_split,
)


def _entry(label: str = "gsm8k", identity: str = "pools/gsm8k@1:aaaa", weight: float | None = 0.7) -> CurriculumEntry:
    return CurriculumEntry(label=label, identity=identity, weight=weight)


def _pool_rows(label: str, count: int) -> tuple[PoolRow, ...]:
    return tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(count))


def _rows_by_label(counts: dict[str, int]) -> dict[str, tuple[PoolRow, ...]]:
    return {label: _pool_rows(label, count) for label, count in counts.items()}


def _validation(seed: int, mixture: dict[str, float], counts: dict[str, int], size: int = 200):
    return draw_validation_split(mixture, _rows_by_label(counts), size=size, seed=seed)


def _weighted(
    entries: tuple[CurriculumEntry, ...] | None = None,
    seed: int = 17,
    validation_size: int = 200,
    counts: dict[str, int] | None = None,
    mixture: dict[str, float] | None = None,
) -> RlCurriculum:
    """A weighted curriculum over the gsm8k/svamp pools with a drawn holdout."""
    validation = draw_validation_split(
        mixture or {"gsm8k": 0.7, "svamp": 0.3},
        _rows_by_label(counts or {"gsm8k": 1000, "svamp": 1000}),
        size=validation_size,
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
    validation_size: int = 200,
    counts: dict[str, int] | None = None,
) -> RlCurriculum:
    if counts is None:
        counts = {"gsm8k": 4096, "svamp": 1000}
    pool_rows = _rows_by_label(counts)
    mixture = {"gsm8k": float(min(4096, counts["gsm8k"])), "svamp": float(counts["svamp"])}
    validation = draw_validation_split(mixture, pool_rows, size=validation_size, seed=seed)
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
    """Pool identities, seed, and stage order affect the fingerprint; curriculum version does not."""
    base = _ordered()
    bumped_pool = _ordered(entries=(_entry(identity="pools/gsm8k@2:bbbb", weight=None), _entry("svamp", weight=None)))
    assert base.fingerprint != bumped_pool.fingerprint
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


def test_curriculum_weights_are_canonical_whatever_the_declaration_order():
    """The trainer breaks ties and seeds shuffles from the mapping's order, so it is sorted."""
    curriculum = _weighted(entries=(_entry("svamp", weight=0.3), _entry(weight=0.7)))
    assert list(curriculum.weights()) == ["gsm8k", "svamp"]
    assert curriculum.fingerprint == _weighted().fingerprint


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
        lambda: _ordered(stages=(CurriculumStage("gsm8k", 0),)),  # non-positive quota
        lambda: _ordered(entries=(_entry(identity="", weight=None), _entry("svamp", weight=None))),  # unpinned pool
    ],
)
def test_curriculum_rejects_invalid_ordered_construction(change):
    with pytest.raises(ValueError):
        change()


@pytest.mark.parametrize(
    "entries",
    [
        (_entry(weight=None), _entry("svamp", weight=None)),  # weighted mode without weights
        (_entry(weight=0.0), _entry("svamp", weight=0.3)),  # non-positive weight
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
    repeated_draw = draw_validation_split(
        {"gsm8k": 0.7, "svamp": 0.3}, _rows_by_label({"gsm8k": 1000, "svamp": 1000}), size=200, seed=17
    )
    assert repeated_draw.validation == first.validation


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
    assert len(split.training["gsm8k"]) == counts["gsm8k"] - len(
        [r for r in split.validation.rows if r.label == "gsm8k"]
    )


def test_validation_split_follows_the_mixture():
    """Per-label holdout counts are the mixture's largest-remainder apportionment of the size."""
    split = _validation(17, {"gsm8k": 0.7, "svamp": 0.3}, {"gsm8k": 1000, "svamp": 1000})
    counts = Counter(row.label for row in split.validation.rows)
    assert counts == {"gsm8k": 140, "svamp": 60}
    thirds = _validation(17, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, {"a": 500, "b": 500, "c": 500})
    assert Counter(row.label for row in thirds.validation.rows) == {"a": 67, "b": 67, "c": 66}
    assert len(thirds.validation.rows) == 200


def test_changing_the_curriculum_changes_the_split():
    """A different seed, mixture, or pool draws a different holdout and a different identity."""
    base = _weighted()
    assert base.validation.rows != _weighted(seed=18).validation.rows
    assert base.fingerprint != _weighted(seed=18).fingerprint
    reweighted = _weighted(entries=(_entry(), _entry("svamp", weight=0.3)), counts={"gsm8k": 900, "svamp": 1000})
    assert base.validation.rows != reweighted.validation.rows
    assert base.fingerprint != reweighted.fingerprint
    reweighted_mix = _weighted(mixture={"gsm8k": 0.5, "svamp": 0.5})
    assert Counter(row.label for row in reweighted_mix.validation.rows) == {"gsm8k": 100, "svamp": 100}
    assert base.fingerprint != reweighted_mix.fingerprint
    rebadged = replace(base, version="2026.11.19")
    assert base.validation == rebadged.validation
    assert base.fingerprint == rebadged.fingerprint


def test_curriculum_fingerprint_covers_the_validation_split():
    """Two curricula identical except for one held-out row are different objects."""
    base = _weighted()
    first = base.validation.rows[0]
    altered_rows = (replace(first, sha256="f" * 64), *base.validation.rows[1:])
    altered = RlCurriculum(
        name=base.name,
        version=base.version,
        mode=base.mode,
        entries=base.entries,
        stages=base.stages,
        seed=base.seed,
        validation=replace(base.validation, rows=altered_rows),
    )
    assert base.validation != altered.validation
    assert base.fingerprint != altered.fingerprint


def test_curriculum_rejects_a_validation_seed_mismatch():
    split = _validation(17, {"gsm8k": 0.7, "svamp": 0.3}, {"gsm8k": 1000, "svamp": 1000})
    validation = replace(split.validation, seed=18)
    with pytest.raises(ValueError, match="validation seed"):
        RlCurriculum(
            name="post-training/math-mix",
            version="2026.11.18",
            mode=SamplingMode.WEIGHTED,
            entries=(_entry(), _entry("svamp", weight=0.3)),
            stages=(),
            seed=17,
            validation=validation,
        )


def test_curriculum_rejects_validation_rows_from_undeclared_sources():
    split = _validation(17, {"gsm8k": 0.7, "svamp": 0.3, "web": 1.0}, {"gsm8k": 100, "svamp": 100, "web": 100})
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
        _validation(17, {"gsm8k": 1.0}, {"gsm8k": 10}, size=200)
    with pytest.raises(ValueError, match="names pools with no rows"):
        _validation(17, {"gsm8k": 0.5, "svamp": 0.5}, {"gsm8k": 100})
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
    curriculum = _ordered(validation_size=20, counts={"gsm8k": 4096, "svamp": 1000})
    counts = Counter(row.label for row in curriculum.validation.rows)
    # consumption: gsm8k quota 4096 (pool 4096), svamp None -> 1000; shares 4096/5096 and 1000/5096
    assert counts == {"gsm8k": 16, "svamp": 4}
