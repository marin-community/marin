# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the versioned RL curriculum type."""

from dataclasses import replace

import pytest
from marin.rl.curriculum import CurriculumEntry, CurriculumStage, RlCurriculum, SamplingMode


def _entry(label: str = "gsm8k", identity: str = "pools/gsm8k@1:aaaa", weight: float | None = 0.7) -> CurriculumEntry:
    return CurriculumEntry(label=label, identity=identity, weight=weight)


def _weighted(entries: tuple[CurriculumEntry, ...] | None = None, seed: int = 17) -> RlCurriculum:
    return RlCurriculum(
        name="post-training/math-mix",
        version="2026.11.18",
        mode=SamplingMode.WEIGHTED,
        entries=entries if entries is not None else (_entry(), _entry("svamp", weight=0.3)),
        stages=(),
        seed=seed,
    )


def _ordered(
    stages: tuple[CurriculumStage, ...] = (CurriculumStage("gsm8k", 4096), CurriculumStage("svamp")),
    entries: tuple[CurriculumEntry, ...] | None = None,
    seed: int = 17,
) -> RlCurriculum:
    return RlCurriculum(
        name="post-training/math-ladder",
        version="2026.11.18",
        mode=SamplingMode.ORDERED,
        entries=entries if entries is not None else (_entry(weight=None), _entry("svamp", weight=None)),
        stages=stages,
        seed=seed,
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
