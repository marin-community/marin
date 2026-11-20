# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the versioned RL curriculum type."""

from dataclasses import replace

import pytest
from marin.rl.curriculum import CurriculumEntry, CurriculumStage, RlCurriculum, SamplingMode, SourcePin

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
