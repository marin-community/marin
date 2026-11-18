# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The catalog binding for RL curricula."""

from pathlib import Path

import pytest

from experiments.post_training.task_curation.curriculum import rl_curriculum
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from marin.rl.curriculum import SamplingMode

_DUMMY_INPUT = Path("/unused/numbers.txt")


def _catalog() -> dict[str, RlDataSource]:
    return {
        "numbers": number_source(_DUMMY_INPUT, "numbers"),
        "letters": number_source(_DUMMY_INPUT, "letters"),
    }


def test_rl_curriculum_records_pinned_source_identities():
    curriculum = rl_curriculum(
        "users/tester/curricula/numbers-mix",
        "2026.11.18",
        mode=SamplingMode.WEIGHTED,
        weights={"numbers": 0.6, "letters": 0.4},
        seed=17,
        catalog=_catalog(),
    )
    assert curriculum.fingerprint
    assert {entry.label for entry in curriculum.entries} == {"numbers", "letters"}
    entry = curriculum.entries[0]
    assert entry.source in ("numbers", "letters")
    assert entry.version == "1"
    assert (entry.pin.name, entry.pin.revision, entry.pin.url) == (
        "local-numbers",
        "pinned",
        "https://example.org/numbers",
    )
    assert entry.recipe.endswith("numbers_pipeline.numbers_pipeline")
    assert curriculum.identity.startswith("users/tester/curricula/numbers-mix@2026.11.18:")


def test_rl_curriculum_ordered_stages_declare_the_sequence():
    curriculum = rl_curriculum(
        "users/tester/curricula/numbers-ladder",
        "2026.11.18",
        mode=SamplingMode.ORDERED,
        stages=("numbers", "letters", "numbers"),
        quotas={"numbers": 8},
        seed=3,
        catalog=_catalog(),
    )
    assert {entry.label for entry in curriculum.entries} == {"numbers", "letters"}
    assert curriculum.sampling_config() == {
        "kind": "ordered",
        "stages": [
            {"source": "numbers", "quota": 8},
            {"source": "letters", "quota": None},
            {"source": "numbers", "quota": 8},
        ],
        "seed": 3,
    }


def test_rl_curriculum_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown RL source 'ghost'"):
        rl_curriculum(
            "users/tester/curricula/broken",
            "2026.11.18",
            mode=SamplingMode.WEIGHTED,
            weights={"ghost": 1.0},
            catalog=_catalog(),
        )


def test_rl_curriculum_rejects_unpinned_source():
    catalog = {**_catalog(), "inventory": RlDataSource(info=SourceInfo(id="fixture:inventory", title="x", origin="f"))}
    with pytest.raises(ValueError, match="no pinned dataset"):
        rl_curriculum(
            "users/tester/curricula/unpinned",
            "2026.11.18",
            mode=SamplingMode.ORDERED,
            stages=("inventory",),
            catalog=catalog,
        )


def test_rl_curriculum_rejects_quotas_for_unstaged_sources():
    with pytest.raises(ValueError, match="no stage samples"):
        rl_curriculum(
            "users/tester/curricula/stray-quota",
            "2026.11.18",
            mode=SamplingMode.ORDERED,
            stages=("numbers",),
            quotas={"letters": 4},
            catalog=_catalog(),
        )
