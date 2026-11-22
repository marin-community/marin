# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pool binding for RL curricula."""

from collections import Counter

import pytest
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.rl.curriculum import CurriculumStage, PoolRow

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.curriculum import ordered_curriculum, weighted_curriculum


def _pool(name: str) -> ArtifactStep[CampaignArtifact]:
    return ArtifactStep.adopt(f"pools/{name}", "2026.11.18", f"s3://test/pools/{name}", kind=CampaignArtifact)


def _pool_rows(label: str, count: int) -> tuple[PoolRow, ...]:
    return tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(count))


def test_weighted_curriculum_pins_the_pools_the_run_trains_on():
    numbers, letters = _pool("numbers"), _pool("letters")
    curriculum = weighted_curriculum(
        "users/tester/curricula/numbers-mix",
        "2026.11.18",
        train={numbers: 0.6, letters: 0.4},
        seed=17,
        validation_size=200,
        pool_rows={
            "pools/numbers": _pool_rows("pools/numbers", 1000),
            "pools/letters": _pool_rows("pools/letters", 1000),
        },
    )
    assert curriculum.fingerprint
    assert {entry.label for entry in curriculum.entries} == {"pools/numbers", "pools/letters"}
    identities = {numbers.name: artifact_identity(numbers), letters.name: artifact_identity(letters)}
    for entry in curriculum.entries:
        assert entry.identity == identities[entry.label]
    assert curriculum.identity.startswith("users/tester/curricula/numbers-mix@2026.11.18:")
    assert curriculum.weights() == {"pools/letters": 0.4, "pools/numbers": 0.6}


def test_weighted_curriculum_draws_the_holdout_from_the_declared_mixture():
    numbers, letters = _pool("numbers"), _pool("letters")
    curriculum = weighted_curriculum(
        "users/tester/curricula/numbers-mix",
        "2026.11.18",
        train={numbers: 0.6, letters: 0.4},
        seed=17,
        validation_size=200,
        pool_rows={
            "pools/numbers": _pool_rows("pools/numbers", 1000),
            "pools/letters": _pool_rows("pools/letters", 1000),
        },
    )
    counts = Counter(row.label for row in curriculum.validation.rows)
    assert counts == {"pools/numbers": 120, "pools/letters": 80}
    assert curriculum.validation.seed == 17
    assert curriculum.validation.size == 200
    assert (
        curriculum.fingerprint
        == weighted_curriculum(
            "users/tester/curricula/numbers-mix",
            "2026.11.18",
            train={numbers: 0.6, letters: 0.4},
            seed=17,
            validation_size=200,
            pool_rows={
                "pools/numbers": _pool_rows("pools/numbers", 1000),
                "pools/letters": _pool_rows("pools/letters", 1000),
            },
        ).fingerprint
    )


def test_ordered_curriculum_takes_stage_objects():
    curriculum = ordered_curriculum(
        "users/tester/curricula/numbers-ladder",
        "2026.11.18",
        stages=(
            CurriculumStage("pools/numbers", 8),
            CurriculumStage("pools/letters"),
            CurriculumStage("pools/numbers", 8),
        ),
        pools={"pools/numbers": _pool("numbers"), "pools/letters": _pool("letters")},
        seed=3,
        validation_size=10,
        pool_rows={
            "pools/numbers": _pool_rows("pools/numbers", 20),
            "pools/letters": _pool_rows("pools/letters", 20),
        },
    )
    assert {entry.label for entry in curriculum.entries} == {"pools/numbers", "pools/letters"}
    assert curriculum.sampling_config() == {
        "kind": "ordered",
        "stages": [
            {"source": "pools/numbers", "quota": 8},
            {"source": "pools/letters", "quota": None},
            {"source": "pools/numbers", "quota": 8},
        ],
        "seed": 3,
    }
    # consumed prompts: numbers 8+8=16, letters 20; allocation of 10 follows 16:36 and 20:36
    counts = Counter(row.label for row in curriculum.validation.rows)
    assert counts == {"pools/numbers": 4, "pools/letters": 6}


def test_curriculum_rejects_two_pools_sharing_a_label():
    with pytest.raises(ValueError, match="two pools"):
        weighted_curriculum(
            "users/tester/curricula/broken",
            "2026.11.18",
            train={_pool("numbers"): 0.5, _pool("numbers"): 0.5},
            seed=1,
            validation_size=10,
            pool_rows={"pools/numbers": _pool_rows("pools/numbers", 100)},
        )


def test_curriculum_rejects_pool_rows_that_do_not_match_the_mixture():
    with pytest.raises(ValueError, match="does not declare"):
        weighted_curriculum(
            "users/tester/curricula/stray-rows",
            "2026.11.18",
            train={_pool("numbers"): 1.0},
            seed=1,
            validation_size=10,
            pool_rows={
                "pools/numbers": _pool_rows("pools/numbers", 100),
                "pools/letters": _pool_rows("pools/letters", 100),
            },
        )


def test_ordered_curriculum_rejects_a_stage_without_a_pool():
    with pytest.raises(ValueError, match="has no pool"):
        ordered_curriculum(
            "users/tester/curricula/unstaged",
            "2026.11.18",
            stages=(CurriculumStage("ghost"),),
            pools={"numbers": _pool("numbers")},
            seed=1,
            validation_size=10,
            pool_rows={"numbers": _pool_rows("numbers", 100)},
        )


def test_ordered_curriculum_rejects_pools_no_stage_visits():
    with pytest.raises(ValueError, match="no stage visits"):
        ordered_curriculum(
            "users/tester/curricula/stray-pool",
            "2026.11.18",
            stages=(CurriculumStage("numbers"),),
            pools={"numbers": _pool("numbers"), "letters": _pool("letters")},
            seed=1,
            validation_size=10,
            pool_rows={"numbers": _pool_rows("numbers", 100), "letters": _pool_rows("letters", 100)},
        )


def test_curriculum_seed_is_explicit():
    with pytest.raises(TypeError):
        weighted_curriculum(  # type: ignore[call-arg]
            "users/tester/curricula/no-seed",
            "2026.11.18",
            train={_pool("numbers"): 1.0},
            validation_size=10,
            pool_rows={},
        )


def test_curriculum_validation_size_is_explicit():
    with pytest.raises(TypeError):
        weighted_curriculum(  # type: ignore[call-arg]
            "users/tester/curricula/no-size",
            "2026.11.18",
            train={_pool("numbers"): 1.0},
            seed=1,
            pool_rows={"pools/numbers": _pool_rows("pools/numbers", 100)},
        )
