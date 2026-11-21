# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pool binding for RL curricula."""

import pytest
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.rl.curriculum import CurriculumStage

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.curriculum import ordered_curriculum, weighted_curriculum


def _pool(name: str) -> ArtifactStep[CampaignArtifact]:
    return ArtifactStep.adopt(f"pools/{name}", "2026.11.18", f"s3://test/pools/{name}", kind=CampaignArtifact)


def test_weighted_curriculum_pins_the_pools_the_run_trains_on():
    numbers, letters = _pool("numbers"), _pool("letters")
    curriculum = weighted_curriculum(
        "users/tester/curricula/numbers-mix",
        "2026.11.18",
        train={numbers: 0.6, letters: 0.4},
        seed=17,
    )
    assert curriculum.fingerprint
    assert {entry.label for entry in curriculum.entries} == {"pools/numbers", "pools/letters"}
    identities = {numbers.name: artifact_identity(numbers), letters.name: artifact_identity(letters)}
    for entry in curriculum.entries:
        assert entry.identity == identities[entry.label]
    assert curriculum.identity.startswith("users/tester/curricula/numbers-mix@2026.11.18:")
    assert curriculum.weights() == {"pools/letters": 0.4, "pools/numbers": 0.6}


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


def test_curriculum_rejects_two_pools_sharing_a_label():
    with pytest.raises(ValueError, match="two pools"):
        weighted_curriculum(
            "users/tester/curricula/broken",
            "2026.11.18",
            train={_pool("numbers"): 0.5, _pool("numbers"): 0.5},
            seed=1,
        )


def test_ordered_curriculum_rejects_a_stage_without_a_pool():
    with pytest.raises(ValueError, match="has no pool"):
        ordered_curriculum(
            "users/tester/curricula/unstaged",
            "2026.11.18",
            stages=(CurriculumStage("ghost"),),
            pools={"numbers": _pool("numbers")},
            seed=1,
        )


def test_ordered_curriculum_rejects_pools_no_stage_visits():
    with pytest.raises(ValueError, match="no stage visits"):
        ordered_curriculum(
            "users/tester/curricula/stray-pool",
            "2026.11.18",
            stages=(CurriculumStage("numbers"),),
            pools={"numbers": _pool("numbers"), "letters": _pool("letters")},
            seed=1,
        )


def test_curriculum_seed_is_explicit():
    with pytest.raises(TypeError):
        weighted_curriculum("users/tester/curricula/no-seed", "2026.11.18", train={_pool("numbers"): 1.0})  # type: ignore[call-arg]
