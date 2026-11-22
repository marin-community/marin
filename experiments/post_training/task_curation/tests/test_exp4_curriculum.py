# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source pins recorded by the Exp4 curriculum."""

import math

from marin.rl.curriculum import PoolRow

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE_URL,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE_URL,
    REASONING_GYM_LABEL,
)
from experiments.post_training.task_curation.exp4_curriculum import (
    AGENTIC_TRAIN_SOURCE,
    NEMOTRON_IF_LABEL,
    exp4_rl_curriculum,
    exp4_validation_split,
)
from experiments.post_training.task_curation.sources import all_sources

_HOLDOUT_POOL_ROWS = {
    label: tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(count))
    for label, count in {
        AGENTIC_TRAIN_SOURCE: 80,
        NEMOTRON_IF_LABEL: 40,
        "open-thoughts/TaskTrove": 30,
        REASONING_GYM_LABEL: 30,
        "nvidia/OpenScience": 30,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": 20,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 20,
        "PRIME-RL/Eurus-2-RL-Data": 20,
    }.items()
}


def _curriculum_with_holdout():
    return exp4_rl_curriculum(exp4_validation_split(_HOLDOUT_POOL_ROWS).validation)


def test_the_entries_pin_the_consumed_releases_and_the_new_source():
    curriculum = _curriculum_with_holdout()
    catalog = all_sources()
    for entry in curriculum.entries:
        if entry.label == NEMOTRON_IF_LABEL:
            source = catalog["MarinSkyRL:nemotron_if"]
            assert entry.source == NEMOTRON_IF_LABEL
            assert entry.pin.name == "nvidia/Llama-Nemotron-Post-Training-Dataset"
            assert entry.pin.revision == source.dataset.revision
        else:
            release_pin = AGENTIC_RELEASE_URL if entry.source == "long-loop-rl-agentic-train" else NONAGENTIC_RELEASE_URL
            release_revision = (
                AGENTIC_MANIFEST_SHA256 if entry.source == "long-loop-rl-agentic-train" else NONAGENTIC_MANIFEST_SHA256
            )
            assert entry.pin.url == release_pin
            assert entry.pin.revision == release_revision


def largest_remainder_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    """The vendored ``weighted_quotas`` apportionment, replicated for this test.

    The marin test environment does not import the vendored trainer (torch), so
    this computes the same largest-remainder apportionment the sampler performs:
    per-source floors of ``total * weight`` with the remainder seats going to the
    largest fractional parts, ties rotating by window index.
    """

    def quotas(batch_index: int) -> dict[str, int]:
        raw = {name: total * weight for name, weight in weights.items()}
        assigned = {name: math.floor(value) for name, value in raw.items()}
        priority = sorted(
            weights,
            key=lambda name: (-(raw[name] - assigned[name]), (list(weights).index(name) - batch_index) % len(weights)),
        )
        for name in priority[: total - sum(assigned.values())]:
            assigned[name] += 1
        return assigned

    window_quotas = [quotas(batch_index) for batch_index in range(len(weights))]
    assert all(window == window_quotas[0] for window in window_quotas), "the apportionment must be window-stable"
    return window_quotas[0]


def test_the_sampler_apportions_every_window_to_22_agentic_groups():
    quotas = largest_remainder_quotas(64, _curriculum_with_holdout().weights())
    assert quotas == {
        AGENTIC_TRAIN_SOURCE: 22,
        NEMOTRON_IF_LABEL: 9,
        "open-thoughts/TaskTrove": 8,
        REASONING_GYM_LABEL: 7,
        "nvidia/OpenScience": 7,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": 4,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 4,
        "PRIME-RL/Eurus-2-RL-Data": 3,
    }
    assert sum(quotas.values()) == 64
    assert quotas[AGENTIC_TRAIN_SOURCE] == 22
    assert 64 - quotas[AGENTIC_TRAIN_SOURCE] == 42
