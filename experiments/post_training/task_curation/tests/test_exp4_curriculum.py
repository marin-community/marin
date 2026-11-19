# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source pins recorded by the Exp4 curriculum."""

import math

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE_URL,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE_URL,
    REASONING_GYM_SOURCE,
)
from experiments.post_training.task_curation.exp4_curriculum import (
    AGENTIC_TRAIN_SOURCE,
    NEMOTRON_IF_LABEL,
    exp4_rl_curriculum,
)
from experiments.post_training.task_curation.sources import all_sources


def test_the_entries_pin_the_consumed_releases_and_the_new_source():
    curriculum = exp4_rl_curriculum()
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
    this computes the same largest-integer apportionment the sampler performs:
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
    """The realized window the PR body and module docstring state: 22/42 routes."""
    quotas = largest_remainder_quotas(64, exp4_rl_curriculum().weights())
    assert quotas == {
        AGENTIC_TRAIN_SOURCE: 22,
        NEMOTRON_IF_LABEL: 9,
        "open-thoughts/TaskTrove": 8,
        REASONING_GYM_SOURCE: 7,
        "nvidia/OpenScience": 7,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": 4,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 4,
        "PRIME-RL/Eurus-2-RL-Data": 3,
    }
    assert sum(quotas.values()) == 64
    assert quotas[AGENTIC_TRAIN_SOURCE] == 22
    assert 64 - quotas[AGENTIC_TRAIN_SOURCE] == 42
