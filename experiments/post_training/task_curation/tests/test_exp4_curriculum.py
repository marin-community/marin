# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp4 curriculum: the .55 mix with the agentic route cut to 35% and the
newly admitted Nemotron instruction-following source taking 15%."""

from dataclasses import replace

import pytest
from marin.rl.curriculum import SamplingMode

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE_URL,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE_URL,
    REASONING_GYM_SOURCE,
)
from experiments.post_training.task_curation.exp3_curriculum import weights as exp3_weights
from experiments.post_training.task_curation.exp4_curriculum import (
    AGENTIC_SHARE,
    AGENTIC_TRAIN_SOURCE,
    BATCH_GROUPS,
    CURRICULUM_NAME,
    CURRICULUM_VERSION,
    EXACT_ROUTE_GROUPS,
    FIXED_ROUTE_QUOTAS,
    NEMOTRON_IF_LABEL,
    NEMOTRON_IF_SHARE,
    SOURCE_ROUTES,
    TARGET_SHARES,
    exp4_rl_curriculum,
)
from experiments.post_training.task_curation.sources import all_sources

CARRY_OVER_SHARES = {
    "open-thoughts/TaskTrove": 0.120165,
    REASONING_GYM_SOURCE: 0.104,
    "nvidia/OpenScience": 0.10395,
    "nvidia/Nemotron-RL-Ultra-Training-Blends": 0.06213,
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 0.05817,
    "PRIME-RL/Eurus-2-RL-Data": 0.051585,
}


def test_the_resolved_shares_equal_the_targets():
    resolved = exp4_rl_curriculum().weights()
    assert resolved == dict(TARGET_SHARES)
    assert sum(resolved.values()) == pytest.approx(1.0)
    assert resolved[AGENTIC_TRAIN_SOURCE] == AGENTIC_SHARE == 0.35
    assert resolved[NEMOTRON_IF_LABEL] == NEMOTRON_IF_SHARE == 0.15


def test_the_carry_over_sources_keep_their_55_shares_exactly():
    """The six .55 nonagentic sources keep the shares .55 consumed, not approximations."""
    exp3 = exp3_weights()
    for label, share in CARRY_OVER_SHARES.items():
        assert exp4_rl_curriculum().weights()[label] == share == exp3[label]


def test_the_route_masses_realize_the_targets_on_the_55_batch():
    """35% agentic / 65% nonagentic: 22.4 and 41.6 groups per 64-group update."""
    route_mass = {}
    for label, weight in exp4_rl_curriculum().weights().items():
        route_mass[SOURCE_ROUTES[label]] = route_mass.get(SOURCE_ROUTES[label], 0.0) + weight
    assert route_mass["agentic"] == pytest.approx(0.35)
    assert route_mass["nonagentic"] == pytest.approx(0.65)
    assert EXACT_ROUTE_GROUPS == {"agentic": 22.4, "nonagentic": 41.6}
    # Fixed integer buffer quotas cannot hit 22.4; the declared rounding is 22/42,
    # and its agentic shortfall stays under one group per window.
    assert FIXED_ROUTE_QUOTAS == {"agentic": 22, "nonagentic": 42}
    assert 0.0 < AGENTIC_SHARE - FIXED_ROUTE_QUOTAS["agentic"] / BATCH_GROUPS < 1 / BATCH_GROUPS


def test_the_entries_pin_the_consumed_releases_and_the_new_source():
    curriculum = exp4_rl_curriculum()
    assert curriculum.name == CURRICULUM_NAME
    assert curriculum.version == CURRICULUM_VERSION
    assert curriculum.mode is SamplingMode.WEIGHTED
    assert {entry.label for entry in curriculum.entries} == set(TARGET_SHARES)
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


def test_the_new_source_is_a_runnable_catalog_source_with_all_its_tasks():
    source = all_sources()["MarinSkyRL:nemotron_if"]
    assert source.name == NEMOTRON_IF_LABEL
    assert source.pipeline is not None
    assert "agentic" not in source.info.tags


def test_the_seed_policy_is_unchanged_from_55():
    """The .55 policy seeds the sampler with the trainer seed 20261020; Exp4 inherits it."""
    curriculum = exp4_rl_curriculum()
    assert curriculum.seed == 20261020
    assert curriculum.sampling_config()["seed"] == 20261020


def test_the_fingerprint_follows_only_the_declared_mix():
    curriculum = exp4_rl_curriculum()
    assert exp4_rl_curriculum().fingerprint == curriculum.fingerprint
    assert replace(curriculum, seed=20261021).fingerprint != curriculum.fingerprint
    rebalanced = replace(curriculum, entries=tuple(replace(e, weight=e.weight / 2) for e in curriculum.entries))
    assert rebalanced.fingerprint != curriculum.fingerprint
    assert replace(curriculum, version="2026.11.19").fingerprint == curriculum.fingerprint
