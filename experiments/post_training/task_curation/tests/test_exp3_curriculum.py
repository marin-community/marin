# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 curriculum records the route mix the .55 run enforced."""

import hashlib
import json
import math
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.rl.curriculum import SamplingMode

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE,
    AGENTIC_RELEASE_URL,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE,
    NONAGENTIC_RELEASE_URL,
    REASONING_GYM_SOURCE,
    AdoptedFile,
    AdoptedRelease,
    adopt_release,
    verify_release,
)
from experiments.post_training.task_curation.exp3_curriculum import (
    BATCH_GROUPS,
    CURRICULUM_NAME,
    CURRICULUM_VERSION,
    ROUTE_QUOTAS,
    SEED,
    WEIGHTS,
    exp3_rl_curriculum,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

NEMOTRON_BLEND_ROWS = 12426
OPEN_SCIENCE_ROWS = 20790
REASONING_GYM_ROWS = 20800
SNOWBALL_SMARTIE_ROWS = 11634
TASKTROVE_ROWS = 24033
EURUS_ROWS = 10317
AGENTIC_TRAIN_ROWS = 5608


def largest_remainder_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    """The vendored ``weighted_quotas`` apportionment, replicated for this test.

    The marin test environment does not import the vendored trainer (torch), so
    this computes the same largest-remainder apportionment the sampler performs:
    per-source floors of ``total * weight`` with the remainder seats going to the
    largest fractional parts, ties rotating by window index.
    """
    names = list(weights)

    def quotas(batch_index: int) -> dict[str, int]:
        raw = {name: total * weight for name, weight in weights.items()}
        assigned = {name: math.floor(value) for name, value in raw.items()}
        priority = sorted(
            names, key=lambda name: (-(raw[name] - assigned[name]), (names.index(name) - batch_index) % len(names))
        )
        for name in priority[: total - sum(assigned.values())]:
            assigned[name] += 1
        return assigned

    window_quotas = [quotas(batch_index) for batch_index in range(len(names))]
    assert all(window == window_quotas[0] for window in window_quotas), "the apportionment must be window-stable"
    return window_quotas[0]


def test_the_curriculum_records_the_enforced_route_mix():
    curriculum = exp3_rl_curriculum()
    assert curriculum.name == CURRICULUM_NAME
    assert curriculum.version == CURRICULUM_VERSION
    assert curriculum.mode is SamplingMode.WEIGHTED
    assert curriculum.seed == SEED == 20261020
    assert curriculum.weights() == dict(WEIGHTS) == {"agentic": 0.5, "nonagentic": 0.5}
    assert {entry.label: entry.source for entry in curriculum.entries} == {
        "agentic": "long-loop-rl-agentic-train",
        "nonagentic": "long-loop-rl-nonagentic-train",
    }
    agentic = next(entry for entry in curriculum.entries if entry.label == "agentic")
    nonagentic = next(entry for entry in curriculum.entries if entry.label == "nonagentic")
    assert agentic.pin.url == AGENTIC_RELEASE_URL
    assert agentic.pin.revision == AGENTIC_MANIFEST_SHA256
    assert nonagentic.pin.url == NONAGENTIC_RELEASE_URL
    assert nonagentic.pin.revision == NONAGENTIC_MANIFEST_SHA256
    assert curriculum.sampling_config() == {
        "kind": "domain-weighted",
        "domain_weights": {"agentic": 0.5, "nonagentic": 0.5},
        "seed": SEED,
    }


def test_the_sampler_apportions_the_consumed_route_quotas_exactly():
    """0.5/0.5 over 64 groups is 32/32 in every window: the quotas .55 consumed."""
    quotas = largest_remainder_quotas(BATCH_GROUPS, dict(WEIGHTS))
    assert quotas == dict(ROUTE_QUOTAS) == {"agentic": 32, "nonagentic": 32}


def test_the_releases_pin_the_per_source_provenance():
    """Per-source composition is release provenance: the pinned row counts."""
    agentic_config = AGENTIC_RELEASE.config
    nonagentic_config = NONAGENTIC_RELEASE.config
    assert agentic_config is not None and nonagentic_config is not None
    train = agentic_config.members[0]
    assert train.rows == AGENTIC_TRAIN_ROWS
    assert train.source_counts == {"long-loop-smartie-swesmith": AGENTIC_TRAIN_ROWS}
    blend = nonagentic_config.members[0]
    assert (
        blend.rows
        == TASKTROVE_ROWS
        + REASONING_GYM_ROWS
        + OPEN_SCIENCE_ROWS
        + NEMOTRON_BLEND_ROWS
        + SNOWBALL_SMARTIE_ROWS
        + EURUS_ROWS
    )
    assert blend.source_counts == {
        "open-thoughts/TaskTrove": TASKTROVE_ROWS,
        "nvidia/OpenScience": OPEN_SCIENCE_ROWS,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": NEMOTRON_BLEND_ROWS,
        REASONING_GYM_SOURCE: REASONING_GYM_ROWS,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": SNOWBALL_SMARTIE_ROWS,
        "PRIME-RL/Eurus-2-RL-Data": EURUS_ROWS,
    }


def test_the_fingerprint_follows_only_the_enforced_mix():
    curriculum = exp3_rl_curriculum()
    assert exp3_rl_curriculum().fingerprint == curriculum.fingerprint
    assert replace(curriculum, seed=SEED + 1).fingerprint != curriculum.fingerprint
    rebalanced = replace(curriculum, entries=tuple(replace(entry, weight=0.25) for entry in curriculum.entries))
    assert rebalanced.fingerprint != curriculum.fingerprint
    assert replace(curriculum, version="2026.11.18").fingerprint == curriculum.fingerprint


def test_verify_release_admits_the_recorded_composition(tmp_path):
    """The adoption stage verifies digests, row counts and provenance against the record."""
    source = _local_release(tmp_path)
    config = source.config
    assert config is not None
    composition = verify_release(config)
    assert composition["train.parquet"].rows == 2
    assert composition["train.parquet"].source_counts == {"alpha": 1, "beta": 1}
    tampered = replace(config, members=tuple(replace(member, sha256="0" * 64) for member in config.members))
    with pytest.raises(ValueError, match="does not match the pinned"):
        verify_release(tampered)
    miscounted = replace(
        config,
        members=tuple(
            replace(member, source_counts={"alpha": 2, "beta": 0}) if member.source_counts else member
            for member in config.members
        ),
    )
    with pytest.raises(ValueError, match="provenance counts"):
        verify_release(miscounted)


def _local_release(tmp_path):
    """A two-row local release shaped like the pinned ones: a task_spec payload column."""
    rows = [
        {"task_spec": json.dumps({"source": {"dataset": "alpha"}, "id": "a1"})},
        {"task_spec": json.dumps({"source": {"dataset": "beta"}, "id": "b1"})},
    ]
    release = tmp_path / "release"
    release.mkdir()
    path = release / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return RlDataSource(
        info=SourceInfo(id="Test:adopted-release", title="adopted release", origin="Test"),
        config=AdoptedRelease(
            name="test-adopted-release",
            version="1",
            url=str(release),
            manifest_sha256="0" * 64,
            members=(AdoptedFile("train.parquet", sha256, 2, {"alpha": 1, "beta": 1}),),
        ),
        pipeline=adopt_release,
    )
