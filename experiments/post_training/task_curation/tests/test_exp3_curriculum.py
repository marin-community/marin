# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 .55 RL run's consumed task mix, pinned as a curriculum.

The consumed record binds three primary artifacts: the .55 resolved launch
(source_routes, consumed_route_quotas 32/32 over a 64-group batch, seed 20261020),
the two release seals (manifest SHA-256s and file digests), and the in-region
per-source scan of the pinned parquets (row counts per ``source.dataset``).
"""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.rl.curriculum import SamplingMode

from experiments.post_training.task_curation.curriculum import recipe_identity
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE_URL,
    AGENTIC_TRAIN_SHA256,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_TRAIN_SHA256,
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
    SOURCE_ROUTES,
    exp3_rl_curriculum,
    weights,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.sources import all_sources

CONSUMED = {
    "long-loop-smartie-swesmith": {
        "route": "agentic",
        "release": "long-loop-rl-agentic-train",
        "rows": 5608,
        "weight": 0.5,
    },
    "open-thoughts/TaskTrove": {
        "route": "nonagentic",
        "release": "long-loop-rl-nonagentic-train",
        "rows": 24033,
        "weight": 0.120165,
    },
    REASONING_GYM_SOURCE: {
        "route": "nonagentic",
        "release": "long-loop-rl-nonagentic-train",
        "rows": 20800,
        "weight": 0.104,
    },
    "nvidia/OpenScience": {
        "route": "nonagentic",
        "release": "long-loop-rl-nonagentic-train",
        "rows": 20790,
        "weight": 0.10395,
    },
    "nvidia/Nemotron-RL-Ultra-Training-Blends": {
        "route": "nonagentic",
        "release": "long-loop-rl-nonagentic-train",
        "rows": 12426,
        "weight": 0.06213,
    },
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": {
        "route": "nonagentic",
        "release": "long-loop-rl-nonagentic-train",
        "rows": 11634,
        "weight": 0.05817,
    },
    "PRIME-RL/Eurus-2-RL-Data": {
        "route": "nonagentic",
        "release": "long-loop-rl-nonagentic-train",
        "rows": 10317,
        "weight": 0.051585,
    },
}

RELEASE_PINS = {
    "long-loop-rl-agentic-train": (AGENTIC_RELEASE_URL, AGENTIC_MANIFEST_SHA256, AGENTIC_TRAIN_SHA256),
    "long-loop-rl-nonagentic-train": (NONAGENTIC_RELEASE_URL, NONAGENTIC_MANIFEST_SHA256, NONAGENTIC_TRAIN_SHA256),
}


def test_the_curriculum_is_what_55_consumed_source_by_source():
    curriculum = exp3_rl_curriculum()
    assert curriculum.name == CURRICULUM_NAME
    assert curriculum.version == CURRICULUM_VERSION
    assert curriculum.mode is SamplingMode.WEIGHTED
    assert curriculum.seed == SEED == 20261020
    assert {entry.label for entry in curriculum.entries} == set(CONSUMED)
    catalog = {source.name: source for source in all_sources().values()}
    for entry in curriculum.entries:
        consumed = CONSUMED[entry.label]
        release = catalog[entry.source]
        assert entry.source == consumed["release"]
        assert entry.version == "1"
        name, revision, _ = RELEASE_PINS[entry.source]
        assert entry.pin.name == name
        assert entry.pin.revision == revision
        assert entry.pin.url == name
        assert entry.weight == consumed["weight"]
        assert entry.recipe == recipe_identity(release.pipeline)
        assert release.dataset.revision == revision
        assert release.info.count == sum(
            consumed["rows"] for consumed in CONSUMED.values() if consumed["release"] == entry.source
        )
    # The run drew groups by teacher route; every source's route must be the consumed one.
    assert {entry.label: SOURCE_ROUTES[entry.label] for entry in curriculum.entries} == {
        label: consumed["route"] for label, consumed in CONSUMED.items()
    }


def test_the_weights_project_the_consumed_route_quotas():
    resolved = weights()
    assert resolved == {label: consumed["weight"] for label, consumed in CONSUMED.items()}
    assert sum(resolved.values()) == 1.0
    # Each route's weight mass is its consumed quota over the 64-group batch.
    route_mass = {}
    for label, weight in resolved.items():
        route_mass[SOURCE_ROUTES[label]] = route_mass.get(SOURCE_ROUTES[label], 0.0) + weight
    assert route_mass == {route: quota / BATCH_GROUPS for route, quota in ROUTE_QUOTAS.items()}
    assert exp3_rl_curriculum().sampling_config() == {
        "kind": "domain-weighted",
        "domain_weights": resolved,
        "seed": SEED,
    }


def test_the_fingerprint_follows_only_the_consumed_mix():
    curriculum = exp3_rl_curriculum()
    assert exp3_rl_curriculum().fingerprint == curriculum.fingerprint
    # A different seed, weight, or pin is a different mix; re-badging is not.
    drifted = replace(curriculum, seed=SEED + 1)
    assert drifted.fingerprint != curriculum.fingerprint
    reweighted = replace(
        curriculum,
        entries=tuple(
            replace(entry, weight=entry.weight / 2 if entry.weight else entry.weight) for entry in curriculum.entries
        ),
    )
    assert reweighted.fingerprint != curriculum.fingerprint
    relabeled = replace(curriculum, version="2026.11.18")
    assert relabeled.fingerprint == curriculum.fingerprint


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
