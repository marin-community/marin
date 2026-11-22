# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 curriculum records the route mix the .55 run enforced."""

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.rl.curriculum import PoolRow, RlCurriculum, SamplingMode, draw_validation_split

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE,
    AGENTIC_RELEASE_URL,
    AGENTIC_ROUTE,
    AGENTIC_TRAIN_ROWS,
    EURUS_ROWS,
    NEMOTRON_BLENDS_ROWS,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_ROUTE,
    OPEN_SCIENCE_ROWS,
    REASONING_GYM_ROWS,
    REASONING_GYM_SOURCE,
    SNOWBALL_SMARTIE_ROWS,
    TASKTROVE_ROWS,
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
    VALIDATION_SIZE,
    WEIGHTS,
    exp3_rl_curriculum,
    release_pool_rows,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def _fixture_pool_rows(counts: dict[str, int]) -> dict[str, tuple[PoolRow, ...]]:
    """Synthetic row identities per route pool, keyed the way the draw consumes them."""
    return {
        route: tuple(PoolRow(id=f"{route}-{index}", sha256=f"{index:064x}") for index in range(count))
        for route, count in counts.items()
    }


def test_the_curriculum_records_the_enforced_route_mix():
    curriculum = exp3_rl_curriculum(_fixture_pool_rows({AGENTIC_ROUTE: AGENTIC_TRAIN_ROWS, NONAGENTIC_ROUTE: 100_000}))
    assert curriculum.name == CURRICULUM_NAME
    assert curriculum.version == CURRICULUM_VERSION
    assert curriculum.mode is SamplingMode.WEIGHTED
    assert curriculum.seed == SEED == 20261020
    assert curriculum.weights() == dict(WEIGHTS) == {AGENTIC_ROUTE: 0.5, NONAGENTIC_ROUTE: 0.5}
    assert {entry.label: entry.source for entry in curriculum.entries} == {
        AGENTIC_ROUTE: "long-loop-rl-agentic-train",
        NONAGENTIC_ROUTE: "long-loop-rl-nonagentic-train",
    }
    agentic = next(entry for entry in curriculum.entries if entry.label == AGENTIC_ROUTE)
    nonagentic = next(entry for entry in curriculum.entries if entry.label == NONAGENTIC_ROUTE)
    assert agentic.pin.url == AGENTIC_RELEASE_URL
    assert agentic.pin.revision == AGENTIC_MANIFEST_SHA256
    assert nonagentic.pin.url == NONAGENTIC_RELEASE_URL
    assert nonagentic.pin.revision == NONAGENTIC_MANIFEST_SHA256
    assert curriculum.sampling_config() == {
        "kind": "domain-weighted",
        "domain_weights": {AGENTIC_ROUTE: 0.5, NONAGENTIC_ROUTE: 0.5},
        "seed": SEED,
    }


def test_the_sampler_apportions_the_consumed_route_quotas_exactly():
    """0.5/0.5 over 64 groups is 32/32 in every window: the quotas .55 consumed."""
    assert dict(ROUTE_QUOTAS) == {AGENTIC_ROUTE: 32, NONAGENTIC_ROUTE: 32}
    assert WEIGHTS[AGENTIC_ROUTE] * BATCH_GROUPS == ROUTE_QUOTAS[AGENTIC_ROUTE]
    assert WEIGHTS[NONAGENTIC_ROUTE] * BATCH_GROUPS == ROUTE_QUOTAS[NONAGENTIC_ROUTE]


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
        + NEMOTRON_BLENDS_ROWS
        + SNOWBALL_SMARTIE_ROWS
        + EURUS_ROWS
    )
    assert blend.source_counts == {
        "open-thoughts/TaskTrove": TASKTROVE_ROWS,
        "nvidia/OpenScience": OPEN_SCIENCE_ROWS,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": NEMOTRON_BLENDS_ROWS,
        REASONING_GYM_SOURCE: REASONING_GYM_ROWS,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": SNOWBALL_SMARTIE_ROWS,
        "PRIME-RL/Eurus-2-RL-Data": EURUS_ROWS,
    }


def test_the_fingerprint_follows_only_the_enforced_mix():
    pool_rows = _fixture_pool_rows({AGENTIC_ROUTE: AGENTIC_TRAIN_ROWS, NONAGENTIC_ROUTE: 100_000})
    curriculum = exp3_rl_curriculum(pool_rows)
    assert exp3_rl_curriculum(pool_rows).fingerprint == curriculum.fingerprint
    shifted = draw_validation_split(dict(WEIGHTS), pool_rows, size=VALIDATION_SIZE, seed=SEED + 1)
    reseeded = RlCurriculum(
        name=curriculum.name,
        version=curriculum.version,
        mode=curriculum.mode,
        entries=curriculum.entries,
        stages=curriculum.stages,
        seed=SEED + 1,
        validation=shifted.validation,
    )
    assert reseeded.fingerprint != curriculum.fingerprint
    rebalanced = replace(curriculum, entries=tuple(replace(entry, weight=0.25) for entry in curriculum.entries))
    assert rebalanced.fingerprint != curriculum.fingerprint
    assert replace(curriculum, version="2026.11.18").fingerprint == curriculum.fingerprint
    with pytest.raises(ValueError, match="validation seed"):
        replace(curriculum, seed=SEED + 1)


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


def test_the_curriculum_holds_out_a_route_proportional_split():
    """The recorded holdout follows the 0.5/0.5 mixture and partitions the pools."""
    counts = {AGENTIC_ROUTE: AGENTIC_TRAIN_ROWS, NONAGENTIC_ROUTE: 100_000}
    curriculum = exp3_rl_curriculum(_fixture_pool_rows(counts))
    assert curriculum.validation.size == VALIDATION_SIZE == 200
    assert curriculum.validation.seed == SEED
    assert Counter(row.label for row in curriculum.validation.rows) == {AGENTIC_ROUTE: 100, NONAGENTIC_ROUTE: 100}
    split = draw_validation_split(dict(WEIGHTS), _fixture_pool_rows(counts), size=VALIDATION_SIZE, seed=SEED)
    assert split.validation == curriculum.validation
    held = {(row.label, row.id) for row in curriculum.validation.rows}
    trained = {(label, row.id) for label, rows in split.training.items() for row in rows}
    assert held.isdisjoint(trained)
    assert held | trained == {(label, f"{label}-{index}") for label, count in counts.items() for index in range(count)}


def _write_release_parquet(directory, rows, name="train.parquet"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_release_pool_rows_stream_the_pinned_parquets(tmp_path):
    """The row identities come from the release files, keyed by each source's admitted route."""
    rows = [
        {"task_spec": json.dumps({"source": {"dataset": "long-loop-smartie-swesmith"}, "id": "swe-1"})},
        {"task_spec": json.dumps({"source": {"dataset": "open-thoughts/TaskTrove"}, "id": "tt-1"})},
    ]
    release = tmp_path / "release"
    path = _write_release_parquet(release, rows)
    pooled = release_pool_rows(str(release), path.name)
    assert set(pooled) == {AGENTIC_ROUTE, NONAGENTIC_ROUTE}
    assert [row.id for row in pooled[AGENTIC_ROUTE]] == ["swe-1"]
    assert [row.id for row in pooled[NONAGENTIC_ROUTE]] == ["tt-1"]


def test_release_pool_rows_reject_unrouted_sources(tmp_path):
    rows = [{"task_spec": json.dumps({"source": {"dataset": "alpha"}, "id": "a1"})}]
    release = tmp_path / "release"
    path = _write_release_parquet(release, rows)
    with pytest.raises(ValueError, match="names no admitted route"):
        release_pool_rows(str(release), path.name)
