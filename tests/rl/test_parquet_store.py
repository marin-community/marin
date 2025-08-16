"""Round-trip tests for marin.rl.parquet_store."""

import time
from dataclasses import asdict

import pyarrow as pa  # noqa: F401  # ensures pyarrow import works in CI

from marin.rl.parquet_store import iter_rollout_groups, write_rollout_groups
from marin.rl.datatypes import RolloutGroup, RolloutRecord, Turn


def _make_sample_groups() -> list[RolloutGroup]:
    ts = time.time()

    r = RolloutRecord(
        environment="dummy_env",
        example_id="ex1",
        policy_version="v0",
        rollout_uid="u1",
        replica_id="rep",
        reward=1.0,
        turns=[
            Turn(
                message="hi",
                logprobs=None,
                role="user",
                reward=None,
                inference_metadata={},
            ),
            Turn(
                message="there",
                logprobs=None,
                role="assistant",
                reward=1.0,
                inference_metadata={},
            ),
        ],
        metadata={"text": "Hi there!"},
        created_ts=ts,
    )

    g1 = RolloutGroup(
        id="g1",
        environment="dummy_env",
        example_id="ex1",
        policy_version="v0",
        segment_idx=0,
        rollouts=[r],
        sealed_ts=ts,
        metadata={"env": "dummy_env"},
    )

    g2 = RolloutGroup(
        id="g2",
        environment="dummy_env",
        example_id="ex1",
        policy_version="v0",
        segment_idx=0,
        rollouts=[r],
        sealed_ts=ts + 1,
        metadata={"env": "dummy_env", "difficulty": "hard"},
    )

    return [g1, g2]


def _sort_by_id(groups: list[RolloutGroup]) -> list[RolloutGroup]:
    return sorted(groups, key=lambda g: g.id)


def _groups_equal(a: RolloutGroup, b: RolloutGroup) -> bool:
    """Deep equality helper via dataclasses.asdict with float tolerance."""

    da, db = asdict(a), asdict(b)
    if abs(da["sealed_ts"] - db["sealed_ts"]) > 1e-9:
        return False
    da["sealed_ts"] = db["sealed_ts"] = 0
    for ra, rb in zip(da["rollouts"], db["rollouts"], strict=False):
        if abs(ra["created_ts"] - rb["created_ts"]) > 1e-9:
            return False
        ra["created_ts"] = rb["created_ts"] = 0
    return da == db


def test_parquet_round_trip(tmp_path):
    groups = _make_sample_groups()

    # Write groups twice to verify appending additional parts is okay.
    write_rollout_groups(groups[:1], str(tmp_path))
    write_rollout_groups(groups[1:], str(tmp_path))

    read_back = list(iter_rollout_groups(str(tmp_path)))

    assert len(read_back) == len(groups)

    original_sorted = _sort_by_id(groups)
    read_sorted = _sort_by_id(read_back)

    for orig, rec in zip(original_sorted, read_sorted, strict=False):
        assert _groups_equal(orig, rec)
