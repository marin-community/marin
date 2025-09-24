# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Round-trip tests for marin.rl.parquet_store."""

import time
from dataclasses import asdict

import pyarrow as pa  # noqa: F401  # ensures pyarrow import works in CI

from marin.rl.parquet_store import iter_rollout_groups, write_rollout_groups
from marin.rl.types import Rollout, RolloutGroup, Turn


def _make_sample_groups() -> list[RolloutGroup]:
    ts = time.time()

    turn1 = Turn(
        message="Hello",
        role="user",
        logprobs=None,
        reward=None,
        inference_metadata={"model": "v0"},
    )
    turn2 = Turn(
        message="Hi there!",
        role="assistant",
        logprobs=[-0.3, -0.2],
        reward=1.0,
        inference_metadata={"model": "v0"},
    )

    rollout = Rollout(turns=[turn1, turn2], metadata={"seed": 42})

    g1 = RolloutGroup(
        id="g1",
        source="dummy_env",
        created=ts,
        rollouts=[rollout],
        metadata={"env": "dummy_env"},
    )

    g2 = RolloutGroup(
        id="g2",
        source="dummy_env",
        created=ts + 1,
        rollouts=[rollout],
        metadata={"env": "dummy_env", "difficulty": "hard"},
    )

    return [g1, g2]


def _sort_by_id(groups: list[RolloutGroup]) -> list[RolloutGroup]:
    return sorted(groups, key=lambda g: g.id)


def _groups_equal(a: RolloutGroup, b: RolloutGroup) -> bool:
    """Deep equality helper via dataclasses.asdict with float tolerance."""

    da, db = asdict(a), asdict(b)
    # Allow tiny float differences in "created"
    if abs(da["created"] - db["created"]) > 1e-9:
        return False
    da["created"] = db["created"] = 0  # normalise
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
