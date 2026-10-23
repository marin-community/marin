# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SkyRL ``train`` box against a fake job that reads the feed and writes graded groups."""

import pytest

# The loop runs in the marin-taskforge environment; other suites skip it.
pytest.importorskip("taskforge")

import asyncio

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import GradedRollout, Rollout
from taskcompendium.grading_result import GradeResult, Outcome

from experiments.post_training.baby_rsi.skyrl_box import (
    CHUNKS,
    GRADED,
    SKIPPED,
    FeedConfig,
    GradedGroup,
    Skipped,
    seq_name,
    skyrl_train,
)
from experiments.post_training.baby_rsi.streams import from_iterable
from experiments.post_training.baby_rsi.tests.test_loop import task


async def fake_job(root: StoragePath, skip: str, stop: asyncio.Event) -> None:
    """Grades each task of the feed once, in seq order; skips the task whose id is ``skip``."""
    done: set[int] = set()
    while not stop.is_set():
        for chunk in sorted((root / CHUNKS / "*.parquet").glob(), key=str):
            with chunk.open("rb") as handle:
                table = pq.read_table(handle)
            for seq, task_json in zip(table["seq"].to_pylist(), table["task"].to_pylist(), strict=True):
                # A resumed job continues after the tasks that it graded.
                if seq in done or (root / GRADED / seq_name(seq, ".json")).exists():
                    continue
                done.add(seq)
                if f'"id":"{skip}"' in task_json:
                    (root / SKIPPED / seq_name(seq, ".json")).write_text(
                        Skipped(uid=str(seq), reason="test").model_dump_json()
                    )
                    continue
                rollout = GradedRollout(Rollout(f"run-{seq}", (), "stop"), GradeResult(Outcome.GRADED, 1.0, None))
                group = GradedGroup(uid=str(seq), policy_step=seq // 2, rollouts=(rollout,))
                (root / GRADED / seq_name(seq, ".json")).write_text(group.model_dump_json())
        await asyncio.sleep(0.01)


async def run(root: StoragePath, ids: list[str], groups: int) -> list[str]:
    stop = asyncio.Event()
    job = asyncio.create_task(fake_job(root, skip="t2", stop=stop))
    config = FeedConfig(root=root, lookahead=3, chunk_rows=2, poll_seconds=0.01, groups=groups)
    trained = [group async for group in skyrl_train(from_iterable(task(i) for i in ids), config)]
    stop.set()
    await job
    assert all(
        group.model == f"skyrl-step-{int(group.rollouts[0].rollout.run_id.split('-')[1]) // 2}" for group in trained
    )
    return [group.task.id for group in trained]


def test_the_box_returns_the_feed_in_order_and_a_resumed_box_continues(tmp_path):
    root = StoragePath(str(tmp_path))

    # The skipped task does not come back.
    assert asyncio.run(run(root, [f"t{k}" for k in range(5)], groups=4)) == ["t0", "t1", "t3", "t4"]
    assert asyncio.run(run(root, ["u0", "u1"], groups=2)) == ["u0", "u1"]

    # A resumed box writes after the last task of the feed.
    feed = {}
    for chunk in (root / CHUNKS / "*.parquet").glob():
        with chunk.open("rb") as handle:
            table = pq.read_table(handle)
        feed.update(zip(table["seq"].to_pylist(), table["task"].to_pylist(), strict=True))
    assert sorted(feed) == list(range(7)) and '"id":"u0"' in feed[5]
