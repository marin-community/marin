# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed an active join-right reducer and query PR #7911's dashboard API.

Run from the repository root: uv run lib/zephyr/scripts/repro_pr7911_join_status.py
The real planner and coordinator test driver run locally; no pipeline is executed.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from fray.actor import ActorContext, _reset_current_actor, _set_current_actor
from starlette.testclient import TestClient
from zephyr.dataset import Dataset
from zephyr.plan import compute_plan
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask
from zephyr.testing.coordinator import TEST_TASK_COST, make_test_coordinator, start_test_stage


def main() -> None:
    right = (
        Dataset.from_list([1])
        .map(lambda value: value + 1)
        .group_by(key=lambda value: value, reducer=lambda key, values: key)
    )
    plan = compute_plan(
        Dataset.from_list([1]).sorted_merge_join(right, left_key=lambda value: value, right_key=lambda value: value)
    )
    stage = "join-right-0-0-stage1"
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name=stage,
        cost=TEST_TASK_COST,
    )
    with TemporaryDirectory() as directory:
        token = _set_current_actor(ActorContext(handle=MagicMock(), index=0, group_name="probe"))
        coordinator = make_test_coordinator(Path(directory))
        try:
            coordinator.register_worker("worker", MagicMock())
            start_test_stage(coordinator, [task], plan=plan, execution_id="probe", stage_name=stage)
            with TestClient(coordinator.web_application) as client:
                response = client.get("/api/status", params={"execution_id": "probe"})
                response.raise_for_status()
                states = {node["node_id"]: node["state"] for node in response.json()["node_statuses"]}
            print(f"Seeded active stage: {stage} (right-side Reduce)")
            expected = {
                "main/stage/0/join/0/right/stage/0": "succeeded",
                "main/stage/0/join/0/right/stage/1": "running",
                "main/stage/0": "pending",
            }
            for node, state in expected.items():
                print(f"{node}: expected={state}, actual={states[node]}")
        finally:
            coordinator.shutdown()
            _reset_current_actor(token)


if __name__ == "__main__":
    main()
