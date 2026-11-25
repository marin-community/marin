# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.rl.curriculum import consumed_route_quotas, validation_payload
from taskcompendium.identity import canonical_sha256

from experiments.post_training.long_loop.exp3_data import partition_tasks
from experiments.post_training.task_curation.curriculum_manifest import curriculum_from_manifest


def test_exp3_pool_materialization_preserves_tasks_and_excludes_random_holdout():
    tasks = {
        route: [
            {"id": f"{route}-{i}", "source": {"dataset": f"source-{route}"}, "prompt": f"original {i}"}
            for i in range(180)
        ]
        for route in ["agentic", "nonagentic"]
    }
    curriculum, train, val = partition_tasks(tasks)
    assert {route: len(rows) for route, rows in train.items()} == {"agentic": 80, "nonagentic": 80}
    assert len(val) == 200
    assert consumed_route_quotas(curriculum, {route: route for route in tasks}, 64) == {"agentic": 32, "nonagentic": 32}
    train_tasks = [json.loads(row["task_spec"]) for rows in train.values() for row in rows]
    val_tasks = [json.loads(row["task_spec"]) for row in val]
    assert not {row["id"] for row in train_tasks} & {row["id"] for row in val_tasks}
    assert not {canonical_sha256(row) for row in train_tasks} & {canonical_sha256(row) for row in val_tasks}
    assert sorted(train_tasks + val_tasks, key=lambda row: row["id"]) == sorted(
        [row for rows in tasks.values() for row in rows], key=lambda row: row["id"]
    )
    reordered = partition_tasks({route: list(reversed(rows)) for route, rows in reversed(list(tasks.items()))})[0]
    assert reordered == curriculum
    rebuilt = curriculum_from_manifest(
        {"curriculum": curriculum.identity, "validation": validation_payload(curriculum.validation)}
    )
    assert rebuilt == curriculum
