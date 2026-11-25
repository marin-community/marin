# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from dataclasses import asdict

from marin.rl.curriculum import consumed_route_quotas, validation_payload

from experiments.post_training.long_loop.exp4_control_data import partition_rows
from experiments.post_training.task_curation.curriculum_manifest import curriculum_from_manifest


def test_original_exp4_partition_preserves_task_bytes_and_curriculum_entries():
    weights = {"swesmith": 0.35, "nemotron_if": 0.15, "original-generator": 0.5}
    parent = {
        "name": "long-loop/exp4-rl",
        "version": "original-version",
        "identity": "long-loop/exp4-rl@original-version:original-fingerprint",
        "mode": "weighted",
        "seed": 20261020,
        "stages": [],
        "entries": [
            {
                "label": label,
                "source": label,
                "version": "1",
                "pin": {"name": f"original/{label}", "revision": "original", "url": f"s3://region/{label}"},
                "recipe": "original.recipe",
                "weight": weight,
            }
            for label, weight in weights.items()
        ],
    }
    rows = [
        {
            "teacher_route": label,
            "task_spec": json.dumps(
                {"id": f"{label}-{index}", "prompt": f"Original spacing and Unicode: \u03b1 {index}"}, indent=2
            ),
        }
        for label, count in {"swesmith": 80, "nemotron_if": 40, "original-generator": 180}.items()
        for index in range(count)
    ]
    prepared = partition_rows(rows, parent, "new-version")
    train = [row for members in prepared.training.values() for row in members]
    assert len(train) == 100 and len(prepared.validation) == 200
    assert Counter((row["teacher_route"], row["task_spec"]) for row in train + prepared.validation) == Counter(
        (row["teacher_route"], row["task_spec"]) for row in rows
    )
    assert not {row["task_spec"] for row in train} & {row["task_spec"] for row in prepared.validation}
    assert prepared.curriculum.weights() == weights
    assert [asdict(entry) for entry in prepared.curriculum.entries] == parent["entries"]
    assert prepared.curriculum.identity != parent["identity"]
    assert partition_rows(list(reversed(rows)), parent, "new-version").curriculum == prepared.curriculum
    definition = asdict(prepared.curriculum)
    definition.pop("fingerprint")
    assert (
        curriculum_from_manifest(
            {
                "curriculum": prepared.curriculum.identity,
                "curriculum_definition": definition,
                "validation": validation_payload(prepared.curriculum.validation),
            }
        )
        == prepared.curriculum
    )
    assert consumed_route_quotas(
        prepared.curriculum, {"swesmith": "agentic", "nemotron_if": "nonagentic", "original-generator": "nonagentic"}, 64
    ) == {"agentic": 22, "nonagentic": 42}
