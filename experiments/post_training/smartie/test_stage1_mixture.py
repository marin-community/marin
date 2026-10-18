# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json

from experiments.post_training.smartie.mixture import structural_sample


def test_structural_sample_preserves_proportions_payloads_and_input_order_independence():
    rows = []
    for family, size in (("object", 8), ("array", 4), ("tool", 4)):
        for index in range(size):
            config = base64.b64encode(json.dumps({"extras": {"family": family}}).encode()).decode()
            rows.append(
                {
                    "teacher_route": "structural",
                    "lowered_task_spec": json.dumps(
                        {
                            "task": {
                                "id": f"{family}-{index}",
                                "resources": {
                                    "verifier": [{"path": "config.json", "source": {"content_base64": config}}]
                                },
                            }
                        }
                    ),
                }
            )
    selected, counts = structural_sample(rows, 8, 42)
    reordered, reordered_counts = structural_sample(list(reversed(rows)), 8, 42)
    assert counts == {"array": 2, "object": 4, "tool": 2}
    assert counts == reordered_counts
    assert selected == reordered
    assert len({row["lowered_task_spec"] for row in selected}) == 8
    assert all(row in rows for row in selected)


def test_structural_sample_allocates_remaining_rows_deterministically():
    rows = []
    for family in ("array", "object", "tool"):
        config = base64.b64encode(json.dumps({"extras": {"family": family}}).encode()).decode()
        for index in range(3):
            rows.append(
                {
                    "lowered_task_spec": json.dumps(
                        {
                            "task": {
                                "id": f"{family}-{index}",
                                "resources": {
                                    "verifier": [{"path": "config.json", "source": {"content_base64": config}}]
                                },
                            }
                        }
                    )
                }
            )
    selected, counts = structural_sample(rows, 5, 42)
    assert counts == {"array": 2, "object": 2, "tool": 1}
    assert len(selected) == 5
