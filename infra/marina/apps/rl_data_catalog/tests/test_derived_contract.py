# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A grader census follows exact task bytes rather than a dataset's display name."""

import pytest

from infra.marina.apps.rl_data_catalog.server.derived_contract import contract_projection


@pytest.mark.parametrize(
    "changed", [None, "dataset_revision", "converter_revision", "task_specs_sha256", "measured_tasks"]
)
def test_changed_population_drops_derived_profile(changed: str | None) -> None:
    binding = dict(
        dataset_id="data",
        dataset_revision="a",
        converter_version="v1",
        converter_revision="b",
        measured_tasks=3,
        task_specs_sha256="c",
    )
    source = {"id": "Marin:source", "actor_census_input": binding}
    receipt = {
        "source_id": source["id"],
        "input": dict(binding),
        "evidence_url": "https://example.org/receipt",
        "scanned_task_specs_sha256": "d",
        "summary": {
            "task_count": 3,
            "grader_type_counts": {"script": 1, "llm-judge": 2},
            "judge_rubric_counts": {"reference": 2},
            "judge_reference_count": 2,
            "judge_no_reference_count": 0,
            "agentic_count": 0,
            "single_turn_count": 3,
            "reply_only_count": 3,
            "workspace_graded_count": 0,
        },
    }
    if changed:
        binding[changed] = 4 if changed == "measured_tasks" else "changed"
    projected = contract_projection(source, [receipt])
    if changed:
        assert projected["grader_type"] == "" and projected.get("derived_contract_receipt") is None
    else:
        assert projected["grader_type"] == "mixed"
        assert projected["grader_type_counts"] == {"script": 1, "llm-judge": 2}
        assert projected["derived_agentic"] is False and projected["derived_turns"] == "single-turn"
        assert projected["derived_contract_receipt"]["uri"] == receipt["evidence_url"]
