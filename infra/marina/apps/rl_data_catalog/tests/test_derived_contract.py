# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A grader census follows exact task bytes rather than a dataset's display name."""

import copy
import json
from pathlib import Path

import pytest

from infra.marina.apps.rl_data_catalog.server.contract_selection import (
    receipt_identity,
    select_contract_receipts,
)
from infra.marina.apps.rl_data_catalog.server.derived_contract import contract_projection
from infra.marina.apps.rl_data_catalog.server.restrictions import PRIVATE_TEXT_MARKER, private_declaration_projection


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
        withheld = private_declaration_projection(
            {
                **projected,
                "data_restriction": "private_always",
                "notes": "private prompt",
                "quality_result": {"issues": ["private reviewer output"]},
            }
        )
        assert withheld["derived_contract_receipt"] == projected["derived_contract_receipt"]
        assert withheld["notes"] == PRIVATE_TEXT_MARKER
        assert "quality_result" not in withheld


PAIRS = json.loads((Path(__file__).parent / "fixtures/contract_supersessions.json").read_text())


@pytest.mark.parametrize("pair", PAIRS, ids=lambda pair: pair["source_id"])
def test_recorded_rule_correction_selects_exact_receipt_without_changing_history(pair):
    source = {"id": pair["source_id"], "actor_census_input": pair["selected"]["input"]}
    old, new = pair["previous"], pair["selected"]
    original = [receipt_identity(old), receipt_identity(new)]
    expected = contract_projection(source, [new])
    assert contract_projection(source, [old, new]) == expected
    assert contract_projection(source, [new, old]) == expected
    assert [receipt_identity(old), receipt_identity(new)] == original
    assert select_contract_receipts(source["id"], [old]) == [old]


@pytest.mark.parametrize("change", ["extra-conflict", "summary", "evidence", "input"])
def test_unlisted_conflict_never_inherits_a_recorded_supersession(change):
    pair = PAIRS[0]
    old, new = copy.deepcopy(pair["previous"]), copy.deepcopy(pair["selected"])
    if change == "extra-conflict":
        other = copy.deepcopy(old)
        other["evidence_url"] += "-unlisted"
        rows = [old, new, other]
    else:
        if change == "summary":
            new["summary"]["single_turn_count"] -= 1
        if change == "evidence":
            new["evidence_url"] += "-unlisted"
        if change == "input":
            new["input"]["dataset_revision"] = "f" * 40
        rows = [old, new]
    with pytest.raises(ValueError, match="Conflicting derived contract summaries"):
        select_contract_receipts(pair["source_id"], rows)
