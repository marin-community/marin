# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict

import pytest

from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CODING_SUITES,
    CodingPanel,
    PanelItem,
    coding_analysis_request,
    coding_evidence_rows,
    protocol_digest,
)


def panel_fixture():
    records, archives, items = [], [], []
    for suite in CODING_SUITES:
        record = {
            "eval": {"name": suite, "source_digest": "dataset-pin", "tasks": [], "evalchemy": {"shots": 3}},
            "provenance": {"eval_runtime": "evaluator-pin"},
            "model": {
                "config": {
                    "identity": "parent",
                    "tokenizer": "tokenizer",
                    "tokenizer_revision": "revision",
                    "serve": {},
                    "generation": {},
                }
            },
            "status": "succeeded",
            "error": None,
        }
        records.append(record)
        rows = []
        for index in range(32):
            prompt = f"{suite} prompt {index}"
            items.append(PanelItem(suite, str(index), hashlib.sha256(prompt.encode()).hexdigest()))
            rows.append(
                {
                    "task": suite,
                    "doc_id": str(index + 100),
                    "kind": "generation",
                    "prompt_text": prompt,
                    "output": f"actual answer {index}",
                    "extracted": None,
                    "doc": json.dumps({"task_id": index, "canonical_solution": "secret-gold", "test": "secret-test"}),
                    "metrics": [("pass_rate", 0.0 if index < 7 else 1.0)],
                    "grading": None,
                    "correct": None,
                    "filter": "none",
                    "trial_id": "",
                }
            )
        archives.append(rows)
    panel = CodingPanel(tuple(items), {record["eval"]["name"]: protocol_digest(record) for record in records})
    return tuple(records), tuple(archives), panel


def test_real_sample_shape_keeps_null_grading_and_excludes_gold_from_feedback():
    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    assert len(rows) == 64
    assert {suite: sum(row.pass_rate for row in rows if row.suite == suite) for suite in CODING_SUITES} == {
        "humanevalplus": 25,
        "mbppplus": 25,
    }
    request = coding_analysis_request({"rows": [asdict(row) for row in rows]})
    failed = json.loads(request["messages"][1]["content"])["failures"]
    assert len(failed) == 14
    assert {row["suite"] for row in failed} == set(CODING_SUITES)
    serialized = json.dumps(request)
    assert "secret-gold" not in serialized and "secret-test" not in serialized
    assert all(row["pass_rate"] == 0 for row in failed)


@pytest.mark.parametrize("change", ["prompt", "task", "missing", "duplicate", "model", "protocol", "metric"])
def test_panel_drift_and_incomplete_measurement_cannot_enter_feedback(change):
    records, archives, panel = panel_fixture()
    if change == "prompt":
        archives[0][0]["prompt_text"] += " changed"
    elif change == "task":
        archives[0][0]["doc"] = json.dumps({"task_id": "not-in-panel"})
    elif change == "missing":
        archives[0].pop()
    elif change == "duplicate":
        archives[0].append(archives[0][0])
    elif change == "model":
        records[0]["model"]["config"]["identity"] = "different-policy"
    elif change == "protocol":
        records[0]["eval"]["evalchemy"]["shots"] = 0
    else:
        archives[0][0]["metrics"] = []
    with pytest.raises(ValueError):
        coding_evidence_rows(records, archives, "parent", panel)
