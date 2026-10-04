# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingPanel,
    coding_evidence_rows,
    protocol_digest,
)
from experiments.post_training.russell_rsi.heldout_evaluation import heldout_rows, paired_scores
from experiments.post_training.russell_rsi.test_coding_eval_feedback import panel_fixture


def final_panel_fixture():
    records, archives, working = panel_fixture()
    for record in records:
        record["eval"]["tasks"] = [{"benchmark": {"n_attempted": 32}}]
        record["eval"]["evalchemy"]["max_eval_instances"] = 32
        record["provenance"]["eval_runtime"] = "evalchemy @ git+https://example.invalid/evalchemy@revision"
    working = CodingPanel(working.items, {record["eval"]["name"]: protocol_digest(record) for record in records})
    for record, rows in zip(records, archives, strict=True):
        record["eval"]["evalchemy"]["max_eval_instances"] = 64
        record["eval"]["tasks"][0]["benchmark"]["n_attempted"] = 64
        for index in range(32, 64):
            row = deepcopy(rows[index - 32])
            row.update(
                doc_id=str(index + 100),
                doc=json.dumps({"task_id": index}),
                prompt_text=f"{record['eval']['name']} final prompt {index}",
                metrics=[("pass_rate", float(index < 48))],
            )
            rows.append(row)
    manifest = {
        "evalchemy_commit": "revision",
        "suites": {record["eval"]["name"]: {"sample_ids": list(range(32, 64))} for record in records},
    }
    return records, archives, working, manifest


def test_final_report_pairs_reordered_rows_and_excludes_working_scores():
    records, archives, working, manifest = final_panel_fixture()
    selected, heldout = heldout_rows(records, archives, working, manifest)
    parent = coding_evidence_rows(records, selected, "parent", heldout)
    candidate_records, candidate_archives = deepcopy(records), deepcopy(archives)
    for record, rows in zip(candidate_records, candidate_archives, strict=True):
        record["model"]["config"]["identity"] = "champion"
        for index in range(32):
            rows[index]["metrics"] = [("pass_rate", 0.0)]
        rows[32]["metrics"] = [("pass_rate", 0.0)]
        rows[48]["metrics"] = [("pass_rate", 1.0)]
        rows[49]["metrics"] = [("pass_rate", 1.0)]
        rows.reverse()
    selected, _ = heldout_rows(candidate_records[::-1], candidate_archives[::-1], working, manifest)
    champion = coding_evidence_rows(candidate_records[::-1], selected, "champion", heldout)
    report = paired_scores(parent, champion)
    for score in report.values():
        assert score["count"] == 32
        assert (score["parent_correct"], score["champion_correct"]) == (16, 17)
        assert (score["gains"], score["losses"], score["both_correct"], score["both_incorrect"]) == (2, 1, 15, 14)
        assert score["difference_pp"] == 3.125
        assert score["ci95_pp"][0] <= score["difference_pp"] <= score["ci95_pp"][1]
        assert {int(row["benchmark_id"]) for row in score["outcomes"]} == set(range(32, 64))


@pytest.mark.parametrize("change", ["missing", "duplicate", "protocol", "working_prompt", "heldout_prompt"])
def test_final_comparison_rejects_incomplete_or_unmatched_archives(change):
    records, archives, working, manifest = final_panel_fixture()
    selected, panel = heldout_rows(records, archives, working, manifest)
    coding_evidence_rows(records, selected, "parent", panel)
    if change == "missing":
        archives[0].pop()
    elif change == "duplicate":
        archives[0].append(deepcopy(archives[0][0]))
    elif change == "protocol":
        records[0]["model"]["config"]["generation"]["max_gen_toks"] = 1
    elif change == "working_prompt":
        archives[0][0]["prompt_text"] += " changed"
    else:
        archives[0][32]["prompt_text"] += " changed"
    with pytest.raises(ValueError):
        selected, _ = heldout_rows(records, archives, working, manifest)
        coding_evidence_rows(records, selected, "parent", panel)
