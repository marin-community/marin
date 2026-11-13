# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source sampling, judge request policies, and classification labels."""

import csv

import pytest
from rigging.filesystem.storage_path import StoragePath
from verifyit.grade import Status
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate

from taskcompendium.importers.evalchemy.judge_prompts import financebench_spec, simpleqa_spec
from taskcompendium.importers.evalchemy.simpleqa import read_simpleqa
from taskcompendium.pipeline.inputs import ConversionContext


def write_simpleqa(file, rows):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metadata", "problem", "answer"))
        writer.writeheader()
        writer.writerows(
            {"problem": row["question"], "answer": row["answer"], "metadata": repr(row["metadata"])} for row in rows
        )


def test_simpleqa_mini_preserves_row_identity_across_repeated_reads(tmp_path):
    file = tmp_path / "simpleqa.csv"
    write_simpleqa(
        file,
        [
            {"question": f"Question {i}?", "answer": f"Answer {i}", "metadata": {"topic": "fixture", "id": i}}
            for i in range(600)
        ],
    )
    rows = list(read_simpleqa(StoragePath(str(file)), ConversionContext({}, None), mini=True))
    assert len(rows) == 500
    assert len({row["source_index"] for row in rows}) == 500
    assert all(
        row["question"] == f"Question {row['source_index']}?" and row["metadata"]["id"] == row["source_index"]
        for row in rows
    )
    assert list(read_simpleqa(StoragePath(str(file)), ConversionContext({}, None), mini=True)) == rows


@pytest.mark.parametrize(
    "spec_builder,reply,metric",
    [
        (financebench_spec, "CORRECT", "accuracy"),
        (financebench_spec, "INCORRECT", "incorrect"),
        (financebench_spec, "NOT_ATTEMPTED", "not_attempted"),
        (simpleqa_spec, "A", "accuracy"),
        (simpleqa_spec, "B", "incorrect"),
        (simpleqa_spec, "C", "not_attempted"),
    ],
)
def test_source_labels_grade_empty_and_exact_candidates_through_judge(
    judge_server, monkeypatch, spec_builder, reply, metric
):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    judge_server.reply = reply
    spec = spec_builder("How much revenue?", "$7")
    connection = JudgeConnection(
        f"http://127.0.0.1:{judge_server.server_port}/v1", "fixture-key", default_model="fixture-judge"
    )
    for candidate in ("", "$7"):
        verdict = grade_judge_candidate(spec, candidate, connection=connection)
        assert verdict.status is Status.SCORED
        assert verdict.reward == float(metric == "accuracy")
        assert verdict.detail["metrics"] == {
            name: float(name == metric) for name in ("accuracy", "incorrect", "not_attempted", "judge_failed")
        }
    assert len(judge_server.requests) == 2
    assert {request["model"] for request in judge_server.requests} == {spec.model or connection.default_model}
    prompt = judge_server.requests[-1]["messages"][-1]["content"]
    assert "How much revenue?" in prompt
    assert "$7" in prompt
