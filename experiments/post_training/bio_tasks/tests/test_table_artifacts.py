# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract, grade_files


def test_intermediate_table_must_pass_even_when_final_summary_is_correct(tmp_path):
    contract = Contract(
        columns={"genes": Column(kind="integer", description="reported genes", unit="genes")},
        expected={"summary": {"genes": 2}},
        tables={
            "fit.tsv": TableContract(
                columns={
                    "effect": Column(kind="number", description="fitted effect", unit="log2", atol=1e-6),
                    "pvalue": Column(kind="number", description="test probability", unit="probability", nullable=True),
                },
                expected={"g1": {"effect": 2.5, "pvalue": 0.01}, "g2": {"effect": -1.0, "pvalue": None}},
                max_bytes=256,
            )
        },
    )
    reference, answer, artifact = (tmp_path / name for name in ("reference.json", "answer.json", "fit.tsv"))
    reference.write_text(contract.model_dump_json())
    answer.write_text(json.dumps([{"id": "summary", "genes": 2}]))
    assert grade_files(reference, answer).reward == 0
    valid = "id\teffect\tpvalue\ng1\t2.5\t0.01\ng2\t-1\tNA\n"
    artifact.write_text(valid)
    assert grade_files(reference, answer).reward == 1
    artifact.write_text("pvalue\tid\teffect\r\nNA\tg2\t-1.0\r\n1e-2\tg1\t2.5000001\r\n")
    assert grade_files(reference, answer).reward == 1
    for invalid in (
        valid.replace("2.5", "-2.5"),
        valid.replace("0.01", "0.1"),
        valid.replace("NA", "0"),
        valid.replace("2.5", "nan"),
        valid.replace("2.5", "inf"),
        valid.replace("g2\t-1\tNA\n", ""),
        valid + "g2\t-1\tNA\n",
        valid + "g3\t0\t1\n",
        valid.replace("pvalue", "effect"),
        valid.replace("2.5", "0" * 256),
        "",
    ):
        artifact.write_text(invalid)
        assert grade_files(reference, answer).reward == 0, invalid
    artifact.unlink()
    external = tmp_path / "external.tsv"
    external.write_text(valid)
    artifact.symlink_to(external)
    assert grade_files(reference, answer).reward == 0


@pytest.mark.parametrize(
    "answer_bytes,reason",
    [
        (b'[{"id":"genes","genes":2},{"id":"samples","samples":12}]', "record_schema"),
        (b"not JSON", "malformed_json"),
        (b"\xff", "malformed_utf8"),
        (None, "missing_or_nonregular_answer"),
    ],
)
@pytest.mark.parametrize("first_table,first_passed", [("id\tcount\ng1\t2\n", True), ("wrong_header\ng1\n", False)])
def test_invalid_summary_still_reports_every_artifact_without_awarding_reward(
    tmp_path, answer_bytes, reason, first_table, first_passed
):
    contract = Contract(
        columns={name: Column(kind="integer", description=name, unit=name) for name in ("genes", "samples")},
        expected={"comparison": {"genes": 2, "samples": 12}},
        tables={
            name: TableContract(
                columns={"count": Column(kind="integer", description="observed count", unit="reads")},
                expected={"g1": {"count": 2}},
                max_bytes=64,
            )
            for name in ("first.tsv", "second.tsv")
        },
    )
    reference, answer = tmp_path / "reference.json", tmp_path / "answer.json"
    reference.write_text(contract.model_dump_json())
    if answer_bytes is not None:
        answer.write_bytes(answer_bytes)
    (tmp_path / "first.tsv").write_text(first_table)
    (tmp_path / "second.tsv").write_text("id\tcount\ng1\t2\n")

    verdict = grade_files(reference, answer)
    assert verdict.reward == 0
    assert verdict.detail["reason"] == reason
    assert set(verdict.detail["artifact_checks"]) == {"first.tsv", "second.tsv"}
    assert verdict.detail["artifact_checks"]["first.tsv"]["passed"] is first_passed
    assert verdict.detail["artifact_checks"]["second.tsv"]["passed"]
