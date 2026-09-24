# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

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
