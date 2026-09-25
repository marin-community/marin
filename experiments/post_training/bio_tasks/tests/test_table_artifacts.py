# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.bio_tasks.contract import Column, Contract, PcaSignContract, TableContract, grade_files


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


@pytest.mark.parametrize(
    "score_sign,loading_sign,correlation_sign,passed",
    [(1, 1, 1, True), (-1, -1, -1, True), (-1, 1, 1, False), (1, -1, 1, False), (-1, -1, 1, False)],
)
def test_pca_component_sign_is_shared_by_scores_loadings_and_correlations(
    tmp_path, score_sign, loading_sign, correlation_sign, passed
):
    # An orthonormal two-gene loading basis with centered, orthogonal scores.
    expected = {
        "scores.tsv": {"c1": {"PC1": 2.0, "PC2": 1.0}, "c2": {"PC1": 0.0, "PC2": -2.0}, "c3": {"PC1": -2.0, "PC2": 1.0}},
        "loadings.tsv": {"g1": {"PC1": 0.8, "PC2": -0.6}, "g2": {"PC1": 0.6, "PC2": 0.8}},
        "diagnostics.tsv": {"PC1": {"depth_correlation": -0.5}, "PC2": {"depth_correlation": 0.2}},
    }
    contract = Contract(
        columns={"cells": Column(kind="integer", description="cells analyzed", unit="cells")},
        expected={"summary": {"cells": 3}},
        tables={
            name: TableContract(
                columns={
                    field: Column(kind="number", description=field, unit="dimensionless", atol=1e-8)
                    for field in next(iter(rows.values()))
                },
                expected=rows,
                max_bytes=2048,
            )
            for name, rows in expected.items()
        },
        pca_signs=PcaSignContract(
            score_table="scores.tsv",
            loading_table="loadings.tsv",
            components=["PC1", "PC2"],
            correlation_table="diagnostics.tsv",
            correlation_column="depth_correlation",
        ),
    )
    reference, answer = tmp_path / "reference.json", tmp_path / "answer.json"
    reference.write_text(contract.model_dump_json())
    answer.write_text(json.dumps(contract.answer()))
    for name, sign in (
        ("scores.tsv", score_sign),
        ("loadings.tsv", loading_sign),
        ("diagnostics.tsv", correlation_sign),
    ):
        rows = expected[name]
        fields = list(reversed(next(iter(rows.values()))))
        lines = ["id\t" + "\t".join(fields)]
        for key, row in reversed(list(rows.items())):
            # Flip only PC1, exercising independent sign choices per component.
            values = [str(row[field] * (sign if field == "PC1" or key == "PC1" else 1)) for field in fields]
            lines.append(key + "\t" + "\t".join(values))
        (tmp_path / name).write_text("\n".join(lines) + "\n")
    verdict = grade_files(reference, answer)
    assert verdict.reward == float(passed)
    assert all(check["passed"] for check in verdict.detail["checks"])
    assert set(verdict.detail["artifact_checks"]) == set(expected)

    (tmp_path / "loadings.tsv").unlink()
    verdict = grade_files(reference, answer)
    assert verdict.reward == 0
    assert set(verdict.detail["artifact_checks"]) == set(expected)
    assert not verdict.detail["artifact_checks"]["loadings.tsv"]["passed"]
