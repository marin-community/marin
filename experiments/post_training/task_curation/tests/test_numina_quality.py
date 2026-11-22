# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the offline quality build and its input boundaries."""

import csv
import hashlib
import json
import sys

import pytest

from experiments.post_training.task_curation.datasets.lean import numina_quality as quality


def report_row(key, label="no"):
    formal = "theorem t (n : Nat) : n = n"
    reference = "For every natural n, n equals n."
    return {
        **dict.fromkeys(quality.FIELDS, ""),
        "numina_row": key,
        "row_key": key,
        "uuid": key,
        "split": "train",
        "question_type": "proof",
        "disclosure": label,
        "formal": formal,
        "reference": reference,
        "formal_sha256": hashlib.sha256(formal.encode()).hexdigest(),
        "included": True,
    }


def test_one_offline_command_builds_both_artifacts_and_filters_rejected_rows(tmp_path, monkeypatch):
    rows = [report_row(str(i)) for i in range(4)]
    rows[-1].update(included=False, cause="answer_disclosed", disclosure="yes")
    labels = tmp_path / "labels.jsonl"
    golden = tmp_path / "golden.jsonl"
    quality.write_records(labels, [dict(row=r["row_key"], label=r["disclosure"]) for r in rows])
    records = [
        dict(
            row_key=r["row_key"],
            golden_control=verdict,
            note="Test verdict",
            judge_model="offline",
            formal_sha256=r["formal_sha256"],
            judge_spec_sha256=quality.spec_digest(
                quality.golden_spec(r["formal"], r["reference"], 2048 if i == 1 else 1024)
            ),
        )
        for i, (r, verdict) in enumerate(zip(rows[:3], ("PASS", "FAIL", "MALFORMED"), strict=True))
    ]
    quality.write_records(golden, records)
    monkeypatch.setattr(quality, "source_rows", lambda *_: [dict(r) for r in rows])
    out = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "numina_quality",
            "source.parquet",
            "--prime-lists",
            str(tmp_path),
            "--labels",
            str(labels),
            "--golden",
            str(golden),
            "--output-dir",
            str(out),
        ],
    )
    quality.main()
    manifest = json.loads((out / "numina_autoinformalization_tasks.json").read_text())
    assert manifest["accepted"] == ["0"]
    assert manifest["task_counts"] == {"train": 1, "eval": 0}
    with (out / "numina_row_quality_report.csv").open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert [r["included"] for r in result] == ["True", "False", "False", "False"]
    assert [r["cause"] for r in result] == ["", "golden_failed", "judge_malformed", "answer_disclosed"]
    before = (out / "numina_row_quality_report.csv").read_bytes()
    quality.write_records(golden, records[:-1])
    with pytest.raises(ValueError, match="Missing golden"):
        quality.main()
    assert (out / "numina_row_quality_report.csv").read_bytes() == before
    records[-1]["formal_sha256"] = "changed"
    quality.write_records(golden, records)
    with pytest.raises(ValueError, match="different formal"):
        quality.main()


@pytest.mark.parametrize("kind", ["MCQ", "math-word-problem"])
def test_type_exclusion_precedes_the_frozen_label(monkeypatch, kind):
    row = {
        "author": "human",
        "uuid": "example",
        "formal_statement": "theorem t : True",
        "problem": "Find n.",
        "question_type": kind,
    }
    monkeypatch.setattr(quality.auto, "ACCEPTED_ROWS", {quality.row_key(row)})
    assert not quality.auto.is_train_row(row, None)
    assert not quality.auto.is_eval_row(row, None)


def test_reviewer_csv_selects_only_passes_and_preserves_provenance(tmp_path, monkeypatch):
    rows = [report_row(str(i)) for i in range(3)]
    monkeypatch.setattr(quality, "source_rows", lambda *_: [dict(r) for r in rows])
    labels = tmp_path / "labels.jsonl"
    quality.write_records(labels, [dict(row=r["row_key"], label="no") for r in rows])
    golden = tmp_path / "review.csv"
    fields = ("numina_row", "uuid", "golden_control", "cause", "note")
    records = [
        dict(
            numina_row=r["numina_row"],
            uuid=r["uuid"],
            golden_control=label,
            cause="judge_wrong" if label == "FAIL" else "",
            note="Reviewer note",
        )
        for r, label in zip(rows, ("PASS", "FAIL", "MALFORMED"), strict=True)
    ]

    def write_csv():
        with golden.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(records)

    write_csv()
    output = tmp_path / "out"
    manifest = quality.build(tmp_path / "raw", tmp_path, labels, golden, output)
    assert manifest["accepted"] == ["0"]
    assert manifest["golden_format"] == "reviewer_csv"
    assert manifest["judge_models"] == []
    with (output / "numina_row_quality_report.csv").open(newline="") as stream:
        report = list(csv.DictReader(stream))
    assert report[1]["golden_cause"] == "judge_wrong"
    assert report[1]["included"] == "False"
    assert report[0]["judge_spec_sha256"] == ""
    records[0]["uuid"] = "wrong-source"
    write_csv()
    with pytest.raises(ValueError, match="Unknown golden CSV"):
        quality.build(tmp_path / "raw", tmp_path, labels, golden, output)


def test_intersection_excludes_rows_rejected_by_either_control(tmp_path, monkeypatch):
    rows = [report_row(str(i)) for i in range(4)]
    monkeypatch.setattr(quality, "source_rows", lambda *_: [dict(r) for r in rows])
    labels = tmp_path / "labels.jsonl"
    quality.write_records(labels, [dict(row=r["row_key"], label="no") for r in rows])
    files = []
    for name, verdicts in [
        ("first", ("PASS", "PASS", "FAIL", "PASS")),
        ("second", ("PASS", "FAIL", "PASS", "MALFORMED")),
    ]:
        path = tmp_path / (name + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ("numina_row", "uuid", "golden_control", "cause", "note"))
            writer.writeheader()
            writer.writerows(
                dict(numina_row=r["numina_row"], uuid=r["uuid"], golden_control=v, cause="", note=name + " explanation")
                for r, v in zip(rows, verdicts, strict=True)
            )
        files.append(path)
    output = tmp_path / "out"
    result = quality.build_intersection(tmp_path / "raw", tmp_path, labels, *files, output)
    assert result["accepted"] == ["0"]
    assert len(result["golden_inputs"]) == 2
    with (output / "numina_row_quality_report.csv").open() as stream:
        report = list(csv.DictReader(stream))
    assert [(r["primary_verdict"], r["required_verdict"], r["included"]) for r in report] == [
        ("PASS", "PASS", "True"),
        ("PASS", "FAIL", "False"),
        ("FAIL", "PASS", "False"),
        ("PASS", "MALFORMED", "False"),
    ]
    assert report[0]["note"] == "first explanation"
    assert report[0]["required_note"] == "second explanation"
