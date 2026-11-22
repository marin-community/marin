# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the offline quality build and its input boundaries."""

import copy
import csv
import hashlib
import json
import sys

import pytest

from experiments.post_training.task_curation.datasets.lean.autoinformalization import build_task_selection as quality

REPORT = "numina_autoinformalization_row_quality_report.csv"


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


def reviewer_records(rows, verdicts, note="Reviewer note"):
    return [
        dict(
            numina_row=r["numina_row"],
            uuid=r["uuid"],
            golden_control=verdict,
            cause="judge_wrong" if verdict == "FAIL" else "",
            note=note,
        )
        for r, verdict in zip(rows, verdicts, strict=True)
    ]


def gemini_records(rows, verdicts, note="Gemini note"):
    return [
        dict(
            row_key=r["row_key"],
            golden_control=verdict,
            note=note,
            formal_sha256=r["formal_sha256"],
            judge_model="offline",
            judge_spec_sha256=quality.spec_digest(
                quality.golden_spec(r["formal"], r["reference"], 2048 if i == 1 else 1024)
            ),
        )
        for i, (r, verdict) in enumerate(zip(rows, verdicts, strict=True))
    ]


def use_inputs(monkeypatch, rows, reviewer, gemini):
    inputs = {
        "labels": [dict(row_key=r["row_key"], label=r["disclosure"]) for r in rows],
        "reviewer": reviewer,
        "gemini": gemini,
    }
    monkeypatch.setattr(quality, "source_rows", lambda *_: [dict(r) for r in rows])
    monkeypatch.setattr(quality, "read_input", lambda name: copy.deepcopy(inputs[name]))
    return inputs


def read_report(output):
    with (output / REPORT).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_one_offline_command_builds_both_artifacts_and_filters_rejected_rows(tmp_path, monkeypatch):
    rows = [report_row(str(i)) for i in range(4)]
    rows[-1].update(included=False, cause="answer_disclosed", disclosure="yes")
    gemini = gemini_records(rows[:3], ("PASS", "FAIL", "MALFORMED"))
    inputs = use_inputs(monkeypatch, rows, reviewer_records(rows[:3], ("PASS",) * 3), gemini)
    out = tmp_path / "output"
    argv = ["quality", "source.parquet", "--prime-lists", str(tmp_path), "--output-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    quality.main()
    manifest = json.loads((out / "tasks.json").read_text())
    assert manifest["tasks"] == {"train": ["0"], "eval": []}
    result = read_report(out)
    assert [r["included"] for r in result] == ["True", "False", "False", "False"]
    assert [r["cause"] for r in result] == ["", "golden_failed", "judge_malformed", "answer_disclosed"]
    before = (out / REPORT).read_bytes()
    inputs["gemini"] = gemini[:-1]
    with pytest.raises(ValueError, match="Missing golden"):
        quality.main()
    assert (out / REPORT).read_bytes() == before
    inputs["gemini"] = copy.deepcopy(gemini)
    inputs["gemini"][-1]["formal_sha256"] = "changed"
    with pytest.raises(ValueError, match="different formal"):
        quality.main()


def test_reviewer_verdicts_keep_their_provenance(tmp_path, monkeypatch):
    rows = [report_row(str(i)) for i in range(3)]
    reviewer = reviewer_records(rows, ("PASS", "FAIL", "MALFORMED"))
    inputs = use_inputs(monkeypatch, rows, reviewer, gemini_records(rows, ("PASS",) * 3))
    output = tmp_path / "out"
    manifest = quality.build(tmp_path / "raw", tmp_path, output)
    assert manifest["tasks"] == {"train": ["0"], "eval": []}
    assert manifest["golden_format"] == "intersection"
    assert manifest["judge_models"] == ["offline"]
    report = read_report(output)
    assert report[1]["golden_cause"] == "judge_wrong"
    assert report[1]["included"] == "False"
    assert report[0]["judge_spec_sha256"] == ""
    inputs["reviewer"][0]["uuid"] = "wrong-source"
    with pytest.raises(ValueError, match="Unknown golden CSV"):
        quality.build(tmp_path / "raw", tmp_path, output)


def test_intersection_excludes_rows_rejected_by_either_control(tmp_path, monkeypatch):
    rows = [report_row(str(i)) for i in range(4)]
    use_inputs(
        monkeypatch,
        rows,
        reviewer_records(rows, ("PASS", "PASS", "FAIL", "PASS"), note="first explanation"),
        gemini_records(rows, ("PASS", "FAIL", "PASS", "MALFORMED"), note="second explanation"),
    )
    output = tmp_path / "out"
    result = quality.build(tmp_path / "raw", tmp_path, output)
    assert result["tasks"] == {"train": ["0"], "eval": []}
    assert len(result["golden_inputs"]) == 2
    report = read_report(output)
    assert [(r["primary_verdict"], r["required_verdict"], r["included"]) for r in report] == [
        ("PASS", "PASS", "True"),
        ("PASS", "FAIL", "False"),
        ("FAIL", "PASS", "False"),
        ("PASS", "MALFORMED", "False"),
    ]
    assert report[0]["note"] == "first explanation"
    assert report[0]["required_note"] == "second explanation"


def test_bucket_inputs_must_match_their_pinned_digest(tmp_path, monkeypatch):
    data = b"row_key,label\n0,no\n"
    (tmp_path / "labels.csv").write_bytes(data)
    monkeypatch.setattr(quality, "BUCKET", str(tmp_path))
    monkeypatch.setattr(quality, "INPUTS", {"labels": ("labels.csv", hashlib.sha256(data).hexdigest())})
    assert quality.read_input("labels") == [{"row_key": "0", "label": "no"}]
    monkeypatch.setattr(quality, "INPUTS", {"labels": ("labels.csv", hashlib.sha256(b"other").hexdigest())})
    with pytest.raises(ValueError, match="pinned SHA-256"):
        quality.read_input("labels")
