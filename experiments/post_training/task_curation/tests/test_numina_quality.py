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
