# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.task_curation.datasets.openresearcher import questions
from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    DATASET,
    DATASET_REVISION,
    RegionalInputs,
)


def test_seed_repetitions_collapse_but_conflicting_gold_answers_remain_visible(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"qid": 1, "question": "Synthetic question?", "answer": "one"},
                {"qid": 1, "question": "Synthetic question?", "answer": "one"},
                {"qid": 1, "question": "Synthetic question?", "answer": "two"},
            ]
        ),
        raw / "seed-fixture.parquet",
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    manifest = json.dumps(
        [{"dataset_id": DATASET, "revision": DATASET_REVISION, "files": [{"path": "seed-fixture.parquet"}]}]
    )
    (stage / "source-manifest.json").write_text(manifest)
    receipt = RegionalInputs(
        path=str(stage),
        source_roots={DATASET: str(raw)},
        source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        license_receipts_sha256="0" * 64,
        verified_files=1,
        verified_bytes=0,
        iris_task_id="synthetic-fixture",
    ).model_dump_json()
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(receipt)
    monkeypatch.setenv("IRIS_TASK_ID", "synthetic-fixture")
    monkeypatch.setattr(questions, "EXPECTED_TRAJECTORIES", 3)
    output = tmp_path / "output"
    result = questions.extract_questions(
        questions.QuestionConfig(str(receipt_file), hashlib.sha256(receipt.encode()).hexdigest(), str(output))
    )
    rows = pq.read_table(output / "questions.parquet").to_pylist()
    assert (
        result.raw_trajectories,
        result.unique_contracts,
        result.unique_questions,
        result.conflicting_answer_questions,
    ) == (3, 2, 1, 1)
    assert sorted(len(json.loads(row["source_refs"])) for row in rows) == [1, 2]
    assert {row["answer"] for row in rows} == {"one", "two"}
