# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.harbor.adapter import TASK_SPEC_FILE
from taskcompendium.identity import canonical_sha256
from taskcompendium.openresearcher import OpenResearcherState

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS,
    CORPUS_REVISION,
    DATASET,
    DATASET_REVISION,
    RegionalInputs,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import CONTRACT_SHA256
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import (
    MODEL_REVISION,
    GroundedQuestions,
)
from experiments.post_training.task_curation.datasets.openresearcher.prepare import (
    PreparationConfig,
    prepare,
    retained_questions,
)
from experiments.post_training.task_curation.datasets.openresearcher.questions import QuestionPool
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license


def test_retained_population_excludes_benchmarks_and_rejects_rewritten_gold(tmp_path):
    def seal(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    def parquet(name, rows):
        path = tmp_path / name
        pq.write_table(pa.Table.from_pylist(rows), path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    rows = [
        {"id": "research", "question": "Synthetic research question?", "answer": "private A", "source_refs": "[]"},
        {"id": "benchmark", "question": "  Heldout\nQuestion? ", "answer": "private B", "source_refs": "[]"},
    ]
    for row in rows:
        row["source_refs"] = json.dumps([{"file": "seed_fixture/train.parquet", "row": 7, "qid": "fixture"}])
    _, licenses_sha = seal(
        "license-receipts.json",
        {
            "OpenResearcher:"
            + dataset.split("/")[1]: (
                fixture_license("OpenResearcher:" + dataset.split("/")[1], dataset, revision).model_dump(mode="json")
            )
            for dataset, revision in ((DATASET, DATASET_REVISION), (CORPUS, CORPUS_REVISION))
        },
    )
    stage_path, stage_sha = seal(
        "stage.json",
        RegionalInputs(
            path=str(tmp_path),
            source_roots={},
            source_manifest_sha256="a" * 64,
            license_receipts_sha256=licenses_sha,
            verified_files=0,
            verified_bytes=0,
            iris_task_id="synthetic",
        ).model_dump(mode="json"),
    )
    parquet_sha = parquet("questions.parquet", rows)
    pool_path, pool_sha = seal(
        "pool.json",
        QuestionPool(
            path=str(tmp_path),
            source_revision=DATASET_REVISION,
            stage_receipt_sha256=stage_sha,
            raw_trajectories=4,
            unique_contracts=2,
            unique_questions=2,
            conflicting_answer_questions=0,
            invalid_rows=0,
            tasks_parquet_sha256=parquet_sha,
        ).model_dump(mode="json"),
    )
    kept_sha = parquet("kept.parquet", rows)
    decisions_sha = parquet(
        "decisions.parquet",
        [
            {
                "id": row["id"],
                "decision_json": json.dumps(
                    {
                        "id": row["id"],
                        "keep": True,
                        "status": "screened",
                        "contract_sha256": CONTRACT_SHA256,
                    }
                ),
            }
            for row in rows
        ],
    )
    seal("backend-identity.json", {"backend": "synthetic"})
    grounded = GroundedQuestions(
        path=str(tmp_path),
        question_receipt_sha256=pool_sha,
        index_receipt_sha256="b" * 64,
        contract_sha256=CONTRACT_SHA256,
        index_manifest_sha256="c" * 64,
        model_revision=MODEL_REVISION,
        backend_identity_sha256=canonical_sha256({"backend": "synthetic"}),
        input_questions=2,
        kept_questions=2,
        dropped_questions=0,
        execution_failures=0,
        kept_parquet_sha256=kept_sha,
        decisions_parquet_sha256=decisions_sha,
    )
    ground_path, ground_sha = seal("grounding.json", grounded.model_dump(mode="json"))
    heldout_path, heldout_sha = seal(
        "heldout.json",
        {
            "normalizer": "whitespace-collapse-then-unicode-casefold-v1",
            "questions": {hashlib.sha256(b"heldout question?").hexdigest(): {"source": "synthetic benchmark"}},
        },
    )
    config = PreparationConfig(
        stage_receipt=stage_path,
        stage_receipt_sha256=stage_sha,
        question_receipt=pool_path,
        question_receipt_sha256=pool_sha,
        grounding_receipt=ground_path,
        grounding_receipt_sha256=ground_sha,
        heldout_hashes=heldout_path,
        heldout_hashes_sha256=heldout_sha,
        actor_image="ghcr.io/example/research@sha256:" + "f" * 64,
        grader_image="ghcr.io/example/grader@sha256:" + "e" * 64,
        state=OpenResearcherState(
            implementation_revision="d" * 40,
            corpus_revision=CORPUS_REVISION,
            index_manifest_sha256="c" * 64,
            tokenizer_revision="e" * 40,
            top_k=5,
            snippet_max_tokens=512,
            include_get_document=True,
        ),
        workdir=str(tmp_path / "work"),
        output_path=str(tmp_path / "converted"),
    )
    retained, _, excluded = retained_questions(config)
    assert retained == [rows[0]]
    assert [row["id"] for row in excluded] == ["benchmark"]
    result = prepare(config)
    output = tmp_path / "converted"
    exported = pq.read_table(output / "tasks.parquet").to_pylist()
    assert len(exported) == 1
    normalized = pq.read_table(output / "normalize/part-00000.parquet").to_pylist()
    assert normalized[0]["task_json"] == exported[0]["task_spec"]
    assert json.loads((output / exported[0]["path"] / TASK_SPEC_FILE).read_text()) == json.loads(
        exported[0]["task_spec"]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert (manifest["input_rows"], manifest["converted_rows"], manifest["rejections"]) == (
        2,
        1,
        {"grounding_drop": 0, "benchmark_heldout": 1},
    )
    assert result.result.outputs["tasks"] == str(output / "tasks.parquet")
    # Even a freshly sealed parquet cannot change the source question's private answer.
    changed = [{**rows[0], "answer": "unreleased substitute"}, rows[1]]
    changed_sha = parquet("kept.parquet", changed)
    _, ground_sha = seal(
        "grounding.json", grounded.model_copy(update={"kept_parquet_sha256": changed_sha}).model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="retained original question contracts"):
        retained_questions(replace(config, grounding_receipt_sha256=ground_sha))
