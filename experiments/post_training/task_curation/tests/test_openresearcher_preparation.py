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
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
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
    curate,
    prepare,
    retained_questions,
    source,
)
from experiments.post_training.task_curation.datasets.openresearcher.questions import QuestionPool
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license, with_license


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
        {"id": "budget", "question": "Synthetic complex question?", "answer": "private C", "source_refs": "[]"},
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
            unique_contracts=3,
            unique_questions=3,
            conflicting_answer_questions=0,
            invalid_rows=0,
            tasks_parquet_sha256=parquet_sha,
        ).model_dump(mode="json"),
    )
    kept_sha = parquet("kept.parquet", rows[:2])
    decisions_sha = parquet(
        "decisions.parquet",
        [
            {
                "id": row["id"],
                "decision_json": json.dumps(
                    {
                        "id": row["id"],
                        "keep": row["id"] != "budget",
                        "status": "screened",
                        "contract_sha256": CONTRACT_SHA256,
                        "reason": "complete_requirements_exceed_sealed_budget" if row["id"] == "budget" else "pass",
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
        input_questions=3,
        kept_questions=2,
        dropped_questions=1,
        drop_reason_counts={"complete_requirements_exceed_sealed_budget": 1},
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
    retained = retained_questions(config)
    assert retained.rows == [rows[0]]
    assert [row["id"] for row in retained.heldout_exclusions] == ["benchmark"]
    stale_licenses = json.loads((tmp_path / "license-receipts.json").read_text())
    stale_licenses["OpenResearcher:OpenResearcher-Corpus"]["revision"] = "0" * 40
    stale_stage = tmp_path / "stale-corpus-license"
    stale_stage.mkdir()
    _, stale_licenses_sha = seal("stale-corpus-license/license-receipts.json", stale_licenses)
    stage = json.loads((tmp_path / "stage.json").read_text())
    stale_stage_path, stale_stage_sha = seal(
        "stale-corpus-license/stage.json",
        {**stage, "path": str(stale_stage), "license_receipts_sha256": stale_licenses_sha},
    )
    stale_config = replace(config, stage_receipt=stale_stage_path, stage_receipt_sha256=stale_stage_sha)
    # The corpus licence is ingest-time provenance (ruling 2026-11-27): a staged receipt
    # recorded at an older corpus revision still authorizes the graph — revisions inherit.
    curate(
        with_license(source(stale_config)),
        PipelineOptions(mode=SourceProcessingMode.QUICK, runtime=CampaignRuntime()),
    )
    foreign_licenses = json.loads((tmp_path / "license-receipts.json").read_text())
    foreign_licenses["OpenResearcher:OpenResearcher-Corpus"]["source_id"] = "OpenResearcher:Other-Corpus"
    foreign_stage = tmp_path / "foreign-corpus-license"
    foreign_stage.mkdir()
    _, foreign_licenses_sha = seal("foreign-corpus-license/license-receipts.json", foreign_licenses)
    foreign_stage_path, foreign_stage_sha = seal(
        "foreign-corpus-license/stage.json",
        {**stage, "path": str(foreign_stage), "license_receipts_sha256": foreign_licenses_sha},
    )
    foreign_config = replace(config, stage_receipt=foreign_stage_path, stage_receipt_sha256=foreign_stage_sha)
    # A receipt that belongs to another source never authorizes this graph.
    with pytest.raises(ValueError, match="belongs to another source: OpenResearcher:OpenResearcher-Corpus"):
        curate(
            with_license(source(foreign_config)),
            PipelineOptions(mode=SourceProcessingMode.QUICK, runtime=CampaignRuntime()),
        )
    assert not (tmp_path / "converted").exists()
    result = prepare(config)
    output = tmp_path / "converted"
    exported = pq.read_table(output / "tasks.parquet").to_pylist()
    assert len(exported) == 1
    actor = json.loads(exported[0]["task_spec"])["environment_requirements"]
    assert actor["docker_image"] == config.actor_image
    assert actor["tool_providers"]["retrieval"]["network_access"] == "service_only"
    normalized = pq.read_table(output / "normalize/part-00000.parquet").to_pylist()
    assert normalized[0]["task_json"] == exported[0]["task_spec"]
    assert json.loads((output / exported[0]["path"] / TASK_SPEC_FILE).read_text()) == json.loads(
        exported[0]["task_spec"]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert (manifest["input_rows"], manifest["converted_rows"], manifest["rejections"]) == (
        3,
        1,
        {"grounding_drop": 1, "benchmark_heldout": 1},
    )
    assert result.result.outputs["tasks"] == str(output / "tasks.parquet")
    _, changed_counts_sha = seal(
        "grounding.json", grounded.model_copy(update={"drop_reason_counts": {}}).model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="Grounding drop counts differ"):
        retained_questions(replace(config, grounding_receipt_sha256=changed_counts_sha))
    original_decisions = (tmp_path / "decisions.parquet").read_bytes()
    technical_decisions = pq.read_table(tmp_path / "decisions.parquet").to_pylist()
    decision = json.loads(technical_decisions[-1]["decision_json"])
    decision.update(status="technical_exclusion", reason="grounding_incomplete_exhausted")
    technical_decisions[-1]["decision_json"] = json.dumps(decision)
    technical_sha = parquet("decisions.parquet", technical_decisions)
    _, technical_ground_sha = seal(
        "grounding.json",
        grounded.model_copy(
            update={
                "decisions_parquet_sha256": technical_sha,
                "drop_reason_counts": {"grounding_incomplete_exhausted": 1},
            }
        ).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="exclusions exceed one percent"):
        retained_questions(replace(config, grounding_receipt_sha256=technical_ground_sha))
    (tmp_path / "decisions.parquet").write_bytes(original_decisions)
    # Even a freshly sealed parquet cannot change the source question's private answer.
    changed = [{**rows[0], "answer": "unreleased substitute"}, rows[1]]
    changed_sha = parquet("kept.parquet", changed)
    _, ground_sha = seal(
        "grounding.json", grounded.model_copy(update={"kept_parquet_sha256": changed_sha}).model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="retained original question contracts"):
        retained_questions(replace(config, grounding_receipt_sha256=ground_sha))
