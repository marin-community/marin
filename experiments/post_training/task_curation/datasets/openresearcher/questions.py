# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure unique released question-answer contracts independently of synthesis seeds."""

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    DATASET,
    DATASET_REVISION,
    REGIONAL_PREFIX,
    RegionalInputs,
    pinned_document,
)
from experiments.post_training.task_curation.local import artifact_prefix, local_client

EXPECTED_TRAJECTORIES = 97_630
QUESTION_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("question", pa.large_string()),
        ("answer", pa.large_string()),
        ("source_refs", pa.large_string()),
    ]
)


class QuestionPool(Artifact):
    source_revision: str
    stage_receipt_sha256: str
    raw_trajectories: int
    unique_contracts: int
    unique_questions: int
    conflicting_answer_questions: int
    invalid_rows: int
    tasks_parquet_sha256: str


@dataclass(frozen=True)
class QuestionConfig:
    stage_receipt_uri: str
    stage_receipt_sha256: str
    output_path: str


def extract_questions(config: QuestionConfig) -> QuestionPool:
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Read the full trajectory release inside the regional Iris CPU job")
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt_uri, config.stage_receipt_sha256))
    stage = StoragePath(staged.path)
    manifest = json.loads(pinned_document(str(stage / "source-manifest.json"), staged.source_manifest_sha256))
    asset = next(asset for asset in manifest if asset["dataset_id"] == DATASET)
    if asset["revision"] != DATASET_REVISION:
        raise ValueError("Question input differs from the approved release")
    source = StoragePath(staged.source_roots[DATASET])
    contracts, question_answers, invalid = {}, {}, []
    count = 0
    for file in asset["files"]:
        with (source / file["path"]).open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            index = 0
            for batch in parquet.iter_batches(batch_size=256, columns=["qid", "question", "answer"]):
                for row in batch.to_pylist():
                    count += 1
                    reference = {"file": file["path"], "row": index, "qid": row["qid"]}
                    index += 1
                    question, answer = row["question"], row["answer"]
                    if (
                        not isinstance(question, str)
                        or not question.strip()
                        or not isinstance(answer, str)
                        or not answer.strip()
                    ):
                        invalid.append({**reference, "reason": "missing_question_or_gold_answer"})
                        continue
                    key = canonical_sha256({"question": question, "answer": answer})
                    if key not in contracts:
                        contracts[key] = {"id": key, "question": question, "answer": answer, "source_refs": []}
                    contracts[key]["source_refs"].append(reference)
                    question_answers.setdefault(hashlib.sha256(question.encode()).hexdigest(), set()).add(answer)
    if count != EXPECTED_TRAJECTORIES:
        raise ValueError(f"Release contains {count} trajectories; expected {EXPECTED_TRAJECTORIES}")
    rows = [
        {**contract, "source_refs": json.dumps(contract["source_refs"], sort_keys=True)}
        for _, contract in sorted(contracts.items())
    ]
    output = StoragePath(config.output_path)
    target = output / "questions.parquet"
    with target.open("wb", auto_mkdir=True) as writer:
        pq.write_table(pa.Table.from_pylist(rows, schema=QUESTION_SCHEMA), writer, compression="zstd")
    (output / "invalid-rows.json").write_text(json.dumps(invalid, indent=2) + "\n")
    conflicts = {key: len(answers) for key, answers in question_answers.items() if len(answers) > 1}
    (output / "conflicting-questions.json").write_text(json.dumps(conflicts, indent=2) + "\n")
    return QuestionPool(
        path=config.output_path,
        source_revision=DATASET_REVISION,
        stage_receipt_sha256=config.stage_receipt_sha256,
        raw_trajectories=count,
        unique_contracts=len(rows),
        unique_questions=len(question_answers),
        conflicting_answer_questions=len(conflicts),
        invalid_rows=len(invalid),
        tasks_parquet_sha256=file_sha256(target),
    )


def question_config(uri: str, sha256: str, context: StepContext) -> QuestionConfig:
    return QuestionConfig(uri, sha256, context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", required=True)
    parser.add_argument("--stage-receipt-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/question-pool",
            version=args.artifact_version,
            artifact_type=QuestionPool,
            run=extract_questions,
            build_config=partial(question_config, args.stage_receipt, args.stage_receipt_sha256),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        with local_client():
            result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))
