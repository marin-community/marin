# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed pilot sample from the sealed v12 grounding decisions.

Selects, under a recorded seed, 300 of the 4,135 ``required_hop`` drops (the
recovery target) plus 100 of the 283 kept questions (the regression target),
and binds their original question rows into a sha-sealed pilot artifact.
"""

import argparse
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    REGIONAL_PREFIX,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import GroundedQuestions
from experiments.post_training.task_curation.datasets.openresearcher.questions import (
    QUESTION_SCHEMA,
    QuestionPool,
)
from experiments.post_training.task_curation.local import artifact_prefix

PILOT_SEED = "openresearcher-hybrid-pilot-v1"
RECOVERY_DROP_REASON = "required_hop_not_grounded_or_citation_not_literal"
DROP_SAMPLE_SIZE = 300
KEPT_SAMPLE_SIZE = 100
EXPECTED_KEPT = 283
EXPECTED_RECOVERY_DROPS = 4_135
EXPECTED_INPUT_QUESTIONS = 6_102
logger = logging.getLogger(__name__)


class PilotSample(Artifact):
    v12_grounding_receipt_uri: str
    v12_grounding_receipt_sha256: str
    v12_decisions_parquet_sha256: str
    question_pool_receipt_sha256: str
    sample_seed: str
    drop_sample_ids: list[str]
    kept_sample_ids: list[str]
    drop_population: int
    kept_population: int
    input_questions: int
    questions_parquet_sha256: str
    selection_manifest_sha256: str
    iris_task_id: str


@dataclass(frozen=True)
class PilotSampleConfig:
    v12_receipt_uri: str
    v12_receipt_sha256: str
    v12_decisions_uri: str
    v12_decisions_sha256: str
    question_pool_uri: str
    question_pool_sha256: str
    output_path: str


def build_pilot_sample(config: PilotSampleConfig) -> PilotSample:
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Pilot sampling reads the regional grounding artifact inside Iris")
    v12 = GroundedQuestions.model_validate_json(pinned_document(config.v12_receipt_uri, config.v12_receipt_sha256))
    if (
        v12.input_questions != EXPECTED_INPUT_QUESTIONS
        or v12.kept_questions != EXPECTED_KEPT
        or v12.drop_reason_counts.get(RECOVERY_DROP_REASON) != EXPECTED_RECOVERY_DROPS
    ):
        raise ValueError("The v12 grounding receipt differs from the sealed pilot baseline")
    if v12.decisions_parquet_sha256 != config.v12_decisions_sha256:
        raise ValueError("Pinned decisions parquet differs from the sealed v12 receipt")
    pool = QuestionPool.model_validate_json(pinned_document(config.question_pool_uri, config.question_pool_sha256))
    decisions = StoragePath(config.v12_decisions_uri)
    if file_sha256(decisions) != config.v12_decisions_sha256:
        raise ValueError("Decisions parquet digest differs before sampling")
    with decisions.open("rb") as reader:
        rows = [{"id": row["id"], **json.loads(row["decision_json"])} for row in pq.read_table(reader).to_pylist()]
    if len(rows) != EXPECTED_INPUT_QUESTIONS:
        raise ValueError("Decisions parquet covers a different question population")
    kept = sorted(row["id"] for row in rows if row["keep"] and row["reason"] == "pass")
    drops = sorted(row["id"] for row in rows if not row["keep"] and row["reason"] == RECOVERY_DROP_REASON)
    if len(kept) != EXPECTED_KEPT or len(drops) != EXPECTED_RECOVERY_DROPS:
        raise ValueError("Decision counts differ from the sealed v12 receipt")
    sampler = random.Random(PILOT_SEED)
    drop_sample = sorted(sampler.sample(drops, DROP_SAMPLE_SIZE))
    kept_sample = sorted(sampler.sample(kept, KEPT_SAMPLE_SIZE))
    sample_ids = set(drop_sample) | set(kept_sample)
    if len(sample_ids) != DROP_SAMPLE_SIZE + KEPT_SAMPLE_SIZE:
        raise ValueError("Recovery and regression samples overlap")
    questions = StoragePath(pool.path) / "questions.parquet"
    if file_sha256(questions) != pool.tasks_parquet_sha256:
        raise ValueError("Question pool changed before pilot sampling")
    with questions.open("rb") as reader:
        pool_rows = pq.read_table(reader).to_pylist()
    selected = [row for row in pool_rows if row["id"] in sample_ids]
    if sorted(row["id"] for row in selected) != sorted(sample_ids):
        raise ValueError("Sampled question ids are not covered exactly once by the pool")
    output = StoragePath(config.output_path)
    with (output / "questions.parquet").open("wb", auto_mkdir=True) as writer:
        pq.write_table(pa.Table.from_pylist(selected, schema=QUESTION_SCHEMA), writer, compression="zstd")
    selection = {
        "sample_seed": PILOT_SEED,
        "v12_receipt_uri": config.v12_receipt_uri,
        "v12_receipt_sha256": config.v12_receipt_sha256,
        "v12_decisions_sha256": config.v12_decisions_sha256,
        "drop_sample_ids": drop_sample,
        "kept_sample_ids": kept_sample,
        "populations": {
            "input_questions": EXPECTED_INPUT_QUESTIONS,
            "recovery_drops": EXPECTED_RECOVERY_DROPS,
            "kept": EXPECTED_KEPT,
        },
        "v12_status_by_id": {
            row["id"]: {"keep": row["keep"], "reason": row["reason"]} for row in rows if row["id"] in sample_ids
        },
    }
    selection_json = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
    (output / "selection.json").write_bytes(selection_json)
    return PilotSample(
        path=config.output_path,
        v12_grounding_receipt_uri=config.v12_receipt_uri,
        v12_grounding_receipt_sha256=config.v12_receipt_sha256,
        v12_decisions_parquet_sha256=config.v12_decisions_sha256,
        question_pool_receipt_sha256=config.question_pool_sha256,
        sample_seed=PILOT_SEED,
        drop_sample_ids=drop_sample,
        kept_sample_ids=kept_sample,
        drop_population=len(drops),
        kept_population=len(kept),
        input_questions=len(rows),
        questions_parquet_sha256=file_sha256(output / "questions.parquet"),
        selection_manifest_sha256=hashlib.sha256(selection_json).hexdigest(),
        iris_task_id=os.environ["IRIS_TASK_ID"],
    )


def pilot_config(
    v12_uri: str,
    v12_sha256: str,
    decisions_uri: str,
    decisions_sha256: str,
    pool_uri: str,
    pool_sha256: str,
    context: StepContext,
) -> PilotSampleConfig:
    return PilotSampleConfig(
        v12_uri, v12_sha256, decisions_uri, decisions_sha256, pool_uri, pool_sha256, context.output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-receipt", required=True)
    parser.add_argument("--v12-receipt-sha256", required=True)
    parser.add_argument("--v12-decisions", required=True)
    parser.add_argument("--v12-decisions-sha256", required=True)
    parser.add_argument("--question-pool-receipt", required=True)
    parser.add_argument("--question-pool-receipt-sha256", required=True)
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
            name="rl-data/openresearcher/hybrid-pilot-sample",
            version=args.artifact_version,
            artifact_type=PilotSample,
            run=build_pilot_sample,
            build_config=partial(
                pilot_config,
                args.v12_receipt,
                args.v12_receipt_sha256,
                args.v12_decisions,
                args.v12_decisions_sha256,
                args.question_pool_receipt,
                args.question_pool_receipt_sha256,
            ),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
