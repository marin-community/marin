# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge-variance control: re-ground the pilot sample with the pure BM25 v1 provider.

Runs the sealed screen over the SAME fixed sample with the SAME contract and
judge as the hybrid pilot, but retrieval stays the untouched BM25 v12 search
server. Differences against the hybrid pilot are therefore attributable to the
retriever, not to judge sampling.
"""

import argparse
import asyncio
import json
import logging
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from transformers import AutoTokenizer

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    REGIONAL_PREFIX,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import (
    CONTRACT_SHA256,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import (
    API_MODEL,
    MODEL,
    MODEL_CONFIG,
    MODEL_REVISION,
    WORKER_TRANSPORT_ENV,
    GroundingConfig,
    grounded_rows,
    local_index,
)
from experiments.post_training.task_curation.datasets.openresearcher.index import BM25Index
from experiments.post_training.task_curation.datasets.openresearcher.pilot_sample import PilotSample
from experiments.post_training.task_curation.datasets.openresearcher.questions import QUESTION_SCHEMA
from experiments.post_training.task_curation.judges import (
    LocalJudgeEndpoint,
    LocalJudgeModel,
    PinnedLocalJudgeEndpoint,
)
from experiments.post_training.task_curation.local import artifact_prefix

logger = logging.getLogger(__name__)


class ControlGroundedQuestions(Artifact):
    question_receipt_sha256: str
    contract_sha256: str
    model_revision: str
    backend_identity_sha256: str
    input_questions: int
    kept_questions: int
    dropped_questions: int
    drop_reason_counts: dict[str, int]
    execution_failures: int
    kept_parquet_sha256: str
    decisions_parquet_sha256: str


@dataclass(frozen=True)
class ControlGroundingConfig:
    question_receipt_uri: str
    question_receipt_sha256: str
    bm25_index_uri: str
    bm25_index_sha256: str
    contract_sha256: str
    model_config_sha256: str
    runtime_freeze_uri: str
    runtime_freeze_sha256: str
    runtime_python: str
    workdir: str
    concurrency: int
    output_path: str


def load_sample(config: ControlGroundingConfig) -> list[dict]:
    """Load and verify the pinned pilot-sample question rows."""
    sample = PilotSample.model_validate_json(
        pinned_document(config.question_receipt_uri, config.question_receipt_sha256)
    )
    questions = StoragePath(sample.path) / "questions.parquet"
    if file_sha256(questions) != sample.questions_parquet_sha256:
        raise ValueError("Pilot question parquet changed before grounding")
    with questions.open("rb") as reader:
        rows = pq.read_table(reader).to_pylist()
    expected = set(sample.drop_sample_ids) | set(sample.kept_sample_ids)
    if {row["id"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Pilot questions differ from the sealed selection")
    return rows


def control_ground_questions(config: ControlGroundingConfig) -> ControlGroundedQuestions:
    info = get_job_info()
    if info is None:
        raise RuntimeError("Screen the question population inside the regional Iris CPU job")
    if (
        config.contract_sha256 != CONTRACT_SHA256
        or file_sha256(StoragePath(str(MODEL_CONFIG))) != config.model_config_sha256
    ):
        raise ValueError("Grounding matcher or serving contract changed after planning")
    frozen_runtime = pinned_document(config.runtime_freeze_uri, config.runtime_freeze_sha256)
    actual_runtime = subprocess.check_output(["uv", "pip", "freeze", "--python", config.runtime_python])
    if actual_runtime != frozen_runtime:
        raise ValueError("Grounding retrieval runtime differs from the sealed BM25 runtime")
    rows = load_sample(config)
    bm25 = BM25Index.model_validate_json(pinned_document(config.bm25_index_uri, config.bm25_index_sha256))
    local_index(bm25, Path(config.workdir) / "bm25")
    output = StoragePath(config.output_path)
    grounding_config = GroundingConfig(
        question_receipt=config.question_receipt_uri,
        question_receipt_sha256=config.question_receipt_sha256,
        index_receipt=config.bm25_index_uri,
        index_receipt_sha256=config.bm25_index_sha256,
        contract_sha256=CONTRACT_SHA256,
        model_config_sha256=config.model_config_sha256,
        runtime_freeze_uri=config.runtime_freeze_uri,
        runtime_freeze_sha256=config.runtime_freeze_sha256,
        runtime_python=config.runtime_python,
        workdir=config.workdir,
        concurrency=config.concurrency,
        output_path="",
    )
    endpoint = PinnedLocalJudgeEndpoint(
        LocalJudgeEndpoint(
            tier=JudgeTier.LIGHT,
            model_config=MODEL_CONFIG,
            context_window_tokens=32768,
            accelerator=AcceleratorChoice(Platform.GPU, gpu_type="H100", gpu_count=1),
            priority=job_pb2.PRIORITY_BAND_BATCH,
            env_vars=WORKER_TRANSPORT_ENV,
            worker_timeout=Duration.from_seconds(259200),
        ),
        LocalJudgeModel(MODEL, MODEL_REVISION, API_MODEL, 32768),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    with endpoint.connect() as connection:
        identity = {
            **endpoint.identity,
            "serving_jobs": endpoint.job_ids,
            "contract_sha256": CONTRACT_SHA256,
            "control": "bm25-v1-on-pilot-sample",
            "parent_task": str(info.task_id),
            "runtime_reported_region": info.worker_region,
            "worker_transport_environment": WORKER_TRANSPORT_ENV,
            "worker_hf_hub_disable_xet": load_model_config(MODEL_CONFIG).serve.hf_hub_disable_xet,
        }
        (output / "backend-identity.json").write_text(json.dumps(identity, indent=2))
        decisions = asyncio.run(grounded_rows(grounding_config, rows, connection, tokenizer, output))
    if len(decisions) != len(rows) or any(
        decision["status"] not in {"screened", "technical_exclusion"} for decision in decisions
    ):
        raise ValueError("Grounding did not finish the complete question set")
    by_id = {row["id"]: row for row in rows}
    kept = [by_id[decision["id"]] for decision in decisions if decision["keep"]]
    target = output / "kept.parquet"
    with target.open("wb", auto_mkdir=True) as writer:
        pq.write_table(pa.Table.from_pylist(kept, schema=QUESTION_SCHEMA), writer, compression="zstd")
    decisions_path = output / "decisions.parquet"
    with decisions_path.open("wb", auto_mkdir=True) as writer:
        pq.write_table(
            pa.table({"id": [d["id"] for d in decisions], "decision_json": [json.dumps(d) for d in decisions]}),
            writer,
            compression="zstd",
        )
    (output / "summary.json").write_text(
        json.dumps(dict(Counter(decision["reason"] for decision in decisions)), indent=2)
    )
    return ControlGroundedQuestions(
        path=config.output_path,
        question_receipt_sha256=config.question_receipt_sha256,
        contract_sha256=CONTRACT_SHA256,
        model_revision=MODEL_REVISION,
        backend_identity_sha256=canonical_sha256(identity),
        input_questions=len(rows),
        kept_questions=len(kept),
        dropped_questions=len(rows) - len(kept),
        drop_reason_counts=dict(Counter(decision["reason"] for decision in decisions if not decision["keep"])),
        execution_failures=0,
        kept_parquet_sha256=file_sha256(target),
        decisions_parquet_sha256=file_sha256(decisions_path),
    )


def control_config(config: ControlGroundingConfig, context: StepContext) -> ControlGroundingConfig:
    return replace(config, output_path=context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "question-receipt",
        "question-receipt-sha256",
        "bm25-index-receipt",
        "bm25-index-receipt-sha256",
        "runtime-freeze",
        "runtime-freeze-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    config = ControlGroundingConfig(
        args.question_receipt,
        args.question_receipt_sha256,
        args.bm25_index_receipt,
        args.bm25_index_receipt_sha256,
        CONTRACT_SHA256,
        file_sha256(StoragePath(str(MODEL_CONFIG))),
        args.runtime_freeze,
        args.runtime_freeze_sha256,
        args.runtime_python,
        args.workdir,
        args.concurrency,
        "",
    )
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/control-grounded-questions",
            version=args.artifact_version,
            artifact_type=ControlGroundedQuestions,
            run=control_ground_questions,
            build_config=partial(control_config, config),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint(), "contract": CONTRACT_SHA256}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
