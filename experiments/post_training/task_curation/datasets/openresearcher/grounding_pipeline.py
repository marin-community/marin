# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind full question screening to immutable regional index, source and judge inputs."""

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePosixPath

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
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS_REVISION,
    DATASET_REVISION,
    REGIONAL_PREFIX,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding import screen_population
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import CONTRACT, CONTRACT_SHA256
from experiments.post_training.task_curation.datasets.openresearcher.index import BM25Index
from experiments.post_training.task_curation.datasets.openresearcher.questions import QUESTION_SCHEMA, QuestionPool
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, LocalJudgeModel, PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.local import artifact_prefix

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
API_MODEL = "openresearcher-grounding-qwen3-30b-a3b"
MODEL_CONFIG = Path(__file__).with_name("grounding-model.yaml")
WORKER_TRANSPORT_ENV = {"MARIN_TEMP_PREFIX": "s3://marin-us-east-02a"}


class GroundedQuestions(Artifact):
    question_receipt_sha256: str
    index_receipt_sha256: str
    contract_sha256: str
    index_manifest_sha256: str
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
class GroundingConfig:
    question_receipt: str
    question_receipt_sha256: str
    index_receipt: str
    index_receipt_sha256: str
    contract_sha256: str
    model_config_sha256: str
    runtime_freeze: str
    runtime_freeze_sha256: str
    runtime_python: str
    workdir: str
    concurrency: int
    output_path: str


def local_index(index: BM25Index, directory: Path) -> None:
    """Materialize only the receipt's digest-verified index files in the same region."""
    manifest = json.loads(
        pinned_document(str(StoragePath(index.path) / "index-manifest.json"), index.index_manifest_sha256)
    )
    directory.mkdir(parents=True, exist_ok=False)
    for item in manifest:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Index manifest contains a path outside the declared index")
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with (StoragePath(index.path) / "bm25" / item["path"]).open("rb") as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if target.stat().st_size != item["bytes"] or file_sha256(StoragePath(str(target))) != item["sha256"]:
            raise ValueError(f"Staged index file differs from the sealed manifest: {relative}")


async def retrieval_response(reader: asyncio.StreamReader) -> dict:
    """Read a complete JSON response even when raw documents exceed the stream limit."""
    payload = bytearray()
    while True:
        try:
            payload.extend(await reader.readuntil(b"\n"))
            return json.loads(payload)
        except asyncio.LimitOverrunError as error:
            payload.extend(await reader.readexactly(error.consumed))


async def grounded_rows(
    config: GroundingConfig,
    rows: list[dict],
    connection: JudgeConnection,
    tokenizer: PreTrainedTokenizerBase,
    output: StoragePath,
) -> list[dict]:
    process = await asyncio.create_subprocess_exec(
        config.runtime_python,
        "-m",
        "experiments.post_training.task_curation.datasets.openresearcher.search_server",
        "--index",
        str(Path(config.workdir) / "bm25"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        limit=16 * 1024 * 1024,
    )
    lock = asyncio.Lock()
    assert process.stdin is not None and process.stdout is not None

    async def retrieve(queries: tuple[str, str]) -> dict[str, str]:
        async with lock:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write((json.dumps(queries) + "\n").encode())
            await process.stdin.drain()
            result = await retrieval_response(process.stdout)
            (output / "retrieval" / (canonical_sha256(queries) + ".json")).write_text(json.dumps(result))
            return result["documents"]

    try:
        return await screen_population(rows, connection, tokenizer, retrieve, output, config.concurrency)
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except TimeoutError:
            process.kill()
            await process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Grounding retrieval process failed: exit={process.returncode}")


def ground_questions(config: GroundingConfig) -> GroundedQuestions:
    info = get_job_info()
    if info is None:
        raise RuntimeError("Screen the complete population inside the regional Iris CPU job")
    if (
        config.contract_sha256 != CONTRACT_SHA256
        or file_sha256(StoragePath(str(MODEL_CONFIG))) != config.model_config_sha256
    ):
        raise ValueError("Grounding matcher or serving contract changed after planning")
    frozen_runtime = pinned_document(config.runtime_freeze, config.runtime_freeze_sha256)
    actual_runtime = subprocess.check_output(["uv", "pip", "freeze", "--python", config.runtime_python])
    if actual_runtime != frozen_runtime:
        raise ValueError("Grounding retrieval runtime differs from the sealed index runtime")
    pool = QuestionPool.model_validate_json(pinned_document(config.question_receipt, config.question_receipt_sha256))
    index = BM25Index.model_validate_json(pinned_document(config.index_receipt, config.index_receipt_sha256))
    if (
        index.content_mismatches
        or index.document_count != 14_878_084
        or index.corpus_revision != CORPUS_REVISION
        or pool.source_revision != DATASET_REVISION
        or pool.stage_receipt_sha256 != index.stage_receipt_sha256
    ):
        raise ValueError("Grounding requires full index/corpus agreement")
    source = StoragePath(pool.path) / "questions.parquet"
    if file_sha256(source) != pool.tasks_parquet_sha256:
        raise ValueError("Question pool changed before grounding")
    with source.open("rb") as reader:
        rows = pq.read_table(reader).to_pylist()
    if len(rows) != pool.unique_contracts or pool.conflicting_answer_questions or pool.invalid_rows:
        raise ValueError("Question pool has unresolved contract defects")
    local_index(index, Path(config.workdir) / "bm25")
    output = StoragePath(config.output_path)
    (output / "contract.json").write_text(json.dumps(CONTRACT, indent=2))
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
            "parent_task": str(info.task_id),
            "runtime_reported_region": info.worker_region,
            "placement_requirement": "cw-us-east-02a",
            "child_placement": "same_controller",
            "accelerator": {"type": "H100", "count": 1},
            "priority": "batch",
            "worker_transport_environment": WORKER_TRANSPORT_ENV,
            "worker_hf_hub_disable_xet": load_model_config(MODEL_CONFIG).serve.hf_hub_disable_xet,
        }
        (output / "backend-identity.json").write_text(json.dumps(identity, indent=2))
        decisions = asyncio.run(grounded_rows(config, rows, connection, tokenizer, output))
    if len(decisions) != len(rows) or any(
        decision["status"] not in {"screened", "technical_exclusion"} for decision in decisions
    ):
        raise ValueError("Grounding did not finish the complete frozen population")
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
    return GroundedQuestions(
        path=config.output_path,
        question_receipt_sha256=config.question_receipt_sha256,
        index_receipt_sha256=config.index_receipt_sha256,
        contract_sha256=CONTRACT_SHA256,
        index_manifest_sha256=index.index_manifest_sha256,
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


def ground_config(config: GroundingConfig, context: StepContext) -> GroundingConfig:
    return replace(config, output_path=context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "question-receipt",
        "question-receipt-sha256",
        "index-receipt",
        "index-receipt-sha256",
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
    config = GroundingConfig(
        args.question_receipt,
        args.question_receipt_sha256,
        args.index_receipt,
        args.index_receipt_sha256,
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
            name="rl-data/openresearcher/grounded-questions",
            version=args.artifact_version,
            artifact_type=GroundedQuestions,
            run=ground_questions,
            build_config=partial(ground_config, config),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint(), "contract": CONTRACT_SHA256}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))
