# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind question screening to the hybrid BM25+dense provider and pinned judge.

Mirrors the sealed v12 grounding run: same contract, same judge deployment,
same decision semantics. Only the retrieval provider changes, so every recovery
against v12 is attributable to retrieval quality.
"""

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
from transformers import AutoTokenizer
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    DATASET_REVISION,
    REGIONAL_PREFIX,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding import screen_population
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import CONTRACT, CONTRACT_SHA256
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import (
    API_MODEL,
    MODEL,
    MODEL_CONFIG,
    MODEL_REVISION,
    WORKER_TRANSPORT_ENV,
    local_index,
)
from experiments.post_training.task_curation.datasets.openresearcher.hybrid_index import HybridIndex
from experiments.post_training.task_curation.datasets.openresearcher.pilot_sample import PilotSample
from experiments.post_training.task_curation.datasets.openresearcher.questions import (
    QUESTION_SCHEMA,
    QuestionPool,
)
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, LocalJudgeModel, PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.local import artifact_prefix

QUESTION_KINDS = ("pool", "pilot_sample")
logger = logging.getLogger(__name__)


class HybridGroundedQuestions(Artifact):
    question_receipt_sha256: str
    question_kind: str
    hybrid_index_receipt_sha256: str
    provider_version: str
    bm25_index_manifest_sha256: str
    dense_index_manifest_sha256: str
    embedding_model: str
    embedding_model_revision: str
    implementation_revision: str
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
    # Present only for pilot runs over the fixed v12 sample.
    v12_recovery_sample: int | None
    v12_regression_sample: int | None
    recovered_questions: int | None
    regressions: int | None


@dataclass(frozen=True)
class HybridGroundingConfig:
    question_receipt_uri: str
    question_receipt_sha256: str
    question_kind: str
    hybrid_index_uri: str
    hybrid_index_sha256: str
    contract_sha256: str
    model_config_sha256: str
    runtime_freeze_uri: str
    runtime_freeze_sha256: str
    runtime_python: str
    implementation_revision: str
    workdir: str
    concurrency: int
    output_path: str


def local_dense(index: HybridIndex, work: Path) -> None:
    """Materialize the sealed dense index and ordered docids beside the BM25 index."""
    manifest = json.loads(
        pinned_document(
            str(StoragePath(index.dense_index.path) / "dense-manifest.json"),
            index.dense_index.index_manifest_sha256,
        )
    )
    directory = work / "dense"
    directory.mkdir(parents=True, exist_ok=False)
    for item in manifest:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dense manifest contains a path outside the declared index")
        source = StoragePath(index.dense_index.path) / item["path"]
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if target.stat().st_size != item["bytes"] or file_sha256(StoragePath(str(target))) != item["sha256"]:
            raise ValueError(f"Staged dense file differs from the sealed manifest: {relative}")


def local_model(index: HybridIndex, work: Path) -> None:
    """Materialize the pinned query encoder from the sealed embedding artifact."""
    manifest = json.loads(
        pinned_document(
            str(StoragePath(index.embedding.path) / "model-manifest.json"),
            index.embedding.model_manifest_sha256,
        )
    )
    directory = work / "model"
    directory.mkdir(parents=True, exist_ok=False)
    for item in manifest:
        source = StoragePath(index.embedding.path) / item["path"]
        target = directory / item["path"]
        with source.open("rb") as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if target.stat().st_size != item["bytes"] or file_sha256(StoragePath(str(target))) != item["sha256"]:
            raise ValueError(f"Staged model file differs from the sealed manifest: {item['path']}")


async def retrieval_response(reader: asyncio.StreamReader) -> dict:
    """Read a complete JSON response even when raw documents exceed the stream limit."""
    payload = bytearray()
    while True:
        try:
            payload.extend(await reader.readuntil(b"\n"))
            return json.loads(payload)
        except asyncio.LimitOverrunError as error:
            payload.extend(await reader.readexactly(error.consumed))


async def hybrid_grounded_rows(
    config: HybridGroundingConfig,
    rows: list[dict],
    connection: JudgeConnection,
    tokenizer,
    index: HybridIndex,
    output: StoragePath,
) -> list[dict]:
    work = Path(config.workdir)

    async def spawn_server() -> asyncio.subprocess.Process:
        server_log = open(work / "search-server.log", "ab")  # noqa: SIM115 - lives as long as the subprocess
        process = await asyncio.create_subprocess_exec(
            config.runtime_python,
            "-m",
            "experiments.post_training.task_curation.datasets.openresearcher.hybrid_search_server",
            "--bm25-index",
            str(work / "bm25"),
            "--dense-index",
            str(work / "dense" / "dense.faiss"),
            "--docids",
            str(work / "dense" / "dense-docids.parquet"),
            "--model-dir",
            str(work / "model"),
            "--candidates",
            str(index.fusion.candidate_depth),
            "--rrf-k",
            str(index.fusion.rrf_k),
            "--top-k",
            "5",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=server_log,
            limit=16 * 1024 * 1024,
        )
        server_log.close()
        return process

    process = await spawn_server()
    lock = asyncio.Lock()
    assert process.stdin is not None and process.stdout is not None
    retrieval_timeout = 900.0

    async def retrieve(queries: tuple[str, str]) -> dict[str, str]:
        async with lock:
            nonlocal process
            assert process.stdin is not None and process.stdout is not None
            request = (json.dumps(queries) + "\n").encode()
            try:
                process.stdin.write(request)
                await process.stdin.drain()
                result = await asyncio.wait_for(retrieval_response(process.stdout), timeout=retrieval_timeout)
            except (asyncio.TimeoutError, TimeoutError) as error:
                # A stalled server blocks every concurrent question; restart once
                # and replay the request rather than hanging the artifact.
                logger.warning("Retrieval server stalled on %s; restarting", canonical_sha256(queries))
                process.kill()
                await process.wait()
                (output / "search-server.log").write_text(
                    (work / "search-server.log").read_text(errors="replace")[-20000:]
                )
                process = await spawn_server()
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(request)
                await process.stdin.drain()
                result = await asyncio.wait_for(retrieval_response(process.stdout), timeout=retrieval_timeout)
            (output / "retrieval" / (canonical_sha256(queries) + ".json")).write_text(json.dumps(result))
            return result["documents"]

    try:
        return await screen_population(rows, connection, tokenizer, retrieve, output, config.concurrency)
    finally:
        server_log.close()
        (output / "search-server.log").write_text((work / "search-server.log").read_text(errors="replace")[-20000:])
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except TimeoutError:
            process.kill()
            await process.wait()
        if process.returncode != 0:
            tail = (work / "search-server.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(f"Grounding retrieval process failed: exit={process.returncode}; tail: {tail}")


def load_questions(config: HybridGroundingConfig, index: HybridIndex) -> list[dict]:
    """Load and verify the pinned question rows for a pool or pilot-sample run."""
    document = pinned_document(config.question_receipt_uri, config.question_receipt_sha256)
    if config.question_kind == "pool":
        pool = QuestionPool.model_validate_json(document)
        if (
            pool.source_revision != DATASET_REVISION
            or pool.stage_receipt_sha256 != index.bm25_index.stage_receipt_sha256
        ):
            raise ValueError("Question pool differs from the staged corpus the indexes seal")
        questions = StoragePath(pool.path) / "questions.parquet"
        if file_sha256(questions) != pool.tasks_parquet_sha256:
            raise ValueError("Question pool changed before grounding")
        with questions.open("rb") as reader:
            return pq.read_table(reader).to_pylist()
    if config.question_kind != "pilot_sample":
        raise ValueError(f"Unsupported question receipt kind: {config.question_kind}")
    sample = PilotSample.model_validate_json(document)
    questions = StoragePath(sample.path) / "questions.parquet"
    if file_sha256(questions) != sample.questions_parquet_sha256:
        raise ValueError("Pilot question parquet changed before grounding")
    with questions.open("rb") as reader:
        rows = pq.read_table(reader).to_pylist()
    expected = set(sample.drop_sample_ids) | set(sample.kept_sample_ids)
    if {row["id"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Pilot questions differ from the sealed selection")
    return rows


def pilot_metrics(config: HybridGroundingConfig, decisions: list[dict]) -> dict:
    """Measure recovery against the sealed v12 decisions for the fixed sample."""
    sample = PilotSample.model_validate_json(
        pinned_document(config.question_receipt_uri, config.question_receipt_sha256)
    )
    selection = json.loads(
        pinned_document(str(StoragePath(sample.path) / "selection.json"), sample.selection_manifest_sha256)
    )
    keep_by_id = {decision["id"]: decision["keep"] for decision in decisions}
    recovered = sum(1 for question_id in selection["drop_sample_ids"] if keep_by_id[question_id])
    regressions = sum(1 for question_id in selection["kept_sample_ids"] if not keep_by_id[question_id])
    return {
        "v12_recovery_sample": len(selection["drop_sample_ids"]),
        "v12_regression_sample": len(selection["kept_sample_ids"]),
        "recovered_questions": recovered,
        "regressions": regressions,
    }


def hybrid_ground_questions(config: HybridGroundingConfig) -> HybridGroundedQuestions:
    info = get_job_info()
    if info is None:
        raise RuntimeError("Screen the question population inside the regional Iris CPU job")
    if config.question_kind not in QUESTION_KINDS:
        raise ValueError(f"Unsupported question kind: {config.question_kind}")
    if (
        config.contract_sha256 != CONTRACT_SHA256
        or file_sha256(StoragePath(str(MODEL_CONFIG))) != config.model_config_sha256
    ):
        raise ValueError("Grounding matcher or serving contract changed after planning")
    frozen_runtime = pinned_document(config.runtime_freeze_uri, config.runtime_freeze_sha256)
    actual_runtime = subprocess.check_output(["uv", "pip", "freeze", "--python", config.runtime_python])
    if actual_runtime != frozen_runtime:
        raise ValueError("Grounding retrieval runtime differs from the sealed hybrid runtime")
    index = HybridIndex.model_validate_json(pinned_document(config.hybrid_index_uri, config.hybrid_index_sha256))
    rows = load_questions(config, index)
    local_index(index.bm25_index, Path(config.workdir) / "bm25")
    local_dense(index, Path(config.workdir))
    local_model(index, Path(config.workdir))
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
            "provider_version": index.provider_version,
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
        decisions = asyncio.run(hybrid_grounded_rows(config, rows, connection, tokenizer, index, output))
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
    pilot = (
        pilot_metrics(config, decisions)
        if config.question_kind == "pilot_sample"
        else {
            "v12_recovery_sample": None,
            "v12_regression_sample": None,
            "recovered_questions": None,
            "regressions": None,
        }
    )
    return HybridGroundedQuestions(
        path=config.output_path,
        question_receipt_sha256=config.question_receipt_sha256,
        question_kind=config.question_kind,
        hybrid_index_receipt_sha256=config.hybrid_index_sha256,
        provider_version=index.provider_version,
        bm25_index_manifest_sha256=index.bm25_index.index_manifest_sha256,
        dense_index_manifest_sha256=index.dense_index.index_manifest_sha256,
        embedding_model=index.embedding_model,
        embedding_model_revision=index.embedding_model_revision,
        implementation_revision=config.implementation_revision,
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
        **pilot,
    )


def grounding_config(config: HybridGroundingConfig, context: StepContext) -> HybridGroundingConfig:
    return replace(config, output_path=context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "question-receipt",
        "question-receipt-sha256",
        "hybrid-index-receipt",
        "hybrid-index-receipt-sha256",
        "runtime-freeze",
        "runtime-freeze-sha256",
        "implementation-revision",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--question-kind", choices=QUESTION_KINDS, required=True)
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
    config = HybridGroundingConfig(
        args.question_receipt,
        args.question_receipt_sha256,
        args.question_kind,
        args.hybrid_index_receipt,
        args.hybrid_index_receipt_sha256,
        CONTRACT_SHA256,
        file_sha256(StoragePath(str(MODEL_CONFIG))),
        args.runtime_freeze,
        args.runtime_freeze_sha256,
        args.runtime_python,
        args.implementation_revision,
        args.workdir,
        args.concurrency,
        "",
    )
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/hybrid-grounded-questions",
            version=args.artifact_version,
            artifact_type=HybridGroundedQuestions,
            run=hybrid_ground_questions,
            build_config=partial(grounding_config, config),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint(), "contract": CONTRACT_SHA256}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
