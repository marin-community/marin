# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a completely screened regional question population into native TRAIN tasks."""

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements
from taskcompendium.openresearcher import OpenResearcherState
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS,
    DATASET,
    DATASET_REVISION,
    REGIONAL_PREFIX,
    RegionalInputs,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding import validate_technical_exclusions
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import CONTRACT_SHA256
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import (
    MODEL_REVISION,
    GroundedQuestions,
)
from experiments.post_training.task_curation.datasets.openresearcher.questions import QuestionPool
from experiments.post_training.task_curation.datasets.openresearcher.tasks import SOURCE_ID, VERSION, convert_question
from experiments.post_training.task_curation.license_binding import licensed_source
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

INITIAL_REQUIREMENTS = ExecutionRequirements(
    cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
)
TASK_SCHEMA = pa.schema(
    [("id", pa.string()), ("task_spec", pa.string()), ("task_sha256", pa.string()), ("path", pa.string())]
)


@dataclass(frozen=True)
class PreparationConfig:
    stage_receipt: str
    stage_receipt_sha256: str
    question_receipt: str
    question_receipt_sha256: str
    grounding_receipt: str
    grounding_receipt_sha256: str
    heldout_hashes: str
    heldout_hashes_sha256: str
    actor_image: str
    grader_image: str
    state: OpenResearcherState
    workdir: str
    output_path: str


@dataclass(frozen=True)
class ResearchDataset:
    preparation: PreparationConfig

    @property
    def name(self) -> str:
        return "openresearcher-train"

    @property
    def version(self) -> str:
        return VERSION

    @property
    def dataset(self) -> HfSource:
        return HfSource(DATASET, DATASET_REVISION, self.files, SourceFormat.PARQUET)

    @property
    def files(self) -> tuple[str, ...]:
        return ("seed_*/train-*.parquet",)

    @property
    def intended_use(self) -> IntendedUse:
        return IntendedUse.TRAIN


@dataclass(frozen=True)
class RetainedQuestions:
    rows: list[dict]
    grounding: GroundedQuestions
    heldout_exclusions: list[dict]


def source_licenses(config: PreparationConfig) -> dict[str, LicenseReceipt]:
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt, config.stage_receipt_sha256))
    rows = json.loads(
        pinned_document(str(StoragePath(staged.path) / "license-receipts.json"), staged.license_receipts_sha256)
    )
    receipts = {}
    for dataset in (DATASET, CORPUS):
        source_id = "OpenResearcher:" + dataset.split("/")[1]
        receipt = LicenseReceipt.model_validate(rows[source_id])
        require_license(receipt, source_id, "train", ban=None)
        receipts[source_id] = receipt
    return receipts


def retained_questions(config: PreparationConfig) -> RetainedQuestions:
    """Require a complete decision bijection and remove exact heldout benchmark matches."""
    pool = QuestionPool.model_validate_json(pinned_document(config.question_receipt, config.question_receipt_sha256))
    grounded = GroundedQuestions.model_validate_json(
        pinned_document(config.grounding_receipt, config.grounding_receipt_sha256)
    )
    if (
        pool.source_revision != DATASET_REVISION
        or pool.stage_receipt_sha256 != config.stage_receipt_sha256
        or pool.conflicting_answer_questions
        or pool.invalid_rows
        or grounded.question_receipt_sha256 != config.question_receipt_sha256
        or grounded.contract_sha256 != CONTRACT_SHA256
        or grounded.model_revision != MODEL_REVISION
        or grounded.index_manifest_sha256 != config.state.index_manifest_sha256
        or grounded.execution_failures
        or grounded.input_questions != pool.unique_contracts
        or grounded.kept_questions + grounded.dropped_questions != grounded.input_questions
    ):
        raise ValueError("Conversion requires complete grounding of the approved question and index pins")
    original = pq.read_table(
        pa.BufferReader(pinned_document(str(StoragePath(pool.path) / "questions.parquet"), pool.tasks_parquet_sha256))
    ).to_pylist()
    kept = pq.read_table(
        pa.BufferReader(pinned_document(str(StoragePath(grounded.path) / "kept.parquet"), grounded.kept_parquet_sha256))
    ).to_pylist()
    decisions = pq.read_table(
        pa.BufferReader(
            pinned_document(str(StoragePath(grounded.path) / "decisions.parquet"), grounded.decisions_parquet_sha256)
        )
    ).to_pylist()
    originals = {row["id"]: row for row in original}
    decisions_by_id = {row["id"]: json.loads(row["decision_json"]) for row in decisions}
    if (
        len(originals) != len(original)
        or len(original) != pool.unique_contracts
        or len(decisions_by_id) != len(decisions)
        or set(decisions_by_id) != set(originals)
        or any(
            d["id"] != key
            or d["status"] not in {"screened", "technical_exclusion"}
            or d["contract_sha256"] != CONTRACT_SHA256
            for key, d in decisions_by_id.items()
        )
    ):
        raise ValueError("Grounding decisions do not cover every original question exactly once")
    expected = {key for key, decision in decisions_by_id.items() if decision["keep"]}
    if grounded.drop_reason_counts != dict(
        Counter(decision["reason"] for decision in decisions_by_id.values() if not decision["keep"])
    ):
        raise ValueError("Grounding drop counts differ from the sealed question decisions")
    validate_technical_exclusions(list(decisions_by_id.values()))
    if (
        len(kept) != grounded.kept_questions
        or len({row["id"] for row in kept}) != len(kept)
        or {row["id"] for row in kept} != expected
        or any(row != originals[row["id"]] for row in kept)
    ):
        raise ValueError("Grounded QA bytes differ from the retained original question contracts")
    backend = json.loads((StoragePath(grounded.path) / "backend-identity.json").read_text())
    if canonical_sha256(backend) != grounded.backend_identity_sha256:
        raise ValueError("Grounding backend identity changed after screening")
    heldout = json.loads(pinned_document(config.heldout_hashes, config.heldout_hashes_sha256))
    if heldout["normalizer"] != "whitespace-collapse-then-unicode-casefold-v1":
        raise ValueError("Unsupported heldout question normalizer")
    hashes = heldout["questions"]
    retained, exclusions = [], []
    for row in kept:
        question_hash = hashlib.sha256(" ".join(row["question"].split()).casefold().encode()).hexdigest()
        if question_hash in hashes:
            exclusions.append({"id": row["id"], "question_sha256": question_hash, "matches": hashes[question_hash]})
        else:
            retained.append(row)
    return RetainedQuestions(retained, grounded, exclusions)


def prepare(config: PreparationConfig) -> CampaignArtifact:
    receipts = source_licenses(config)
    for image in (config.actor_image, config.grader_image):
        if re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image) is None:
            raise ValueError("Actor and grader images require immutable digests")
    retained = retained_questions(config)
    grounded = retained.grounding
    output = StoragePath(config.output_path)
    if (output / "tasks.parquet").exists() or (output / "task-directories").exists():
        raise FileExistsError("Conversion requires a fresh output version")
    records, normalized = [], []
    seen = set()
    work = Path(config.workdir)
    work.mkdir(parents=True, exist_ok=False)
    for row in retained.rows:
        task = convert_question(
            row, state=config.state, actor_image=config.actor_image, grader_image=config.grader_image
        ).model_copy(
            update={
                "execution_requirements": INITIAL_REQUIREMENTS,
                "verifier_execution_requirements": INITIAL_REQUIREMENTS,
            }
        )
        if task.id in seen:
            raise ValueError("Retained questions have duplicate actor identities")
        seen.add(task.id)
        directory = work / task.id
        write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
        for path in directory.rglob("*"):
            if path.is_file():
                (output / "task-directories" / task.id / path.relative_to(directory).as_posix()).write_bytes(
                    path.read_bytes()
                )
        encoded = task.model_dump_json()
        records.append(
            {
                "id": task.id,
                "task_spec": encoded,
                "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "path": f"task-directories/{task.id}",
            }
        )
        normalized.append(
            {**normalization_columns(task.id, task.source, task, None, (), task.source.row), "task_json": encoded}
        )
    for name, data, schema in (
        ("tasks.parquet", records, TASK_SCHEMA),
        ("normalize/part-00000.parquet", normalized, NORMALIZED_SCHEMA),
    ):
        with (output / name).open("wb", auto_mkdir=True) as writer:
            pq.write_table(pa.Table.from_pylist(data, schema=schema), writer, compression="zstd")
    manifest = {
        "mode": "quick",
        "source": "openresearcher-train",
        "source_id": SOURCE_ID,
        "source_dataset": DATASET,
        "source_revision": DATASET_REVISION,
        "recipe_revision": VERSION,
        "intended_use": "train",
        "input_rows": grounded.input_questions,
        "converted_rows": len(records),
        "rejections": {
            "grounding_drop": grounded.dropped_questions,
            "benchmark_heldout": len(retained.heldout_exclusions),
        },
        "grounding_receipt": config.grounding_receipt,
        "grounding_receipt_sha256": config.grounding_receipt_sha256,
        "heldout_hashes": config.heldout_hashes,
        "heldout_hashes_sha256": config.heldout_hashes_sha256,
        "actor_image": config.actor_image,
        "grader_image": config.grader_image,
        "retrieval": config.state.model_dump(mode="json"),
        "task_format": "taskcompendium",
        "oracle_solutions": False,
        "reviewed": False,
        "admitted": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "heldout-drops.json").write_text(json.dumps(retained.heldout_exclusions, indent=2) + "\n")
    (output / "license-receipts.json").write_text(
        json.dumps({key: value.model_dump(mode="json") for key, value in receipts.items()}, indent=2) + "\n"
    )
    return CampaignArtifact(
        path=config.output_path,
        status=SourceStatus.COMPLETED,
        result=PipelineResult(
            SourceStatus.COMPLETED,
            {"normalize": str(output / "normalize"), "tasks": str(output / "tasks.parquet")},
            {"manifest": str(output / "manifest.json")},
            ("grounding-binding", "heldout-exclusion", "conversion", "task-directories"),
        ),
    )


def prepare_config(config: PreparationConfig, context: StepContext) -> PreparationConfig:
    return replace(config, output_path=context.output_path)


def curate(source: RlDataSource[ResearchDataset], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    source = licensed_source(source)
    if options.mode is not SourceProcessingMode.QUICK:
        raise ValueError("Prepare QUICK first, then qualify and review its frozen private release")
    assert source.config is not None
    source_licenses(source.config.preparation)
    return ArtifactStep(
        name="rl-data/openresearcher/converted",
        version=VERSION,
        artifact_type=CampaignArtifact,
        run=prepare,
        build_config=partial(prepare_config, source.config.preparation),
    )


def source(config: PreparationConfig) -> RlDataSource[ResearchDataset]:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            origin=SourceOrigin.OPENRESEARCHER,
            upstream_dataset="OpenResearcher-Dataset",
            upstream_name="OpenResearcher TRAIN",
            build_path="Grounded offline research intake",
            family="offline-search",
            tags=("license:mit", "corpus-license:odc-by"),
            task_format=TaskFormat.TASKSPEC,
        ),
        config=ResearchDataset(config),
        pipeline=curate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Exact PreparationConfig JSON")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text())
    config = PreparationConfig(**{**raw, "state": OpenResearcherState.model_validate(raw["state"])})
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    selected = source(config)
    with artifact_prefix(args.artifact_root):
        step = curate(selected, PipelineOptions(mode=SourceProcessingMode.QUICK, runtime=CampaignRuntime()))
        if not args.run:
            StoragePath(args.output).write_text(
                json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}, indent=2) + "\n"
            )
            return
        with local_client():
            result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
