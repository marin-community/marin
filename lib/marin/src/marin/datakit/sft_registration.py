# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify completed Datakit artifacts before issuing SFT qualification."""

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import finestore.sft as receipt_models
import numpy as np
import pyarrow.parquet as pq
from finestore.sft import (
    REQUIRED_SFT_EVAL_FAMILIES,
    DatasetVariantInput,
    SftEvidenceReference,
    SftPreparationReceipt,
    SftRecord,
    SftStage,
)
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import load_tokenizer
from openai_harmony import Message
from pydantic import Field
from rigging.filesystem.step_status import STATUS_SUCCESS
from rigging.filesystem.storage_path import StoragePath

from marin.datakit.chat_normalize import ChatSourceData, validate_chat_messages, validate_tool_definitions
from marin.datakit.chat_render import _inference_messages
from marin.datakit.decon import (
    DeconAttributes,
    EvalBloom,
    NGramConfig,
    _validate_best_effort_eval_manifest,
    _validate_required_eval_manifest,
)
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import ARTIFACT_LOAD_CONTEXT_KEY, ArtifactRecord
from marin.execution.step_status import StatusFile

SFT_NGRAM_LENGTH = 5


class SftFileManifest(SftRecord):
    """A complete file selection, pinned as one content identity."""

    files: tuple[SftEvidenceReference, ...] = Field(min_length=1)
    rows: int = Field(ge=0)


class SftFilterResult(SftRecord):
    input: DatasetVariantInput
    output_manifest: SftEvidenceReference
    input_rows: int = Field(ge=0)
    filtered_rows: int = Field(ge=0)


@dataclass(frozen=True)
class EncodedChat:
    input_ids: np.ndarray
    assistant_masks: np.ndarray
    encoding_sha256: str


@dataclass(frozen=True)
class TrainingSummary:
    rows: int
    chat_rows: int
    input_tokens: int
    supervised_tokens: int
    sha256: str


class SftEvalRequirement(SftRecord):
    manifest: SftEvidenceReference
    version: str
    names: tuple[str, ...] = ()
    required_tasks: dict[str, int] = Field(default_factory=dict)
    source_catalog_sha256: str | None = None


class SftRegistrationPlan(SftRecord):
    """Pinned inputs to a terminal check; this plan does not convert data."""

    input: DatasetVariantInput
    registry_source: str
    input_manifest: SftEvidenceReference
    stages: dict[SftStage, SftEvidenceReference]
    stage_files: dict[SftStage, SftEvidenceReference]
    evals: dict[str, SftEvalRequirement]
    output_manifest: SftEvidenceReference
    tokenizer: SftEvidenceReference
    chat_template: SftEvidenceReference


def qualification_implementation_sha256() -> str:
    """Pin the validator and its public receipt contract in the step identity."""
    return hashlib.sha256(Path(__file__).read_bytes() + Path(receipt_models.__file__).read_bytes()).hexdigest()


def read_pinned(reference: SftEvidenceReference) -> bytes:
    """Read a small manifest or record only if its byte digest matches."""
    with StoragePath(reference.uri).open("rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError(f"SFT evidence digest mismatch: {reference.uri}")
    return raw


def _verify_file(reference: SftEvidenceReference) -> None:
    with StoragePath(reference.uri).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != reference.sha256:
            raise ValueError(f"SFT dataset file changed: {reference.uri}")


def _parquet_rows(paths: Iterable[StoragePath]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=256):
                yield from batch.to_pylist()


def _rows(files: tuple[SftEvidenceReference, ...]) -> Iterator[dict[str, Any]]:
    for reference in files:
        _verify_file(reference)
        yield from _parquet_rows([StoragePath(reference.uri)])


def _stage_rows(
    plan: SftRegistrationPlan,
    stage: SftStage,
    record: ArtifactRecord,
    schema: type[ChatSourceData] | type[NormalizedData] | type[DeconAttributes],
) -> Iterator[dict[str, Any]]:
    value = schema.model_validate(record.result, context={ARTIFACT_LOAD_CONTEXT_KEY: True})
    manifest = SftFileManifest.model_validate_json(read_pinned(plan.stage_files[stage]))
    actual = {str(path) for path in (StoragePath(value.main_output_dir) / "*.parquet").glob()}
    if actual != {str(StoragePath(ref.uri)) for ref in manifest.files}:
        raise ValueError("SFT stage files do not match its pinned complete manifest")
    count = 0
    for row in _rows(manifest.files):
        count += 1
        yield row
    if count != manifest.rows:
        raise ValueError("SFT stage manifest row count differs from its files")


def _chat_digest(row: dict[str, Any]) -> str:
    value = {"messages": row["messages"], "chat_template_kwargs": row.get("chat_template_kwargs")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _encode_chat(row: dict[str, Any], processor: ChatProcessor, template: str) -> EncodedChat:
    messages = [Message.from_dict(message) for message in row["messages"]]
    validate_chat_messages(messages)
    kwargs = json.loads(row["chat_template_kwargs"]) if row.get("chat_template_kwargs") else {}
    if kwargs.get("chat_template", template) != template:
        raise ValueError("SFT row overrides the pinned chat template")
    validate_tool_definitions(kwargs.get("tools", []), messages)
    actual = processor([{"messages": list(_inference_messages(messages)), "chat_template_kwargs": kwargs}])[0]
    pair = json.dumps([actual["input_ids"].tolist(), actual["assistant_masks"].tolist()], separators=(",", ":")).encode()
    return EncodedChat(actual["input_ids"], actual["assistant_masks"], hashlib.sha256(pair).hexdigest())


def _validate_evals(plan: SftRegistrationPlan, bloom: ArtifactRecord) -> str:
    evidence = EvalBloom.model_validate(bloom.result, context={ARTIFACT_LOAD_CONTEXT_KEY: True})
    if evidence.n_eval_records <= 0:
        raise ValueError("SFT Bloom has no measured eval records")
    if set(plan.evals) != REQUIRED_SFT_EVAL_FAMILIES:
        raise ValueError("SFT qualification requires every mandatory eval family")
    config = bloom.config or {}
    mark_config = ArtifactRecord.model_validate_json(read_pinned(plan.stages[SftStage.DECONTAM])).config or {}
    policy_keys = (
        "text_field",
        "ngram_length",
        "overlap_threshold",
        "min_matched_features",
        "paragraph_delimiter",
        "feature_filter_version",
    )
    if any(key not in config or mark_config.get(key) != config[key] for key in policy_keys):
        raise ValueError("SFT decontamination marks and Bloom use different matching policies")
    required = sorted(sorted(plan.evals[key].names) for key in ("aa", "uncheatable"))
    if config.get("ngram_length") != SFT_NGRAM_LENGTH or config.get("required_eval_suites") != required:
        raise ValueError("SFT Bloom does not bind the required suites and five-word policy")
    if config.get("required_eval_corpus_version") != plan.evals["aa"].version:
        raise ValueError("SFT Bloom uses another eval corpus")
    lmh = plan.evals["lm_eval"]
    if dict(config.get("best_effort_required_tasks", [])) != lmh.required_tasks or not lmh.required_tasks:
        raise ValueError("SFT Bloom lacks required lm-eval leaf-task coverage")
    supplement = plan.evals["atlas_references"]
    additional = config.get("additional_required_eval_manifests", [])
    expected = [supplement.manifest.uri, supplement.version, list(supplement.names), supplement.source_catalog_sha256]
    if (
        not isinstance(additional, list)
        or expected not in additional
        or config.get("best_effort_eval_corpus_version") != lmh.version
    ):
        raise ValueError("SFT Bloom does not bind the pinned reference supplement and lm-eval version")
    for key, requirement in plan.evals.items():
        read_pinned(requirement.manifest)
        if key == "lm_eval":
            _validate_best_effort_eval_manifest(
                requirement.manifest.uri, requirement.version, requirement.required_tasks
            )
        else:
            _validate_required_eval_manifest(
                requirement.manifest.uri,
                requirement.version,
                requirement.names,
                text_field="text",
                ngram=NGramConfig(
                    ngram_length=SFT_NGRAM_LENGTH,
                    overlap_threshold=float(config["overlap_threshold"]),
                    min_matched_features=int(config["min_matched_features"]),
                    paragraph_delimiter=str(config["paragraph_delimiter"]),
                ),
                expected_source_catalog_sha256=requirement.source_catalog_sha256,
            )
    return hashlib.sha256(
        json.dumps(
            {
                "evals": {k: v.model_dump(mode="json") for k, v in plan.evals.items()},
                "matching_policy": {key: config[key] for key in policy_keys},
                "bloom": plan.stages[SftStage.EVAL_BLOOM].model_dump(mode="json"),
                "decontamination": plan.stages[SftStage.DECONTAM].model_dump(mode="json"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _load_completed_stages(plan: SftRegistrationPlan) -> tuple[SftFileManifest, dict[SftStage, ArtifactRecord]]:
    if set(plan.stages) != set(SftStage):
        raise ValueError("SFT qualification requires all completed Datakit stages")
    if plan.input_manifest.sha256 != plan.input.data_sha256:
        raise ValueError("SFT input manifest does not match the dataset variant")
    inputs = SftFileManifest.model_validate_json(read_pinned(plan.input_manifest))
    if sum(1 for _ in _rows(inputs.files)) != inputs.rows:
        raise ValueError("SFT input manifest row count differs from its files")
    if set(plan.stage_files) != {SftStage.CHAT, SftStage.NORMALIZE, SftStage.DECONTAM}:
        raise ValueError("SFT preparation requires content-pinned stage manifests")
    records = {stage: ArtifactRecord.model_validate_json(read_pinned(ref)) for stage, ref in plan.stages.items()}
    for stage, record in records.items():
        if plan.stages[stage].uri != str(StoragePath(record.output_path) / ".artifact.json"):
            raise ValueError("SFT stage record does not identify its pinned output")
        if not record.name or StatusFile(record.output_path, worker_id="sft-registration").status != STATUS_SUCCESS:
            raise ValueError("SFT qualification refuses incomplete stage records")
    download = records[SftStage.DOWNLOAD]
    download_config = download.config or {}
    if (download_config.get("hf_dataset_id"), download_config.get("revision")) != (
        plan.input.repository,
        plan.input.revision,
    ):
        raise ValueError("SFT download does not match the selected repository and revision")
    for reference in inputs.files:
        location = StoragePath(reference.uri)
        if any(segment in {".", ".."} for segment in location.segments):
            raise ValueError("SFT input file has an ambiguous storage path")
        if not location.relative_to(StoragePath(download.output_path)):
            raise ValueError("SFT input files are outside the pinned download")
    pending = [records[SftStage.TRAINING].output_path]
    ancestors = set()
    while pending:
        path = pending.pop()
        if path in ancestors:
            continue
        ancestors.add(path)
        artifact_path = StoragePath(path) / ".artifact.json"
        pinned = next((ref for ref in plan.stages.values() if StoragePath(ref.uri) == artifact_path), None)
        raw = read_pinned(pinned) if pinned is not None else artifact_path.read_bytes()
        record = ArtifactRecord.model_validate_json(raw)
        if record.output_path != path or StatusFile(path, worker_id="sft-registration").status != STATUS_SUCCESS:
            raise ValueError("SFT lineage contains an incomplete upstream stage")
        pending.extend(record.dep_paths)
    if any(record.output_path not in ancestors for record in records.values()):
        raise ValueError("SFT preparation does not descend from every required stage")
    return inputs, records


def _load_processor(root: Path, files: SftFileManifest, template: str) -> ChatProcessor:
    for reference in files.files:
        name = StoragePath(reference.uri).name
        if not name.startswith(("tokenizer", "special_tokens", "added_tokens", "vocab", "merges", "chat_template")):
            raise ValueError("SFT tokenizer manifest contains a non-tokenizer file")
        (root / name).write_bytes(read_pinned(reference))
    return ChatProcessor(load_tokenizer(str(root)), chat_template=template)


def _audit_training(
    plan: SftRegistrationPlan,
    inputs: SftFileManifest,
    records: dict[SftStage, ArtifactRecord],
    output: SftFileManifest,
    filtering: SftFilterResult,
) -> TrainingSummary:
    template = read_pinned(plan.chat_template).decode("utf-8")
    tokenizer_files = SftFileManifest.model_validate_json(read_pinned(plan.tokenizer))
    count = tokens = supervised = 0
    training_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="datakit-sft-registration-") as temporary:
        root = Path(temporary)
        processor = _load_processor(root, tokenizer_files, template)
        with sqlite3.connect(root / "selection.sqlite") as db:
            db.execute("CREATE TABLE chat (id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
            db.execute("CREATE TABLE text (id TEXT PRIMARY KEY, chat_id TEXT NOT NULL)")
            db.execute("CREATE TABLE marks (id TEXT PRIMARY KEY, contaminated INTEGER NOT NULL)")
            db.execute("CREATE TABLE selected (id TEXT PRIMARY KEY, encoding_sha TEXT UNIQUE NOT NULL)")
            for row in _stage_rows(plan, SftStage.CHAT, records[SftStage.CHAT], ChatSourceData):
                db.execute("INSERT INTO chat VALUES (?,?)", (row["id"], _chat_digest(row)))
            chat_rows = db.execute("SELECT COUNT(*) FROM chat").fetchone()[0]
            if chat_rows > inputs.rows:
                raise ValueError("SFT chat population exceeds the downloaded input population")
            for row in _stage_rows(plan, SftStage.NORMALIZE, records[SftStage.NORMALIZE], NormalizedData):
                db.execute("INSERT INTO text VALUES (?,?)", (row["id"], row["source_id"]))
            for row in _stage_rows(plan, SftStage.DECONTAM, records[SftStage.DECONTAM], DeconAttributes):
                if not isinstance(row["contaminated"], bool):
                    raise ValueError("SFT decontamination contains an unmeasured row")
                db.execute("INSERT INTO marks VALUES (?,?)", (row["id"], bool(row["contaminated"])))
            if db.execute("SELECT COUNT(*) FROM text LEFT JOIN marks USING(id) WHERE marks.id IS NULL").fetchone() != (
                0,
            ) or db.execute("SELECT COUNT(*) FROM marks LEFT JOIN text USING(id) WHERE text.id IS NULL").fetchone() != (
                0,
            ):
                raise ValueError("SFT decontamination does not cover the complete normalized population")
            for row in _rows(output.files):
                original = db.execute("SELECT digest FROM chat WHERE id=?", (row["id"],)).fetchone()
                mark = db.execute(
                    "SELECT text.chat_id,marks.contaminated FROM text JOIN marks USING(id) WHERE id=?",
                    (row["normalized_id"],),
                ).fetchone()
                if original != (_chat_digest(row),) or mark != (row["id"], 0):
                    raise ValueError("SFT output changed an original chat or retained unmeasured/contaminated text")
                encoded = _encode_chat(row, processor, template)
                if not np.array_equal(encoded.input_ids, row["input_ids"]) or not np.array_equal(
                    encoded.assistant_masks, row["assistant_masks"]
                ):
                    raise ValueError("SFT tokens or assistant masks differ from the pinned tokenizer/template")
                db.execute("INSERT INTO selected VALUES (?,?)", (row["id"], encoded.encoding_sha256))
                training_digest.update(json.dumps([row["id"], encoded.encoding_sha256]).encode() + b"\n")
                count += 1
                tokens += len(encoded.input_ids)
                supervised += int(encoded.assistant_masks.sum())
            for row in _stage_rows(plan, SftStage.CHAT, records[SftStage.CHAT], ChatSourceData):
                if db.execute("SELECT 1 FROM selected WHERE id=?", (row["id"],)).fetchone():
                    continue
                contaminated = db.execute(
                    "SELECT 1 FROM text JOIN marks USING(id) WHERE chat_id=? AND contaminated=1", (row["id"],)
                ).fetchone()
                if contaminated:
                    continue
                encoded = _encode_chat(row, processor, template)
                if not db.execute("SELECT 1 FROM selected WHERE encoding_sha=?", (encoded.encoding_sha256,)).fetchone():
                    raise ValueError("SFT filtering discarded a distinct clean conversation")
            if (
                count != output.rows
                or filtering.input_rows != inputs.rows
                or filtering.filtered_rows != chat_rows - count
            ):
                raise ValueError("SFT preparation does not reconcile its complete population")
    return TrainingSummary(count, chat_rows, tokens, supervised, training_digest.hexdigest())


def qualify_sft_variant(plan: SftRegistrationPlan, output_path: str) -> SftPreparationReceipt:
    """Check full input/output content, original chats, filtering, and actual masks.

    Run beside the data. Prepared Parquet retains canonical messages and their
    normalized-text ID beside input_ids/assistant_masks; it is the training input,
    not a sampled audit. No teacher calls, conversion or label backfill occur here.
    """
    inputs, records = _load_completed_stages(plan)
    policy_sha256 = _validate_evals(plan, records[SftStage.EVAL_BLOOM])
    filtering = SftFilterResult.model_validate(records[SftStage.FILTER].result)
    if filtering.input != plan.input or filtering.output_manifest != plan.output_manifest:
        raise ValueError("SFT filtering evidence belongs to another input or training selection")
    output = SftFileManifest.model_validate_json(read_pinned(plan.output_manifest))
    training = _audit_training(plan, inputs, records, output, filtering)
    receipt = SftPreparationReceipt(
        input=plan.input,
        registry_source=plan.registry_source,
        implementation_sha256=qualification_implementation_sha256(),
        stages=plan.stages,
        eval_manifests={key: value.manifest for key, value in plan.evals.items()},
        eval_policy_sha256=policy_sha256,
        output_manifest=plan.output_manifest,
        tokenizer=plan.tokenizer,
        chat_template_sha256=plan.chat_template.sha256,
        input_rows=inputs.rows,
        training_rows=training.rows,
        filtered_rows=training.chat_rows - training.rows,
        normalization_filtered_rows=inputs.rows - training.chat_rows,
        stage_files=plan.stage_files,
        input_tokens=training.input_tokens,
        supervised_tokens=training.supervised_tokens,
        training_sha256=training.sha256,
        completed_at=datetime.now(UTC),
    )
    destination_path = StoragePath(output_path)
    if destination_path.exists():
        previous = SftPreparationReceipt.model_validate_json(destination_path.read_bytes())
        if previous.model_dump(exclude={"completed_at"}) != receipt.model_dump(exclude={"completed_at"}):
            raise ValueError("Existing SFT receipt belongs to different preparation evidence")
        return previous
    destination_path.parent.mkdirs()
    with destination_path.open("wt") as destination:
        destination.write(receipt.model_dump_json(indent=2) + "\n")
    return receipt
