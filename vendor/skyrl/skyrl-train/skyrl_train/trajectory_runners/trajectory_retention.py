from collections import OrderedDict
from collections.abc import Mapping, Sequence
import asyncio
import concurrent.futures
from dataclasses import dataclass, replace
import gzip
import hashlib
from io import BytesIO
import json
import posixpath
import threading
from typing import Any, Protocol
import zipfile

from omegaconf import DictConfig
from loguru import logger
import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizerBase

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.trajectory_runners.types import (
    TrajectoryRequestBatch,
    TrajectoryBatch,
    RewardShapingComponents,
    RewardShapingLoopSpan,
    TrajectoryID,
    VerifierTestCollection,
)
from skyrl_train.trajectory_runners.trajectory_retention_config import (
    TrajectoryRetentionConfig,
    parse_trajectory_retention_config,
)
from skyrl_train.trajectory_runners.trajectory_retention_publisher import (
    ProcessTrajectoryPublisher,
    PublicationOperation,
    PublicationRequest,
    PublicationResult,
    TrajectoryPublisher,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import (
    NormalizedReward,
    aggregate_reward_shaping_components,
)
from skyrl_gym.verification import VerificationResult
from skyrl_train.json_serialization import canonical_json_bytes, to_jsonable
from skyrl_train.io import io


RETENTION_METRIC_PREFIX = "generate/trajectory_retention"
RETENTION_SCHEMA_VERSION = 1
TRAJECTORY_RECORD_SCHEMA_VERSION = 7
_LEDGER_NAME = "_retention_ledger.json"
_SELECTION_COUNT = "count"
_SELECTION_FRACTION = "fraction"
_SELECTION_MANDATORY = "mandatory"
_ARCHIVE_DIRECTORY = "archives"
_ARCHIVE_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class _LedgerEntry:
    path: str
    bytes: int
    step: str
    reasons: tuple[str, ...]
    sample_score: int

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "_LedgerEntry":
        return cls(
            path=str(value["path"]),
            bytes=int(value["bytes"]),
            step=str(value["step"]),
            reasons=tuple(str(reason) for reason in value.get("reasons", ())),
            sample_score=int(value.get("sample_score", 0)),
        )


@dataclass(frozen=True)
class _ArchiveEntry:
    bytes: int
    step: str
    record_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "_ArchiveEntry":
        return cls(
            bytes=int(value["bytes"]),
            step=str(value["step"]),
            record_ids=tuple(str(record_id) for record_id in value.get("record_ids", ())),
        )


@dataclass
class _RetentionLedger:
    total_bytes: int
    step_bytes: dict[str, int]
    records: dict[str, _LedgerEntry]
    archives: dict[str, _ArchiveEntry]
    schema_version: int = RETENTION_SCHEMA_VERSION

    @classmethod
    def from_json(cls, value: Mapping[str, Any] | None) -> "_RetentionLedger":
        if value is None:
            return cls(total_bytes=0, step_bytes={}, records={}, archives={})
        schema_version = int(value.get("schema_version", 0))
        if schema_version != RETENTION_SCHEMA_VERSION:
            raise ValueError("trajectory retention ledger schema version is incompatible")
        return cls(
            schema_version=schema_version,
            total_bytes=int(value.get("total_bytes", 0)),
            step_bytes={str(step): int(byte_count) for step, byte_count in value.get("step_bytes", {}).items()},
            records={
                str(record_id): _LedgerEntry.from_json(entry) for record_id, entry in value.get("records", {}).items()
            },
            archives={str(path): _ArchiveEntry.from_json(entry) for path, entry in value.get("archives", {}).items()},
        )


@dataclass(frozen=True)
class _TrajectoryRows:
    trajectory_id: TrajectoryID
    row_indices: tuple[int, ...]
    prompt_index: int


@dataclass(frozen=True)
class _Selection:
    reasons: tuple[str, ...]
    displaced_id: str | None = None


@dataclass(frozen=True)
class _SelectedRecord:
    record: "TrajectoryRecord"
    payload: bytes
    selection: _Selection


@dataclass(frozen=True)
class _TrajectoryIdentity:
    instance_id: str
    repetition_id: int
    row_indices: tuple[int, ...]
    environment_class: str
    environment_extras: Any


@dataclass(frozen=True)
class _PromptTrace:
    messages: Any
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _StepBoundary:
    row_index: int
    prompt_token_ids: tuple[int, ...]
    token_start: int
    token_end: int
    stop_reason: str | None
    is_last_step: bool | None


@dataclass(frozen=True)
class _ResponseTrace:
    messages: tuple[dict[str, Any], ...] | None
    text: str | None
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    trainable_spans: tuple[RewardShapingLoopSpan, ...]
    step_boundaries: tuple[_StepBoundary, ...]
    stop_reason: str | None
    generation_limit: int | None
    loop_spans: tuple[RewardShapingLoopSpan, ...]


@dataclass(frozen=True)
class _RewardTrace:
    outcome: float
    shaped: float
    components: RewardShapingComponents | None


@dataclass(frozen=True)
class _DispositionTrace:
    exception_type: str | None
    error_treatment: str | None
    server_error: dict[str, Any] | None


@dataclass(frozen=True)
class _ProvenanceTrace:
    runner: str
    inference_backend: str | None
    model_path: str | None
    model_source_identity: str | None
    resume_path: str | None
    model_version_step: int
    sampling: dict[str, Any]
    reward_shaping_schema_version: int | None


@dataclass(frozen=True)
class TrajectoryRecord:
    schema_version: int
    record_id: str
    run_id: str
    global_step: int
    phase: str
    trajectory: _TrajectoryIdentity
    prompt: _PromptTrace
    response: _ResponseTrace
    reward: _RewardTrace
    disposition: _DispositionTrace
    verifier: VerifierTestCollection | None
    verification_result: VerificationResult | None
    metrics: dict[str, Any]
    provenance: _ProvenanceTrace

    def to_json(self) -> dict[str, Any]:
        return to_jsonable(self)


class TrajectoryWriter(Protocol):
    def exists(self, relative_path: str) -> bool: ...

    def read_json(self, relative_path: str) -> dict[str, Any] | None: ...

    def read_bytes(self, relative_path: str) -> bytes: ...

    def write_bytes(self, relative_path: str, payload: bytes) -> None: ...

    def write_json(self, relative_path: str, value: dict[str, Any]) -> None: ...

    def find_files(self) -> dict[str, int]: ...


class _FilesystemTrajectoryWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def _path(self, relative_path: str) -> str:
        return join_resource_path(self.output_path, relative_path)

    def exists(self, relative_path: str) -> bool:
        return io.exists(self._path(relative_path))

    def read_json(self, relative_path: str) -> dict[str, Any] | None:
        path = self._path(relative_path)
        if not io.exists(path):
            return None
        with io.open_file(path, "r") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError(f"trajectory retention ledger must contain an object: {path}")
        return value

    def read_bytes(self, relative_path: str) -> bytes:
        return io.read_bytes(self._path(relative_path))

    def write_bytes(self, relative_path: str, payload: bytes) -> None:
        self._write(relative_path, payload)

    def write_json(self, relative_path: str, value: dict[str, Any]) -> None:
        self._write(relative_path, canonical_json_bytes(value))

    def find_files(self) -> dict[str, int]:
        absolute_files = io.find_files(self.output_path)
        normalized_root = self.output_path.split("://", 1)[-1].rstrip("/")
        return {posixpath.relpath(path, normalized_root): size for path, size in absolute_files.items()}

    def _write(self, relative_path: str, payload: bytes) -> None:
        io.write_bytes_atomic(self._path(relative_path), payload)


def _trajectory_key(trajectory_id: TrajectoryID) -> tuple[str, int]:
    return trajectory_id.instance_id, trajectory_id.repetition_id


def _group_rows(input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> list[_TrajectoryRows]:
    batch_size = len(output["response_ids"])
    output_ids = output.get("trajectory_ids")
    input_ids = input_batch.trajectory_ids
    if output_ids is None:
        if input_ids is None or len(input_ids) != batch_size:
            raise ValueError("trajectory retention requires one trajectory ID per normalized output row")
        output_ids = input_ids
    if len(output_ids) != batch_size:
        raise ValueError("trajectory IDs must align with normalized output rows")

    input_index = {_trajectory_key(trajectory_id): index for index, trajectory_id in enumerate(input_ids or [])}
    groups: OrderedDict[tuple[str, int], tuple[TrajectoryID, list[int], int]] = OrderedDict()
    for row_index, trajectory_id in enumerate(output_ids):
        key = _trajectory_key(trajectory_id)
        prompt_index = input_index.get(key, row_index)
        if key not in groups:
            groups[key] = trajectory_id, [], prompt_index
        groups[key][1].append(row_index)
    return [
        _TrajectoryRows(trajectory_id=trajectory_id, row_indices=tuple(row_indices), prompt_index=prompt_index)
        for trajectory_id, row_indices, prompt_index in groups.values()
    ]


def _active_spans(loss_mask: Sequence[int]) -> list[RewardShapingLoopSpan]:
    spans = []
    start = None
    for index, active in enumerate(loss_mask):
        if active and start is None:
            start = index
        if not active and start is not None:
            spans.append({"start": start, "end": index})
            start = None
    if start is not None:
        spans.append({"start": start, "end": len(loss_mask)})
    return spans


def _decode(tokenizer: PreTrainedTokenizerBase | None, token_ids: Sequence[int]) -> str | None:
    if tokenizer is None:
        return None
    return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _step_boundaries(
    output: TrajectoryBatch, row_indices: Sequence[int], stop_reasons: Sequence[str | None]
) -> list[_StepBoundary]:
    boundaries = []
    token_start = 0
    final_steps = output.get("is_last_step")
    for row_index in row_indices:
        token_end = token_start + len(output["response_ids"][row_index])
        boundaries.append(
            _StepBoundary(
                row_index=row_index,
                prompt_token_ids=tuple(output["prompt_token_ids"][row_index]),
                token_start=token_start,
                token_end=token_end,
                stop_reason=stop_reasons[row_index],
                is_last_step=None if final_steps is None else final_steps[row_index],
            )
        )
        token_start = token_end
    return boundaries


def _redact(record: dict[str, Any], fields: Sequence[str]) -> None:
    for field in fields:
        parts = field.split(".")
        current: Any = record
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, dict) and parts[-1] in current:
            current[parts[-1]] = "[REDACTED]"


def _serialized_record(record: TrajectoryRecord, redact_fields: Sequence[str]) -> dict[str, Any]:
    serialized = record.to_json()
    _redact(serialized, redact_fields)
    return serialized


def build_trajectory_records(
    input_batch: TrajectoryRequestBatch,
    output: TrajectoryBatch,
    config: TrajectoryRetentionConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    runner_name: str,
) -> list[TrajectoryRecord]:
    """Build runner-neutral records from one normalized batch."""
    metadata = input_batch.batch_metadata
    if metadata is None:
        raise ValueError("trajectory retention requires batch metadata")
    stop_reasons = output.get("stop_reasons") or [None] * len(output["response_ids"])
    exception_types = output.get("exception_types") or [None] * len(output["response_ids"])
    error_treatments = output.get("error_treatments") or [None] * len(output["response_ids"])
    server_errors = output.get("server_errors") or [None] * len(output["response_ids"])
    if len(exception_types) != len(output["response_ids"]):
        raise ValueError("exception types must have one entry per trajectory row")
    if len(error_treatments) != len(output["response_ids"]):
        raise ValueError("error treatments must have one entry per trajectory row")
    if len(server_errors) != len(output["response_ids"]):
        raise ValueError("server errors must have one entry per trajectory row")
    unshaped = output.get("unshaped_rewards")
    components = output.get("reward_shaping_components")
    loop_spans = output.get("reward_shaping_loop_spans")
    shaping_versions = output.get("reward_shaping_versions")
    verifier_tests = output.get("verifier_tests")
    if verifier_tests is not None and len(verifier_tests) != len(output["response_ids"]):
        raise ValueError("verifier tests must have one entry per trajectory row")
    # A request's step names the policy version being trained, which has completed one fewer update.
    model_version_step = max(0, metadata.global_step - 1)
    prompts = input_batch.public_prompts
    rows = input_batch.metadata

    records = []
    for group in _group_rows(input_batch, output):
        trajectory_id = group.trajectory_id
        row_indices = group.row_indices
        prompt_index = group.prompt_index
        final_index = row_indices[-1]
        response_ids = [token for index in row_indices for token in output["response_ids"][index]]
        loss_mask = [value for index in row_indices for value in output["loss_masks"][index]]
        prompt_ids = output["prompt_token_ids"][row_indices[0]]
        normalized_reward = NormalizedReward.from_output(output["rewards"][final_index])
        shaped_reward = normalized_reward.total
        outcome = float(unshaped[final_index]) if unshaped is not None else normalized_reward.outcome
        trajectory_components = None
        if components is not None:
            trajectory_components = aggregate_reward_shaping_components(components, row_indices)
        response_text = _decode(tokenizer, response_ids)
        record = TrajectoryRecord(
            schema_version=TRAJECTORY_RECORD_SCHEMA_VERSION,
            record_id="",
            run_id=config.run_id,
            global_step=metadata.global_step,
            phase=metadata.training_phase,
            trajectory=_TrajectoryIdentity(
                instance_id=trajectory_id.instance_id,
                repetition_id=trajectory_id.repetition_id,
                row_indices=row_indices,
                environment_class=rows[prompt_index].metric_class,
                environment_extras={
                    "data_source": rows[prompt_index].data_source,
                    "teacher_route": rows[prompt_index].teacher_route,
                },
            ),
            prompt=_PromptTrace(
                messages=to_jsonable(prompts[prompt_index]),
                token_ids=tuple(prompt_ids),
            ),
            response=_ResponseTrace(
                messages=(
                    tuple(output["evidence_messages"][final_index])
                    if output.get("evidence_messages") is not None
                    and output["evidence_messages"][final_index] is not None
                    else None
                    if response_text is None
                    else ({"role": "assistant", "content": response_text},)
                ),
                text=response_text,
                token_ids=tuple(response_ids),
                loss_mask=tuple(loss_mask),
                trainable_spans=tuple(_active_spans(loss_mask)),
                step_boundaries=tuple(_step_boundaries(output, row_indices, stop_reasons)),
                stop_reason=stop_reasons[final_index],
                generation_limit=(input_batch.sampling_params or {}).get(
                    "max_tokens", (input_batch.sampling_params or {}).get("max_generate_length")
                ),
                loop_spans=tuple(() if loop_spans is None else loop_spans[final_index]),
            ),
            reward=_RewardTrace(
                outcome=outcome,
                shaped=shaped_reward,
                components=trajectory_components,
            ),
            disposition=_DispositionTrace(
                exception_type=exception_types[final_index],
                error_treatment=error_treatments[final_index],
                server_error=server_errors[final_index],
            ),
            verifier=None if verifier_tests is None else verifier_tests[final_index],
            verification_result=(
                output["verification_results"][final_index] if output.get("verification_results") is not None else None
            ),
            metrics=to_jsonable(output.get("rollout_metrics") or {}),
            provenance=_ProvenanceTrace(
                runner=runner_name,
                inference_backend=config.inference_backend,
                model_path=config.model_path,
                model_source_identity=config.model_source_identity,
                resume_path=config.resume_path,
                model_version_step=model_version_step,
                sampling=to_jsonable(input_batch.sampling_params or {}),
                reward_shaping_schema_version=None if shaping_versions is None else shaping_versions[final_index],
            ),
        )
        digest_source = canonical_json_bytes(_serialized_record(record, config.redact_fields))
        records.append(replace(record, record_id=hashlib.sha256(digest_source).hexdigest()))
    return records


class TrajectoryRetentionPublicationError(RuntimeError):
    """A required trajectory-retention publication failed."""


class TrajectoryRetentionPublicationTimeout(TrajectoryRetentionPublicationError):
    """A required trajectory-retention publication exceeded its deadline."""


def _copy_ledger(ledger: _RetentionLedger) -> _RetentionLedger:
    return _RetentionLedger(
        schema_version=ledger.schema_version,
        total_bytes=ledger.total_bytes,
        step_bytes=dict(ledger.step_bytes),
        records=dict(ledger.records),
        archives=dict(ledger.archives),
    )


def _archive_payload(selected: Sequence[_SelectedRecord]) -> bytes:
    manifest_records = []
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for item in selected:
            entry_name = f"records/{item.record.record_id}.json.gz"
            info = zipfile.ZipInfo(entry_name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, item.payload)
            manifest_records.append(
                {
                    "record_id": item.record.record_id,
                    "step": f"{item.record.phase}:{item.record.global_step}",
                    "reasons": item.selection.reasons,
                    "sample_score": TrajectorySink._sample_score(item.record),
                    "bytes": len(item.payload),
                    "entry": entry_name,
                }
            )
        manifest = {"schema_version": RETENTION_SCHEMA_VERSION, "records": manifest_records}
        info = zipfile.ZipInfo(_ARCHIVE_MANIFEST)
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, canonical_json_bytes(manifest))
    return buffer.getvalue()


def _read_archive_manifest(payload: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(BytesIO(payload), mode="r") as archive:
        value = json.loads(archive.read(_ARCHIVE_MANIFEST))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != RETENTION_SCHEMA_VERSION:
        raise ValueError("trajectory retention archive manifest is incompatible")
    return value


def _remove_ledger_path(ledger: _RetentionLedger, path: str) -> None:
    archive = ledger.archives.pop(path, None)
    if archive is not None:
        ledger.total_bytes -= archive.bytes
        ledger.step_bytes[archive.step] -= archive.bytes
        if ledger.step_bytes[archive.step] == 0:
            del ledger.step_bytes[archive.step]
        for record_id in archive.record_ids:
            ledger.records.pop(record_id, None)
        return

    removed = [record_id for record_id, entry in ledger.records.items() if entry.path == path]
    for record_id in removed:
        entry = ledger.records.pop(record_id)
        ledger.total_bytes -= entry.bytes
        ledger.step_bytes[entry.step] -= entry.bytes
        if ledger.step_bytes[entry.step] == 0:
            del ledger.step_bytes[entry.step]


def _add_archive_to_ledger(ledger: _RetentionLedger, path: str, payload: bytes) -> None:
    manifest = _read_archive_manifest(payload)
    record_ids = []
    step = None
    for value in manifest.get("records", ()):
        record_id = str(value["record_id"])
        record_step = str(value["step"])
        step = record_step if step is None else step
        if step != record_step:
            raise ValueError(f"trajectory retention archive mixes steps: {path}")
        ledger.records[record_id] = _LedgerEntry(
            path=path,
            bytes=int(value["bytes"]),
            step=record_step,
            reasons=tuple(str(reason) for reason in value.get("reasons", ())),
            sample_score=int(value["sample_score"]),
        )
        record_ids.append(record_id)
    if step is None:
        raise ValueError(f"trajectory retention archive contains no records: {path}")
    archive_bytes = len(payload)
    ledger.archives[path] = _ArchiveEntry(bytes=archive_bytes, step=step, record_ids=tuple(record_ids))
    ledger.total_bytes += archive_bytes
    ledger.step_bytes[step] = ledger.step_bytes.get(step, 0) + archive_bytes


def _sample_score(
    run_id: str,
    phase: str,
    global_step: int,
    instance_id: str,
    repetition_id: int,
) -> int:
    sample_key = ":".join(
        (
            run_id,
            phase,
            str(global_step),
            instance_id,
            str(repetition_id),
        )
    )
    return int(hashlib.sha256(sample_key.encode("utf-8")).hexdigest(), 16)


def _serialized_sample_score(value: Mapping[str, Any]) -> int:
    trajectory = value["trajectory"]
    return _sample_score(
        str(value["run_id"]),
        str(value["phase"]),
        int(value["global_step"]),
        str(trajectory["instance_id"]),
        int(trajectory["repetition_id"]),
    )


def _is_mandatory(
    *,
    outcome: float,
    stop_reason: str | None,
    has_loops: bool,
    config: TrajectoryRetentionConfig,
) -> bool:
    normalized_stop = None if stop_reason is None else str(stop_reason).strip().lower()
    return any(
        (
            config.always_retain_failures and outcome <= 0.0,
            config.always_retain_non_terminating and normalized_stop not in config.accepted_stop_reasons,
            config.always_retain_loops and has_loops,
            config.reward_below is not None and outcome <= config.reward_below,
            config.reward_above is not None and outcome >= config.reward_above,
        )
    )


def _orphan_reasons(value: Mapping[str, Any], config: TrajectoryRetentionConfig) -> tuple[str, ...]:
    reward = value["reward"]
    response = value["response"]
    outcome = float(reward["outcome"])
    stop_reason = response.get("stop_reason")
    mandatory = _is_mandatory(
        outcome=outcome,
        stop_reason=None if stop_reason is None else str(stop_reason),
        has_loops=bool(response.get("loop_spans")),
        config=config,
    )
    if mandatory:
        return (_SELECTION_MANDATORY,)
    score = _serialized_sample_score(value)
    if score / (1 << 256) < config.sample_fraction:
        return (_SELECTION_FRACTION,)
    return ()


def _recompute_count_samples(ledger: _RetentionLedger, config: TrajectoryRetentionConfig) -> None:
    by_step: dict[str, list[tuple[str, _LedgerEntry]]] = {}
    for record_id, entry in ledger.records.items():
        reasons = tuple(reason for reason in entry.reasons if reason != _SELECTION_COUNT)
        ledger.records[record_id] = replace(entry, reasons=reasons)
        if _SELECTION_MANDATORY not in reasons:
            by_step.setdefault(entry.step, []).append((record_id, ledger.records[record_id]))
    for candidates in by_step.values():
        for record_id, entry in sorted(candidates, key=lambda item: item[1].sample_score)[
            : config.sample_count_per_step
        ]:
            ledger.records[record_id] = replace(entry, reasons=entry.reasons + (_SELECTION_COUNT,))


def _reconciled_ledger(writer: TrajectoryWriter, config: TrajectoryRetentionConfig) -> _RetentionLedger:
    ledger = _RetentionLedger.from_json(writer.read_json(_LEDGER_NAME))
    files = writer.find_files()
    known_paths = set(ledger.archives)
    known_paths.update(entry.path for entry in ledger.records.values() if entry.path not in ledger.archives)
    for missing_path in known_paths - files.keys():
        _remove_ledger_path(ledger, missing_path)

    for path in sorted(files.keys() - known_paths):
        if path.endswith(".zip") and f"/{_ARCHIVE_DIRECTORY}/" in f"/{path}":
            _add_archive_to_ledger(ledger, path, writer.read_bytes(path))
            continue
        if not path.endswith(".json.gz"):
            continue
        payload = writer.read_bytes(path)
        value = json.loads(gzip.decompress(payload))
        record_id = str(value["record_id"])
        if record_id in ledger.records:
            continue
        step = f"{value['phase']}:{value['global_step']}"
        ledger.records[record_id] = _LedgerEntry(
            path=path,
            bytes=len(payload),
            step=step,
            reasons=_orphan_reasons(value, config),
            sample_score=_serialized_sample_score(value),
        )
        ledger.total_bytes += len(payload)
        ledger.step_bytes[step] = ledger.step_bytes.get(step, 0) + len(payload)
    _recompute_count_samples(ledger, config)
    return ledger


def _initialize_publication(request: PublicationRequest) -> _RetentionLedger:
    writer = _FilesystemTrajectoryWriter(request.output_path)
    config = parse_trajectory_retention_config(request.retention_config)
    return _reconciled_ledger(writer, config)


def _publish_archives(request: PublicationRequest) -> _RetentionLedger:
    if request.archives is None or request.ledger is None:
        raise ValueError("trajectory publication request is incomplete")
    writer = _FilesystemTrajectoryWriter(request.output_path)
    for path, payload in request.archives.items():
        if not writer.exists(path):
            writer.write_bytes(path, payload)
    ledger = _RetentionLedger.from_json(request.ledger)
    writer.write_json(_LEDGER_NAME, to_jsonable(ledger))
    return ledger


def execute_publication(request: PublicationRequest) -> PublicationResult:
    """Perform one storage operation, reporting any failure in the result rather than raising."""
    try:
        if request.operation is PublicationOperation.INITIALIZE:
            ledger = _initialize_publication(request)
        elif request.operation is PublicationOperation.PUBLISH:
            ledger = _publish_archives(request)
        else:
            raise ValueError(f"unknown trajectory publication operation: {request.operation}")
    except Exception as error:
        return PublicationResult(
            request_id=request.request_id,
            record_count=request.record_count,
            error=f"{type(error).__name__}: {error}",
        )
    return PublicationResult(
        request_id=request.request_id,
        record_count=request.record_count,
        ledger=to_jsonable(ledger),
    )


def _empty_metrics() -> dict[str, float]:
    return {
        f"{RETENTION_METRIC_PREFIX}/candidates": 0.0,
        f"{RETENTION_METRIC_PREFIX}/selected": 0.0,
        f"{RETENTION_METRIC_PREFIX}/written": 0.0,
        f"{RETENTION_METRIC_PREFIX}/duplicates": 0.0,
        f"{RETENTION_METRIC_PREFIX}/dropped_by_bounds": 0.0,
        f"{RETENTION_METRIC_PREFIX}/write_errors": 0.0,
        f"{RETENTION_METRIC_PREFIX}/bytes_written": 0.0,
        f"{RETENTION_METRIC_PREFIX}/enqueued": 0.0,
        f"{RETENTION_METRIC_PREFIX}/publish_timeouts": 0.0,
    }


class RetentionSink(Protocol):
    """Where trajectory runners send finished batches for retention."""

    config: TrajectoryRetentionConfig

    def bind_runner(self, runner_name: str) -> None: ...

    def retain(self, input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> dict[str, float]: ...

    def close(self) -> None: ...


class TrajectorySink:
    """Select, bound, and persist normalized trajectory records.

    A best-effort sink runs one storage operation at a time. Records selected while one is in flight wait in a
    queue, and the next operation publishes all of them, one archive per step, so a run that retains one group at
    a time keeps every record it selects.
    """

    def __init__(
        self,
        config: TrajectoryRetentionConfig,
        tokenizer: PreTrainedTokenizerBase,
        publisher: TrajectoryPublisher | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.publisher = publisher or ProcessTrajectoryPublisher(
            execute_publication,
            publish_timeout_seconds=config.publish_timeout_seconds,
            shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        )
        self._lock = threading.Lock()
        # Includes the queued records, so selection sees every record the sink has accepted.
        self._ledger: _RetentionLedger | None = None
        self._queued: list[_SelectedRecord] = []
        self._runner_name: str | None = None
        self._pending_operation: PublicationOperation | None = None
        self._pending_archive_paths: tuple[str, ...] = ()
        self._pending_archive_bytes = 0
        if config.enabled:
            result = self.publisher.execute(self._initialization_request())
            if result.error is None:
                self._ledger = _RetentionLedger.from_json(result.ledger)
            elif config.required:
                self._raise_publication_error(result, _LEDGER_NAME)
            else:
                logger.error("Failed to initialize trajectory retention: {}", result.error)

    def bind_runner(self, runner_name: str) -> None:
        """Bind the runner identity once when the sink is attached."""
        if self._runner_name not in (None, runner_name):
            raise ValueError(f"trajectory sink is already bound to {self._runner_name}")
        self._runner_name = runner_name

    def retain(
        self,
        input_batch: TrajectoryRequestBatch,
        output: TrajectoryBatch,
    ) -> dict[str, float]:
        """Retain eligible records, or return no metrics when retention does not apply."""
        if not self.config.enabled:
            return {}
        metadata = input_batch.batch_metadata
        if metadata is None or metadata.training_phase not in self.config.phases:
            return {}
        if self._runner_name is None:
            raise ValueError("trajectory sink must be bound to a runner before retention")

        with self._lock:
            return self._retain_locked(input_batch, output)

    def close(self) -> None:
        """Finish or cancel the pending publication within the shutdown deadline, then publish queued records."""
        with self._lock:
            result = self.publisher.close()
            if result is not None:
                self._finish_publication(result, _empty_metrics())
            if not self._queued:
                return
            result = self.publisher.execute(self._queued_publication())
            if result.error is not None:
                logger.error("Trajectory retention publication did not finish during shutdown: {}", result.error)

    def _retain_locked(
        self,
        input_batch: TrajectoryRequestBatch,
        output: TrajectoryBatch,
    ) -> dict[str, float]:
        metrics = _empty_metrics()
        self._drain_publication(metrics)
        records = build_trajectory_records(
            input_batch,
            output,
            self.config,
            self.tokenizer,
            runner_name=self._runner_name,
        )
        metrics[f"{RETENTION_METRIC_PREFIX}/candidates"] = float(len(records))
        if not self._ensure_ledger(metrics, len(records)):
            return metrics
        self._queue_selected(records, metrics)
        if self._queued and self._pending_operation is None:
            self._publish_queued(metrics)
        return metrics

    def _ensure_ledger(self, metrics: dict[str, float], record_count: int) -> bool:
        if self._ledger is not None:
            return True
        if not self.config.required:
            self._start_initialization()
            metrics[f"{RETENTION_METRIC_PREFIX}/write_errors"] += float(record_count)
            return False
        result = self.publisher.execute(self._initialization_request())
        if result.error is not None:
            self._raise_publication_error(result, _LEDGER_NAME)
        self._ledger = _RetentionLedger.from_json(result.ledger)
        return True

    def _publish_queued(self, metrics: dict[str, float]) -> None:
        request = self._queued_publication()
        assert request.archives is not None
        archive_bytes = sum(len(payload) for payload in request.archives.values())
        if self.config.required:
            result = self.publisher.execute(request)
            if result.error is not None:
                self._raise_publication_error(result, ", ".join(request.archives))
            metrics[f"{RETENTION_METRIC_PREFIX}/written"] += float(request.record_count)
            metrics[f"{RETENTION_METRIC_PREFIX}/bytes_written"] += float(archive_bytes)
            return

        submitted = self.publisher.submit(request)
        assert submitted, "the sink submits only when no publication is pending"
        self._pending_operation = PublicationOperation.PUBLISH
        self._pending_archive_paths = tuple(request.archives)
        self._pending_archive_bytes = archive_bytes
        metrics[f"{RETENTION_METRIC_PREFIX}/enqueued"] += float(request.record_count)

    def _queued_publication(self) -> PublicationRequest:
        """Move the queued records into one archive per step, and return the request that publishes them."""
        assert self._ledger is not None
        by_step: dict[str, list[_SelectedRecord]] = {}
        for selected_record in self._queued:
            by_step.setdefault(self._step_key(selected_record.record), []).append(selected_record)
        record_count = len(self._queued)
        self._queued = []
        archives = {}
        for step_key, selected in by_step.items():
            payload = _archive_payload(selected)
            path = self._archive_path(selected[0].record, payload)
            record_ids = tuple(item.record.record_id for item in selected)
            for record_id in record_ids:
                self._ledger.records[record_id] = replace(self._ledger.records[record_id], path=path)
            self._ledger.archives[path] = _ArchiveEntry(bytes=len(payload), step=step_key, record_ids=record_ids)
            self._ledger.step_bytes[step_key] = self._ledger.step_bytes.get(step_key, 0) + len(payload)
            self._ledger.total_bytes += len(payload)
            archives[path] = payload
        return PublicationRequest(
            request_id=hashlib.sha256("".join(archives).encode("utf-8")).hexdigest(),
            operation=PublicationOperation.PUBLISH,
            output_path=self.config.output_path,
            archives=archives,
            ledger=to_jsonable(self._ledger),
            retention_config=to_jsonable(self.config),
            record_count=record_count,
        )

    def _queue_selected(self, records: Sequence[TrajectoryRecord], metrics: dict[str, float]) -> None:
        """Queue the payload-bounded priority prefix of the records that qualify for retention."""
        assert self._ledger is not None
        selection_ledger = _copy_ledger(self._ledger)
        selected = []
        for record in sorted(records, key=self._sample_score):
            if record.record_id in selection_ledger.records:
                metrics[f"{RETENTION_METRIC_PREFIX}/duplicates"] += 1.0
                continue
            selection = self._selection(record, selection_ledger)
            if selection is None:
                continue
            metrics[f"{RETENTION_METRIC_PREFIX}/selected"] += 1.0
            payload = self._encode_record(record)
            selected_record = _SelectedRecord(record=record, payload=payload, selection=selection)
            self._add_selected_record(selection_ledger, selected_record)
            selected.append(selected_record)

        if not selected:
            return

        step_key = self._step_key(selected[0].record)
        # Queued records count against the bounds before their archives exist.
        queued_bytes = sum(len(item.payload) for item in self._queued)
        queued_step_bytes = sum(len(item.payload) for item in self._queued if self._step_key(item.record) == step_key)
        remaining_limits = []
        if self.config.max_bytes_per_step is not None:
            remaining_limits.append(
                self.config.max_bytes_per_step - self._ledger.step_bytes.get(step_key, 0) - queued_step_bytes
            )
        if self.config.max_bytes_per_run is not None:
            remaining_limits.append(self.config.max_bytes_per_run - self._ledger.total_bytes - queued_bytes)
        max_payload_bytes = min(remaining_limits) if remaining_limits else None
        retained = []
        retained_payload_bytes = 0
        for selected_record in selected:
            candidate_bytes = retained_payload_bytes + len(selected_record.payload)
            if max_payload_bytes is not None and candidate_bytes > max_payload_bytes:
                break
            retained.append(selected_record)
            retained_payload_bytes = candidate_bytes
        metrics[f"{RETENTION_METRIC_PREFIX}/dropped_by_bounds"] += float(len(selected) - len(retained))

        for selected_record in retained:
            self._add_selected_record(self._ledger, selected_record)
        # A queued record displaced as a count sample, with no other reason to keep it, is no longer retained.
        self._queued = [item for item in (*self._queued, *retained) if item.record.record_id in self._ledger.records]

    def _add_selected_record(self, ledger: _RetentionLedger, selected: _SelectedRecord) -> None:
        record = selected.record
        if selected.selection.displaced_id is not None:
            self._remove_count_reason(ledger, selected.selection.displaced_id)
        ledger.records[record.record_id] = _LedgerEntry(
            path="",
            bytes=len(selected.payload),
            step=self._step_key(record),
            reasons=selected.selection.reasons,
            sample_score=self._sample_score(record),
        )

    def _encode_record(self, record: TrajectoryRecord) -> bytes:
        return gzip.compress(
            canonical_json_bytes(_serialized_record(record, self.config.redact_fields)),
            mtime=0,
        )

    def _initialization_request(self) -> PublicationRequest:
        return PublicationRequest(
            request_id="initialize",
            operation=PublicationOperation.INITIALIZE,
            output_path=self.config.output_path,
            retention_config=to_jsonable(self.config),
        )

    def _start_initialization(self) -> None:
        if self._pending_operation is not None:
            return
        if self.publisher.submit(self._initialization_request()):
            self._pending_operation = PublicationOperation.INITIALIZE

    def _drain_publication(self, metrics: dict[str, float]) -> None:
        result = self.publisher.poll()
        if result is not None:
            self._finish_publication(result, metrics)

    def _finish_publication(self, result: PublicationResult, metrics: dict[str, float]) -> None:
        operation = self._pending_operation
        archive_paths = self._pending_archive_paths
        archive_bytes = self._pending_archive_bytes
        self._pending_operation = None
        self._pending_archive_paths = ()
        self._pending_archive_bytes = 0
        if result.error is not None:
            logger.error("Trajectory retention {} failed: {}", operation, result.error)
            # The ledger now describes records storage may not hold, so reload it from storage and drop the
            # records queued against it.
            metrics[f"{RETENTION_METRIC_PREFIX}/write_errors"] += float(max(1, result.record_count) + len(self._queued))
            metrics[f"{RETENTION_METRIC_PREFIX}/publish_timeouts"] += float(result.timed_out)
            self._ledger = None
            self._queued = []
            return
        if operation is PublicationOperation.INITIALIZE:
            self._ledger = _RetentionLedger.from_json(result.ledger)
            return
        # The sink's ledger already includes this publication and every record queued since.
        metrics[f"{RETENTION_METRIC_PREFIX}/written"] += float(result.record_count)
        metrics[f"{RETENTION_METRIC_PREFIX}/bytes_written"] += float(archive_bytes)
        logger.info("Published trajectory retention archives {}", ", ".join(archive_paths))

    @staticmethod
    def _raise_publication_error(result: PublicationResult, path: str) -> None:
        message = f"trajectory retention publication failed for {path}: {result.error}"
        if result.timed_out:
            raise TrajectoryRetentionPublicationTimeout(message)
        raise TrajectoryRetentionPublicationError(message)

    def _selection(self, record: TrajectoryRecord, ledger: _RetentionLedger) -> _Selection | None:
        reasons = []
        if _is_mandatory(
            outcome=record.reward.outcome,
            stop_reason=record.response.stop_reason,
            has_loops=bool(record.response.loop_spans),
            config=self.config,
        ):
            return _Selection((_SELECTION_MANDATORY,))

        score = self._sample_score(record)
        if score / (1 << 256) < self.config.sample_fraction:
            reasons.append(_SELECTION_FRACTION)

        displaced_id = None
        count_samples = self._count_samples(ledger, self._step_key(record))
        if self.config.sample_count_per_step > 0:
            if len(count_samples) < self.config.sample_count_per_step:
                reasons.append(_SELECTION_COUNT)
            else:
                largest_id, largest_entry = max(count_samples, key=lambda item: item[1].sample_score)
                if score < largest_entry.sample_score:
                    reasons.append(_SELECTION_COUNT)
                    displaced_id = largest_id

        return _Selection(tuple(reasons), displaced_id) if reasons else None

    @staticmethod
    def _count_samples(ledger: _RetentionLedger, step_key: str) -> list[tuple[str, _LedgerEntry]]:
        return [
            (record_id, entry)
            for record_id, entry in ledger.records.items()
            if entry.step == step_key and _SELECTION_COUNT in entry.reasons
        ]

    @staticmethod
    def _remove_count_reason(ledger: _RetentionLedger, displaced_id: str) -> None:
        displaced = ledger.records[displaced_id]
        remaining_reasons = tuple(reason for reason in displaced.reasons if reason != _SELECTION_COUNT)
        if remaining_reasons:
            ledger.records[displaced_id] = replace(displaced, reasons=remaining_reasons)
            return
        del ledger.records[displaced_id]

    @staticmethod
    def _sample_score(record: TrajectoryRecord) -> int:
        return _sample_score(
            record.run_id,
            record.phase,
            record.global_step,
            record.trajectory.instance_id,
            record.trajectory.repetition_id,
        )

    @staticmethod
    def _step_key(record: TrajectoryRecord) -> str:
        return f"{record.phase}:{record.global_step}"

    @staticmethod
    def _archive_path(record: TrajectoryRecord, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        return posixpath.join(
            f"schema_v{record.schema_version}",
            _ARCHIVE_DIRECTORY,
            f"phase={record.phase}",
            f"step={record.global_step:08d}",
            f"{digest}.zip",
        )


# Its own thread: the sink serialises on one lock anyway, and the default executor is shared with
# the training step, which a burst of finished rollout groups would queue in front of.
_RETENTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="retention")


async def retain_trajectories(
    sink: RetentionSink,
    input_batch: TrajectoryRequestBatch,
    output: TrajectoryBatch,
) -> None:
    """Persist normalized trajectories without blocking the trainer event loop."""
    try:
        metrics = await asyncio.get_running_loop().run_in_executor(
            _RETENTION_EXECUTOR,
            sink.retain,
            input_batch,
            output,
        )
    except Exception:
        # `required: false` promises best effort, and the sink already honours that for publication
        # failures. Retention runs inside the rollout worker, whose own handler exits the process, so
        # without this a serialization or schema bug in retention ends the training run.
        if sink.config.required:
            raise
        logger.exception("Trajectory retention failed; continuing without it")
        # Seeded from the zero-filled keyset: an absent `written` reads as no data, not as none written.
        metrics = _empty_metrics()
        metrics[f"{RETENTION_METRIC_PREFIX}/write_errors"] = float(len(output.get("response_ids") or ()))
    if not metrics:
        return
    rollout_metrics = output.get("rollout_metrics") or {}
    rollout_metrics.update(metrics)
    output["rollout_metrics"] = rollout_metrics


class SharedTrajectorySink:
    """A picklable handle to the run's one ``TrajectorySink`` actor.

    Rollout workers retain from many processes, and the retention ledger admits a single writer. Only the
    trainer, which owns the sink, closes it.
    """

    def __init__(self, config: TrajectoryRetentionConfig, actor: ActorHandle):
        self.config = config
        self._actor = actor

    def bind_runner(self, runner_name: str) -> None:
        ray.get(self._actor.bind_runner.remote(runner_name))

    def retain(self, input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> dict[str, float]:
        return ray.get(self._actor.retain.remote(input_batch, output))

    def close(self) -> None:
        ray.get(self._actor.close.remote())
        ray.kill(self._actor)


class DisabledTrajectorySink:
    """The sink of a run that retains nothing, which needs no actor."""

    def __init__(self, config: TrajectoryRetentionConfig):
        self.config = config

    def bind_runner(self, runner_name: str) -> None:
        """Nothing is retained, so no runner identity is recorded."""

    def retain(self, input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> dict[str, float]:
        return {}

    def close(self) -> None:
        """The sink holds no resources."""


def make_trajectory_sink(config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> RetentionSink:
    """Start the run's sink on this node, where a local ``output_path`` resolves as it does here."""
    retention = parse_trajectory_retention_config(config.get("trajectory_retention"))
    if not retention.enabled:
        return DisabledTrajectorySink(retention)
    node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    actor = ray.remote(TrajectorySink).options(num_cpus=0, scheduling_strategy=node).remote(retention, tokenizer)
    return SharedTrajectorySink(retention, actor)
