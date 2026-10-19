"""Stable sample identity and coverage shared by every evaluation route."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SAMPLE_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_NAMESPACE = "default"
DEFAULT_SAMPLE_GROUP = "examples"


class SampleManifestError(RuntimeError):
    """Base class for invalid sample manifests."""


class SampleIdentityError(SampleManifestError):
    """Raised before generation when sample identity is absent or unstable."""


class SampleCoverageError(SampleManifestError):
    """Raised when planned, generated, and scored sample coverage diverges."""


@dataclass(frozen=True)
class SampleRequest:
    """Benchmark-provided identity plus framework-owned execution coordinates."""

    source_id: Any
    ordinal: int
    namespace: str = DEFAULT_SAMPLE_NAMESPACE
    shard: int | None = None
    repeat: int | None = None


@dataclass(frozen=True)
class SampleEntry:
    """One validated unit in a task's sample manifest."""

    sample_id: str
    unit_id: str
    source_id: Any
    ordinal: int
    namespace: str
    shard: int | None
    repeat: int | None

    def resume_unit(self, task_name: str) -> dict[str, str]:
        """Return the shared opaque key used by resume storage."""
        return {"task": task_name, "sample_id": self.unit_id}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe identity and coordinates for artifacts and resume guards."""
        return {
            "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "unit_id": self.unit_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "namespace": self.namespace,
            "shard": self.shard,
            "repeat": self.repeat,
        }


class SampleManifest:
    """Thread-safe lifecycle ledger for one requested benchmark task."""

    def __init__(self, task_name: str):
        if not isinstance(task_name, str) or not task_name:
            raise SampleIdentityError("sample manifest task_name must be a non-empty string")
        self.task_name = task_name
        self._entries: dict[str, SampleEntry] = {}
        self._generated: set[str] = set()
        self._scored: set[str] = set()
        self._lock = threading.RLock()

    def plan_batch(self, requests: Sequence[SampleRequest]) -> tuple[SampleEntry, ...]:
        """Validate and register a request batch before any model call."""
        entries = tuple(self._entry(request) for request in requests)
        unit_ids = [entry.unit_id for entry in entries]
        if len(unit_ids) != len(set(unit_ids)):
            raise SampleIdentityError(f"{self.task_name}: duplicate sample identity in one generation batch")

        with self._lock:
            for entry in entries:
                prior = self._entries.get(entry.unit_id)
                if prior is not None and prior != entry:
                    if prior.ordinal != entry.ordinal:
                        detail = f"moved from ordinal {prior.ordinal} to {entry.ordinal}"
                    else:
                        detail = "changed execution coordinates"
                    raise SampleIdentityError(f"{self.task_name}: sample identity {entry.source_id!r} {detail}")
                self._entries[entry.unit_id] = entry
        return entries

    def mark_generated(self, entries: Sequence[SampleEntry], outputs: Sequence[Any]) -> None:
        """Mark an exact request batch generated; reject zip-style truncation."""
        self.validate_output_count(len(entries), outputs)

        with self._lock:
            for entry in entries:
                if entry.unit_id not in self._entries:
                    raise SampleCoverageError(f"{self.task_name}: generated an unplanned sample unit {entry.unit_id}")
                self._generated.add(entry.unit_id)

    def validate_output_count(self, expected_count: int, outputs: Sequence[Any]) -> None:
        """Reject non-sequences and short/extra transport responses."""
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
            raise SampleCoverageError(f"{self.task_name}: model outputs must be a sequence")
        if len(outputs) != expected_count:
            raise SampleCoverageError(
                f"{self.task_name}: model returned {len(outputs)} outputs for {expected_count} planned units"
            )

    def adopt_entries(self, entries: Sequence[SampleEntry]) -> None:
        """Register already-validated entries gathered from another rank."""
        with self._lock:
            for entry in entries:
                prior = self._entries.get(entry.unit_id)
                if prior is not None and prior != entry:
                    raise SampleIdentityError(
                        f"{self.task_name}: distributed ranks disagree about sample " f"identity {entry.source_id!r}"
                    )
                self._entries[entry.unit_id] = entry

    def validate_prior_entries(
        self,
        restored_payloads: Sequence[Mapping[str, Any]],
        current_entries: Sequence[SampleEntry],
    ) -> None:
        """Reject persisted identities absent from the equivalent current batch."""
        current_coordinates = {(entry.namespace, entry.repeat) for entry in current_entries}
        current_units = {entry.unit_id for entry in current_entries}
        for payload in restored_payloads:
            samples = payload.get("samples")
            if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
                samples = [payload.get("sample")]
            for sample in samples:
                if not isinstance(sample, Mapping):
                    continue
                coordinates = (sample.get("namespace"), sample.get("repeat"))
                if coordinates not in current_coordinates:
                    continue
                prior_unit = sample.get("unit_id")
                if prior_unit not in current_units:
                    raise SampleIdentityError(
                        f"{self.task_name}: persisted sample identity "
                        f"{sample.get('source_id')!r} is absent from the current request batch"
                    )

    def mark_scored(self, scored_sample_count: int | None) -> None:
        """Reconcile a grader's sample count, then close generated units as scored."""
        if scored_sample_count is not None and scored_sample_count != self.generated_sample_count:
            raise SampleCoverageError(
                f"{self.task_name}: grader reported {scored_sample_count} samples but the manifest "
                f"contains {self.generated_sample_count} generated samples"
            )
        with self._lock:
            self._scored = set(self._generated)

    def validate_generated(self) -> None:
        """Require every planned unit, and no extra unit, to have generated output."""
        with self._lock:
            missing = set(self._entries) - self._generated
            extra = self._generated - set(self._entries)
        if missing or extra:
            raise SampleCoverageError(
                f"{self.task_name}: sample manifest generation coverage differs "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

    def validate_scored(self) -> None:
        """Require generated and scored unit sets to match exactly."""
        self.validate_generated()
        with self._lock:
            missing = self._generated - self._scored
            extra = self._scored - self._generated
        if missing or extra:
            raise SampleCoverageError(
                f"{self.task_name}: sample manifest scoring coverage differs "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

    @property
    def expected_unit_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def generated_unit_count(self) -> int:
        with self._lock:
            return len(self._generated)

    @property
    def expected_sample_count(self) -> int:
        with self._lock:
            return self._sample_count(set(self._entries))

    @property
    def generated_sample_count(self) -> int:
        return self._sample_count(self._generated)

    @property
    def scored_sample_count(self) -> int:
        return self._sample_count(self._scored)

    def sample_entries(self, namespace: str | None = None) -> tuple[SampleEntry, ...]:
        """Return one ordered entry per source sample for artifact serialization."""
        with self._lock:
            entries = [entry for entry in self._entries.values() if namespace is None or entry.namespace == namespace]
        by_sample: dict[str, SampleEntry] = {}
        for entry in entries:
            by_sample.setdefault(entry.sample_id, entry)
        unique_entries = tuple(by_sample.values())
        if namespace is None:
            return unique_entries
        return tuple(sorted(unique_entries, key=lambda entry: entry.ordinal))

    def unit_entries(self, namespace: str | None = None) -> tuple[SampleEntry, ...]:
        """Return every generated unit, including independently repeated trials."""
        with self._lock:
            entries = tuple(
                entry for entry in self._entries.values() if namespace is None or entry.namespace == namespace
            )
        return entries

    def _sample_count(self, unit_ids: set[str]) -> int:
        with self._lock:
            return len({self._entries[unit_id].sample_id for unit_id in unit_ids})

    def _entry(self, request: SampleRequest) -> SampleEntry:
        _validate_request(request, self.task_name)
        sample_payload = {
            "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
            "task": self.task_name,
            "namespace": request.namespace,
            "source_id": request.source_id,
        }
        sample_id = _opaque_id(sample_payload)
        unit_id = _opaque_id({**sample_payload, "repeat": request.repeat})
        return SampleEntry(
            sample_id=sample_id,
            unit_id=unit_id,
            source_id=request.source_id,
            ordinal=request.ordinal,
            namespace=request.namespace,
            shard=request.shard,
            repeat=request.repeat,
        )


def _opaque_id(value: Mapping[str, Any]) -> str:
    encoded = canonical_json_identity(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_identity(value: Any) -> str:
    """Return deterministic JSON for any identity value accepted by the schema."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SampleIdentityError(f"identity must be stable JSON data: {exc}") from exc


def _validate_request(request: SampleRequest, task_name: str) -> None:
    if request.source_id is None:
        raise SampleIdentityError(f"{task_name}: sample source_id is missing")
    if not isinstance(request.ordinal, int) or isinstance(request.ordinal, bool) or request.ordinal < 0:
        raise SampleIdentityError(f"{task_name}: sample ordinal must be a non-negative integer")
    if not isinstance(request.namespace, str) or not request.namespace:
        raise SampleIdentityError(f"{task_name}: sample namespace must be a non-empty string")
    for name, value in (("shard", request.shard), ("repeat", request.repeat)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise SampleIdentityError(f"{task_name}: sample {name} must be a non-negative integer or null")
    try:
        canonical_json_identity(request.source_id)
    except SampleIdentityError as exc:
        raise SampleIdentityError(f"{task_name}: sample source_id {exc}") from exc
