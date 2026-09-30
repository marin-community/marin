"""Catalog ingestion, corruption auditing, and deterministic pilot selection.

The source catalog is controller-sized (roughly one MiB), so parsing it locally is
both cheaper and more reliable than spending inference capacity on transport and
schema recovery.  The parser is deliberately fail-closed: it only emits JSON
objects that ``json.JSONDecoder`` decoded completely.  It never repairs a partial
string or fabricates missing curriculum metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

# One capability per subject keeps the pilot broad while emphasizing cases where
# realistic artifacts and objective checks can be built.  Order is stable and is
# the order proposal-generation jobs should use. The first twenty selections are
# the original corrupt-catalog pilot; the final fourteen cover the curricula that
# became available when the authoritative catalog was restored.
DEFAULT_PILOT_SELECTIONS: tuple[tuple[str, str], ...] = (
    (
        "c03.backup-restoration",
        "Stateful host workflow with destructive-looking failure modes and an objective restore check.",
    ),
    (
        "logic-verification.system-verification",
        "Supports pure reasoning and executable invariant checking with exact counterexamples.",
    ),
    (
        "c21.spreadsheets.calculate",
        "Produces inspectable office artifacts whose formulas and outputs can be recalculated.",
    ),
    (
        "c01.3.3.2",
        "Exercises multi-file implementation, concurrency semantics, and behavior-test verification.",
    ),
    (
        "c06.pipeline_recovery",
        "Naturally yields a realistic failed ETL environment with replay and data-integrity checks.",
    ),
    (
        "c07.experiments-health.experimental-design.ab-analysis",
        "Combines statistical judgment with reproducible numeric and claim-calibration checks.",
    ),
    (
        "c23.guides.troubleshooting",
        "Grounded documentation can be judged against a runnable faulty system and evidence rubric.",
    ),
    (
        "c25.2.3",
        "Tests retrieval, source assessment, and corroboration over a controlled document collection.",
    ),
    (
        "minan.accounting.reconciliation",
        "Structured ledgers provide realistic ambiguity plus exact balance and discrepancy checks.",
    ),
    (
        "c19.1.5",
        "A sandboxed vulnerable service enables exploit regression tests and remediation review.",
    ),
    (
        "c09.1",
        "Schema-bound extraction spans language reasoning and deterministic structural validation.",
    ),
    (
        "c15.6.4",
        "Simulation construction exposes model assumptions while enabling numerical behavior tests.",
    ),
    (
        "c02.flaky_test_repair",
        "Repository fixtures can reproduce nondeterminism and verify stability across repeated runs.",
    ),
    (
        "c12.simulation.model_validation",
        "Pairs an embodied simulator with quantitative fidelity metrics and diagnostic reasoning.",
    ),
    (
        "c34.lab_data.lineage_audit",
        "A synthetic LIMS dataset supports realistic forensic tracing and exact referential checks.",
    ),
    (
        "c04.infrastructure.drift-reconciliation",
        "Declarative state and a local cloud simulation support plan, mutation, and convergence checks.",
    ),
    (
        "c10.table_structure",
        "Document images and cell topology provide grounded multimodal work with exact graph scoring.",
    ),
    (
        "c16.digital.waveform_debug",
        "Trace artifacts and an RTL testbench make diagnosis causal and repair objectively testable.",
    ),
    (
        "c18.idempotent_message_processing",
        "A broker-backed service can inject duplicates and verify externally observable exactly-once effects.",
    ),
    (
        "c29.procurement.bid_evaluation",
        "Messy bid documents combine policy reasoning, spreadsheet work, and auditable ranking constraints.",
    ),
    (
        "c05.analysis.protocol_trace",
        "Packet captures and endpoint logs support realistic reconstruction with exact protocol-state checks.",
    ),
    (
        "c08.models.optimization_diagnosis",
        "Training traces and controlled reruns make model-debugging hypotheses empirically testable.",
    ),
    (
        "c11.signal.fingerprint_matching",
        "Synthetic and transformed audio fixtures provide measurable matching and robustness criteria.",
    ),
    (
        "c13.graphs.dynamic_rollback",
        "Combines algorithm design, adversarial operation sequences, and exact complexity-aware tests.",
    ),
    (
        "c17.widget_lifecycle_accessibility",
        "A browser-ready widget exposes semantic, keyboard, focus, and lifecycle behavior to tests.",
    ),
    (
        "c22.workflow_runtime_diagnosis",
        "Versioned workflow state and execution history enable causal diagnosis and replay verification.",
    ),
    (
        "c24.experiment.reproducibility",
        "Research artifacts can be rebuilt in isolation and audited for provenance and exact outputs.",
    ),
    (
        "c26.assessment.grading",
        "Blind learner work and anchor responses support rubric calibration and feedback-quality review.",
    ),
    (
        "c28.compliance.evidence_assessment",
        "A controlled evidence room supports traceable control judgments without relying on live legal facts.",
    ),
    (
        "c30.growth.performance_reporting",
        "Cross-channel exports create a realistic reconciliation task with exact totals and claim checks.",
    ),
    (
        "c31.page.preflight",
        "Supplied production files permit deterministic checks for fonts, color, bleed, and packaging defects.",
    ),
    (
        "c32.geometry_topology_repair",
        "Geospatial fixtures expose invalid topology and allow exact validity and conservation checks.",
    ),
    (
        "c33.records.exchange_diagnose",
        "Synthetic clinical messages support standards-conformance diagnosis without private patient data.",
    ),
    (
        "c35.persistence_migration",
        "Versioned save fixtures and a game-state oracle make compatibility and losslessness testable.",
    ),
)


class CatalogError(ValueError):
    """Raised when a catalog or pilot fails structural validation."""


@dataclass(frozen=True)
class CapabilityRecord:
    """One completely decoded capability and its known source context."""

    capability_id: str
    subject_id: str | None
    subject_name: str | None
    capability: dict[str, Any]
    curriculum_complete: bool

    @property
    def capability_sha256(self) -> str:
        return canonical_sha256(self.capability)


@dataclass(frozen=True)
class CatalogAudit:
    """Machine-readable evidence about how much of a source was recoverable."""

    status: str
    source_bytes: int
    source_sha256: str
    first_nul_byte: int | None
    parse_error_byte: int | None
    parse_error: str | None
    recoverable_through_byte: int
    dropped_bytes: int
    dropped_records_minimum: int
    dropped_records_exact: int | None
    complete_curricula: int
    incomplete_curricula: int
    recovered_sections: int
    recovered_capabilities: int
    orphaned_capabilities: int
    limitation: str | None


@dataclass(frozen=True)
class CatalogIngestion:
    """Recovered catalog data and the audit trail for the recovery decision."""

    catalog_version: str | None
    records: tuple[CapabilityRecord, ...]
    audit: CatalogAudit
    learning_progression: dict[str, Any] | None = None


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for provenance hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[:char_offset].encode("utf-8"))


def _skip_space(text: str, offset: int) -> int:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    return offset


def _find_key_value_start(
    text: str, object_start: int, key_to_find: str, decoder: json.JSONDecoder
) -> int:
    """Locate a direct object's value without decoding later, possibly damaged data."""

    offset = _skip_space(text, object_start)
    if offset >= len(text) or text[offset] != "{":
        raise CatalogError(f"expected object at character {offset}")
    offset += 1
    while True:
        offset = _skip_space(text, offset)
        if offset >= len(text):
            raise CatalogError(f"object ended before key {key_to_find!r}")
        if text[offset] == "}":
            raise CatalogError(f"object has no key {key_to_find!r}")
        key, offset = decoder.raw_decode(text, offset)
        if not isinstance(key, str):
            raise CatalogError(f"non-string object key at character {offset}")
        offset = _skip_space(text, offset)
        if offset >= len(text) or text[offset] != ":":
            raise CatalogError(f"missing colon after key {key!r}")
        value_start = _skip_space(text, offset + 1)
        if key == key_to_find:
            return value_start
        try:
            _, offset = decoder.raw_decode(text, value_start)
        except json.JSONDecodeError as exc:
            raise CatalogError(
                f"could not pass key {key!r} while locating {key_to_find!r}"
            ) from exc
        offset = _skip_space(text, offset)
        if offset < len(text) and text[offset] == ",":
            offset += 1
            continue
        if offset < len(text) and text[offset] == "}":
            raise CatalogError(f"object has no key {key_to_find!r}")
        raise CatalogError(f"invalid object separator at character {offset}")


def _catalog_version_before_curricula(
    text: str, decoder: json.JSONDecoder, curricula_start: int
) -> str | None:
    """Decode the version only when it occurs in the intact top-level prefix."""

    try:
        value_start = _find_key_value_start(text, 0, "catalog_version", decoder)
        if value_start >= curricula_start:
            return None
        value, _ = decoder.raw_decode(text, value_start)
    except (CatalogError, json.JSONDecodeError):
        return None
    return value if isinstance(value, str) else None


def _validate_curriculum(wrapper: Any) -> tuple[str, str, list[Any]]:
    if not isinstance(wrapper, dict):
        raise CatalogError("curricula entries must be objects")
    curriculum = wrapper.get("curriculum")
    if not isinstance(curriculum, dict):
        raise CatalogError("curricula entry is missing an object curriculum")
    subject_id = curriculum.get("subject_id")
    subject_name = curriculum.get("subject_name")
    sections = curriculum.get("sections")
    if not isinstance(subject_id, str) or not isinstance(subject_name, str):
        raise CatalogError("complete curriculum has invalid subject metadata")
    if not isinstance(sections, list):
        raise CatalogError("complete curriculum has no sections array")
    return subject_id, subject_name, sections


def _record(
    section: Any,
    subject_id: str | None,
    subject_name: str | None,
    curriculum_complete: bool,
) -> CapabilityRecord | None:
    if not isinstance(section, dict) or section.get("kind") != "capability":
        return None
    capability_id = section.get("id")
    if not isinstance(capability_id, str) or not capability_id:
        raise CatalogError("decoded capability is missing a non-empty string id")
    # A fresh shallow dict prevents later wrapper mutation from changing provenance.
    return CapabilityRecord(
        capability_id=capability_id,
        subject_id=subject_id,
        subject_name=subject_name,
        capability=dict(section),
        curriculum_complete=curriculum_complete,
    )


def _records_from_complete_document(
    document: Any,
) -> tuple[str | None, list[CapabilityRecord], int]:
    if not isinstance(document, dict):
        raise CatalogError("catalog root must be an object")
    curricula = document.get("curricula")
    if not isinstance(curricula, list):
        raise CatalogError("catalog root must contain a curricula array")
    version = document.get("catalog_version")
    if version is not None and not isinstance(version, str):
        raise CatalogError("catalog_version must be a string")
    records: list[CapabilityRecord] = []
    section_count = 0
    for wrapper in curricula:
        subject_id, subject_name, sections = _validate_curriculum(wrapper)
        section_count += len(sections)
        for section in sections:
            recovered = _record(section, subject_id, subject_name, True)
            if recovered is not None:
                records.append(recovered)
    return version, records, section_count


def ingest_catalog(path: str | Path) -> CatalogIngestion:
    """Load a catalog, salvaging only fully decoded records if it is truncated.

    For a valid document the result has ``audit.status == "complete"``.  For a
    damaged document, complete curricula are recovered first.  If the final
    curriculum is partial, each complete section object before the damage is also
    recovered, but unavailable subject metadata remains ``None``.
    """

    source_path = Path(path)
    data = source_path.read_bytes()
    source_digest = hashlib.sha256(data).hexdigest()
    first_nul = data.find(b"\0")
    first_nul_byte = first_nul if first_nul >= 0 else None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Only the strict UTF-8 prefix can safely be presented to the JSON decoder.
        text = data[: exc.start].decode("utf-8")
    decoder = json.JSONDecoder()

    document_error: json.JSONDecodeError | None = None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        document_error = exc
    else:
        version, records, section_count = _records_from_complete_document(document)
        audit = CatalogAudit(
            status="complete",
            source_bytes=len(data),
            source_sha256=source_digest,
            first_nul_byte=first_nul_byte,
            parse_error_byte=None,
            parse_error=None,
            recoverable_through_byte=len(data),
            dropped_bytes=0,
            dropped_records_minimum=0,
            dropped_records_exact=0,
            complete_curricula=len(document["curricula"]),
            incomplete_curricula=0,
            recovered_sections=section_count,
            recovered_capabilities=len(records),
            orphaned_capabilities=0,
            limitation=None,
        )
        progression = document.get("learning_progression")
        if progression is not None:
            if (
                not isinstance(progression, dict)
                or progression.get("catalog_version") != version
            ):
                raise CatalogError("learning progression has invalid catalog version")
            edges = progression.get("edges")
            ids = {record.capability_id for record in records}
            if not isinstance(edges, list) or any(
                not isinstance(edge, dict)
                or edge.get("dependent_id") not in ids
                or edge.get("prerequisite_id") not in ids
                for edge in edges
            ):
                raise CatalogError("learning progression has invalid capability edges")
        return CatalogIngestion(version, tuple(records), audit, progression)

    try:
        curricula_value = _find_key_value_start(text, 0, "curricula", decoder)
    except CatalogError as exc:
        raise CatalogError(
            "catalog is damaged before a curricula array can be located"
        ) from exc
    if curricula_value >= len(text) or text[curricula_value] != "[":
        raise CatalogError("curricula value is not an array")

    version = _catalog_version_before_curricula(text, decoder, curricula_value)
    records: list[CapabilityRecord] = []
    complete_curricula = 0
    recovered_sections = 0
    offset = curricula_value + 1
    partial_wrapper_start: int | None = None
    last_recovered_end = offset

    while True:
        offset = _skip_space(text, offset)
        if offset < len(text) and text[offset] == ",":
            offset = _skip_space(text, offset + 1)
        if offset >= len(text) or text[offset] == "]":
            break
        entry_start = offset
        try:
            wrapper, entry_end = decoder.raw_decode(text, entry_start)
        except json.JSONDecodeError:
            partial_wrapper_start = entry_start
            break
        subject_id, subject_name, sections = _validate_curriculum(wrapper)
        recovered_sections += len(sections)
        for section in sections:
            recovered = _record(section, subject_id, subject_name, True)
            if recovered is not None:
                records.append(recovered)
        complete_curricula += 1
        last_recovered_end = entry_end
        offset = entry_end

    orphaned_capabilities = 0
    incomplete_curricula = 0
    partial_record_detected = False
    if partial_wrapper_start is not None:
        incomplete_curricula = 1
        try:
            curriculum_start = _find_key_value_start(
                text, partial_wrapper_start, "curriculum", decoder
            )
            sections_start = _find_key_value_start(
                text, curriculum_start, "sections", decoder
            )
        except CatalogError:
            sections_start = -1
        if (
            sections_start >= 0
            and sections_start < len(text)
            and text[sections_start] == "["
        ):
            section_offset = sections_start + 1
            while True:
                section_offset = _skip_space(text, section_offset)
                if section_offset < len(text) and text[section_offset] == ",":
                    section_offset = _skip_space(text, section_offset + 1)
                if section_offset >= len(text) or text[section_offset] == "]":
                    break
                try:
                    section, section_end = decoder.raw_decode(text, section_offset)
                except json.JSONDecodeError:
                    partial_record_detected = bool(text[section_offset:].strip())
                    break
                recovered_sections += 1
                recovered = _record(section, None, None, False)
                if recovered is not None:
                    records.append(recovered)
                    orphaned_capabilities += 1
                last_recovered_end = section_end
                section_offset = section_end

    assert document_error is not None
    parse_error_byte = _byte_offset(text, document_error.pos)
    recovered_byte = _byte_offset(text, last_recovered_end)
    dropped_bytes = max(0, len(data) - recovered_byte)
    audit = CatalogAudit(
        status="incomplete",
        source_bytes=len(data),
        source_sha256=source_digest,
        first_nul_byte=first_nul_byte,
        parse_error_byte=parse_error_byte,
        parse_error=f"{document_error.msg} (character {document_error.pos})",
        recoverable_through_byte=recovered_byte,
        dropped_bytes=dropped_bytes,
        dropped_records_minimum=1 if partial_record_detected else 0,
        dropped_records_exact=None,
        complete_curricula=complete_curricula,
        incomplete_curricula=incomplete_curricula,
        recovered_sections=recovered_sections,
        recovered_capabilities=len(records),
        orphaned_capabilities=orphaned_capabilities,
        limitation=(
            "The damaged suffix may have contained additional records. Its exact record "
            "count and missing text are unknowable from this file; no missing content was inferred."
        ),
    )
    return CatalogIngestion(version, tuple(records), audit)


def build_pilot(
    ingestion: CatalogIngestion,
    selections: Sequence[tuple[str, str]],
    *,
    source_path: str,
) -> dict[str, Any]:
    """Build a deterministic pilot from ``(capability_id, rationale)`` pairs."""

    indexed: dict[str, CapabilityRecord] = {}
    duplicates: set[str] = set()
    for record in ingestion.records:
        if record.capability_id in indexed:
            duplicates.add(record.capability_id)
        indexed[record.capability_id] = record
    requested_ids = [capability_id for capability_id, _ in selections]
    if len(requested_ids) != len(set(requested_ids)):
        raise CatalogError("pilot selections contain duplicate capability ids")
    missing = [
        capability_id for capability_id in requested_ids if capability_id not in indexed
    ]
    ambiguous = [
        capability_id for capability_id in requested_ids if capability_id in duplicates
    ]
    if missing:
        raise CatalogError(f"pilot capabilities not recovered: {', '.join(missing)}")
    if ambiguous:
        raise CatalogError(
            f"pilot capability ids are ambiguous: {', '.join(ambiguous)}"
        )

    capabilities: list[dict[str, Any]] = []
    for capability_id, rationale in selections:
        record = indexed[capability_id]
        if record.subject_id is None or record.subject_name is None:
            raise CatalogError(
                f"pilot capability {capability_id!r} lacks source subject metadata"
            )
        capabilities.append(
            {
                "capability_id": record.capability_id,
                "subject_id": record.subject_id,
                "subject_name": record.subject_name,
                "capability_sha256": record.capability_sha256,
                "selection_rationale": rationale,
                "capability": record.capability,
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "catalog_version": ingestion.catalog_version,
            "path": source_path,
            "sha256": ingestion.audit.source_sha256,
        },
        "catalog_audit": asdict(ingestion.audit),
        "capabilities": capabilities,
    }
    if ingestion.learning_progression is not None:
        selected = set(requested_ids)
        progression = ingestion.learning_progression
        edges = [
            edge for edge in progression["edges"] if edge["dependent_id"] in selected
        ]
        result["learning_progression"] = {
            "catalog_version": progression["catalog_version"],
            "prompt_version": progression.get("prompt_version"),
            "scope_subject_ids": progression.get("scope_subject_ids"),
            "source_sha256": canonical_sha256(progression),
            "edges": edges,
            "edges_sha256": canonical_sha256(edges),
            "selection_rule": "edges whose dependent_id is a selected capability",
        }
    return result


def build_subject_cohort(
    ingestion: CatalogIngestion,
    *,
    per_subject: int = 1,
    source_path: str = "catalog.json",
) -> dict[str, Any]:
    """Select a stable development cohort without changing the historical pilot."""
    if ingestion.audit.status != "complete" or per_subject < 1:
        raise CatalogError(
            "subject cohort requires a complete catalog and positive size"
        )
    grouped: dict[str, list[CapabilityRecord]] = {}
    for record in ingestion.records:
        if record.subject_id is None:
            raise CatalogError(
                "subject cohort contains a capability without subject metadata"
            )
        grouped.setdefault(record.subject_id, []).append(record)
    selections = []
    for subject_id in sorted(grouped):
        # Hash order avoids favoring short, early, or superficially easy records.
        ranked = sorted(
            grouped[subject_id],
            key=lambda record: (record.capability_sha256, record.capability_id),
        )
        for record in ranked[:per_subject]:
            selections.append(
                (
                    record.capability_id,
                    "Deterministic development cohort; full-catalog coverage remains required.",
                )
            )
    return build_pilot(ingestion, selections, source_path=source_path)


def build_default_pilot(
    ingestion: CatalogIngestion, *, source_path: str = "catalog.json"
) -> dict[str, Any]:
    """Build the checked-in, one-per-curriculum 34-capability pilot."""

    return build_pilot(ingestion, DEFAULT_PILOT_SELECTIONS, source_path=source_path)


def build_full_manifest(
    ingestion: CatalogIngestion, *, source_path: str = "catalog.json"
) -> dict[str, Any]:
    """Use the same pipeline input contract for every catalog capability."""
    if ingestion.audit.status != "complete":
        raise CatalogError("full-catalog generation requires a complete source catalog")
    selections = [
        (
            record.capability_id,
            "Full-catalog generation: every source capability is included.",
        )
        for record in ingestion.records
    ]
    result = build_pilot(ingestion, selections, source_path=source_path)
    validate_pilot(result)
    return result


def validate_pilot(pilot: Mapping[str, Any]) -> None:
    """Validate pilot structure, identity, hashes, and subject diversity."""

    if pilot.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(
            f"unsupported pilot schema_version: {pilot.get('schema_version')!r}"
        )
    source = pilot.get("source")
    audit = pilot.get("catalog_audit")
    if not isinstance(source, dict) or not isinstance(audit, dict):
        raise CatalogError("pilot must contain source and catalog_audit objects")
    source_hash = source.get("sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise CatalogError("pilot source sha256 is invalid")
    if audit.get("source_sha256") != source_hash:
        raise CatalogError("pilot source and audit hashes differ")
    if audit.get("status") not in ("complete", "incomplete"):
        raise CatalogError("pilot catalog audit status is invalid")
    if audit.get("status") == "complete" and (
        audit.get("dropped_bytes") != 0 or audit.get("dropped_records_exact") != 0
    ):
        raise CatalogError("complete catalog audit reports dropped data")
    progression = pilot.get("learning_progression")
    if progression is not None:
        if not isinstance(progression, dict) or progression.get(
            "catalog_version"
        ) != source.get("catalog_version"):
            raise CatalogError("pilot learning progression version mismatch")
        edges = progression.get("edges")
        if not isinstance(edges, list) or progression.get(
            "edges_sha256"
        ) != canonical_sha256(edges):
            raise CatalogError("pilot learning progression edge digest mismatch")
        selected_ids = {
            row.get("capability_id")
            for row in pilot.get("capabilities", [])
            if isinstance(row, dict)
        }
        if any(
            not isinstance(edge, dict) or edge.get("dependent_id") not in selected_ids
            for edge in edges
        ):
            raise CatalogError(
                "pilot learning progression references an unselected dependent"
            )
        digest_value = progression.get("source_sha256")
        if (
            not isinstance(digest_value, str)
            or len(digest_value) != 64
            or any(c not in "0123456789abcdef" for c in digest_value)
        ):
            raise CatalogError("pilot learning progression source digest is invalid")
    capabilities = pilot.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise CatalogError("pilot capabilities must be a non-empty array")
    ids: set[str] = set()
    for index, item in enumerate(capabilities):
        if not isinstance(item, dict):
            raise CatalogError(f"pilot capability {index} is not an object")
        capability = item.get("capability")
        capability_id = item.get("capability_id")
        if not isinstance(capability, dict) or not isinstance(capability_id, str):
            raise CatalogError(
                f"pilot capability {index} has invalid identity or payload"
            )
        if capability.get("id") != capability_id:
            raise CatalogError(
                f"pilot capability {capability_id!r} id does not match payload"
            )
        if capability.get("kind") != "capability":
            raise CatalogError(f"pilot record {capability_id!r} is not a capability")
        if capability_id in ids:
            raise CatalogError(f"duplicate pilot capability id: {capability_id}")
        ids.add(capability_id)
        expected_hash = canonical_sha256(capability)
        if item.get("capability_sha256") != expected_hash:
            raise CatalogError(f"pilot capability {capability_id!r} hash mismatch")
        if not isinstance(item.get("subject_id"), str) or not isinstance(
            item.get("subject_name"), str
        ):
            raise CatalogError(
                f"pilot capability {capability_id!r} lacks subject metadata"
            )
        if (
            not isinstance(item.get("selection_rationale"), str)
            or not item["selection_rationale"].strip()
        ):
            raise CatalogError(f"pilot capability {capability_id!r} lacks a rationale")


def load_pilot(path: str | Path) -> dict[str, Any]:
    pilot = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(pilot, dict):
        raise CatalogError("pilot root must be an object")
    validate_pilot(pilot)
    return pilot


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument(
        "--audit-json", type=Path, help="write the ingestion audit as JSON"
    )
    parser.add_argument(
        "--pilot-json",
        type=Path,
        help="write the deterministic 34-capability pilot as JSON",
    )
    parser.add_argument(
        "--all-json",
        type=Path,
        help="write every capability as a production input manifest",
    )
    parser.add_argument(
        "--cohort-json", type=Path, help="write a deterministic development cohort"
    )
    parser.add_argument(
        "--cohort-per-subject",
        type=int,
        default=1,
        help="number of hash-ranked capabilities per subject in the development cohort",
    )
    args = parser.parse_args(argv)
    ingestion = ingest_catalog(args.catalog)
    audit = asdict(ingestion.audit)
    if args.audit_json:
        _write_json(args.audit_json, audit)
    if args.pilot_json:
        pilot = build_default_pilot(ingestion, source_path=str(args.catalog))
        validate_pilot(pilot)
        _write_json(args.pilot_json, pilot)
    if args.all_json:
        _write_json(
            args.all_json, build_full_manifest(ingestion, source_path=str(args.catalog))
        )
    if args.cohort_json:
        cohort = build_subject_cohort(
            ingestion,
            per_subject=args.cohort_per_subject,
            source_path=str(args.catalog),
        )
        validate_pilot(cohort)
        _write_json(args.cohort_json, cohort)
    print(json.dumps(audit, indent=2))
    return 0 if ingestion.audit.status == "complete" else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the public API
    raise SystemExit(_cli())
