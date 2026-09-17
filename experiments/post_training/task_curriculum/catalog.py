# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated curriculum and macro-area catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CatalogError(ValueError):
    """A curriculum catalog violates its schema or references."""


@dataclass(frozen=True)
class MicroArea:
    id: str
    name: str
    taxonomy_unit_count: int
    credited_task_count: int
    mapped_source_count: int
    coverage_status: str


@dataclass(frozen=True)
class MacroArea:
    id: str
    name: str
    micro_areas: tuple[MicroArea, ...]


@dataclass(frozen=True)
class MacroCatalog:
    snapshot: str
    macro_areas: tuple[MacroArea, ...]

    @property
    def macros_by_id(self) -> dict[str, MacroArea]:
        return {area.id: area for area in self.macro_areas}

    @property
    def micros_by_id(self) -> dict[str, tuple[str, MicroArea]]:
        return {micro.id: (macro.id, micro) for macro in self.macro_areas for micro in macro.micro_areas}


class MicroExtensionStatus(StrEnum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"


class EvidenceSplit(StrEnum):
    DISCOVERY = "discovery"
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"


@dataclass(frozen=True)
class ExtensionEvidence:
    task_id: str
    source: str
    path: str
    split: EvidenceSplit
    instruction_sha256: str
    task_archive_sha256: str
    macro_area_id: str
    micro_vocabulary_gap: bool


@dataclass(frozen=True)
class MicroExtension:
    id: str
    name: str
    parent_macro_area_id: str
    status: MicroExtensionStatus
    evidence: tuple[ExtensionEvidence, ...]


@dataclass(frozen=True)
class MicroExtensionCatalog:
    version: str
    macro_snapshot: str
    tasktrove_release: str
    extensions: tuple[MicroExtension, ...]

    @property
    def extensions_by_id(self) -> dict[str, MicroExtension]:
        return {extension.id: extension for extension in self.extensions}


@dataclass(frozen=True)
class CurriculumUnit:
    id: str
    name: str
    outcome: str
    macro_area_id: str
    micro_area_ids: tuple[str, ...]
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    prerequisites: tuple[str, ...]
    positive_examples: tuple[str, ...]


@dataclass(frozen=True)
class CurriculumCatalog:
    version: str
    macro_snapshot: str
    micro_extension_version: str
    units: tuple[CurriculumUnit, ...]


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CatalogError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context} must be a non-empty string")
    return value


def nonnegative_integer(value: object, context: str) -> int:
    """Return a nonnegative JSON integer, rejecting booleans."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CatalogError(f"{context} must be a nonnegative integer")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CatalogError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _objects(value: object, context: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise CatalogError(f"{context} must be a list")
    return [_object(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _strings(value: object, context: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CatalogError(f"{context} must be a list")
    strings = tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(value))
    if len(strings) < minimum:
        raise CatalogError(f"{context} must contain at least {minimum} entries")
    if len(set(strings)) != len(strings):
        raise CatalogError(f"{context} contains duplicates")
    return strings


def _read_json_object(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text()), str(path))


def _reject_extra_keys(value: dict[str, object], allowed: set[str], context: str) -> None:
    extras = sorted(value.keys() - allowed)
    if extras:
        raise CatalogError(f"{context} has unknown fields: {', '.join(extras)}")


def _validate_prerequisite_graph(units: tuple[CurriculumUnit, ...]) -> None:
    units_by_id = {unit.id: unit for unit in units}
    known_units = set(units_by_id)
    for unit in units:
        unknown = sorted(set(unit.prerequisites) - known_units)
        if unknown:
            raise CatalogError(f"unit {unit.id} has unknown prerequisites: {', '.join(unknown)}")

    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(unit_id: str) -> None:
        if unit_id in visiting:
            raise CatalogError(f"curriculum prerequisite cycle includes {unit_id}")
        if unit_id in visited:
            return
        visiting.add(unit_id)
        for prerequisite in units_by_id[unit_id].prerequisites:
            visit(prerequisite)
        visiting.remove(unit_id)
        visited.add(unit_id)

    for unit_id in units_by_id:
        visit(unit_id)


def load_macro_catalog(path: Path) -> MacroCatalog:
    """Load the versioned macro-area inventory and validate its hierarchy."""

    raw = _read_json_object(path)
    _reject_extra_keys(
        raw,
        {"schema_version", "snapshot", "source_url", "source_issue", "methodology", "macro_areas"},
        str(path),
    )
    if raw.get("schema_version") != "task-curriculum-macro-areas-v1":
        raise CatalogError(f"{path} has an unsupported schema_version")

    macros = []
    for macro_raw in _objects(raw.get("macro_areas"), "macro_areas"):
        _reject_extra_keys(macro_raw, {"id", "name", "micro_areas"}, "macro area")
        macro_id = _string(macro_raw.get("id"), "macro area id")
        micros = []
        for micro_raw in _objects(macro_raw.get("micro_areas"), f"{macro_id}.micro_areas"):
            _reject_extra_keys(
                micro_raw,
                {
                    "id",
                    "name",
                    "taxonomy_unit_count",
                    "credited_task_count",
                    "mapped_source_count",
                    "coverage_status",
                },
                "micro area",
            )
            micro_id = _string(micro_raw.get("id"), "micro area id")
            if not micro_id.startswith(f"{macro_id}."):
                raise CatalogError(f"micro area {micro_id} does not belong to {macro_id}")
            micros.append(
                MicroArea(
                    id=micro_id,
                    name=_string(micro_raw.get("name"), f"{micro_id}.name"),
                    taxonomy_unit_count=nonnegative_integer(
                        micro_raw.get("taxonomy_unit_count"), f"{micro_id}.taxonomy_unit_count"
                    ),
                    credited_task_count=nonnegative_integer(
                        micro_raw.get("credited_task_count"), f"{micro_id}.credited_task_count"
                    ),
                    mapped_source_count=nonnegative_integer(
                        micro_raw.get("mapped_source_count"), f"{micro_id}.mapped_source_count"
                    ),
                    coverage_status=_string(micro_raw.get("coverage_status"), f"{micro_id}.coverage_status"),
                )
            )
        macros.append(
            MacroArea(
                id=macro_id,
                name=_string(macro_raw.get("name"), f"{macro_id}.name"),
                micro_areas=tuple(micros),
            )
        )

    catalog = MacroCatalog(snapshot=_string(raw.get("snapshot"), "snapshot"), macro_areas=tuple(macros))
    macro_ids = [macro.id for macro in catalog.macro_areas]
    micro_ids = [micro.id for macro in catalog.macro_areas for micro in macro.micro_areas]
    if len(set(macro_ids)) != len(macro_ids):
        raise CatalogError("macro area IDs must be unique")
    if len(set(micro_ids)) != len(micro_ids):
        raise CatalogError("micro area IDs must be unique")

    methodology = _object(raw.get("methodology"), "methodology")
    if methodology.get("active_macro_area_count") != len(macro_ids):
        raise CatalogError("active_macro_area_count does not match macro_areas")
    if methodology.get("active_micro_area_count") != len(micro_ids):
        raise CatalogError("active_micro_area_count does not match macro_areas")
    return catalog


def load_micro_extensions(path: Path, macro_catalog: MacroCatalog) -> MicroExtensionCatalog:
    """Load project-owned micro areas layered over a published macro snapshot."""

    raw = _read_json_object(path)
    _reject_extra_keys(
        raw,
        {"schema_version", "version", "macro_snapshot", "tasktrove_release", "extensions"},
        str(path),
    )
    if raw.get("schema_version") != "task-curriculum-micro-extensions-v1":
        raise CatalogError(f"{path} has an unsupported schema_version")
    macro_snapshot = _string(raw.get("macro_snapshot"), "macro_snapshot")
    if macro_snapshot != macro_catalog.snapshot:
        raise CatalogError(f"extension macro snapshot {macro_snapshot} does not match {macro_catalog.snapshot}")

    extensions = []
    for extension_raw in _objects(raw.get("extensions"), "extensions"):
        _reject_extra_keys(
            extension_raw,
            {"id", "name", "parent_macro_area_id", "status", "evidence"},
            "micro extension",
        )
        extension_id = _string(extension_raw.get("id"), "micro extension id")
        parent_macro_id = _string(extension_raw.get("parent_macro_area_id"), f"{extension_id}.parent_macro_area_id")
        if parent_macro_id not in macro_catalog.macros_by_id:
            raise CatalogError(f"micro extension {extension_id} has unknown parent macro {parent_macro_id}")
        if not extension_id.startswith(f"{parent_macro_id}.ext."):
            raise CatalogError(f"micro extension {extension_id} does not belong to {parent_macro_id}")
        try:
            status = MicroExtensionStatus(_string(extension_raw.get("status"), f"{extension_id}.status"))
        except ValueError as error:
            raise CatalogError(f"micro extension {extension_id} has unknown status") from error
        evidence = []
        for evidence_raw in _objects(extension_raw.get("evidence"), f"{extension_id}.evidence"):
            _reject_extra_keys(
                evidence_raw,
                {
                    "task_id",
                    "source",
                    "path",
                    "split",
                    "instruction_sha256",
                    "task_archive_sha256",
                    "macro_area_id",
                    "micro_vocabulary_gap",
                },
                "extension evidence",
            )
            try:
                split = EvidenceSplit(_string(evidence_raw.get("split"), "extension evidence split"))
            except ValueError as error:
                raise CatalogError(f"micro extension {extension_id} has unknown evidence split") from error
            evidence_macro_id = _string(evidence_raw.get("macro_area_id"), "extension evidence macro_area_id")
            if evidence_macro_id != parent_macro_id:
                raise CatalogError(f"micro extension {extension_id} evidence belongs to macro {evidence_macro_id}")
            micro_vocabulary_gap = evidence_raw.get("micro_vocabulary_gap")
            if micro_vocabulary_gap is not True:
                raise CatalogError(f"micro extension {extension_id} evidence must record a micro-vocabulary gap")
            evidence.append(
                ExtensionEvidence(
                    task_id=_string(evidence_raw.get("task_id"), "extension evidence task_id"),
                    source=_string(evidence_raw.get("source"), "extension evidence source"),
                    path=_string(evidence_raw.get("path"), "extension evidence path"),
                    split=split,
                    instruction_sha256=_sha256(
                        evidence_raw.get("instruction_sha256"), "extension evidence instruction_sha256"
                    ),
                    task_archive_sha256=_sha256(
                        evidence_raw.get("task_archive_sha256"), "extension evidence task_archive_sha256"
                    ),
                    macro_area_id=evidence_macro_id,
                    micro_vocabulary_gap=micro_vocabulary_gap,
                )
            )
        if not evidence:
            raise CatalogError(f"micro extension {extension_id} requires evidence")
        evidence_keys = [(row.source, row.path) for row in evidence]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise CatalogError(f"micro extension {extension_id} has duplicate evidence tasks")
        if len({row.task_id for row in evidence}) != len(evidence):
            raise CatalogError(f"micro extension {extension_id} has duplicate evidence task IDs")
        extensions.append(
            MicroExtension(
                id=extension_id,
                name=_string(extension_raw.get("name"), f"{extension_id}.name"),
                parent_macro_area_id=parent_macro_id,
                status=status,
                evidence=tuple(evidence),
            )
        )

    catalog = MicroExtensionCatalog(
        version=_string(raw.get("version"), "version"),
        macro_snapshot=macro_snapshot,
        tasktrove_release=_string(raw.get("tasktrove_release"), "tasktrove_release"),
        extensions=tuple(extensions),
    )
    extension_ids = [extension.id for extension in catalog.extensions]
    if len(set(extension_ids)) != len(extension_ids):
        raise CatalogError("micro extension IDs must be unique")
    collisions = sorted(set(extension_ids) & macro_catalog.micros_by_id.keys())
    if collisions:
        raise CatalogError(f"micro extension IDs collide with source micro areas: {', '.join(collisions)}")
    return catalog


def load_curriculum(
    path: Path,
    macro_catalog: MacroCatalog,
    extension_catalog: MicroExtensionCatalog,
) -> CurriculumCatalog:
    """Load a curriculum and validate its macro and prerequisite references."""

    raw = _read_json_object(path)
    _reject_extra_keys(
        raw,
        {"schema_version", "version", "macro_snapshot", "micro_extension_version", "notes", "units"},
        str(path),
    )
    if raw.get("schema_version") != "task-curriculum-v1":
        raise CatalogError(f"{path} has an unsupported schema_version")
    macro_snapshot = _string(raw.get("macro_snapshot"), "macro_snapshot")
    if macro_snapshot != macro_catalog.snapshot:
        raise CatalogError(f"curriculum macro snapshot {macro_snapshot} does not match {macro_catalog.snapshot}")
    extension_version = _string(raw.get("micro_extension_version"), "micro_extension_version")
    if extension_version != extension_catalog.version:
        raise CatalogError(
            f"curriculum micro extension version {extension_version} does not match {extension_catalog.version}"
        )
    if extension_catalog.macro_snapshot != macro_snapshot:
        raise CatalogError("curriculum and micro extensions use different macro snapshots")

    macros_by_id = macro_catalog.macros_by_id
    micros_by_id = macro_catalog.micros_by_id
    extensions_by_id = extension_catalog.extensions_by_id
    units = []
    for unit_raw in _objects(raw.get("units"), "units"):
        _reject_extra_keys(
            unit_raw,
            {
                "id",
                "name",
                "outcome",
                "macro_area_id",
                "micro_area_ids",
                "includes",
                "excludes",
                "prerequisites",
                "positive_examples",
            },
            "curriculum unit",
        )
        unit_id = _string(unit_raw.get("id"), "unit id")
        macro_id = _string(unit_raw.get("macro_area_id"), f"{unit_id}.macro_area_id")
        if macro_id not in macros_by_id:
            raise CatalogError(f"unit {unit_id} refers to unknown macro area {macro_id}")
        micro_ids = _strings(unit_raw.get("micro_area_ids"), f"{unit_id}.micro_area_ids")
        for micro_id in micro_ids:
            parent = micros_by_id.get(micro_id)
            extension = extensions_by_id.get(micro_id)
            if parent is None and extension is None:
                raise CatalogError(f"unit {unit_id} refers to unknown micro area or extension {micro_id}")
            if parent is not None:
                parent_macro_id = parent[0]
            else:
                assert extension is not None
                parent_macro_id = extension.parent_macro_area_id
            if parent_macro_id != macro_id:
                raise CatalogError(f"micro area {micro_id} does not belong to unit macro {macro_id}")
        units.append(
            CurriculumUnit(
                id=unit_id,
                name=_string(unit_raw.get("name"), f"{unit_id}.name"),
                outcome=_string(unit_raw.get("outcome"), f"{unit_id}.outcome"),
                macro_area_id=macro_id,
                micro_area_ids=micro_ids,
                includes=_strings(unit_raw.get("includes"), f"{unit_id}.includes", minimum=1),
                excludes=_strings(unit_raw.get("excludes"), f"{unit_id}.excludes", minimum=1),
                prerequisites=_strings(unit_raw.get("prerequisites"), f"{unit_id}.prerequisites"),
                positive_examples=_strings(unit_raw.get("positive_examples"), f"{unit_id}.positive_examples", minimum=3),
            )
        )

    catalog = CurriculumCatalog(
        version=_string(raw.get("version"), "version"),
        macro_snapshot=macro_snapshot,
        micro_extension_version=extension_version,
        units=tuple(units),
    )
    unit_ids = [unit.id for unit in catalog.units]
    if len(set(unit_ids)) != len(unit_ids):
        raise CatalogError("curriculum unit IDs must be unique")
    _validate_prerequisite_graph(catalog.units)
    return catalog
