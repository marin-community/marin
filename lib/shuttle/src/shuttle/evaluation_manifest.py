# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Closed schema for versioned Shuttle evaluation scorecards."""

import hashlib
import itertools
import json
import math
import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from shuttle.options import Numerics

SCHEMA_VERSION = 1
ORACLE_LATENCY_RATIO = 1.20
STRETCH_ORACLE_LATENCY_RATIO = 1.10
EXPECTED_TARGETS = {
    0: "jax_xla_plugin_and_stablehlo_conversion",
    1: "rmsnorm",
    2: "layernorm",
    3: "dot_product_attention",
    4: "msa",
    5: "gated_delta_net",
    6: "dense_transformer_block",
    7: "local_moe_mlp",
    8: "distributed_moe",
    9: "grug_moe",
    10: "complete_grug_train_step",
}
EXPECTED_TARGET_DIMENSIONS = {
    0: ({"forward", "backward"}, {"tanh_dot_primary_f32"}, {"cpu", "h100", "gb200_or_b200"}),
    1: (
        {"forward", "backward", "composed_forward_backward"},
        {"rmsnorm_bf16_2048x4096"},
        {"h100", "gb200_or_b200"},
    ),
    2: (
        {"forward", "backward", "composed_forward_backward"},
        {"layernorm_representative_pending"},
        {"h100", "gb200_or_b200"},
    ),
    3: (
        {"forward", "backward", "composed_forward_backward"},
        {"dot_product_attention_representative_pending"},
        {"h100", "gb200_or_b200"},
    ),
    4: (
        {"forward", "backward", "composed_forward_backward"},
        {"msa_representative_pending"},
        {"h100", "gb200_or_b200"},
    ),
    5: (
        {"forward", "backward", "composed_forward_backward"},
        {"gated_delta_net_representative_pending"},
        {"h100", "gb200_or_b200"},
    ),
    6: (
        {"forward", "backward", "composed_forward_backward"},
        {"dense_transformer_block_representative_pending"},
        {"h100", "gb200_or_b200"},
    ),
    7: (
        {"forward", "backward", "composed_forward_backward"},
        {"local_moe_mlp_representative_pending"},
        {"h100", "gb200_or_b200"},
    ),
    8: (
        {"forward", "backward", "composed_forward_backward"},
        {"distributed_moe_representative_pending"},
        {"gb200x4"},
    ),
    9: (
        {"forward", "backward", "composed_forward_backward"},
        {"grug_moe_representative_pending"},
        {"gb200x4"},
    ),
    10: (
        {"forward", "backward", "composed_forward_backward", "train_step"},
        {"complete_grug_train_step_representative_pending"},
        {"gb200x4"},
    ),
}
REQUIRED_POLICIES = frozenset(Numerics)
PENDING_SHAPE_SUFFIX = "_pending"
IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationManifestError(ValueError):
    """The evaluation manifest violates its closed scorecard contract."""


class Identified(Protocol):
    """Structural contract for manifest records with stable IDs."""

    @property
    def id(self) -> str: ...


class Status(StrEnum):
    """Only states admitted by the evaluation contract."""

    NOT_STARTED = "not_started"
    BLOCKED = "blocked"
    FAILED = "failed"
    PARTIAL = "partial"
    ACCEPTED = "accepted"


class Boundary(StrEnum):
    """Public execution boundary represented by a scorecard cell."""

    FORWARD = "forward"
    BACKWARD = "backward"
    COMPOSED_FORWARD_BACKWARD = "composed_forward_backward"
    TRAIN_STEP = "train_step"


class Hardware(StrEnum):
    """Hardware requirement represented by a scorecard cell."""

    CPU = "cpu"
    H100 = "h100"
    GB200_OR_B200 = "gb200_or_b200"
    GB200_X4 = "gb200x4"


class Blocker(StrEnum):
    """Closed reasons why an evaluation cell cannot currently execute."""

    ARCHITECTURE_PATH_UNAVAILABLE = "architecture_path_unavailable"
    ORACLE_NOT_PINNED = "oracle_not_pinned"
    PERFORMANCE_EVIDENCE_MISSING = "performance_evidence_missing"
    REPRESENTATIVE_SHAPE_NOT_DECLARED = "representative_shape_not_declared"


class ArchitectureConformance(StrEnum):
    """Whether evidence came through the current ordinary-JAX MLIR path."""

    CONFORMING = "conforming"
    NONCONFORMING = "nonconforming"


class Gate(StrEnum):
    """Independent acceptance gate supported by an evidence artifact."""

    ARCHITECTURE = "architecture"
    NUMERICAL = "numerical"
    PERFORMANCE = "performance"


class EvidenceFormat(StrEnum):
    """Artifact format with a closed claim extractor."""

    CPU_ORDINARY_JAX_ACCEPTANCE_V1 = "cpu_ordinary_jax_acceptance_v1"
    ARCHITECTURE_STATUS_DOCUMENT_V1 = "architecture_status_document_v1"


class ExclusionReason(StrEnum):
    """Why evidence is retained outside the acceptance scorecard."""

    ARCHITECTURE_NONCONFORMING = "architecture_nonconforming"


@dataclass(frozen=True)
class CellIdentity:
    """Complete identity of one evaluation matrix cell."""

    target_id: int
    boundary: Boundary
    shape: str
    policy: Numerics
    hardware: Hardware


@dataclass(frozen=True)
class EvidenceGateClaim:
    """Exact status established for one independent gate."""

    gate: Gate
    status: Status


@dataclass(frozen=True)
class EvidenceClaim:
    """Exact cell and gate states supported by one evidence artifact."""

    identity: CellIdentity
    gates: tuple[EvidenceGateClaim, ...]


@dataclass(frozen=True)
class Evidence:
    """One content-addressed evidence record."""

    id: str
    path: PurePosixPath
    source_commit: str
    record_commit: str
    sha256: str
    architecture_conformance: ArchitectureConformance
    format: EvidenceFormat
    claims: tuple[EvidenceClaim, ...]


@dataclass(frozen=True)
class ExcludedEvidence:
    """Evidence retained for research but forbidden from scorecard cells."""

    evidence: Evidence
    exclusion_reason: ExclusionReason


@dataclass(frozen=True)
class ShapeDeclaration:
    """Versioned representative-shape coordinate."""

    id: str
    declaration_status: Status
    specification: str


@dataclass(frozen=True)
class EvaluationCell:
    """Separate gate states for one complete matrix identity."""

    identity: CellIdentity
    status: Status
    architecture_status: Status
    numerical_status: Status
    performance_status: Status
    evidence_ids: tuple[str, ...]
    blockers: tuple[Blocker, ...]


@dataclass(frozen=True)
class Target:
    """One target and the exact dimensions required by this revision."""

    id: int
    name: str
    status: Status
    required_boundaries: tuple[Boundary, ...]
    required_shapes: tuple[str, ...]
    required_policies: tuple[Numerics, ...]
    required_hardware: tuple[Hardware, ...]
    blockers: tuple[Blocker, ...]


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Versioned oracle-relative performance policy."""

    oracle_latency_ratio: float
    stretch_oracle_latency_ratio: float


@dataclass(frozen=True)
class EvaluationManifest:
    """Validated Shuttle evaluation scorecard."""

    manifest_id: str
    baseline_commit: str
    acceptance_thresholds: AcceptanceThresholds
    shapes: tuple[ShapeDeclaration, ...]
    evidence: tuple[Evidence, ...]
    excluded_evidence: tuple[ExcludedEvidence, ...]
    targets: tuple[Target, ...]
    cells: tuple[EvaluationCell, ...]


def load_evaluation_manifest(path: Path, *, source_root: Path) -> EvaluationManifest:
    """Load and validate a scorecard, including every linked artifact hash."""
    document = json.loads(path.read_text(), object_pairs_hook=_unique_json_object)
    return validate_evaluation_manifest(document, source_root=source_root)


def validate_evaluation_manifest(document: object, *, source_root: Path) -> EvaluationManifest:
    """Validate a decoded manifest and return its immutable representation."""
    root = _closed_mapping(
        document,
        "manifest",
        {
            "schema_version",
            "manifest_id",
            "baseline_commit",
            "acceptance_thresholds",
            "shapes",
            "evidence",
            "excluded_evidence",
            "targets",
            "cells",
        },
    )
    schema_version = _integer(root["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise EvaluationManifestError(f"schema_version must be {SCHEMA_VERSION}")

    manifest_id = _identifier(root["manifest_id"], "manifest_id")
    baseline_commit = _sha1(root["baseline_commit"], "baseline_commit")
    _verify_git_commit(baseline_commit, source_root=source_root, name="baseline_commit")
    thresholds = _acceptance_thresholds(root["acceptance_thresholds"])
    shapes = tuple(_shape(value, index) for index, value in enumerate(_list(root["shapes"], "shapes")))
    shape_by_id = _unique_by_id(shapes, "shape")

    evidence = tuple(
        _evidence(value, f"evidence[{index}]", source_root=source_root)
        for index, value in enumerate(_list(root["evidence"], "evidence"))
    )
    for item in evidence:
        _verify_git_evidence(item, baseline_commit=baseline_commit, source_root=source_root)
    evidence_by_id = _unique_by_id(evidence, "evidence")
    for item in evidence:
        if item.architecture_conformance is not ArchitectureConformance.CONFORMING:
            raise EvaluationManifestError(f"evidence {item.id!r} is not architecturally conforming")
        if not item.claims:
            raise EvaluationManifestError(f"active evidence {item.id!r} must declare cell-scoped gates")

    excluded_evidence = tuple(
        _excluded_evidence(value, index, source_root=source_root)
        for index, value in enumerate(_list(root["excluded_evidence"], "excluded_evidence"))
    )
    for item in excluded_evidence:
        _verify_git_evidence(item.evidence, baseline_commit=baseline_commit, source_root=source_root)
    excluded_by_id = _unique_by_id((item.evidence for item in excluded_evidence), "excluded evidence")
    if evidence_by_id.keys() & excluded_by_id.keys():
        raise EvaluationManifestError("active and excluded evidence IDs must be disjoint")

    targets = tuple(_target(value, index) for index, value in enumerate(_list(root["targets"], "targets")))
    target_by_id = _targets_by_id(targets)
    cells = tuple(_cell(value, index) for index, value in enumerate(_list(root["cells"], "cells")))
    _validate_cells(
        cells,
        target_by_id=target_by_id,
        shape_by_id=shape_by_id,
        evidence_by_id=evidence_by_id,
        excluded_evidence_ids=frozenset(excluded_by_id),
    )
    _validate_target_statuses(targets, cells, shape_by_id=shape_by_id)

    return EvaluationManifest(
        manifest_id=manifest_id,
        baseline_commit=baseline_commit,
        acceptance_thresholds=thresholds,
        shapes=shapes,
        evidence=evidence,
        excluded_evidence=excluded_evidence,
        targets=targets,
        cells=cells,
    )


def _acceptance_thresholds(value: object) -> AcceptanceThresholds:
    mapping = _closed_mapping(
        value,
        "acceptance_thresholds",
        {"oracle_latency_ratio", "stretch_oracle_latency_ratio"},
    )
    oracle_ratio = _number(mapping["oracle_latency_ratio"], "acceptance_thresholds.oracle_latency_ratio")
    stretch_ratio = _number(
        mapping["stretch_oracle_latency_ratio"],
        "acceptance_thresholds.stretch_oracle_latency_ratio",
    )
    if not 0 < stretch_ratio <= oracle_ratio:
        raise EvaluationManifestError("stretch_oracle_latency_ratio must be positive and no greater than the limit")
    if oracle_ratio != ORACLE_LATENCY_RATIO or stretch_ratio != STRETCH_ORACLE_LATENCY_RATIO:
        raise EvaluationManifestError("acceptance thresholds differ from Shuttle evaluation schema version 1")
    return AcceptanceThresholds(oracle_latency_ratio=oracle_ratio, stretch_oracle_latency_ratio=stretch_ratio)


def _shape(value: object, index: int) -> ShapeDeclaration:
    name = f"shapes[{index}]"
    mapping = _closed_mapping(value, name, {"id", "declaration_status", "specification"})
    return ShapeDeclaration(
        id=_identifier(mapping["id"], f"{name}.id"),
        declaration_status=_enum(Status, mapping["declaration_status"], f"{name}.declaration_status"),
        specification=_nonempty_string(mapping["specification"], f"{name}.specification"),
    )


def _evidence(value: object, name: str, *, source_root: Path) -> Evidence:
    mapping = _closed_mapping(
        value,
        name,
        {"id", "path", "source_commit", "record_commit", "sha256", "architecture_conformance", "format"},
    )
    path = _relative_path(mapping["path"], f"{name}.path")
    sha256 = _sha256(mapping["sha256"], f"{name}.sha256")
    artifact = _verified_artifact(path, sha256, source_root=source_root, name=name)
    source_commit = _sha1(mapping["source_commit"], f"{name}.source_commit")
    architecture_conformance = _enum(
        ArchitectureConformance,
        mapping["architecture_conformance"],
        f"{name}.architecture_conformance",
    )
    evidence_format = _enum(EvidenceFormat, mapping["format"], f"{name}.format")
    claims = _artifact_claims(
        evidence_format,
        artifact,
        source_commit=source_commit,
        architecture_conformance=architecture_conformance,
        name=name,
    )
    return Evidence(
        id=_identifier(mapping["id"], f"{name}.id"),
        path=path,
        source_commit=source_commit,
        record_commit=_sha1(mapping["record_commit"], f"{name}.record_commit"),
        sha256=sha256,
        architecture_conformance=architecture_conformance,
        format=evidence_format,
        claims=claims,
    )


def _excluded_evidence(value: object, index: int, *, source_root: Path) -> ExcludedEvidence:
    name = f"excluded_evidence[{index}]"
    mapping = _closed_mapping(
        value,
        name,
        {
            "id",
            "path",
            "source_commit",
            "record_commit",
            "sha256",
            "architecture_conformance",
            "format",
            "exclusion_reason",
        },
    )
    evidence = _evidence(
        {key: item for key, item in mapping.items() if key != "exclusion_reason"},
        name,
        source_root=source_root,
    )
    reason = _enum(ExclusionReason, mapping["exclusion_reason"], f"{name}.exclusion_reason")
    if evidence.architecture_conformance is not ArchitectureConformance.NONCONFORMING:
        raise EvaluationManifestError(f"{name} must be architecture-nonconforming evidence")
    return ExcludedEvidence(evidence=evidence, exclusion_reason=reason)


def _artifact_claims(
    evidence_format: EvidenceFormat,
    artifact: bytes,
    *,
    source_commit: str,
    architecture_conformance: ArchitectureConformance,
    name: str,
) -> tuple[EvidenceClaim, ...]:
    if evidence_format is EvidenceFormat.ARCHITECTURE_STATUS_DOCUMENT_V1:
        if architecture_conformance is not ArchitectureConformance.NONCONFORMING:
            raise EvaluationManifestError(f"{name} architecture status document must remain nonconforming")
        text = artifact.decode("utf-8")
        if "This checkpoint is `architecture_nonconforming`." not in text:
            raise EvaluationManifestError(f"{name} does not contain its required architecture status")
        return ()

    if architecture_conformance is not ArchitectureConformance.CONFORMING:
        raise EvaluationManifestError(f"{name} CPU acceptance artifact must remain conforming")
    fields = _key_value_artifact(artifact, name=name)
    expected = {
        "canonical_marin_sha": source_commit,
        "terminal_state": "succeeded",
        "exit_code": "0",
        "failures": "0",
        "preemptions": "0",
        "retries": "0",
        "jaxlib_cpu_acceptance": "PASS",
        "concurrency.policies": "source_ordered,source_ordered,fast,fast",
        "concurrency.fixtures": "forward,vjp,forward,vjp",
        "concurrency.forward_source_ordered.bitwise": "true",
        "concurrency.vjp_source_ordered.bitwise": "true",
        "concurrency.forward_fast.bitwise": "true",
        "concurrency.vjp_fast.bitwise": "true",
    }
    if any(fields.get(key) != expected_value for key, expected_value in expected.items()):
        raise EvaluationManifestError(f"{name} does not satisfy the CPU ordinary-JAX acceptance format")
    return tuple(
        EvidenceClaim(
            identity=CellIdentity(
                target_id=0,
                boundary=boundary,
                shape="tanh_dot_primary_f32",
                policy=policy,
                hardware=Hardware.CPU,
            ),
            gates=(
                EvidenceGateClaim(Gate.ARCHITECTURE, Status.ACCEPTED),
                EvidenceGateClaim(Gate.NUMERICAL, Status.PARTIAL),
            ),
        )
        for boundary, policy in itertools.product(
            (Boundary.FORWARD, Boundary.BACKWARD),
            (Numerics.SOURCE_ORDERED, Numerics.FAST),
        )
    )


def _key_value_artifact(artifact: bytes, *, name: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in artifact.decode("utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise EvaluationManifestError(f"{name} is not a unique key-value artifact")
        fields[key] = value
    return fields


def _target(value: object, index: int) -> Target:
    name = f"targets[{index}]"
    mapping = _closed_mapping(
        value,
        name,
        {
            "id",
            "name",
            "status",
            "required_boundaries",
            "required_shapes",
            "required_policies",
            "required_hardware",
            "blockers",
        },
    )
    target_id = _integer(mapping["id"], f"{name}.id")
    return Target(
        id=target_id,
        name=_identifier(mapping["name"], f"{name}.name"),
        status=_enum(Status, mapping["status"], f"{name}.status"),
        required_boundaries=_unique_enum_list(Boundary, mapping["required_boundaries"], f"{name}.required_boundaries"),
        required_shapes=_unique_identifier_list(mapping["required_shapes"], f"{name}.required_shapes"),
        required_policies=_unique_enum_list(Numerics, mapping["required_policies"], f"{name}.required_policies"),
        required_hardware=_unique_enum_list(Hardware, mapping["required_hardware"], f"{name}.required_hardware"),
        blockers=_unique_enum_list(Blocker, mapping["blockers"], f"{name}.blockers", allow_empty=True),
    )


def _cell(value: object, index: int) -> EvaluationCell:
    name = f"cells[{index}]"
    mapping = _closed_mapping(
        value,
        name,
        {
            "target_id",
            "boundary",
            "shape",
            "policy",
            "hardware",
            "status",
            "architecture_status",
            "numerical_status",
            "performance_status",
            "evidence_ids",
            "blockers",
        },
    )
    return EvaluationCell(
        identity=CellIdentity(
            target_id=_integer(mapping["target_id"], f"{name}.target_id"),
            boundary=_enum(Boundary, mapping["boundary"], f"{name}.boundary"),
            shape=_identifier(mapping["shape"], f"{name}.shape"),
            policy=_enum(Numerics, mapping["policy"], f"{name}.policy"),
            hardware=_enum(Hardware, mapping["hardware"], f"{name}.hardware"),
        ),
        status=_enum(Status, mapping["status"], f"{name}.status"),
        architecture_status=_enum(Status, mapping["architecture_status"], f"{name}.architecture_status"),
        numerical_status=_enum(Status, mapping["numerical_status"], f"{name}.numerical_status"),
        performance_status=_enum(Status, mapping["performance_status"], f"{name}.performance_status"),
        evidence_ids=_unique_identifier_list(mapping["evidence_ids"], f"{name}.evidence_ids", allow_empty=True),
        blockers=_unique_enum_list(Blocker, mapping["blockers"], f"{name}.blockers", allow_empty=True),
    )


def _targets_by_id(targets: tuple[Target, ...]) -> dict[int, Target]:
    by_id: dict[int, Target] = {}
    for target in targets:
        if target.id in by_id:
            raise EvaluationManifestError(f"duplicate target ID {target.id}")
        by_id[target.id] = target
    if set(by_id) != set(EXPECTED_TARGETS):
        raise EvaluationManifestError("targets must cover exactly IDs 0 through 10")
    for target_id, expected_name in EXPECTED_TARGETS.items():
        target = by_id[target_id]
        if target.name != expected_name:
            raise EvaluationManifestError(f"target {target_id} must be named {expected_name!r}")
        if frozenset(target.required_policies) != REQUIRED_POLICIES:
            raise EvaluationManifestError(f"target {target_id} must require source_ordered and fast policies")
        expected_boundaries, expected_shapes, expected_hardware = EXPECTED_TARGET_DIMENSIONS[target_id]
        if {value.value for value in target.required_boundaries} != expected_boundaries:
            raise EvaluationManifestError(f"target {target_id} boundaries differ from schema version 1")
        if set(target.required_shapes) != expected_shapes:
            raise EvaluationManifestError(f"target {target_id} shapes differ from schema version 1")
        if {value.value for value in target.required_hardware} != expected_hardware:
            raise EvaluationManifestError(f"target {target_id} hardware differs from schema version 1")
    return by_id


def _validate_cells(
    cells: tuple[EvaluationCell, ...],
    *,
    target_by_id: Mapping[int, Target],
    shape_by_id: Mapping[str, ShapeDeclaration],
    evidence_by_id: Mapping[str, Evidence],
    excluded_evidence_ids: frozenset[str],
) -> None:
    cell_by_identity: dict[CellIdentity, EvaluationCell] = {}
    for cell in cells:
        identity = cell.identity
        if identity in cell_by_identity:
            raise EvaluationManifestError(f"duplicate cell identity {identity}")
        cell_by_identity[identity] = cell
        target = target_by_id.get(identity.target_id)
        if target is None:
            raise EvaluationManifestError(f"cell references unknown target {identity.target_id}")
        shape = shape_by_id.get(identity.shape)
        if shape is None:
            raise EvaluationManifestError(f"cell references unknown shape {identity.shape!r}")
        _validate_cell_status(
            cell,
            shape=shape,
            evidence_by_id=evidence_by_id,
            excluded_evidence_ids=excluded_evidence_ids,
        )

    expected_identities: set[CellIdentity] = set()
    for target in target_by_id.values():
        for shape_id in target.required_shapes:
            if shape_id not in shape_by_id:
                raise EvaluationManifestError(f"target {target.id} references unknown required shape {shape_id!r}")
        expected_identities.update(
            CellIdentity(target.id, boundary, shape, policy, hardware)
            for boundary, shape, policy, hardware in itertools.product(
                target.required_boundaries,
                target.required_shapes,
                target.required_policies,
                target.required_hardware,
            )
        )
    actual_identities = set(cell_by_identity)
    if actual_identities != expected_identities:
        missing = len(expected_identities - actual_identities)
        extra = len(actual_identities - expected_identities)
        raise EvaluationManifestError(
            f"cell matrix does not match required dimensions: {missing} missing, {extra} extra"
        )
    for evidence in evidence_by_id.values():
        if any(claim.identity not in expected_identities for claim in evidence.claims):
            raise EvaluationManifestError(f"evidence {evidence.id!r} claims a cell outside the required matrix")


def _validate_cell_status(
    cell: EvaluationCell,
    *,
    shape: ShapeDeclaration,
    evidence_by_id: Mapping[str, Evidence],
    excluded_evidence_ids: frozenset[str],
) -> None:
    for evidence_id in cell.evidence_ids:
        if evidence_id in excluded_evidence_ids:
            raise EvaluationManifestError(f"cell references architecture-nonconforming evidence {evidence_id!r}")
        if evidence_id not in evidence_by_id:
            raise EvaluationManifestError(f"cell references unknown evidence {evidence_id!r}")
        if not any(claim.identity == cell.identity for claim in evidence_by_id[evidence_id].claims):
            raise EvaluationManifestError(f"evidence {evidence_id!r} does not support cell {cell.identity}")

    gate_statuses = {
        Gate.ARCHITECTURE: cell.architecture_status,
        Gate.NUMERICAL: cell.numerical_status,
        Gate.PERFORMANCE: cell.performance_status,
    }
    expected_status = _cell_status(gate_statuses.values(), has_blockers=bool(cell.blockers))
    if cell.status is not expected_status:
        raise EvaluationManifestError(f"cell {cell.identity} aggregate status does not match its gates and blockers")
    evidenced_gate = any(status in {Status.FAILED, Status.PARTIAL, Status.ACCEPTED} for status in gate_statuses.values())
    if evidenced_gate and not cell.evidence_ids:
        raise EvaluationManifestError(f"cell {cell.identity} has an evidenced gate without durable evidence")
    if cell.status is Status.BLOCKED and not cell.blockers:
        raise EvaluationManifestError(f"blocked cell {cell.identity} requires a closed blocker")
    if cell.status is Status.ACCEPTED and cell.blockers:
        raise EvaluationManifestError(f"accepted cell {cell.identity} must not retain blockers")
    if cell.performance_status is Status.ACCEPTED and (
        cell.architecture_status is not Status.ACCEPTED or cell.numerical_status is not Status.ACCEPTED
    ):
        raise EvaluationManifestError("performance acceptance requires architecture and numerical acceptance")
    for gate, status in gate_statuses.items():
        if status in {Status.FAILED, Status.PARTIAL, Status.ACCEPTED} and not any(
            EvidenceGateClaim(gate, status) in claim.gates
            for evidence_id in cell.evidence_ids
            for claim in evidence_by_id[evidence_id].claims
            if claim.identity == cell.identity
        ):
            raise EvaluationManifestError(f"{status.value} {gate.value} gate lacks cell-scoped evidence")
    if shape.declaration_status is not Status.ACCEPTED and cell.status not in {Status.NOT_STARTED, Status.BLOCKED}:
        raise EvaluationManifestError(f"cell {cell.identity} cannot execute against an undeclared representative shape")
    if shape.id.endswith(PENDING_SHAPE_SUFFIX) and shape.declaration_status is Status.ACCEPTED:
        raise EvaluationManifestError(f"pending shape {shape.id!r} cannot be declared accepted")
    shape_blocker = Blocker.REPRESENTATIVE_SHAPE_NOT_DECLARED in cell.blockers
    if shape_blocker == (shape.declaration_status is Status.ACCEPTED):
        raise EvaluationManifestError(f"cell {cell.identity} representative-shape blocker disagrees with declaration")


def _cell_status(gate_statuses: Iterable[Status], *, has_blockers: bool) -> Status:
    statuses = frozenset(gate_statuses)
    if statuses == {Status.ACCEPTED}:
        return Status.ACCEPTED
    if Status.FAILED in statuses:
        return Status.FAILED
    if statuses & {Status.PARTIAL, Status.ACCEPTED}:
        return Status.PARTIAL
    if Status.BLOCKED in statuses or has_blockers:
        return Status.BLOCKED
    return Status.NOT_STARTED


def _validate_target_statuses(
    targets: tuple[Target, ...],
    cells: tuple[EvaluationCell, ...],
    *,
    shape_by_id: Mapping[str, ShapeDeclaration],
) -> None:
    cells_by_target = {target.id: [] for target in targets}
    for cell in cells:
        cells_by_target[cell.identity.target_id].append(cell)
    for target in targets:
        statuses = {cell.status for cell in cells_by_target[target.id]}
        expected_status = _target_status(statuses)
        if target.status is not expected_status:
            raise EvaluationManifestError(f"target {target.id} aggregate status does not match its required cells")
        if target.status is Status.ACCEPTED and target.blockers:
            raise EvaluationManifestError(f"accepted target {target.id} must not retain blockers")
        if target.status is Status.BLOCKED and (not target.blockers or Status.BLOCKED not in statuses):
            raise EvaluationManifestError(f"blocked target {target.id} requires blockers and blocked cells")
        has_undeclared_shape = any(
            shape_by_id[shape_id].declaration_status is not Status.ACCEPTED for shape_id in target.required_shapes
        )
        shape_blocker = Blocker.REPRESENTATIVE_SHAPE_NOT_DECLARED in target.blockers
        if shape_blocker is not has_undeclared_shape:
            raise EvaluationManifestError(f"target {target.id} representative-shape blocker disagrees with declaration")


def _target_status(cell_statuses: set[Status]) -> Status:
    if cell_statuses == {Status.ACCEPTED}:
        return Status.ACCEPTED
    if Status.FAILED in cell_statuses:
        return Status.FAILED
    if cell_statuses & {Status.PARTIAL, Status.ACCEPTED}:
        return Status.PARTIAL
    if Status.BLOCKED in cell_statuses:
        return Status.BLOCKED
    return Status.NOT_STARTED


def _closed_mapping(value: object, name: str, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise EvaluationManifestError(f"{name} must be an object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise EvaluationManifestError(f"{name} keys must be strings")
    typed = cast(dict[str, object], mapping)
    missing = fields - typed.keys()
    unknown = typed.keys() - fields
    if missing or unknown:
        raise EvaluationManifestError(
            f"{name} fields differ from schema: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return typed


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise EvaluationManifestError(f"{name} must be a list")
    return cast(list[object], value)


def _unique_by_id[IdentifiedType: Identified](
    values: Iterable[IdentifiedType], description: str
) -> dict[str, IdentifiedType]:
    by_id: dict[str, IdentifiedType] = {}
    for value in values:
        if value.id in by_id:
            raise EvaluationManifestError(f"duplicate {description} ID {value.id!r}")
        by_id[value.id] = value
    return by_id


def _enum[EnumType: StrEnum](enum_type: type[EnumType], value: object, name: str) -> EnumType:
    raw = _nonempty_string(value, name)
    try:
        return enum_type(raw)
    except ValueError as error:
        raise EvaluationManifestError(f"{name} has unknown {enum_type.__name__} value {raw!r}") from error


def _unique_enum_list[EnumType: StrEnum](
    enum_type: type[EnumType],
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[EnumType, ...]:
    values = tuple(_enum(enum_type, item, f"{name}[{index}]") for index, item in enumerate(_list(value, name)))
    if not allow_empty and not values:
        raise EvaluationManifestError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise EvaluationManifestError(f"{name} must not contain duplicates")
    return values


def _unique_identifier_list(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = tuple(_identifier(item, f"{name}[{index}]") for index, item in enumerate(_list(value, name)))
    if not allow_empty and not values:
        raise EvaluationManifestError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise EvaluationManifestError(f"{name} must not contain duplicates")
    return values


def _identifier(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    if IDENTIFIER_PATTERN.fullmatch(text) is None:
        raise EvaluationManifestError(f"{name} must be a lowercase machine identifier")
    return text


def _nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise EvaluationManifestError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise EvaluationManifestError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise EvaluationManifestError(f"{name} must be a number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise EvaluationManifestError(f"{name} must be finite")
    return result


def _sha1(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    if SHA1_PATTERN.fullmatch(text) is None:
        raise EvaluationManifestError(f"{name} must be a full lowercase Git SHA")
    return text


def _sha256(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise EvaluationManifestError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _relative_path(value: object, name: str) -> PurePosixPath:
    text = _nonempty_string(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise EvaluationManifestError(f"{name} must be a normalized repository-relative path")
    return path


def _verified_artifact(path: PurePosixPath, expected_sha256: str, *, source_root: Path, name: str) -> bytes:
    resolved_root = source_root.resolve()
    resolved = (resolved_root / Path(path)).resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise EvaluationManifestError(f"{name}.path does not resolve to a repository file")
    artifact = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(artifact).hexdigest()
    if actual_sha256 != expected_sha256:
        raise EvaluationManifestError(f"{name}.sha256 does not match the linked artifact")
    return artifact


def _verify_git_evidence(evidence: Evidence, *, baseline_commit: str, source_root: Path) -> None:
    _verify_git_commit(evidence.source_commit, source_root=source_root, name=f"evidence {evidence.id!r} source_commit")
    _verify_git_commit(evidence.record_commit, source_root=source_root, name=f"evidence {evidence.id!r} record_commit")
    _require_git_ancestor(
        evidence.source_commit,
        baseline_commit,
        source_root=source_root,
        name=f"evidence {evidence.id!r} source_commit",
    )
    _require_git_ancestor(
        evidence.record_commit,
        baseline_commit,
        source_root=source_root,
        name=f"evidence {evidence.id!r} record_commit",
    )
    artifact_at_record = _git(
        ("show", f"{evidence.record_commit}:{evidence.path.as_posix()}"),
        source_root=source_root,
        name=f"evidence {evidence.id!r} artifact provenance",
    ).stdout
    if hashlib.sha256(artifact_at_record).hexdigest() != evidence.sha256:
        raise EvaluationManifestError(f"evidence {evidence.id!r} digest does not match its record commit")


def _verify_git_commit(commit: str, *, source_root: Path, name: str) -> None:
    _git(("cat-file", "-e", f"{commit}^{{commit}}"), source_root=source_root, name=name)


def _require_git_ancestor(ancestor: str, descendant: str, *, source_root: Path, name: str) -> None:
    _git(("merge-base", "--is-ancestor", ancestor, descendant), source_root=source_root, name=name)


def _git(arguments: tuple[str, ...], *, source_root: Path, name: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ("git", "-C", str(source_root.resolve()), *arguments),
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise EvaluationManifestError(f"could not verify {name} with Git") from error
    if result.returncode != 0:
        raise EvaluationManifestError(f"Git could not verify {name}")
    return result
