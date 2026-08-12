# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact ABI 7 observer contracts for the six BF16 rowwise fixtures."""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from acceptance_contract import SUCCESS_PHASES, ObserverIdentity

SourceRef = tuple[int, int, int, int]


@dataclass(frozen=True)
class ZeroResultOperation:
    name: str
    operands: tuple[SourceRef, ...]
    operation_ref: tuple[int, ...]

    @property
    def attribute(self) -> str:
        operands = _attribute_array(_source_ref(reference) for reference in self.operands)
        operation_ref = ", ".join(str(value) for value in self.operation_ref)
        return (
            '{classification = "terminator", fingerprint = {attributes = {}, '
            f'name = "{self.name}", result_types = []}}, operands = {operands}, '
            f"operation_ref = array<i64: {operation_ref}>}}"
        )


@dataclass(frozen=True)
class Target1FixtureExpectation:
    shape_id: str
    boundary: str
    complete: tuple[SourceRef, ...]
    function_result_anchors: tuple[SourceRef, ...]
    zero_result_operations: tuple[ZeroResultOperation, ...]
    final_normalized_fingerprint: str
    excluded_manifest: str = "[]"

    @property
    def label(self) -> str:
        return f"{self.shape_id}_{self.boundary}"

    @property
    def region_membership(self) -> str:
        return _attribute_array((_attribute_array(_source_ref(reference) for reference in self.complete),))

    @property
    def unsupported_fingerprint(self) -> str:
        return hashlib.sha256(self.excluded_manifest.encode()).hexdigest()

    def coverage_manifest(self, identity: ObserverIdentity) -> str:
        complete = _attribute_array(_source_ref(reference) for reference in self.complete)
        function_results = _attribute_array(
            ("{anchor = " f"{_source_ref(anchor)}, function = 0 : i64, result = {result} : i64" "}")
            for result, anchor in enumerate(self.function_result_anchors)
        )
        zero_results = _attribute_array(operation.attribute for operation in self.zero_result_operations)
        return (
            "{canonical_options = "
            f"{_mlir_string(identity.canonical_options)}, "
            f"canonical_tuning = {_mlir_string(identity.canonical_tuning)}, "
            f"complete = {complete}, "
            f"excluded = {self.excluded_manifest}, "
            f"function_results = {function_results}, "
            f'policy = "{identity.policy}", '
            f'policy_digest = "{identity.policy_digest}", '
            f"selected_regions = {self.region_membership}, "
            f'tuning_digest = "{identity.tuning_digest}", '
            "version = 2 : i64, "
            f"zero_result_operations = {zero_results}"
            "}"
        )


def _source_ref(reference: SourceRef) -> str:
    return "#shuttle.source_ref<" + ", ".join(str(value) for value in reference) + ">"


def _attribute_array(items: Iterable[str]) -> str:
    return "[" + ", ".join(items) + "]"


def _mlir_string(value: str) -> str:
    if any(ord(character) < 0x20 or character in "\\" for character in value):
        raise ValueError("fixture options contain an unsupported MLIR string character")
    return '"' + value.replace('"', r"\22") + '"'


def _top(operation: int) -> SourceRef:
    return (0, 0, operation, 0)


def _nested(region: int) -> SourceRef:
    return (0, region, 0, 0)


def _refs(parts: Sequence[range | int]) -> tuple[SourceRef, ...]:
    references = []
    for part in parts:
        if isinstance(part, int):
            references.append(_nested(part))
        else:
            references.extend(_top(operation) for operation in part)
    return tuple(references)


FORWARD_REFS = _refs((range(0, 6), 1, range(6, 19)))
BACKWARD_REFS = _refs(
    (range(0, 7), 1, range(7, 23), 2, range(23, 27), 3, range(27, 30), 4, range(30, 37), 5, range(37, 43))
)
COMPOSED_REFS = _refs(
    (range(0, 7), 1, range(7, 26), 2, range(26, 30), 3, range(30, 33), 4, range(33, 40), 5, range(40, 46))
)


def _zero_results(boundary: str) -> tuple[ZeroResultOperation, ...]:
    reducer_count = 1 if boundary == "forward" else 5
    if boundary == "forward":
        anchors = (_top(18),)
        return_ordinal = 19
    elif boundary == "backward":
        anchors = (_top(42), _top(27))
        return_ordinal = 43
    else:
        anchors = (_top(22), _top(45), _top(30))
        return_ordinal = 46
    reducers = tuple(
        ZeroResultOperation("stablehlo.return", (_nested(region),), (0, region, 1))
        for region in range(1, reducer_count + 1)
    )
    return (*reducers, ZeroResultOperation("func.return", anchors, (0, 0, return_ordinal)))


_FINGERPRINTS = {
    ("44d152ecc3e9ff18", "forward"): "b0a69e21331c5ebc86681c44e8a2ae00ff7d7f21bc1b95539ba0e2496678af7e",
    ("44d152ecc3e9ff18", "backward"): "f6f78a1bc29a210e6c0a4a4b632307c93a6410508bc24be76a99cad7205ada92",
    ("44d152ecc3e9ff18", "composed"): "8c0e9200787e5fc2b91f36ac83b62ab8d30979287edb4fd2b6dfcb12fd73ecbe",
    ("81928ab3539c0f03", "forward"): "541d20db0599d64d0d7e853b8a3a8ed00004a19b5a67aea01105609d9bc83616",
    ("81928ab3539c0f03", "backward"): "048bb8757432cb7b2ed65666189861ce92f339d19981f93aa71d084ec2d5f7b1",
    ("81928ab3539c0f03", "composed"): "b23d862fc56af03ec3538f32c857b84c19982b19fd3bca27111d5ff43976fe98",
}
_BOUNDARY_REFS = {
    "forward": (FORWARD_REFS, (_top(18),)),
    "backward": (BACKWARD_REFS, (_top(42), _top(27))),
    "composed": (COMPOSED_REFS, (_top(22), _top(45), _top(30))),
}

TARGET1_EXPECTATIONS = tuple(
    Target1FixtureExpectation(
        shape_id=shape_id,
        boundary=boundary,
        complete=_BOUNDARY_REFS[boundary][0],
        function_result_anchors=_BOUNDARY_REFS[boundary][1],
        zero_result_operations=_zero_results(boundary),
        final_normalized_fingerprint=fingerprint,
    )
    for (shape_id, boundary), fingerprint in _FINGERPRINTS.items()
)
_EXPECTATION_BY_IDENTITY = {(item.shape_id, item.boundary): item for item in TARGET1_EXPECTATIONS}


def target1_expectation(shape_id: str, boundary: str) -> Target1FixtureExpectation:
    try:
        return _EXPECTATION_BY_IDENTITY[(shape_id, boundary)]
    except KeyError as error:
        raise ValueError(f"unknown Target 1 fixture identity: {shape_id}/{boundary}") from error


def validate_target1_success_events(
    events: Sequence[Mapping[str, Any]],
    identity: ObserverIdentity,
    fixture: Target1FixtureExpectation,
) -> dict[str, Any]:
    """Require exact total source coverage and final semantic erasure."""
    if len(events) != len(SUCCESS_PHASES) or tuple(event["phase"] for event in events) != SUCCESS_PHASES:
        raise AssertionError("one successful compilation must emit the three ordered observer phases")
    if len({event["invocation_id"] for event in events}) != 1:
        raise AssertionError("observer phases do not share one invocation ID")
    expected_manifest = fixture.coverage_manifest(identity)
    for event in events:
        if event["policy"] != identity.policy or event["policy_digest"] != identity.policy_digest:
            raise AssertionError("observer policy identity differs from compiler options")
        if event["tuning_digest"] != identity.tuning_digest:
            raise AssertionError("observer tuning identity differs from compiler options")
        if event["failure_pass"] != "":
            raise AssertionError("successful compilation emitted a failure pass")
    for event in events[:2]:
        if event["region_membership"] != fixture.region_membership:
            raise AssertionError(f"{fixture.label}: selected-region membership changed")
        if event["coverage_manifest"] != expected_manifest:
            raise AssertionError(f"{fixture.label}: coverage manifest changed")
        if event["unsupported_fingerprint"] != fixture.unsupported_fingerprint:
            raise AssertionError(f"{fixture.label}: unsupported structural island changed")
        if event["normalized_module_fingerprint"] != "" or event["no_shuttle_semantics"] is not False:
            raise AssertionError("pre-final observer phase erased provenance early")
    final = events[2]
    if final["region_membership"] != "" or final["coverage_manifest"] != "":
        raise AssertionError("final observer phase retained provenance")
    if final["unsupported_fingerprint"] != "":
        raise AssertionError("final observer phase retained an unsupported-island fingerprint")
    if final["normalized_module_fingerprint"] != fixture.final_normalized_fingerprint:
        raise AssertionError(f"{fixture.label}: final fingerprint changed")
    if final["no_shuttle_semantics"] is not True:
        raise AssertionError("final observer phase did not prove erasure")
    return {
        "invocation_id": events[0]["invocation_id"],
        "shape_id": fixture.shape_id,
        "boundary": fixture.boundary,
        "policy": identity.policy,
        "policy_digest": identity.policy_digest,
        "tuning_digest": identity.tuning_digest,
        "complete_source_results": len(fixture.complete),
        "excluded_source_results": 0,
        "coverage_manifest": expected_manifest,
        "final_fingerprint": fixture.final_normalized_fingerprint,
    }
