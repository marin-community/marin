# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure structural expectations for the pinned ordinary-JAX fixtures."""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SUCCESS_PHASES = ("algebra_coverage", "lowered_coverage", "final_erasure")
EVENT_FIELDS = (
    "invocation_id",
    "phase",
    "policy",
    "policy_digest",
    "tuning_digest",
    "region_membership",
    "coverage_manifest",
    "unsupported_fingerprint",
    "normalized_module_fingerprint",
    "no_shuttle_semantics",
    "failure_pass",
)


@dataclass(frozen=True)
class ObserverIdentity:
    policy: str
    policy_digest: str
    tuning_digest: str
    canonical_options: str
    canonical_tuning: str


@dataclass(frozen=True)
class FixtureExpectation:
    name: str
    complete_operations: tuple[int, ...]
    selected_regions: tuple[tuple[int, ...], ...]
    excluded_manifest: str
    function_result_anchors: tuple[int, ...]
    final_normalized_fingerprint: str

    @property
    def region_membership(self) -> str:
        return _attribute_array(
            _attribute_array(_source_ref(ordinal) for ordinal in region) for region in self.selected_regions
        )

    @property
    def unsupported_fingerprint(self) -> str:
        return hashlib.sha256(self.excluded_manifest.encode()).hexdigest()

    def coverage_manifest(self, identity: ObserverIdentity) -> str:
        complete = _attribute_array(_source_ref(ordinal) for ordinal in self.complete_operations)
        function_results = _attribute_array(
            ("{anchor = " f"{_source_ref(anchor)}, function = 0 : i64, result = {result} : i64" "}")
            for result, anchor in enumerate(self.function_result_anchors)
        )
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
            "version = 1 : i64, "
            f"zero_result_operations = {self._zero_result_operations()}"
            "}"
        )

    def _zero_result_operations(self) -> str:
        operands = _attribute_array(_source_ref(anchor) for anchor in self.function_result_anchors)
        return (
            '[{classification = "terminator", fingerprint = {attributes = {}, '
            'name = "func.return", result_types = []}, '
            f"operands = {operands}, operation_ref = array<i64: 0, 0, {len(self.complete_operations)}>"
            "}]"
        )


def _source_ref(operation: int) -> str:
    return f"#shuttle.source_ref<0, 0, {operation}, 0>"


def _attribute_array(items: Iterable[str]) -> str:
    return "[" + ", ".join(items) + "]"


def _mlir_string(value: str) -> str:
    if any(ord(character) < 0x20 or character in "\\" for character in value):
        raise ValueError("fixture options contain an unsupported MLIR string character")
    return '"' + value.replace('"', r"\22") + '"'


# These immutable structural oracles are audited from the pinned ordinary-JAX
# exports in ../test/Inputs/jax-0.10.1-tanh-dot-{forward,vjp}.mlir. They encode
# source ordinals and normalized operation structure, never callable names.
VJP_EXCLUDED_MANIFEST = (
    "[{fingerprint = {attributes = {value = dense<1.000000e+00> : tensor<f32>}, "
    'name = "stablehlo.constant", result_types = [tensor<f32>]}, operands = [], '
    'reason = "unsupported_operation", source = #shuttle.source_ref<0, 0, 2, 0>}, '
    "{fingerprint = {attributes = {broadcast_dimensions = array<i64>}, "
    'name = "stablehlo.broadcast_in_dim", result_types = [tensor<2x4xf32>]}, '
    'operands = [#shuttle.source_ref<0, 0, 2, 0>], reason = "unsupported_operation", '
    "source = #shuttle.source_ref<0, 0, 3, 0>}, "
    '{fingerprint = {attributes = {}, name = "stablehlo.subtract", '
    "result_types = [tensor<2x4xf32>]}, operands = [#shuttle.source_ref<0, 0, 3, 0>, "
    '#shuttle.source_ref<0, 0, 1, 0>], reason = "unsupported_operation", '
    "source = #shuttle.source_ref<0, 0, 4, 0>}]"
)

FORWARD_EXPECTATION = FixtureExpectation(
    name="forward",
    complete_operations=(0, 1, 2),
    selected_regions=((0, 1, 2),),
    excluded_manifest="[]",
    function_result_anchors=(2,),
    final_normalized_fingerprint="01539d7d3febf0814ccf67320863712fa19e0425bdda9a716b4716fbe2efc944",
)
VJP_EXPECTATION = FixtureExpectation(
    name="vjp",
    complete_operations=tuple(range(14)),
    selected_regions=((0, 1), (5, 6), (7, 8, 9, 10, 11, 12, 13)),
    excluded_manifest=VJP_EXCLUDED_MANIFEST,
    function_result_anchors=(13, 12, 6),
    final_normalized_fingerprint="2d557bd5d2f259a053335a6e004f9c5290d19713961e2c41787ed197ed042891",
)
FIXTURE_EXPECTATIONS = (FORWARD_EXPECTATION, VJP_EXPECTATION)


def decode_native_snapshot(records: object) -> tuple[dict[str, Any], ...]:
    """Validate and decode the immutable native observer snapshot."""
    if type(records) is not tuple:
        raise AssertionError("native observer snapshot must be an immutable tuple")
    if any(type(record) is not tuple for record in records):
        raise AssertionError("native observer records must be immutable tuples")
    if any(len(record) != len(EVENT_FIELDS) for record in records):
        raise AssertionError("native observer record schema changed")
    return tuple(dict(zip(EVENT_FIELDS, record, strict=True)) for record in records)


def validate_success_events(
    events: Sequence[Mapping[str, Any]],
    identity: ObserverIdentity,
    fixture: FixtureExpectation,
) -> dict[str, Any]:
    if len(events) != len(SUCCESS_PHASES):
        raise AssertionError("one successful compilation must emit exactly three observer phases")
    if tuple(event["phase"] for event in events) != SUCCESS_PHASES:
        raise AssertionError("one successful compilation must emit the three ordered observer phases")
    if len({event["invocation_id"] for event in events}) != 1:
        raise AssertionError("observer phases do not share one invocation ID")

    for event in events:
        if event["policy"] != identity.policy:
            raise AssertionError("observer policy differs from compiler options")
        if event["policy_digest"] != identity.policy_digest:
            raise AssertionError("observer policy digest differs from the full canonical options")
        if event["tuning_digest"] != identity.tuning_digest:
            raise AssertionError("observer tuning digest differs from canonical tuning")
        if event["failure_pass"] != "":
            raise AssertionError("successful compilation emitted a failure pass")

    expected_manifest = fixture.coverage_manifest(identity)
    for event in events[:2]:
        if event["region_membership"] != fixture.region_membership:
            raise AssertionError(f"{fixture.name}: normalized selected-region membership changed")
        if event["coverage_manifest"] != expected_manifest:
            raise AssertionError(f"{fixture.name}: complete coverage manifest changed")
        if event["unsupported_fingerprint"] != fixture.unsupported_fingerprint:
            raise AssertionError(f"{fixture.name}: unsupported structural island changed")
        if event["normalized_module_fingerprint"] != "" or event["no_shuttle_semantics"] is not False:
            raise AssertionError("pre-final observer phase erased provenance early")

    final = events[2]
    if final["region_membership"] != "" or final["coverage_manifest"] != "":
        raise AssertionError("final observer phase retained region or manifest provenance")
    if final["unsupported_fingerprint"] != "":
        raise AssertionError("final observer phase retained the unsupported-island fingerprint")
    if final["normalized_module_fingerprint"] != fixture.final_normalized_fingerprint:
        raise AssertionError(f"{fixture.name}: final normalized module fingerprint changed")
    if final["no_shuttle_semantics"] is not True:
        raise AssertionError("final observer phase did not prove Shuttle erasure")

    return {
        "invocation_id": events[0]["invocation_id"],
        "fixture": fixture.name,
        "policy": identity.policy,
        "policy_digest": identity.policy_digest,
        "tuning_digest": identity.tuning_digest,
        "region_membership": fixture.region_membership,
        "coverage_manifest": expected_manifest,
        "unsupported_fingerprint": fixture.unsupported_fingerprint,
        "final_fingerprint": fixture.final_normalized_fingerprint,
    }
