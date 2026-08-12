# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ABI 6 Target 1 installed-wheel contract."""

import copy

import pytest
from acceptance_contract import ObserverIdentity
from target1_acceptance_contract import (
    TARGET1_EXPECTATIONS,
    target1_expectation,
    validate_target1_success_events,
)


def identity(policy: str) -> ObserverIdentity:
    return ObserverIdentity(
        policy=policy,
        policy_digest=f"{policy}-policy-digest",
        tuning_digest="tuning-digest",
        canonical_options=f'{{"execution_mode":"stablehlo_round_trip","numerics":"{policy}","pipeline_abi_version":6}}',
        canonical_tuning="{}",
    )


def successful_events(shape_id: str, boundary: str, policy: str = "source_ordered") -> list[dict[str, object]]:
    fixture = target1_expectation(shape_id, boundary)
    observer_identity = identity(policy)
    common = {
        "invocation_id": 11,
        "policy": observer_identity.policy,
        "policy_digest": observer_identity.policy_digest,
        "tuning_digest": observer_identity.tuning_digest,
        "unsupported_fingerprint": fixture.unsupported_fingerprint,
        "failure_pass": "",
    }
    manifest = fixture.coverage_manifest(observer_identity)
    return [
        {
            **common,
            "phase": phase,
            "region_membership": fixture.region_membership,
            "coverage_manifest": manifest,
            "normalized_module_fingerprint": "",
            "no_shuttle_semantics": False,
        }
        for phase in ("algebra_coverage", "lowered_coverage")
    ] + [
        {
            **common,
            "phase": "final_erasure",
            "region_membership": "",
            "coverage_manifest": "",
            "unsupported_fingerprint": "",
            "normalized_module_fingerprint": fixture.final_normalized_fingerprint,
            "no_shuttle_semantics": True,
        }
    ]


def test_contract_covers_six_shape_boundaries_without_workload_names() -> None:
    assert len(TARGET1_EXPECTATIONS) == 6
    assert {(item.shape_id, item.boundary) for item in TARGET1_EXPECTATIONS} == {
        (shape_id, boundary)
        for shape_id in ("44d152ecc3e9ff18", "81928ab3539c0f03")
        for boundary in ("forward", "backward", "composed")
    }
    assert {len(item.complete) for item in TARGET1_EXPECTATIONS if item.boundary == "forward"} == {20}
    assert {len(item.complete) for item in TARGET1_EXPECTATIONS if item.boundary == "backward"} == {48}
    assert {len(item.complete) for item in TARGET1_EXPECTATIONS if item.boundary == "composed"} == {51}
    assert all(item.excluded_manifest == "[]" for item in TARGET1_EXPECTATIONS)


@pytest.mark.parametrize("policy", ("source_ordered", "fast"))
@pytest.mark.parametrize("fixture", TARGET1_EXPECTATIONS, ids=lambda fixture: fixture.label)
def test_exact_twelve_cell_coverage_and_erasure_contract(fixture, policy: str) -> None:
    evidence = validate_target1_success_events(
        successful_events(fixture.shape_id, fixture.boundary, policy),
        identity(policy),
        fixture,
    )
    assert evidence["shape_id"] == fixture.shape_id
    assert evidence["boundary"] == fixture.boundary
    assert evidence["policy"] == policy
    assert evidence["complete_source_results"] == len(fixture.complete)
    assert evidence["excluded_source_results"] == 0
    assert evidence["final_fingerprint"] == fixture.final_normalized_fingerprint


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda events: events[0].__setitem__("coverage_manifest", "{}"), "coverage manifest"),
        (lambda events: events[1].__setitem__("region_membership", "[]"), "selected-region"),
        (lambda events: events[2].__setitem__("no_shuttle_semantics", False), "erasure"),
        (
            lambda events: events[2].__setitem__("normalized_module_fingerprint", "0" * 64),
            "fingerprint",
        ),
    ),
)
def test_contract_fails_closed_on_architecture_mutations(mutation, message: str) -> None:
    events = copy.deepcopy(successful_events("44d152ecc3e9ff18", "forward"))
    mutation(events)
    with pytest.raises(AssertionError, match=message):
        validate_target1_success_events(
            events,
            identity("source_ordered"),
            target1_expectation("44d152ecc3e9ff18", "forward"),
        )


@pytest.mark.parametrize("policy", ("source_ordered", "fast"))
@pytest.mark.parametrize("fixture", TARGET1_EXPECTATIONS, ids=lambda fixture: fixture.label)
def test_each_cell_rejects_wrong_policy_attribution(fixture, policy: str) -> None:
    events = successful_events(fixture.shape_id, fixture.boundary, policy)
    events[1]["policy"] = "fast" if policy == "source_ordered" else "source_ordered"
    with pytest.raises(AssertionError, match="policy identity"):
        validate_target1_success_events(events, identity(policy), fixture)


def test_unknown_shape_or_boundary_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown Target 1 fixture identity"):
        target1_expectation("unknown", "forward")
    with pytest.raises(ValueError, match="unknown Target 1 fixture identity"):
        target1_expectation("44d152ecc3e9ff18", "training")
