# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversarial tests for the pure ordinary-JAX observer contract."""

import hashlib
import json
import operator
from copy import deepcopy
from pathlib import Path

import pytest
from acceptance_contract import (
    EVENT_FIELDS,
    FORWARD_EXPECTATION,
    VJP_EXPECTATION,
    ObserverIdentity,
    decode_native_snapshot,
    match_fixture_contract,
    validate_success_events,
)
from verify_acceptance_fixture_oracles import (
    EXPECTATIONS,
    audited_fingerprint,
    audited_hook_boundary_fingerprint,
    derived_hook_boundary_fingerprint,
    fixture_path,
    verify_oracles,
)

IDENTITY = ObserverIdentity(
    policy="source_ordered",
    policy_digest="a" * 64,
    tuning_digest="b" * 64,
    canonical_options='{"numerics":"source_ordered"}',
    canonical_tuning='{"tile_sizes":[64,128]}',
)


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "test" / "Inputs"


def valid_events(fixture=VJP_EXPECTATION):
    manifest = fixture.coverage_manifest(IDENTITY)
    common = {
        "invocation_id": 17,
        "policy": IDENTITY.policy,
        "policy_digest": IDENTITY.policy_digest,
        "tuning_digest": IDENTITY.tuning_digest,
        "failure_pass": "",
    }
    return [
        {
            **common,
            "phase": phase,
            "region_membership": fixture.region_membership,
            "coverage_manifest": manifest,
            "unsupported_fingerprint": fixture.unsupported_fingerprint,
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
            "normalized_module_fingerprint": audited_fingerprint(fixture_path(FIXTURE_DIRECTORY, fixture)),
            "no_shuttle_semantics": True,
        }
    ]


@pytest.mark.parametrize(
    ("event_index", "field", "replacement"),
    [
        (0, "phase", "lowered_coverage"),
        (1, "invocation_id", 18),
        (0, "policy", "fast"),
        (1, "policy_digest", "c" * 64),
        (0, "tuning_digest", "d" * 64),
        (0, "region_membership", "[]"),
        (1, "coverage_manifest", "{}"),
        (0, "unsupported_fingerprint", "e" * 64),
        (0, "normalized_module_fingerprint", "f" * 64),
        (1, "no_shuttle_semantics", True),
        (2, "region_membership", VJP_EXPECTATION.region_membership),
        (2, "coverage_manifest", VJP_EXPECTATION.coverage_manifest(IDENTITY)),
        (2, "unsupported_fingerprint", VJP_EXPECTATION.unsupported_fingerprint),
        (2, "normalized_module_fingerprint", "0" * 64),
        (2, "no_shuttle_semantics", False),
        (2, "failure_pass", "shuttle-strip-source-provenance"),
    ],
)
def test_rejects_wrong_or_phase_illegal_evidence(event_index, field, replacement):
    events = deepcopy(valid_events())
    events[event_index][field] = replacement

    with pytest.raises(AssertionError):
        validate_success_events(events, IDENTITY, VJP_EXPECTATION)


def test_rejects_manifest_from_a_different_audited_fixture():
    events = valid_events(VJP_EXPECTATION)
    events[0]["coverage_manifest"] = FORWARD_EXPECTATION.coverage_manifest(IDENTITY)
    events[1]["coverage_manifest"] = FORWARD_EXPECTATION.coverage_manifest(IDENTITY)

    with pytest.raises(AssertionError, match="complete coverage manifest"):
        validate_success_events(events, IDENTITY, VJP_EXPECTATION)


def test_vjp_oracle_is_the_exact_audited_unsupported_chain():
    excluded = VJP_EXPECTATION.excluded_manifest

    assert excluded.count('reason = "unsupported_operation"') == 3
    assert excluded.index('name = "stablehlo.constant"') < excluded.index('name = "stablehlo.broadcast_in_dim"')
    assert excluded.index('name = "stablehlo.broadcast_in_dim"') < excluded.index('name = "stablehlo.subtract"')
    assert VJP_EXPECTATION.unsupported_fingerprint == "1a9aad82650111cbc134fcc17d1afcb051f9ae729f6cdfd48105d1e8dc210201"


def test_rejects_pre_xla_hook_vjp_oracle():
    events = valid_events(VJP_EXPECTATION)
    old_region_membership = VJP_EXPECTATION.region_membership.replace("<0, 0, 1, 0>", "<0, 0, 0, 0>", 1)
    events[0]["region_membership"] = old_region_membership
    events[1]["region_membership"] = old_region_membership

    with pytest.raises(AssertionError, match="selected-region membership"):
        validate_success_events(events, IDENTITY, VJP_EXPECTATION)


def test_accepts_only_the_complete_fixture_contract():
    evidence = validate_success_events(valid_events(), IDENTITY, VJP_EXPECTATION)

    assert evidence["fixture"] == "vjp"
    assert evidence["final_fingerprint"] == VJP_EXPECTATION.final_normalized_fingerprint


def test_contract_mismatch_reports_every_field_and_fixture_without_structural_payloads():
    events = valid_events(VJP_EXPECTATION)
    structural_payload = "private-structural-payload-" * 40
    events[1]["coverage_manifest"] = structural_payload

    with pytest.raises(AssertionError) as failure:
        match_fixture_contract(events, IDENTITY)

    prefix = "observer invocation did not match exactly one audited fixture contract: "
    message = str(failure.value)
    assert message.startswith(prefix)
    diagnostic = json.loads(message.removeprefix(prefix))
    assert diagnostic["event_count"] == 3
    assert diagnostic["omitted_event_count"] == 0
    assert [result["fixture"] for result in diagnostic["contract_results"]] == ["forward", "vjp"]
    assert all(result["status"] == "mismatch" and result["reason"] for result in diagnostic["contract_results"])
    assert all(set(event["fields"]) == set(EVENT_FIELDS) for event in diagnostic["events"])
    manifest = diagnostic["events"][1]["fields"]["coverage_manifest"]
    assert manifest == {
        "length": len(structural_payload),
        "sha256": hashlib.sha256(structural_payload.encode()).hexdigest(),
    }
    assert structural_payload not in message
    assert len(message.encode()) <= 12_000
    with pytest.raises(AssertionError) as repeated:
        match_fixture_contract(events, IDENTITY)
    assert str(repeated.value) == message


def test_contract_match_preserves_the_single_fixture_oracle():
    evidence = match_fixture_contract(valid_events(VJP_EXPECTATION), IDENTITY)

    assert evidence["fixture"] == "vjp"
    assert evidence["final_fingerprint"] == VJP_EXPECTATION.final_normalized_fingerprint


def test_contract_mismatch_bounds_anomalous_event_counts():
    events = valid_events(VJP_EXPECTATION) * 100

    with pytest.raises(AssertionError) as failure:
        match_fixture_contract(events, IDENTITY)

    diagnostic = json.loads(str(failure.value).partition(": ")[2])
    assert diagnostic["event_count"] == 300
    assert len(diagnostic["events"]) == 4
    assert diagnostic["omitted_event_count"] == 296
    assert len(str(failure.value).encode()) <= 12_000


@pytest.mark.parametrize("expectation", EXPECTATIONS)
def test_acceptance_fingerprint_matches_independent_fixture_audit(expectation):
    assert audited_fingerprint(fixture_path(FIXTURE_DIRECTORY, expectation)) == expectation.final_normalized_fingerprint


@pytest.mark.parametrize("expectation", EXPECTATIONS)
def test_xla_hook_boundary_fingerprint_is_rederived_from_pinned_jaxlib(expectation):
    path = fixture_path(FIXTURE_DIRECTORY, expectation)

    assert derived_hook_boundary_fingerprint(path) == audited_hook_boundary_fingerprint(path)


def test_fixture_audit_tool_rejects_oracle_drift(tmp_path):
    for expectation in EXPECTATIONS:
        source = fixture_path(FIXTURE_DIRECTORY, expectation).read_text()
        fixture_path(tmp_path, expectation).write_text(source)
    vjp_path = fixture_path(tmp_path, VJP_EXPECTATION)
    vjp_path.write_text(vjp_path.read_text().replace("D4DAD86C", "04DAD86C"))

    with pytest.raises(ValueError, match="acceptance fixture fingerprint drift"):
        verify_oracles(tmp_path)


def test_fixture_audit_tool_rejects_preprocessing_drift(tmp_path):
    for expectation in EXPECTATIONS:
        source = fixture_path(FIXTURE_DIRECTORY, expectation).read_text()
        fixture_path(tmp_path, expectation).write_text(source)
    vjp_path = fixture_path(tmp_path, VJP_EXPECTATION)
    vjp_path.write_text(vjp_path.read_text().replace("B73249E4", "073249E4"))

    with pytest.raises(ValueError, match="acceptance fixture fingerprint drift"):
        verify_oracles(tmp_path)


def test_decodes_only_immutable_complete_native_records():
    record = (
        17,
        "final_erasure",
        "source_ordered",
        "policy-digest",
        "tuning-digest",
        "",
        "",
        "",
        "module-fingerprint",
        True,
        "",
    )
    records = (record,)

    assert decode_native_snapshot(records) == (
        {
            "invocation_id": 17,
            "phase": "final_erasure",
            "policy": "source_ordered",
            "policy_digest": "policy-digest",
            "tuning_digest": "tuning-digest",
            "region_membership": "",
            "coverage_manifest": "",
            "unsupported_fingerprint": "",
            "normalized_module_fingerprint": "module-fingerprint",
            "no_shuttle_semantics": True,
            "failure_pass": "",
        },
    )
    with pytest.raises(TypeError):
        operator.setitem(records, 0, record)
    with pytest.raises(TypeError):
        operator.setitem(record, 0, -1)


@pytest.mark.parametrize(
    "records",
    [
        [],
        [tuple(range(len(EVENT_FIELDS)))],
        (list(range(len(EVENT_FIELDS))),),
        (tuple(range(len(EVENT_FIELDS) - 1)),),
        (tuple(range(len(EVENT_FIELDS) + 1)),),
    ],
)
def test_rejects_mutable_or_wrong_width_native_records(records):
    with pytest.raises(AssertionError):
        decode_native_snapshot(records)
