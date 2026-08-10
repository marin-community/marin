# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversarial tests for the pure ordinary-JAX observer contract."""

from copy import deepcopy
from pathlib import Path

import pytest
from acceptance_contract import (
    FORWARD_EXPECTATION,
    VJP_EXPECTATION,
    ObserverIdentity,
    validate_success_events,
)
from verify_acceptance_fixture_oracles import EXPECTATIONS, audited_fingerprint, fixture_path, verify_oracles

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
    assert VJP_EXPECTATION.unsupported_fingerprint == "ef0a137534d479ad4a98caf0250938782bb874c4f0f13e2c3c5e8930667c7d05"


def test_accepts_only_the_complete_fixture_contract():
    evidence = validate_success_events(valid_events(), IDENTITY, VJP_EXPECTATION)

    assert evidence["fixture"] == "vjp"
    assert evidence["final_fingerprint"] == VJP_EXPECTATION.final_normalized_fingerprint


@pytest.mark.parametrize("expectation", EXPECTATIONS)
def test_acceptance_fingerprint_matches_independent_fixture_audit(expectation):
    assert audited_fingerprint(fixture_path(FIXTURE_DIRECTORY, expectation)) == expectation.final_normalized_fingerprint


def test_fixture_audit_tool_rejects_oracle_drift(tmp_path):
    for expectation in EXPECTATIONS:
        source = fixture_path(FIXTURE_DIRECTORY, expectation).read_text()
        fixture_path(tmp_path, expectation).write_text(source)
    vjp_path = fixture_path(tmp_path, VJP_EXPECTATION)
    vjp_path.write_text(vjp_path.read_text().replace("2D557BD5", "0D557BD5"))

    with pytest.raises(ValueError, match="acceptance fixture fingerprint drift"):
        verify_oracles(tmp_path)
