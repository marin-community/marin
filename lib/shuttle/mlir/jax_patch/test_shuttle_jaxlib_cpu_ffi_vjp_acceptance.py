# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contracts for the unbuilt ABI 9 identity-policy Host proof."""

import hashlib
import json

import numpy as np
import pytest
from acceptance_contract import ObserverIdentity
from shuttle_jaxlib_cpu_ffi_vjp_acceptance import (
    BOUNDARIES,
    CPU_BUNDLE_FINAL_FINGERPRINTS,
    PIPELINE_ABI_VERSION,
    POLICIES,
    SHAPE,
    arrays,
    boundary_function,
    cell_identities,
    fixed_inputs,
    load_baseline,
    ready,
    save_baseline,
    subject_options,
    validate_cpu_bundle_success_events,
)
from target1_acceptance_contract import target1_expectation

from shuttle import Numerics


def _cpu_bundle_events() -> list[dict[str, object]]:
    identity = _historical_identity(Numerics.SOURCE_ORDERED)
    fixture = target1_expectation(SHAPE.shape_id, "forward")
    common = {
        "invocation_id": 17,
        "policy": identity.policy,
        "policy_digest": identity.policy_digest,
        "tuning_digest": identity.tuning_digest,
        "failure_pass": "",
    }
    return [
        {
            **common,
            "phase": "algebra_coverage",
            "region_membership": fixture.region_membership,
            "coverage_manifest": fixture.coverage_manifest(identity),
            "unsupported_fingerprint": fixture.unsupported_fingerprint,
            "normalized_module_fingerprint": "",
            "no_shuttle_semantics": False,
        },
        {
            **common,
            "phase": "final_erasure",
            "region_membership": "",
            "coverage_manifest": "",
            "unsupported_fingerprint": "",
            "normalized_module_fingerprint": CPU_BUNDLE_FINAL_FINGERPRINTS[
                (SHAPE.shape_id, "forward", "source_ordered")
            ],
            "no_shuttle_semantics": True,
        },
    ]


def _historical_identity(numerics: Numerics) -> ObserverIdentity:
    payload = {
        "execution_mode": "cpu_executable_bundle",
        "numerics": numerics.value,
        "pipeline_abi_version": 9,
        "schema_version": 1,
        "tuning": {
            "cluster_shape": [],
            "materialization": "automatic",
            "maximum_candidates": 1,
            "pipeline_stages": 1,
            "tile_sizes": [],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    canonical_tuning = json.dumps(payload["tuning"], sort_keys=True, separators=(",", ":"))
    return ObserverIdentity(
        policy=numerics.value,
        policy_digest=hashlib.sha256(canonical.encode()).hexdigest(),
        tuning_digest=hashlib.sha256(canonical_tuning.encode()).hexdigest(),
        canonical_options=canonical,
        canonical_tuning=canonical_tuning,
    )


def test_identity_policy_host_driver_rejects_current_abi10_options() -> None:
    assert PIPELINE_ABI_VERSION == 9
    assert BOUNDARIES == ("forward", "backward", "composed")
    assert POLICIES == (Numerics.SOURCE_ORDERED, Numerics.FAST)
    assert cell_identities() == (
        ("forward", "source_ordered"),
        ("backward", "source_ordered"),
        ("composed", "source_ordered"),
        ("forward", "fast"),
        ("backward", "fast"),
        ("composed", "fast"),
    )
    for numerics in POLICIES:
        with pytest.raises(AssertionError, match="requires pipeline ABI 9"):
            subject_options(numerics)
    assert _historical_identity(Numerics.SOURCE_ORDERED) != _historical_identity(Numerics.FAST)


def test_identity_policy_host_driver_preserves_public_jax_result_order() -> None:
    forward = arrays(ready(boundary_function("forward")(*fixed_inputs(SHAPE, "forward"))))
    backward = arrays(ready(boundary_function("backward")(*fixed_inputs(SHAPE, "backward"))))
    composed = arrays(ready(boundary_function("composed")(*fixed_inputs(SHAPE, "composed"))))
    assert [value.shape for value in forward] == [(7, 13)]
    assert [value.shape for value in backward] == [(7, 13), (13,)]
    assert [value.shape for value in composed] == [(7, 13), (7, 13), (13,)]
    assert composed[0].tobytes() == forward[0].tobytes()
    assert composed[1].tobytes() == backward[0].tobytes()
    assert composed[2].tobytes() == backward[1].tobytes()


def test_identity_policy_host_baseline_roundtrips_closed_bf16_bits(tmp_path) -> None:
    path = tmp_path / "baseline.npz"
    save_baseline(path)
    forward = load_baseline(path, "forward")
    backward = load_baseline(path, "backward")
    composed = load_baseline(path, "composed")
    assert [(value.dtype, value.shape) for value in forward] == [
        (np.dtype(np.uint16), (7, 13)),
    ]
    assert [(value.dtype, value.shape) for value in backward] == [
        (np.dtype(np.uint16), (7, 13)),
        (np.dtype(np.uint16), (13,)),
    ]
    assert [(value.dtype, value.shape) for value in composed] == [
        (np.dtype(np.uint16), (7, 13)),
        (np.dtype(np.uint16), (7, 13)),
        (np.dtype(np.uint16), (13,)),
    ]

    stored = {key: value.copy() for key, value in np.load(path).items()}
    stored["backward_0"] = stored["backward_0"].reshape(13, 7)
    np.savez(path, **stored)
    with pytest.raises(AssertionError, match="bit payload 0 changed"):
        load_baseline(path, "backward")


def test_cpu_bundle_observer_accepts_exact_two_phase_contract() -> None:
    identity = _historical_identity(Numerics.SOURCE_ORDERED)
    fixture = target1_expectation(SHAPE.shape_id, "forward")
    report = validate_cpu_bundle_success_events(_cpu_bundle_events(), identity, fixture)
    assert report["invocation_id"] == 17
    assert report["complete_source_results"] == len(fixture.complete)
    assert report["excluded_source_results"] == 0
    assert report["final_fingerprint"] == CPU_BUNDLE_FINAL_FINGERPRINTS[(SHAPE.shape_id, "forward", "source_ordered")]


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("missing", "exactly two observer phases"),
        ("extra", "exactly two observer phases"),
        ("reordered", "two ordered observer phases"),
        ("wrong_policy", "policy identity differs"),
    ],
)
def test_cpu_bundle_observer_rejects_phase_contract_drift(mutation: str, diagnostic: str) -> None:
    events = _cpu_bundle_events()
    if mutation == "missing":
        events.pop()
    elif mutation == "extra":
        events.insert(1, dict(events[0]))
    elif mutation == "reordered":
        events.reverse()
    elif mutation == "wrong_policy":
        events[0]["policy"] = "fast"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    identity = _historical_identity(Numerics.SOURCE_ORDERED)
    fixture = target1_expectation(SHAPE.shape_id, "forward")
    with pytest.raises(AssertionError, match=diagnostic):
        validate_cpu_bundle_success_events(events, identity, fixture)
