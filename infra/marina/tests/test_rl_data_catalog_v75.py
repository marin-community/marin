# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The retained v75 renderer treats every other measured state as Unsafe."""

from infra.marina.apps.rl_data_catalog.server.daytona import daytona


def test_v75_api_marks_zero_actor_and_grandfathered_sources_without_false_unsafe():
    no_actor = {
        "id": "tool-conversation",
        "tags": ["agentic"],
        "daytona_safety": "not_applicable",
        "daytona_machine_slots": 0,
        "daytona_snapshot_count": 0,
        "daytona_execution_failures": 0,
        "daytona_count_basis": "Complete TaskSpec census",
    }
    grandfathered = {
        "id": "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        "tags": ["agentic"],
        "daytona_safety": "unassessed",
        "daytona_snapshot_count": None,
    }
    # v75 recognizes 'na' and renders N/A; it has no dedicated dash state.
    assert daytona(no_actor, 10)["state"] == "na"
    exempt = daytona(grandfathered, 10)
    assert exempt["state"] == "na" and exempt["exempt"]
    locked = {
        "id": "actor-lock",
        "tags": ["agentic"],
        "daytona_safety": "unsafe",
        "daytona_machine_slots": 100,
        "daytona_snapshot_count": 0,
        "daytona_unsafe_reasons": ["host_image_required"],
        "daytona_unsafe_details": {"host_image_required": "Factory requires a registry image"},
        "daytona_unsafe_counts": {"host_image_required": 100},
    }
    refusal = daytona(locked, 10)
    assert refusal["state"] == "unsafe" and refusal["snapshots"] == 0
    assert refusal["unsafe_reasons"] == ["host_image_required"]
    assert refusal["unsafe_counts"] == {"host_image_required": 100}


def test_v75_api_requires_census_evidence_and_exact_frozen_exemption():
    for declaration in (
        {"id": "conversation", "tags": []},
        {"id": "missing-proof", "daytona_safety": "not_applicable", "daytona_machine_slots": 0},
        {"id": "MarinSkyRL:new-agentic-blend", "tags": ["agentic"]},
    ):
        assert daytona(declaration, 10)["state"] == "unmeasured"
