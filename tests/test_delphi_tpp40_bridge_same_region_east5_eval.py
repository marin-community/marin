# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_same_region_east5_eval as same_region


def _frozen_manifest() -> dict:
    trees = []
    for index, relative_path in enumerate(("checkpoints/step-21855", "checkpoints/step-27335", "hf/step-27335")):
        payload_identity = {
            "payload_sha256": str(index) * 64,
            "objects": (4, 7, 5)[index],
            "bytes": (4_299_698_408, 4_299_706_606, 1_450_461_380)[index],
        }
        trees.append(
            {
                "relative_path": relative_path,
                "source_path": f"gs://source/{relative_path}",
                "destination_path": f"{same_region.MIRROR_ROOT}/{relative_path}",
                "source_identity": payload_identity,
                "destination_identity": payload_identity,
            }
        )
    return {
        "europe_mirror_root": same_region.MIRROR_ROOT,
        "storage_transfer_service_used": False,
        "trees": trees,
    }


def _install_frozen_inputs(monkeypatch: pytest.MonkeyPatch) -> dict:
    manifest = _frozen_manifest()
    contract = {
        "bridge": {
            "east5_reference_mirror": {
                "manifest_sha256": same_region.EXPECTED_MIRROR_MANIFEST_SHA256,
            }
        }
    }
    monkeypatch.setattr(
        same_region,
        "_load_frozen_json",
        lambda path, **_: contract if path == same_region.CONTRACT_PATH else manifest,
    )
    return manifest


def test_mirror_audit_rejects_extra_object_outside_frozen_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _install_frozen_inputs(monkeypatch)
    identities = {tree["source_path"]: tree["source_identity"] for tree in manifest["trees"]} | {
        tree["destination_path"]: tree["destination_identity"] for tree in manifest["trees"]
    }
    identities[same_region.MIRROR_ROOT] = {
        "payload_sha256": "f" * 64,
        "objects": same_region.EXPECTED_MIRROR_OBJECTS + 1,
        "bytes": same_region.EXPECTED_MIRROR_BYTES + 1,
    }
    monkeypatch.setattr(same_region, "tree_payload_identity", identities.__getitem__)

    with pytest.raises(ValueError, match="outside the three frozen trees"):
        same_region.audit_east5_reference_mirror()


def test_mirror_audit_rejects_destination_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _install_frozen_inputs(monkeypatch)
    identities = {tree["source_path"]: tree["source_identity"] for tree in manifest["trees"]} | {
        tree["destination_path"]: tree["destination_identity"] for tree in manifest["trees"]
    }
    identities[manifest["trees"][0]["destination_path"]] = {
        **manifest["trees"][0]["destination_identity"],
        "payload_sha256": "f" * 64,
    }
    identities[same_region.MIRROR_ROOT] = {
        "payload_sha256": "e" * 64,
        "objects": same_region.EXPECTED_MIRROR_OBJECTS,
        "bytes": same_region.EXPECTED_MIRROR_BYTES,
    }
    monkeypatch.setattr(same_region, "tree_payload_identity", identities.__getitem__)

    with pytest.raises(ValueError, match="Europe mirror tree changed"):
        same_region.audit_east5_reference_mirror()
