# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import stat

import pytest
from iac.imports import (
    IMPORT_ID_PLACEHOLDER,
    ImportIdentity,
    ImportSpec,
    import_manifest_summary,
    resolve_import_manifest,
    validate_reviewed_manifest,
    write_import_manifest,
)

IAM_COMPONENT_TYPE = "marin:gcp:GcpIam"
IAM_MEMBER_TYPE = "gcp:projects/iAMMember:IAMMember"
IAM_COMPONENT_URN = f"urn:pulumi:marin::marin-iac::{IAM_COMPONENT_TYPE}::iam"
GCP_PROVIDER_URN = "urn:pulumi:marin::marin-iac::pulumi:providers:gcp::gcp::provider-id"
MEMBER_NAME = "project-roles-viewer-serviceaccount-reader-example-com"
MEMBER_IMPORT_ID = "hai-gcp-models roles/viewer serviceAccount:reader@example.com"


def _generated_manifest(*, component_is_new: bool = False) -> dict:
    name_table = {"gcp": GCP_PROVIDER_URN}
    resources = []
    if component_is_new:
        resources.append({"type": IAM_COMPONENT_TYPE, "name": "iam", "component": True})
    else:
        name_table["iam"] = IAM_COMPONENT_URN
    resources.append(
        {
            "type": IAM_MEMBER_TYPE,
            "name": "project_member",
            "logicalName": MEMBER_NAME,
            "id": IMPORT_ID_PLACEHOLDER,
            "parent": "iam",
            "provider": "gcp",
        }
    )
    return {"nameTable": name_table, "resources": resources}


def _member_spec() -> ImportSpec:
    return ImportSpec(
        identity=ImportIdentity(
            resource_type=IAM_MEMBER_TYPE,
            logical_name=MEMBER_NAME,
            parent_type=IAM_COMPONENT_TYPE,
            parent_name="iam",
        ),
        provider_id=MEMBER_IMPORT_ID,
    )


@pytest.mark.parametrize("component_is_new", [False, True])
def test_resolve_import_manifest_preserves_ancestry_and_fills_provider_id(component_is_new):
    generated = _generated_manifest(component_is_new=component_is_new)

    resolved = resolve_import_manifest(generated, [_member_spec()])

    assert resolved["nameTable"] == generated["nameTable"]
    assert resolved["resources"][:-1] == generated["resources"][:-1]
    assert resolved["resources"][-1] == {
        **generated["resources"][-1],
        "id": MEMBER_IMPORT_ID,
    }
    assert generated["resources"][-1]["id"] == IMPORT_ID_PLACEHOLDER


def test_resolve_import_manifest_leaves_uncataloged_create_for_operator_decision():
    generated = _generated_manifest()

    resolved = resolve_import_manifest(generated, [])
    summary = import_manifest_summary(resolved)

    assert resolved["resources"][-1]["id"] == IMPORT_ID_PLACEHOLDER
    assert summary.imports_by_type == {}
    assert summary.unresolved_by_type == {IAM_MEMBER_TYPE: 1}


def test_resolve_import_manifest_preserves_parameterized_provider_dependency():
    generated = _generated_manifest(component_is_new=True)
    generated["resources"].insert(
        1,
        {
            "type": "pulumi:providers:coreweave",
            "name": "coreweave",
            "logicalName": "coreweave-object-storage",
        },
    )

    resolved = resolve_import_manifest(generated, [_member_spec()])
    summary = import_manifest_summary(resolved)

    assert resolved["resources"][1] == generated["resources"][1]
    assert summary.imports_by_type == {IAM_MEMBER_TYPE: 1}
    assert summary.unresolved_by_type == {}


def test_validate_reviewed_manifest_accepts_unchanged_subset():
    current = resolve_import_manifest(_generated_manifest(), [_member_spec()])
    current["resources"].append(
        {
            **current["resources"][0],
            "name": "other_member",
            "logicalName": "project-roles-viewer-serviceaccount-other-example-com",
            "id": "hai-gcp-models roles/viewer serviceAccount:other@example.com",
        }
    )
    reviewed = {
        "nameTable": current["nameTable"],
        "resources": current["resources"][:1],
    }

    summary = validate_reviewed_manifest(reviewed, current)

    assert summary.imports_by_type == {IAM_MEMBER_TYPE: 1}
    assert summary.component_count == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["resources"][-1].update(id="wrong provider id"),
        lambda manifest: manifest["resources"][-1].update(logicalName="stale-name"),
        lambda manifest: manifest["nameTable"].update(gcp="urn:pulumi:stale-provider"),
    ],
)
def test_validate_reviewed_manifest_rejects_stale_or_edited_entries(mutate):
    current = resolve_import_manifest(_generated_manifest(), [_member_spec()])
    reviewed = json.loads(json.dumps(current))
    mutate(reviewed)

    with pytest.raises(ValueError):
        validate_reviewed_manifest(reviewed, current)


def test_validate_reviewed_manifest_rejects_unresolved_placeholder():
    current = resolve_import_manifest(_generated_manifest(), [])

    with pytest.raises(ValueError):
        validate_reviewed_manifest(current, current)


def test_write_import_manifest_uses_owner_only_permissions(tmp_path):
    manifest_path = tmp_path / "import.json"
    manifest_path.write_text("stale")
    manifest_path.chmod(0o644)
    manifest = resolve_import_manifest(_generated_manifest(), [_member_spec()])

    write_import_manifest(manifest_path, manifest)

    assert json.loads(manifest_path.read_text()) == manifest
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
