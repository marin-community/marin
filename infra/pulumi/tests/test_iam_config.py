# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from iac.gcp.iam import GcpEncryptedMember, GcpRoleGrant
from iac.gcp.iam_config import (
    IAM_DATA_PATH,
    GcpIamConfig,
    GcpPrincipal,
    grant_project_roles,
    load_iam_config,
    replace_principals,
    write_iam_config,
)


def _config(*, principals: tuple[GcpPrincipal, ...] = ()) -> GcpIamConfig:
    return GcpIamConfig(
        kms_location="us-central1",
        kms_key_ring="test-key-ring",
        kms_key="test-key",
        principals=principals,
        custom_roles=(),
        owned_service_accounts=(),
        project_grants=(
            GcpRoleGrant(
                role="roles/logging.viewer",
                members=("serviceAccount:logs@example.iam.gserviceaccount.com",),
            ),
        ),
        kms_grants=(),
        secrets=(),
        buckets=(),
        artifact_repositories=(),
        service_accounts=(),
    )


def test_grant_project_roles_reuses_principal_across_roles(tmp_path: Path) -> None:
    config = _config(principals=(GcpPrincipal(principal_id="human-001", ciphertext="existing-ciphertext"),))

    updated = grant_project_roles(
        config,
        "person@example.com",
        ("roles/logging.viewer", "roles/monitoring.viewer"),
        decrypt_ciphertext=lambda ciphertext: {"existing-ciphertext": "user:person@example.com"}[ciphertext],
        encrypt_email=lambda _: pytest.fail("existing principal must not be re-encrypted"),
    )
    output_path = tmp_path / "iam_data.yaml"
    write_iam_config(updated, output_path)
    loaded = load_iam_config(output_path)

    grants = {grant.role: grant for grant in loaded.project_grants}
    encrypted_member = GcpEncryptedMember("existing-ciphertext")
    assert loaded.principals == config.principals
    assert grants["roles/logging.viewer"].members[-1] == encrypted_member
    assert grants["roles/monitoring.viewer"].members == (encrypted_member,)


def test_grant_project_roles_encrypts_new_principal_once(tmp_path: Path) -> None:
    encrypted_emails = []

    def encrypt_email(email: str) -> str:
        encrypted_emails.append(email)
        return "new-ciphertext"

    updated = grant_project_roles(
        _config(),
        "new-person@example.com",
        ("roles/logging.viewer", "roles/monitoring.viewer"),
        decrypt_ciphertext=lambda _: pytest.fail("an empty registry must not decrypt"),
        encrypt_email=encrypt_email,
    )
    output_path = tmp_path / "iam_data.yaml"
    write_iam_config(updated, output_path)
    loaded = load_iam_config(output_path)

    grants = {grant.role: grant for grant in loaded.project_grants}
    encrypted_member = GcpEncryptedMember("new-ciphertext")
    assert encrypted_emails == ["new-person@example.com"]
    assert loaded.principals == (GcpPrincipal(principal_id="human-001", ciphertext="new-ciphertext"),)
    assert grants["roles/logging.viewer"].members[-1] == encrypted_member
    assert grants["roles/monitoring.viewer"].members == (encrypted_member,)


def test_replace_principals_preserves_grant_references(tmp_path: Path) -> None:
    config = grant_project_roles(
        _config(),
        "person@example.com",
        ("roles/logging.viewer",),
        decrypt_ciphertext=lambda _: pytest.fail("an empty registry must not decrypt"),
        encrypt_email=lambda _: "old-ciphertext",
    )
    rotated = replace_principals(
        config,
        (GcpPrincipal(principal_id="human-001", ciphertext="new-ciphertext"),),
    )
    output_path = tmp_path / "iam_data.yaml"
    write_iam_config(rotated, output_path)
    loaded = load_iam_config(output_path)

    assert loaded.principals == rotated.principals
    assert loaded.project_grants[0].members[-1] == GcpEncryptedMember("new-ciphertext")


def test_load_iam_config_rejects_dangling_principal_reference(tmp_path: Path) -> None:
    path = tmp_path / "iam_data.yaml"
    write_iam_config(_config(), path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["project_grants"][0]["members"] = [{"principal": "human-999"}]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown principal 'human-999'"):
        load_iam_config(path)


@pytest.mark.parametrize(
    ("member", "message"),
    [
        ("user:person@example.com", "plaintext user principal"),
        ("person@example.com", "invalid plain IAM member"),
    ],
)
def test_load_iam_config_rejects_plaintext_human_members(tmp_path: Path, member: str, message: str) -> None:
    path = tmp_path / "iam_data.yaml"
    write_iam_config(_config(), path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["project_grants"][0]["members"] = [member]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_iam_config(path)


def test_load_iam_config_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "iam_data.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate YAML mapping key 'schema_version'"):
        load_iam_config(path)


def test_checked_in_iam_yaml_is_canonical(tmp_path: Path) -> None:
    output_path = tmp_path / "iam_data.yaml"
    write_iam_config(load_iam_config(), output_path)

    assert output_path.read_text(encoding="utf-8") == IAM_DATA_PATH.read_text(encoding="utf-8")
