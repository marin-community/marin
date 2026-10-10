# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from google.cloud import kms_v1
from iac.gcp.iam_config import GcpPrincipal, load_iam_config
from iris.cluster.config import PRINCIPAL_REFERENCE_PREFIX, load_config
from iris.cluster.controller.auth import create_controller_auth
from iris.rpc import job_pb2

from infra.pulumi import iam_principal


@pytest.mark.parametrize("gpu", [False, True])
def test_render_iris_preserves_personal_identity_and_existing_access(tmp_path: Path, monkeypatch, gpu: bool) -> None:
    principal_id = "human-999"
    reference = f"{PRINCIPAL_REFERENCE_PREFIX}{principal_id}"
    email = "collaborator@example.com"
    operator = "operator@example.com"
    encrypted_user = b"encrypted-person"
    iam = load_iam_config()
    iam = replace(
        iam, principals=(*iam.principals, GcpPrincipal(principal_id, base64.b64encode(encrypted_user).decode()))
    )
    monkeypatch.setattr(iam_principal, "load_iam_config", lambda: iam)

    class FakeKms:
        def decrypt(self, *, name: str, ciphertext: bytes):
            assert name.endswith(f"/cryptoKeys/{iam.kms_key}")
            assert ciphertext == encrypted_user
            return kms_v1.DecryptResponse(plaintext=email.encode())

    monkeypatch.setattr(kms_v1, "KeyManagementServiceClient", FakeKms)
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "auth": {
                    "iap": {
                        "signed_header_audience": "/projects/1/global/backendServices/2",
                        "unprovisioned_role": "admin",
                    },
                    "user_roles": {} if gpu else {reference: "user"},
                    "allowed_submitters": [operator, reference] if gpu else [],
                },
                "user_budgets": (
                    [] if gpu else [{"user_ids": [reference], "budget_limit": 0, "max_band": "PRIORITY_BAND_BATCH"}]
                ),
            }
        )
    )
    original = source.read_bytes()
    # Unrendered auth must not silently fall through to the hub's admin default.
    with pytest.raises(ValueError, match="render-iris"):
        create_controller_auth(load_config(source).auth, cluster_name="test")
    output = tmp_path / "rendered.yaml"
    iam_principal.render_iris(source, output)
    config = load_config(output)
    auth = create_controller_auth(config.auth, cluster_name="test")
    assert auth.role_policy is not None
    assert auth.role_policy.role_for(operator) == "admin"
    if gpu:
        assert auth.allowed_submitters == (operator, email)
    else:
        assert auth.role_policy.role_for(email) == "user"
        assert config.user_budgets[0].user_ids == [email]
        assert config.user_budgets[0].budget_limit == 0
        assert config.user_budgets[0].max_band == job_pb2.PRIORITY_BAND_BATCH
    assert source.read_bytes() == original
    assert output.stat().st_mode & 0o777 == 0o600
