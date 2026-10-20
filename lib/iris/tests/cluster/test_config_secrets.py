# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SecretRefSpec config fields: runtime resolution and the #6873 render guard."""

import sys
from types import SimpleNamespace

import pytest
import yaml
from iris.cluster.config import (
    AuthConfig,
    IapAuthConfig,
    IrisClusterConfig,
    assert_no_inlined_secrets,
    resolve_config_secrets,
    resolve_principal_references,
)
from iris.cluster.platforms.gcp.controller_bootstrap import build_controller_bootstrap_script_from_config
from rigging.secrets import SecretResolutionError

_INLINE_PEM = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"


def _config(auth: AuthConfig) -> IrisClusterConfig:
    return IrisClusterConfig(name="c", auth=auth)


def test_resolve_config_secrets_resolves_signing_key_reference(monkeypatch):
    monkeypatch.setenv("IRIS_SIGNING_KEY", "resolved-pem")
    config = _config(AuthConfig(signing_key="env:IRIS_SIGNING_KEY"))

    resolved = resolve_config_secrets(config)

    assert resolved.auth.signing_key == "resolved-pem"
    # The original config is untouched (a copy is returned).
    assert config.auth.signing_key == "env:IRIS_SIGNING_KEY"


def test_resolve_config_secrets_leaves_unset_signing_key_empty():
    config = _config(AuthConfig())
    resolved = resolve_config_secrets(config)
    assert resolved.auth.signing_key == ""


def test_resolve_config_secrets_resolves_nested_iap_secret(monkeypatch):
    monkeypatch.setenv("IRIS_OAUTH", "sekret")
    config = _config(AuthConfig(iap=IapAuthConfig(oauth_client_secret="env:IRIS_OAUTH")))

    resolved = resolve_config_secrets(config)

    assert resolved.auth.iap.oauth_client_secret == "sekret"


def test_resolve_config_secrets_raises_when_source_absent(monkeypatch):
    monkeypatch.delenv("IRIS_SIGNING_KEY", raising=False)
    config = _config(AuthConfig(signing_key="env:IRIS_SIGNING_KEY"))
    with pytest.raises(SecretResolutionError):  # all sources absent
        resolve_config_secrets(config)


def test_assert_no_inlined_secrets_rejects_raw_signing_key():
    config = _config(AuthConfig(signing_key=_INLINE_PEM))
    with pytest.raises(ValueError, match="inlined secret"):
        assert_no_inlined_secrets(config)


def test_assert_no_inlined_secrets_passes_reference():
    config = _config(AuthConfig(signing_key="gcp-secret://projects/p/secrets/s/versions/1"))
    assert_no_inlined_secrets(config)  # a reference does not raise


def test_assert_no_inlined_secrets_passes_when_unset():
    config = _config(AuthConfig())
    assert_no_inlined_secrets(config)  # an unset field does not raise


def test_gcp_bootstrap_keeps_identity_references_after_runtime_resolution(monkeypatch):
    payload = '{"human-999":"private@example.com"}'
    resource = "projects/test/secrets/principal-directory/versions/1"

    class SecretManager:
        def access_secret_version(self, *, name):
            assert name == resource
            return SimpleNamespace(payload=SimpleNamespace(data=payload.encode()))

    secretmanager = SimpleNamespace(SecretManagerServiceClient=SecretManager)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", secretmanager)
    monkeypatch.setattr("google.cloud.secretmanager", secretmanager, raising=False)
    config = _config(
        AuthConfig(principal_directory=f"gcp-secret://{resource}", user_roles={"principal:human-999": "user"})
    )
    runtime = resolve_principal_references(resolve_config_secrets(config))

    bootstrap = build_controller_bootstrap_script_from_config(config, lambda image, _zone: image)

    assert "private@example.com" not in bootstrap
    config_yaml = bootstrap.split("<< 'IRIS_CONFIG_EOF'\n", 1)[1].split("\nIRIS_CONFIG_EOF", 1)[0]
    deployed_auth = yaml.safe_load(config_yaml)["auth"]
    assert deployed_auth["principal_directory"] == f"gcp-secret://{resource}"
    assert deployed_auth["user_roles"] == {"principal:human-999": "user"}
    # Accidentally deploying the resolved copy must fail without echoing the payload.
    with pytest.raises(ValueError) as exc:
        build_controller_bootstrap_script_from_config(runtime, lambda image, _zone: image)
    assert "private@example.com" not in str(exc.value)
