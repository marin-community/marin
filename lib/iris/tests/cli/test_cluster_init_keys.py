# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`iris cluster init-keys` against an in-memory Secret Manager.

A cluster's signing key is its identity, so the command reads an existing key back
instead of overwriting it; only --rotate replaces one.
"""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from google.api_core.exceptions import AlreadyExists, FailedPrecondition, NotFound
from iris.cli import cluster as cluster_cli
from rigging.token_authority import generate_ed25519_keypair, signing_key_from_private_pem

RESOURCE = "projects/test-project/secrets/iris-test-signing-key"


class _FakeBindings(list):
    """The subset of a proto repeated-binding field the IAM grant path uses."""

    def add(self, role: str, members: list[str]) -> None:
        self.append(SimpleNamespace(role=role, members=list(members)))


class _FakeSecretManager:
    """In-memory Secret Manager holding one secret's versions and IAM policy.

    Version names are 1-based, matching Secret Manager, so a reference pinned to
    `versions/1` in a test means the first payload ever stored.
    """

    def __init__(self, payloads: list[bytes] | None = None):
        self.payloads = list(payloads or [])
        self.exists = bool(self.payloads)
        self.policy = SimpleNamespace(bindings=_FakeBindings())

    def access_secret_version(self, request):
        assert request["name"] == f"{RESOURCE}/versions/latest"
        if not self.payloads:
            raise NotFound("no such version")
        return SimpleNamespace(
            name=f"{RESOURCE}/versions/{len(self.payloads)}",
            payload=SimpleNamespace(data=self.payloads[-1]),
        )

    def create_secret(self, request):
        if self.exists:
            raise AlreadyExists("secret exists")
        self.exists = True

    def add_secret_version(self, request):
        self.payloads.append(request["payload"]["data"])
        return SimpleNamespace(name=f"{RESOURCE}/versions/{len(self.payloads)}")

    def get_iam_policy(self, request):
        return self.policy

    def set_iam_policy(self, request):
        self.policy = request["policy"]


@pytest.fixture
def existing_key():
    return signing_key_from_private_pem(generate_ed25519_keypair().private_pem)


def _run(args):
    return CliRunner().invoke(cluster_cli.cluster_init_keys, args, obj={}, catch_exceptions=False)


@pytest.fixture(autouse=True)
def _fake_client(monkeypatch):
    """Bind ``_secret_manager_client`` to whatever fake the test installs."""
    holder = {}

    def install(secrets):
        holder["secrets"] = secrets
        return secrets

    monkeypatch.setattr(cluster_cli, "_secret_manager_client", lambda: holder["secrets"])
    return install


def test_reuses_the_key_already_in_secret_manager(_fake_client, existing_key):
    secrets = _fake_client(_FakeSecretManager([existing_key.private_pem.encode()]))

    result = _run(["--gcp-secret", RESOURCE])

    assert result.exit_code == 0, result.output
    assert len(secrets.payloads) == 1, "reading a key back must not store a new version"
    assert existing_key.public_pem.strip() in result.output
    assert existing_key.kid in result.output
    assert f"auth.signing_key: gcp-secret://{RESOURCE}/versions/1" in result.output


def test_rotate_stores_a_new_key_as_the_next_version(_fake_client, existing_key):
    secrets = _fake_client(_FakeSecretManager([existing_key.private_pem.encode()]))

    result = _run(["--gcp-secret", RESOURCE, "--rotate"])

    assert result.exit_code == 0, result.output
    assert len(secrets.payloads) == 2
    rotated = signing_key_from_private_pem(secrets.payloads[-1].decode())
    assert rotated.public_pem != existing_key.public_pem
    assert rotated.public_pem.strip() in result.output
    assert f"auth.signing_key: gcp-secret://{RESOURCE}/versions/2" in result.output


def test_mints_and_stores_a_key_when_the_secret_is_empty(_fake_client):
    secrets = _fake_client(_FakeSecretManager())

    result = _run(["--gcp-secret", RESOURCE])

    assert result.exit_code == 0, result.output
    assert len(secrets.payloads) == 1
    minted = signing_key_from_private_pem(secrets.payloads[0].decode())
    assert minted.public_pem.strip() in result.output
    assert f"auth.signing_key: gcp-secret://{RESOURCE}/versions/1" in result.output


def test_a_secret_holding_something_other_than_a_signing_key_is_an_error(_fake_client):
    secrets = _fake_client(_FakeSecretManager([b"not a pem"]))

    result = _run(["--gcp-secret", RESOURCE])

    assert result.exit_code != 0
    assert "does not hold an Ed25519 private key" in result.output
    assert len(secrets.payloads) == 1, "a bad payload must not be papered over with a new version"


def test_a_disabled_latest_version_is_an_error_not_a_silent_rotation(_fake_client, existing_key):
    secrets = _fake_client(_FakeSecretManager([existing_key.private_pem.encode()]))

    def disabled(request):
        raise FailedPrecondition("version is disabled")

    secrets.access_secret_version = disabled

    result = _run(["--gcp-secret", RESOURCE])

    assert result.exit_code != 0
    assert "disabled or destroyed" in result.output
    assert len(secrets.payloads) == 1, "an unreadable key must not be replaced with a fresh identity"


def test_reuse_still_grants_the_accessor(_fake_client, existing_key):
    secrets = _fake_client(_FakeSecretManager([existing_key.private_pem.encode()]))

    result = _run(["--gcp-secret", RESOURCE, "--accessor", "iris-controller@test.iam.gserviceaccount.com"])

    assert result.exit_code == 0, result.output
    assert len(secrets.payloads) == 1
    (binding,) = secrets.policy.bindings
    assert binding.role == "roles/secretmanager.secretAccessor"
    assert binding.members == ["serviceAccount:iris-controller@test.iam.gserviceaccount.com"]


def test_rotate_without_a_secret_is_a_usage_error(_fake_client):
    secrets = _fake_client(_FakeSecretManager())
    runner = CliRunner()

    result = runner.invoke(cluster_cli.cluster_init_keys, ["--rotate"], obj={})

    assert result.exit_code != 0
    assert "--rotate only applies with --gcp-secret" in result.output
    assert secrets.payloads == []
