# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import stat

import pytest

from infra.auth import staging
from infra.auth.bootstrap import OperatorClient
from infra.auth.login import private_directory
from infra.auth.staging import prepare, private_file, rotate_broker


def test_prepare_preserves_deployment_credentials_and_limits_file_access(tmp_path):
    directory = private_directory(tmp_path / "credentials")
    prepare(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert len({original[f"{service}.key"] for service in ("openbao", "openfga", "broker")}) == 3
    prepare(directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir())


def test_broker_rotation_resumes_after_replica_failure_before_revoking_old_token(tmp_path, monkeypatch):
    directory = private_directory(tmp_path / "credentials")
    private_file(directory / "initialization.json", b'{"root_token":"operator"}')
    private_file(directory / "broker.token", b"previous")
    events = []
    unavailable = True

    def request(self, method, path, payload=None):
        events.append(path)
        if path == "auth/token/create":
            assert payload["policies"] == ["marin-auth-broker"]
            assert payload["no_default_policy"] is True
            return {"auth": {"client_token": "replacement"}}
        if path == "auth/token/lookup":
            assert payload == {"token": "replacement"}
            return {"data": {"policies": ["marin-auth-broker"], "ttl": 3600, "expire_time": "later"}}
        assert path == "auth/token/revoke" and payload == {"token": "previous"}
        assert events[-3:-1] == ["projected:first", "projected:second"]
        return {}

    def kubectl(arguments, **kwargs):
        if "get" in arguments:
            return json.dumps({"items": [{"metadata": {"name": name}} for name in ("first", "second")]}).encode()
        name = arguments[3]
        if name == "second" and unavailable:
            raise OSError("replica unavailable")
        events.append(f"projected:{name}")
        return hashlib.sha256(b"replacement").hexdigest().encode()

    def install_secret(namespace, name, data):
        assert namespace == staging.NAMESPACE and name == "broker-token"
        assert data == {"token": "replacement"}

    # OperatorClient's TLS setup is real; use the fixture CA without opening a connection.
    prepare(directory)
    monkeypatch.setattr(OperatorClient, "request", request)
    monkeypatch.setattr(staging, "kubectl", kubectl)
    monkeypatch.setattr(staging, "install_secret", install_secret)
    with pytest.raises(OSError, match="replica unavailable"):
        rotate_broker(directory, "https://localhost:18200")
    assert "auth/token/revoke" not in events
    assert (directory / "broker.token").read_bytes() == b"previous"
    assert stat.S_IMODE((directory / "broker-next.token").stat().st_mode) == 0o600

    unavailable = False
    result = rotate_broker(directory, "https://localhost:18200")
    assert result["replicas_refreshed"] is True
    assert events.count("auth/token/create") == 1
    assert (directory / "broker.token").read_bytes() == b"replacement"
    assert not (directory / "broker-next.token").exists()
