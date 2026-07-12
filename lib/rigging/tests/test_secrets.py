# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for rigging.secrets: ordered resolution + absent-vs-failed."""

import sys
import types

import pytest
from rigging.secrets import (
    ResolvedSecret,
    SecretResolutionError,
    as_secret_spec,
    default_secret_spec,
    is_secret_reference,
    resolve_secret_spec,
)


def test_env_reference_is_resolved_and_trimmed(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "  s3cr3t\n")
    resolved = resolve_secret_spec("env:MY_TOKEN")
    assert resolved == ResolvedSecret(value="s3cr3t", source="env:MY_TOKEN")


def test_unset_env_is_absent_not_error(monkeypatch):
    monkeypatch.delenv("MISSING", raising=False)
    with pytest.raises(SecretResolutionError, match="no secret source produced a value"):
        resolve_secret_spec("env:MISSING")


def test_file_reference_is_resolved_and_trimmed(tmp_path):
    secret_file = tmp_path / "delegation_key"
    secret_file.write_text("abcdef0123456789\n")
    resolved = resolve_secret_spec(f"file:{secret_file}")
    assert resolved.value == "abcdef0123456789"
    assert resolved.source == f"file:{secret_file}"


def test_missing_file_is_absent(tmp_path):
    with pytest.raises(SecretResolutionError, match="no secret source produced a value"):
        resolve_secret_spec(f"file:{tmp_path / 'nope'}")


def test_unreadable_file_source_fails_hard_not_skipped(tmp_path):
    # A directory at the path is a configured-but-erroring source (IsADirectoryError,
    # an OSError) — it must raise, never fall through to a weaker source.
    a_directory = tmp_path / "dir"
    a_directory.mkdir()
    fallback = tmp_path / "fallback"
    fallback.write_text("weaker")
    with pytest.raises(SecretResolutionError):
        resolve_secret_spec((f"file:{a_directory}", f"file:{fallback}"))


def test_ordered_path_first_present_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("PRIMARY", raising=False)
    secret_file = tmp_path / "secret"
    secret_file.write_text("from-file")
    resolved = resolve_secret_spec(("env:PRIMARY", f"file:{secret_file}"))
    assert resolved.value == "from-file"
    assert resolved.source == f"file:{secret_file}"


def test_earlier_source_shadows_later(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIMARY", "from-env")
    secret_file = tmp_path / "secret"
    secret_file.write_text("from-file")
    resolved = resolve_secret_spec(("env:PRIMARY", f"file:{secret_file}"))
    assert resolved.value == "from-env"


def test_unknown_scheme_raises(monkeypatch):
    with pytest.raises(SecretResolutionError, match="unknown secret-source scheme"):
        resolve_secret_spec("vault:secret/path")


def test_bare_literal_is_rejected():
    with pytest.raises(SecretResolutionError, match="not a secret reference"):
        resolve_secret_spec("hunter2")


def test_empty_spec_raises():
    with pytest.raises(SecretResolutionError, match="empty secret spec"):
        resolve_secret_spec(())


def test_empty_env_name_raises():
    with pytest.raises(SecretResolutionError, match="empty variable name"):
        resolve_secret_spec("env:")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("env:X", True),
        ("file:/etc/x", True),
        ("gcp-secret://projects/p/secrets/s/versions/1", True),
        ("hunter2", False),
        ("vault:x", False),
    ],
)
def test_is_secret_reference(value, expected):
    assert is_secret_reference(value) is expected


def test_as_secret_spec_normalizes_bare_and_list():
    assert as_secret_spec("env:X") == ("env:X",)
    assert as_secret_spec(["env:X", "file:/y"]) == ("env:X", "file:/y")


def test_default_secret_spec_shape():
    spec = default_secret_spec("delegation_key", env_prefix="IRIS", secrets_dir="/etc/iris/secrets/")
    assert spec == ("env:IRIS_DELEGATION_KEY", "file:/etc/iris/secrets/delegation_key")


# --- gcp-secret:// path, exercised against a fake Secret Manager module ---


def test_gcp_secret_requires_explicit_version():
    with pytest.raises(SecretResolutionError, match="explicit version"):
        resolve_secret_spec("gcp-secret://projects/p/secrets/s")


def _install_fake_secretmanager(monkeypatch, *, result):
    """Inject a fake google.cloud.secretmanager + google.api_core.exceptions.

    `result` is ("ok", b"bytes") | ("notfound",) | ("error",).
    """

    class FakeNotFound(Exception):
        pass

    class FakeApiError(Exception):
        pass

    exceptions_mod = types.ModuleType("google.api_core.exceptions")
    exceptions_mod.NotFound = FakeNotFound
    exceptions_mod.GoogleAPICallError = FakeApiError

    class FakeClient:
        def access_secret_version(self, *, name):
            assert name == "projects/p/secrets/s/versions/3"
            if result[0] == "ok":
                payload = types.SimpleNamespace(data=result[1])
                return types.SimpleNamespace(payload=payload)
            if result[0] == "notfound":
                raise FakeNotFound()
            raise FakeApiError("permission denied")

    sm_mod = types.ModuleType("google.cloud.secretmanager")
    sm_mod.SecretManagerServiceClient = FakeClient

    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", sm_mod)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", exceptions_mod)


def test_gcp_secret_success(monkeypatch):
    _install_fake_secretmanager(monkeypatch, result=("ok", b"signing-key-material\n"))
    resolved = resolve_secret_spec("gcp-secret://projects/p/secrets/s/versions/3")
    assert resolved.value == "signing-key-material"


def test_gcp_secret_not_found_is_absent(monkeypatch, tmp_path):
    _install_fake_secretmanager(monkeypatch, result=("notfound",))
    fallback = tmp_path / "fallback"
    fallback.write_text("from-file")
    resolved = resolve_secret_spec(("gcp-secret://projects/p/secrets/s/versions/3", f"file:{fallback}"))
    assert resolved.value == "from-file"


def test_gcp_secret_api_error_fails_hard(monkeypatch, tmp_path):
    _install_fake_secretmanager(monkeypatch, result=("error",))
    fallback = tmp_path / "fallback"
    fallback.write_text("from-file")
    with pytest.raises(SecretResolutionError, match="permission denied"):
        resolve_secret_spec(("gcp-secret://projects/p/secrets/s/versions/3", f"file:{fallback}"))
