# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import builtins
import sys
import types

import pytest
from marin.execution.identity import resolve_marin_user


@pytest.fixture(autouse=True)
def clear_marin_user(monkeypatch):
    """Ensure MARIN_USER does not leak in from the ambient environment."""
    monkeypatch.delenv("MARIN_USER", raising=False)


def _force_getpass(monkeypatch, name: str) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: name)


def _make_iris_unimportable(monkeypatch) -> None:
    """Make `import iris.cluster.client.job_info` raise ImportError."""
    monkeypatch.delitem(sys.modules, "iris.cluster.client.job_info", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "iris.cluster.client.job_info" or name.startswith("iris."):
            raise ImportError(f"no iris: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _make_iris_return(monkeypatch, name: str) -> None:
    """Stub iris resolve_job_user() to return `name`."""
    module = types.ModuleType("iris.cluster.client.job_info")
    module.resolve_job_user = lambda explicit_user=None: name  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "iris.cluster.client.job_info", module)


def test_override_takes_precedence_over_env_and_getpass(monkeypatch):
    monkeypatch.setenv("MARIN_USER", "from_env")
    _force_getpass(monkeypatch, "from_getpass")
    assert resolve_marin_user("alice") == "alice"


def test_override_is_stripped(monkeypatch):
    _force_getpass(monkeypatch, "from_getpass")
    assert resolve_marin_user("  bob  ") == "bob"


def test_blank_override_falls_through_to_env(monkeypatch):
    monkeypatch.setenv("MARIN_USER", "carol")
    _force_getpass(monkeypatch, "from_getpass")
    assert resolve_marin_user("   ") == "carol"


def test_env_takes_precedence_over_iris_and_getpass(monkeypatch):
    monkeypatch.setenv("MARIN_USER", "dave")
    _make_iris_return(monkeypatch, "from_iris")
    _force_getpass(monkeypatch, "from_getpass")
    assert resolve_marin_user() == "dave"


def test_iris_used_when_importable_and_no_override_or_env(monkeypatch):
    _make_iris_return(monkeypatch, "erin")
    _force_getpass(monkeypatch, "from_getpass")
    assert resolve_marin_user() == "erin"


def test_getpass_fallback_when_iris_import_fails(monkeypatch):
    _make_iris_unimportable(monkeypatch)
    _force_getpass(monkeypatch, "power")
    assert resolve_marin_user() == "power"


def test_accepts_normal_names(monkeypatch):
    _make_iris_unimportable(monkeypatch)
    _force_getpass(monkeypatch, "wmoss")
    assert resolve_marin_user() == "wmoss"


@pytest.mark.parametrize("bad", ["../shared", "alice/extra", "al\x00ice", "\x07bell"])
def test_sanitizer_rejects_invalid_names(bad):
    with pytest.raises(ValueError):
        resolve_marin_user(bad)


def test_sanitizer_returns_none_for_empty_resolved_name(monkeypatch):
    # A blank override/env falls through, so an empty name only reaches the
    # classifier from a resolving source (here getpass). Empty is treated as
    # "no usable per-user owner", not malformed, so it returns None.
    _make_iris_unimportable(monkeypatch)
    _force_getpass(monkeypatch, "")
    assert resolve_marin_user() is None


@pytest.mark.parametrize("generic", ["root", "runner", "nobody", "user"])
def test_generic_name_returns_none(monkeypatch, generic):
    _make_iris_return(monkeypatch, generic)
    assert resolve_marin_user() is None
