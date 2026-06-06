# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import pytest
from marin.execution.artifact import Artifact, PathMetadata
from marin.execution.artifact_registry import (
    DEFAULT_REGISTRY_ENV,
    DEFAULT_REGISTRY_ROOT,
    ArtifactAlreadyExistsError,
    ArtifactNotFoundError,
    ArtifactRegistryError,
    FilesystemArtifactRegistry,
    InvalidArtifactIdError,
    get_default_registry,
    set_default_registry,
    use_default_registry,
)
from marin.execution.executor_step_status import STATUS_SUCCESS, get_status_path
from pydantic import BaseModel


class _Payload(BaseModel):
    path: str
    value: int


@pytest.fixture
def registry(tmp_path):
    return FilesystemArtifactRegistry(str(tmp_path / "registry"))


@pytest.fixture
def artifact_path(tmp_path):
    """A directory holding a saved artifact, returned as an absolute uri."""
    base = tmp_path / "artifact"
    base.mkdir()
    Artifact.save(_Payload(path=str(base), value=7), str(base))
    return str(base)


def _save_payload(base_dir, value):
    """Save a `_Payload` artifact at `base_dir`, returning its absolute uri."""
    base_dir.mkdir(parents=True, exist_ok=True)
    Artifact.save(_Payload(path=str(base_dir), value=value), str(base_dir))
    return str(base_dir)


def _registry_warnings(caplog):
    return [r for r in caplog.records if r.name == "marin.execution.artifact" and r.levelno == logging.WARNING]


# --- id / version validation (through the public register surface) ----------


@pytest.mark.parametrize("bad_id", ["foo", "foo/", "/bar", "a/b/c", "a b/c", "ns/na me", "", "ns//name"])
def test_register_rejects_malformed_id(registry, bad_id):
    with pytest.raises(InvalidArtifactIdError):
        registry.register(bad_id, "2026.05.29", "/tmp/x")


@pytest.mark.parametrize("good_version", ["2026.05.29", "2026.10.01-fall-hero", "2026.10.01-rc1", "2024.02.29"])
def test_register_accepts_calver_version(registry, good_version):
    assert registry.register("datasets/foo", good_version, "/tmp/x").version == good_version


@pytest.mark.parametrize(
    "bad_version",
    [
        "v3",
        "2026.5.29",
        "2026.02.30",
        "2026.13.01",
        "2026.10.01-",
        "2026.10.01--foo",
        "2026.10.01-foo/bar",
        "2025.02.29",
    ],
)
def test_register_rejects_non_calver_version(registry, bad_version):
    with pytest.raises(InvalidArtifactIdError):
        registry.register("datasets/foo", bad_version, "/tmp/x")


# --- register / lookup ------------------------------------------------------


def test_register_then_lookup_round_trips(registry, artifact_path):
    # register returns the entry; lookup reads back an equal one (id, version, uri, relative_path).
    entry = registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    assert entry.uri == artifact_path
    assert registry.lookup("datasets/fineweb", "2026.05.29") == entry


def test_lookup_missing_raises(registry):
    with pytest.raises(ArtifactNotFoundError) as exc:
        registry.lookup("datasets/fineweb", "2026.05.29")
    # Subclasses KeyError (dict-style consumers keep working) and exposes the missing key.
    assert isinstance(exc.value, KeyError)
    assert (exc.value.id, exc.value.version) == ("datasets/fineweb", "2026.05.29")


def test_register_is_append_only(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    with pytest.raises(ArtifactAlreadyExistsError) as exc:
        registry.register("datasets/fineweb", "2026.05.29", "/some/other/uri")
    # The existing entry is untouched and exposed for comparison.
    assert exc.value.existing.uri == artifact_path
    assert registry.lookup("datasets/fineweb", "2026.05.29").uri == artifact_path


def test_register_rejects_relative_uri(registry):
    with pytest.raises(InvalidArtifactIdError):
        registry.register("datasets/fineweb", "2026.05.29", "relative/path")


@pytest.mark.parametrize("local_uri", ["/local/path/to/data", "file:///local/path/to/data"])
def test_register_rejects_local_uri_in_remote_registry(local_uri):
    # A local path stored in a shared cloud registry is unresolvable for other readers.
    remote = FilesystemArtifactRegistry("gs://foobar2000")
    with pytest.raises(InvalidArtifactIdError):
        remote.register("datasets/fineweb", "2026.05.29", local_uri)


def test_register_allows_remote_uri_in_local_registry(registry):
    # The reverse is fine: a gs:// pointer recorded in a local registry resolves anywhere.
    uri = "gs://foobar2000/documents/fineweb-8c2f3a"
    assert registry.register("datasets/fineweb", "2026.05.29", uri).uri == uri


def test_manifest_layout(registry, artifact_path):
    # A registered entry lands at {root}/{namespace}/{name}/{version}.json with the wire-format JSON.
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    with open(f"{registry._root}/datasets/fineweb/2026.05.29.json") as fd:
        on_disk = json.load(fd)
    # artifact_path is outside marin_prefix() in tests, so relative_path is null.
    assert on_disk == {"id": "datasets/fineweb", "version": "2026.05.29", "uri": artifact_path, "relative_path": None}


def test_lookup_corrupt_manifest_raises(registry):
    path = registry._entry_path("datasets/fineweb", "2026.05.29")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fd:
        fd.write("not json{")
    with pytest.raises(ArtifactRegistryError):
        registry.lookup("datasets/fineweb", "2026.05.29")


# --- construction -----------------------------------------------------------


def test_root_normalizes_trailing_slash(tmp_path):
    assert FilesystemArtifactRegistry(f"{tmp_path}/reg///")._root == f"{tmp_path}/reg"


def test_empty_root_raises():
    with pytest.raises(ValueError):
        FilesystemArtifactRegistry("")


# --- Artifact.from_id -------------------------------------------------------


def test_from_id_round_trips_typed(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    loaded = Artifact.from_id("datasets/fineweb", "2026.05.29", _Payload, registry=registry)
    assert loaded == _Payload(path=artifact_path, value=7)


def test_from_id_untyped_returns_dict(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    loaded = Artifact.from_id("datasets/fineweb", "2026.05.29", registry=registry)
    assert loaded == {"path": artifact_path, "value": 7}


def test_from_id_propagates_not_found(registry):
    with pytest.raises(ArtifactNotFoundError):
        Artifact.from_id("datasets/fineweb", "2026.05.29", registry=registry)


def test_from_id_uses_default_registry(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    with use_default_registry(registry):
        loaded = Artifact.from_id("datasets/fineweb", "2026.05.29", _Payload)
        assert loaded == _Payload(path=artifact_path, value=7)


def test_use_default_registry_restores_previous(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    before = get_default_registry()
    with use_default_registry(registry) as scoped:
        assert get_default_registry() is scoped is registry
    # The override is scoped to the block; the prior default is restored on exit.
    assert get_default_registry() is before


# --- region-aware (relative-path) resolution --------------------------------


def test_register_records_relative_path(registry, tmp_path, monkeypatch):
    region = tmp_path / "region_a"
    monkeypatch.setenv("MARIN_PREFIX", str(region))
    uri = _save_payload(region / "documents" / "foo", 7)
    entry = registry.register("datasets/foo", "2026.05.29", uri)
    assert entry.relative_path == "documents/foo"
    assert entry.uri == uri


def test_register_relative_path_none_outside_prefix(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "region_a"))
    uri = _save_payload(tmp_path / "elsewhere" / "foo", 7)
    entry = registry.register("datasets/foo", "2026.05.29", uri)
    assert entry.relative_path is None


def test_from_id_prefers_region_local_replica(registry, tmp_path, monkeypatch, caplog):
    # Register under region A's prefix.
    region_a = tmp_path / "region_a"
    monkeypatch.setenv("MARIN_PREFIX", str(region_a))
    registry.register("datasets/foo", "2026.05.29", _save_payload(region_a / "documents" / "foo", 7))

    # A reader in region B with a local replica resolves the replica, not the registration uri.
    region_b = tmp_path / "region_b"
    monkeypatch.setenv("MARIN_PREFIX", str(region_b))
    _save_payload(region_b / "documents" / "foo", 99)

    with caplog.at_level(logging.WARNING):
        loaded = Artifact.from_id("datasets/foo", "2026.05.29", _Payload, registry=registry)
    assert loaded.value == 99
    assert _registry_warnings(caplog) == []  # region-local hit, no cross-region warning


def test_from_id_falls_back_to_absolute_and_warns(registry, tmp_path, monkeypatch, caplog):
    region_a = tmp_path / "region_a"
    monkeypatch.setenv("MARIN_PREFIX", str(region_a))
    registry.register("datasets/foo", "2026.05.29", _save_payload(region_a / "documents" / "foo", 7))

    # Region B has no replica → fall back to the absolute (region A) uri, with a cross-region warning.
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "region_b"))
    with caplog.at_level(logging.WARNING):
        loaded = Artifact.from_id("datasets/foo", "2026.05.29", _Payload, registry=registry)
    assert loaded.value == 7
    assert len(_registry_warnings(caplog)) == 1


# --- default registry singleton ---------------------------------------------


@pytest.fixture
def reset_default():
    set_default_registry(None)
    yield
    set_default_registry(None)


def test_get_default_registry_reads_env(monkeypatch, tmp_path, reset_default):
    monkeypatch.setenv(DEFAULT_REGISTRY_ENV, str(tmp_path / "envroot"))
    assert get_default_registry()._root == str(tmp_path / "envroot")


def test_get_default_registry_falls_back_to_canonical_root(monkeypatch, reset_default):
    monkeypatch.delenv(DEFAULT_REGISTRY_ENV, raising=False)
    assert get_default_registry()._root == DEFAULT_REGISTRY_ROOT


def test_from_path_fallback_via_id(registry, tmp_path):
    # An artifact dir with no sidecar but SUCCESS status synthesizes PathMetadata, even via from_id.
    base = tmp_path / "step-output"
    base.mkdir()
    with open(get_status_path(str(base)), "w") as fd:
        fd.write(STATUS_SUCCESS)
    registry.register("models/checkpoint", "2026.05.29", str(base))
    loaded = Artifact.from_id("models/checkpoint", "2026.05.29", registry=registry)
    assert loaded == PathMetadata(path=str(base))
