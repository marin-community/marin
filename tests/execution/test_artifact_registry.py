# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from marin.execution.artifact import Artifact, PathMetadata
from marin.execution.artifact_registry import (
    DEFAULT_REGISTRY_ENV,
    ArtifactAlreadyExistsError,
    ArtifactEntry,
    ArtifactNotFoundError,
    ArtifactRegistryError,
    FilesystemArtifactRegistry,
    InvalidArtifactIdError,
    get_default_registry,
    set_default_registry,
    validate_id,
    validate_version,
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


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["foo", "foo/", "/bar", "a/b/c", "a b/c", "ns/na me", "", "ns//name"])
def test_validate_id_rejects_malformed(bad_id):
    with pytest.raises(InvalidArtifactIdError):
        validate_id(bad_id)


def test_validate_id_returns_segments():
    assert validate_id("datasets/fineweb-resiliparse") == ("datasets", "fineweb-resiliparse")


@pytest.mark.parametrize(
    "good_version",
    ["2026.05.29", "2026.10.01-fall-hero", "2026.10.01-rc1", "2024.02.29"],
)
def test_validate_version_accepts_calver(good_version):
    assert validate_version(good_version) == good_version


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
def test_validate_version_rejects_non_calver(bad_version):
    with pytest.raises(InvalidArtifactIdError):
        validate_version(bad_version)


# --- register / lookup ------------------------------------------------------


def test_register_returns_entry(registry, artifact_path):
    entry = registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    assert entry == ArtifactEntry(id="datasets/fineweb", version="2026.05.29", uri=artifact_path)


def test_lookup_round_trips(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    assert registry.lookup("datasets/fineweb", "2026.05.29").uri == artifact_path


def test_lookup_missing_raises(registry):
    with pytest.raises(ArtifactNotFoundError) as exc:
        registry.lookup("datasets/fineweb", "2026.05.29")
    assert exc.value.id == "datasets/fineweb"
    assert exc.value.version == "2026.05.29"


def test_lookup_missing_is_keyerror(registry):
    # Subclasses KeyError so dict-style consumers keep working.
    with pytest.raises(KeyError):
        registry.lookup("datasets/fineweb", "2026.05.29")


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


def test_register_validates_id_and_version(registry, artifact_path):
    with pytest.raises(InvalidArtifactIdError):
        registry.register("no-namespace", "2026.05.29", artifact_path)
    with pytest.raises(InvalidArtifactIdError):
        registry.register("datasets/fineweb", "v1", artifact_path)


def test_manifest_layout(registry, artifact_path):
    registry.register("datasets/fineweb", "2026.05.29", artifact_path)
    expected = f"{registry.root}/datasets/fineweb/2026.05.29.json"
    assert registry.entry_path("datasets/fineweb", "2026.05.29") == expected
    with open(expected) as fd:
        on_disk = json.load(fd)
    assert on_disk == {"id": "datasets/fineweb", "version": "2026.05.29", "uri": artifact_path}


def test_lookup_corrupt_manifest_raises(registry):
    path = registry.entry_path("datasets/fineweb", "2026.05.29")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fd:
        fd.write("not json{")
    with pytest.raises(ArtifactRegistryError):
        registry.lookup("datasets/fineweb", "2026.05.29")


# --- construction -----------------------------------------------------------


def test_root_normalizes_trailing_slash(tmp_path):
    assert FilesystemArtifactRegistry(f"{tmp_path}/reg///").root == f"{tmp_path}/reg"


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
    set_default_registry(registry)
    try:
        loaded = Artifact.from_id("datasets/fineweb", "2026.05.29", _Payload)
        assert loaded == _Payload(path=artifact_path, value=7)
    finally:
        set_default_registry(None)


# --- default registry singleton ---------------------------------------------


@pytest.fixture
def reset_default():
    set_default_registry(None)
    yield
    set_default_registry(None)


def test_get_default_registry_reads_env(monkeypatch, tmp_path, reset_default):
    monkeypatch.setenv(DEFAULT_REGISTRY_ENV, str(tmp_path / "envroot"))
    assert get_default_registry().root == str(tmp_path / "envroot")


def test_get_default_registry_caches(monkeypatch, tmp_path, reset_default):
    monkeypatch.setenv(DEFAULT_REGISTRY_ENV, str(tmp_path / "envroot"))
    assert get_default_registry() is get_default_registry()


def test_set_default_registry_clears_cache(monkeypatch, tmp_path, reset_default):
    monkeypatch.setenv(DEFAULT_REGISTRY_ENV, str(tmp_path / "first"))
    first = get_default_registry()
    set_default_registry(None)
    monkeypatch.setenv(DEFAULT_REGISTRY_ENV, str(tmp_path / "second"))
    assert get_default_registry().root == str(tmp_path / "second")
    assert get_default_registry() is not first


def test_from_path_fallback_via_id(registry, tmp_path):
    # An artifact dir with no sidecar but SUCCESS status synthesizes PathMetadata, even via from_id.
    base = tmp_path / "step-output"
    base.mkdir()
    with open(get_status_path(str(base)), "w") as fd:
        fd.write(STATUS_SUCCESS)
    registry.register("models/checkpoint", "2026.05.29", str(base))
    loaded = Artifact.from_id("models/checkpoint", "2026.05.29", registry=registry)
    assert loaded == PathMetadata(path=str(base))
