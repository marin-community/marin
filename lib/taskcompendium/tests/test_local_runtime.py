# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The worker runtime resolved from a lock artifact and the files stored beside it."""

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from shellbox.machine import Mount

from taskcompendium.runtime.local import EnvironmentManifest, LocalRuntime, extract_files, local_runtime


@pytest.mark.parametrize(
    ("cache_mounted", "expected_parent"), [(True, Path("/cache/task-curation")), (False, Path("/tmp"))]
)
def test_runtime_parent_matches_executable_mounts(monkeypatch, cache_mounted, expected_parent):
    monkeypatch.setattr(Path, "is_mount", lambda self: cache_mounted and self == Path("/cache"))

    runtime = LocalRuntime("fixture.lock", "0" * 64)
    assert runtime.root.parent == expected_parent


def files_tar(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def test_the_manifest_beside_the_lock_mounts_extracted_files_at_their_paths_and_carries_the_env(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"six==1.17.0\n")
    guest = "/srv/nltk_data/tokenizers/punkt_tab/english"
    env = {"NLTK_DATA": "/srv/nltk_data"}
    archive = files_tar({f"{guest.removeprefix('/')}/abbrev.tab": b"etc\n"})
    (tmp_path / "files.tar").write_bytes(archive)
    manifest = EnvironmentManifest((guest,), hashlib.sha256(archive).hexdigest(), env)
    (tmp_path / "environment.json").write_text(
        json.dumps({"files": [guest], "files_sha256": manifest.files_sha256, "env": env})
    )

    runtime = replace(local_runtime(str(lock)), parent=tmp_path / "runtime")
    extract_files(archive, runtime.root / "files")

    (mount,) = runtime.mounts
    assert mount == Mount(runtime.root / "files" / guest.removeprefix("/"), PurePosixPath(guest))
    assert (mount.host / "abbrev.tab").read_bytes() == b"etc\n"
    assert runtime.env == env
    bare = LocalRuntime(str(lock), runtime.lock_sha256)
    other_archive = replace(runtime, manifest=replace(manifest, files_sha256="0" * 64))
    other_env = replace(runtime, manifest=replace(manifest, env={}))
    assert len({runtime.identity, other_archive.identity, other_env.identity, bare.identity}) == 4


def test_a_lock_without_a_manifest_mounts_nothing_and_sets_no_variables(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"")
    runtime = local_runtime(str(lock))
    assert (runtime.mounts, runtime.env) == ((), {})


def test_files_archive_members_cannot_escape_the_files_directory(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        extract_files(files_tar({"../outside.txt": b"x"}), tmp_path / "files")
    assert not (tmp_path / "outside.txt").exists()
