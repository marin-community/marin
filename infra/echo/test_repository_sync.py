# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo's rolling GitHub file index."""

import base64
import io
import tarfile
from datetime import UTC, datetime, timedelta

from sync import github_repository as echo_sync


def test_github_blob_accepts_api_line_wrapped_base64(monkeypatch):
    encoded = base64.encodebytes(b"def scheduler():\n    pass\n").decode()
    monkeypatch.setattr(
        echo_sync,
        "github_json",
        lambda path, token: {"encoding": "base64", "content": encoded},
    )

    assert echo_sync.github_blob("marin-community/marin", "abc", "token") == b"def scheduler():\n    pass\n"


def test_incremental_repository_files_fetches_only_eligible_changed_blobs():
    blobs = {
        "kept": b"def kept():\n    return True\n",
        "renamed": b"# Renamed runbook\n\nUse the new path.\n",
    }
    requested = []

    def load_blob(sha):
        requested.append(sha)
        return blobs[sha]

    comparison = {
        "status": "ahead",
        "files": [
            {"filename": "src/kept.py", "status": "modified", "sha": "kept"},
            {"filename": "docs/removed.md", "status": "removed", "size": 0, "sha": "removed"},
            {
                "filename": "docs/renamed.md",
                "previous_filename": "docs/old-name.md",
                "status": "renamed",
                "sha": "renamed",
            },
            {"filename": "vendor/copied.py", "status": "modified", "size": 20, "sha": "vendor"},
            {
                "filename": "docs/huge.md",
                "status": "modified",
                "size": 300_000,
                "sha": "huge",
            },
        ],
    }

    changes = echo_sync.incremental_repository_files(comparison, load_blob)

    assert changes is not None
    assert {file.path for file in changes.files} == {"src/kept.py", "docs/renamed.md"}
    assert changes.replaced_paths == frozenset(
        {
            "src/kept.py",
            "docs/removed.md",
            "docs/renamed.md",
            "docs/old-name.md",
            "vendor/copied.py",
            "docs/huge.md",
        }
    )
    assert requested == ["kept", "renamed"]


def test_large_github_comparison_requests_full_rebuild():
    comparison = {
        "status": "ahead",
        "files": [
            {"filename": f"src/file_{index}.py", "status": "modified", "size": 1, "sha": str(index)}
            for index in range(echo_sync.MAX_COMPARE_FILES)
        ],
    }

    assert echo_sync.incremental_repository_files(comparison, lambda _: b"x") is None


def test_archive_repository_files_uses_repository_relative_safety_filters():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path, contents in {
            "marin-sha/src/kept.py": b"def kept():\n    return True\n",
            "marin-sha/vendor/copied.py": b"def copied():\n    return True\n",
            "marin-sha/keys/deploy.pem": b"private material",
        }.items():
            info = tarfile.TarInfo(path)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    stream.seek(0)

    files = echo_sync.archive_repository_files(stream)

    assert [file.path for file in files] == ["src/kept.py"]


def test_repository_check_is_due_once_per_hour():
    now = datetime(2026, 7, 29, 20, tzinfo=UTC)

    assert echo_sync.repository_check_due(None, now)
    assert not echo_sync.repository_check_due(
        echo_sync.RepositoryState("abc", now - timedelta(minutes=59)),
        now,
    )
    assert echo_sync.repository_check_due(
        echo_sync.RepositoryState("abc", now - timedelta(hours=1)),
        now,
    )
