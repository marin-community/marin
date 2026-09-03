# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo's activity and repository synchronization."""

import base64
import io
import sqlite3
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest
import repository_files
from sync import github_repository as echo_sync
from sync import main as activity_sync


class _TurnResult:
    def __init__(self, row=None):
        self._row = row

    def first(self):
        return self._row

    def scalar(self):
        return self._row


class _TurnEngine:
    """Fake the database boundary needed by one otherwise-real sync execution."""

    def __init__(self, turn_state, *, repository_locked=True, repository_state=None):
        self.turn_state = turn_state
        self.repository_locked = repository_locked
        self.repository_state = repository_state
        self.repository_lock_attempts = 0
        self.repository_checked_at = None

    def connect(self):
        return self

    def begin(self):
        return self

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def execute(self, statement):
        rendered = str(statement)
        table = getattr(getattr(statement, "table", None), "name", None)
        if "pg_try_advisory_lock" in rendered:
            self.repository_lock_attempts += 1
            return _TurnResult(self.repository_locked)
        if "pg_advisory_unlock" in rendered:
            return _TurnResult(True)
        if table == "repository_sync_turn" and statement.is_insert:
            self.turn_state.setdefault("next_target", 0)
            return _TurnResult()
        if table == "repository_sync_turn" and statement.is_update:
            self.turn_state["next_target"] = statement.compile().params["next_target"]
            return _TurnResult()
        if table == "repository_index_state" and statement.is_update:
            self.repository_checked_at = statement.compile().params["checked_at"]
            return _TurnResult()
        if "FROM repository_sync_turn" in rendered:
            return _TurnResult(SimpleNamespace(next_target=self.turn_state["next_target"]))
        if "FROM sync_state" in rendered:
            return _TurnResult(1)
        if "FROM repository_index_state" in rendered:
            if self.repository_state is None:
                return _TurnResult()
            return _TurnResult(
                SimpleNamespace(
                    commit_sha=self.repository_state.commit_sha,
                    checked_at=self.repository_state.checked_at,
                )
            )
        if "FROM repository_index_builds" in rendered:
            return _TurnResult()
        raise AssertionError(f"unexpected statement: {statement}")


def _current_activity_manifest(_token):
    return {"built_at_epoch": 1}


def test_activity_corpus_reader_accepts_marinmirror_schema(tmp_path):
    corpus = tmp_path / "corpus.db"
    with sqlite3.connect(corpus) as database:
        database.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                ref TEXT,
                parent TEXT,
                title TEXT,
                author TEXT,
                date TEXT,
                url TEXT NOT NULL,
                text TEXT,
                hash TEXT,
                embedding BLOB,
                part INTEGER NOT NULL,
                n_parts INTEGER NOT NULL
            )
            """
        )
        database.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                7,
                "discord",
                "message",
                "ref",
                None,
                "deployments",
                "operator",
                "2026-07-29T21:00:00+00:00",
                "https://discord.com/channels/1/2/3",
                "Echo deployed",
                "digest",
                None,
                0,
                1,
            ),
        )
        row = activity_sync.corpus_chunk_cursor(database).fetchone()

    assert activity_sync.chunk_record(row) == {
        "id": 7,
        "source": "discord",
        "kind": "message",
        "ref": "ref",
        "parent": None,
        "title": "deployments",
        "author": "operator",
        "date": datetime(2026, 7, 29, 21, tzinfo=UTC),
        "url": "https://discord.com/channels/1/2/3",
        "text": "Echo deployed",
        "hash": "digest",
        "embedding": None,
        "part": 0,
        "n_parts": 1,
    }


def test_github_blob_accepts_api_line_wrapped_base64(monkeypatch):
    encoded = base64.encodebytes(b"def scheduler():\n    pass\n").decode()
    monkeypatch.setattr(
        echo_sync,
        "github_json",
        lambda _path, _token: {"encoding": "base64", "content": encoded},
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


def test_selected_repository_checks_head_even_if_recently_checked(monkeypatch):
    now = datetime(2026, 7, 29, 20, tzinfo=UTC)
    target = echo_sync.RepositoryTarget("marin-community/marin", "main")
    engine = _TurnEngine(
        {},
        repository_state=echo_sync.RepositoryState("abc", now - timedelta(minutes=1)),
    )
    requested = []
    monkeypatch.setattr(
        echo_sync, "github_head", lambda requested_target, _token: requested.append(requested_target) or "abc"
    )

    echo_sync.sync_repository_locked(engine, target, "token", now)

    assert requested == [target]
    assert engine.repository_checked_at == now


def test_repository_resume_keeps_only_files_missing_from_durable_checkpoint():
    files = []
    for path in ("docs/a.md", "docs/b.md", "docs/c.md"):
        file = repository_files.indexed_file(PurePosixPath(path), f"# {path}\n".encode())
        assert file is not None
        files.append(file)

    remaining = echo_sync.remaining_repository_files(tuple(files), frozenset({"docs/a.md", "docs/c.md"}))

    assert [file.path for file in remaining] == ["docs/b.md"]


def test_repository_turn_is_durable_across_failure_and_process_restart(monkeypatch):
    state = {}
    attempts = []

    monkeypatch.setattr(activity_sync, "fetch_manifest", _current_activity_manifest)

    def fail_github_head(target, _token):
        attempts.append(target.repository)
        raise RuntimeError("injected GitHub failure")

    monkeypatch.setattr(activity_sync.github_repository, "github_head", fail_github_head)

    with pytest.raises(RuntimeError, match="injected GitHub failure"):
        activity_sync.run(_TurnEngine(state), "token")

    assert state == {"next_target": 1}

    # A fresh engine stands in for a new Cloud Run process reading the same row.
    with pytest.raises(RuntimeError, match="injected GitHub failure"):
        activity_sync.run(_TurnEngine(state), "token")
    assert attempts == ["marin-community/marin", "marin-community/vllm"]
    assert state == {"next_target": 2}


def test_repository_lock_loser_does_activity_but_does_not_consume_turn(monkeypatch):
    state = {}
    engine = _TurnEngine(state, repository_locked=False)
    monkeypatch.setattr(activity_sync, "fetch_manifest", _current_activity_manifest)
    monkeypatch.setattr(
        activity_sync.github_repository,
        "github_head",
        lambda _target, _token: pytest.fail("lock loser reached GitHub"),
    )

    assert activity_sync.run(engine, "token") == 0
    assert engine.repository_lock_attempts == 1
    assert state == {}


def test_activity_failure_prevents_repository_lock_and_turn(monkeypatch):
    state = {}
    engine = _TurnEngine(state)
    monkeypatch.setattr(
        activity_sync,
        "fetch_manifest",
        lambda _token: (_ for _ in ()).throw(RuntimeError("injected MarinMirror failure")),
    )
    monkeypatch.setattr(
        activity_sync.github_repository,
        "github_head",
        lambda _target, _token: pytest.fail("activity failure reached GitHub"),
    )

    with pytest.raises(RuntimeError, match="injected MarinMirror failure"):
        activity_sync.run(engine, "token")
    assert engine.repository_lock_attempts == 0
    assert state == {}
