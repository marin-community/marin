# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from rigging.filesystem.atomic import atomic_rename, fetch_file_atomic, unique_temp_path


def test_unique_temp_path_produces_distinct_paths():
    paths = {unique_temp_path("/some/output.txt") for _ in range(10)}
    assert len(paths) == 10
    assert all(path.startswith("/some/output.txt.tmp.") for path in paths)


def test_atomic_rename_replaces_output(tmp_path):
    output = tmp_path / "out.txt"

    with atomic_rename(str(output)) as temp_path:
        Path(temp_path).write_text("data")

    assert output.read_text() == "data"
    assert not Path(temp_path).exists()


def test_atomic_rename_cleans_up_on_error(tmp_path):
    output = str(tmp_path / "out.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_rename(output) as temp_path:
            Path(temp_path).write_text("bad")
            raise RuntimeError("boom")

    assert not Path(temp_path).exists()
    assert not Path(output).exists()


def test_fetch_file_atomic_copies_source(tmp_path):
    source = tmp_path / "remote" / "tokenizer.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"version": 1}')
    destination = tmp_path / "cache" / "tokenizer.json"
    destination.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(source), str(destination)) is True
    assert destination.read_bytes() == b'{"version": 1}'


def test_fetch_file_atomic_missing_source_returns_false(tmp_path):
    destination = tmp_path / "cache" / "tokenizer.json"
    destination.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(tmp_path / "remote" / "absent.json"), str(destination)) is False
    assert not destination.exists()


def test_fetch_file_atomic_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    source = tmp_path / "remote" / "tokenizer.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"version": 2}')
    destination = tmp_path / "cache" / "tokenizer.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'{"version": 1}')

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated failure finalizing the fetch")

    monkeypatch.setattr("os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated failure"):
        fetch_file_atomic(str(source), str(destination))

    assert destination.read_bytes() == b'{"version": 1}'
    assert [path.name for path in destination.parent.iterdir()] == ["tokenizer.json"]
