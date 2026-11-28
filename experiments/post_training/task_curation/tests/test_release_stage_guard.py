# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the R40-gate submission guard (release_stages/guard.py)."""

import json
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.task_curation.release_stages import guard

_LOAD_ID_OFFENDER = 'LOAD = "0123456789abcdef0123456789abcdef"\n'
_SNAPSHOT_OFFENDER = 'SNAPSHOT = "r39-curriculum-corpus"\n'
_BENIGN = 'from pathlib import Path\nROOT = Path("/atlas-releases")\n'


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _init_repo(tmp_path: Path) -> str:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "--quiet")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    _write(tmp_path / "file.txt", "one\n")
    git("add", "file.txt")
    git("commit", "--quiet", "-m", "one")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_scanner_flags_load_id_and_snapshot_literals(tmp_path):
    sources = [
        _write(tmp_path / "with_load.py", _LOAD_ID_OFFENDER),
        _write(tmp_path / "with_snapshot.py", _SNAPSHOT_OFFENDER),
    ]
    violations = guard.scan_release_literals(sources)
    flagged = {v.path.name: v.value for v in violations}
    assert flagged == {
        "with_load.py": "0123456789abcdef0123456789abcdef",
        "with_snapshot.py": "r39-curriculum-corpus",
    }


@pytest.mark.parametrize("release", ["atlas-2026.11.27", "atlas-2026.11.27.2"])
def test_scanner_refuses_calendar_release_literals_in_stage_code(tmp_path, release):
    source = _write(tmp_path / "stage.py", f'ROOT = "s3://bucket/{release}/outputs"\n')
    assert [row.value for row in guard.scan_release_literals([source])] == [f"s3://bucket/{release}/outputs"]


def test_scanner_allows_benign_source(tmp_path):
    source = _write(tmp_path / "benign.py", _BENIGN)
    assert guard.scan_release_literals([source]) == []


def test_scanner_flags_hex_inside_longer_string(tmp_path):
    source = _write(
        tmp_path / "embedded.py",
        'URI = "s3://bucket/loads/0123456789abcdef0123456789abcdef/part"\n',
    )
    assert [v.value for v in guard.scan_release_literals([source])] == [
        "s3://bucket/loads/0123456789abcdef0123456789abcdef/part"
    ]


def test_release_stages_package_self_scan_is_clean():
    guard.assert_no_release_literals(guard.PACKAGE_DIR)


def test_head_sha_is_ancestor_of_itself(tmp_path):
    head = _init_repo(tmp_path)
    assert guard.sha_is_ancestor(head, head, tmp_path)


def test_older_head_is_ancestor_of_tip(tmp_path):
    head = _init_repo(tmp_path)
    subprocess.run(
        ["git", "commit", "--quiet", "--allow-empty", "-m", "two"], cwd=tmp_path, check=True, capture_output=True
    )
    tip = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert guard.sha_is_ancestor(head, tip, tmp_path)
    assert not guard.sha_is_ancestor(tip, head, tmp_path)


def test_landing_log_membership(tmp_path):
    log = tmp_path / "landings.jsonl"
    log.write_text(json.dumps({"sha": "a" * 40, "what": "release"}) + "\n", encoding="utf-8")
    assert guard.sha_in_landing_log("a" * 40, log)
    assert not guard.sha_in_landing_log("b" * 40, log)
    assert not guard.sha_in_landing_log("b" * 40, tmp_path / "missing.jsonl")


def test_refuses_sources_outside_release_stages(tmp_path):
    package = tmp_path / "experiments/post_training/task_curation/other_stage"
    package.mkdir(parents=True)
    _write(package / "stage.py", "x = 1\n")
    with pytest.raises(ValueError, match="release_stages"):
        guard.assert_stage_sources_landed(tmp_path, package)


def test_refuses_unlanded_head(tmp_path, monkeypatch):
    package = tmp_path / "experiments/post_training/task_curation/release_stages"
    package.mkdir(parents=True)
    _write(package / "stage.py", "x = 1\n")
    _init_repo(tmp_path)
    monkeypatch.setattr(guard, "atlas_landed_tip", lambda repo_root: "f" * 40)
    monkeypatch.setattr(guard, "landing_log_path", lambda: tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="not landed"):
        guard.assert_stage_sources_landed(tmp_path, package)


def test_accepts_head_recorded_in_landing_log(tmp_path, monkeypatch):
    package = tmp_path / "experiments/post_training/task_curation/release_stages"
    package.mkdir(parents=True)
    _write(package / "stage.py", "x = 1\n")
    head = _init_repo(tmp_path)
    log = tmp_path / "landings.jsonl"
    log.write_text(json.dumps({"sha": head, "what": "release"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(guard, "landing_log_path", lambda: log)
    guard.assert_stage_sources_landed(tmp_path, package)
