# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The large-data merge guard: per-file and per-directory limits with a shrink-only allowlist."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "precommit_under_test", Path(__file__).resolve().parents[2] / "infra" / "pre-commit.py"
)
assert _SPEC is not None and _SPEC.loader is not None
precommit = importlib.util.module_from_spec(_SPEC)
sys.modules["precommit_under_test"] = precommit
# Keep infra/ namespaces from shadowing optional packages in other tests.
with pytest.MonkeyPatch.context() as import_patch:
    import_patch.setattr(sys, "path", list(sys.path))
    _SPEC.loader.exec_module(precommit)


def _git(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repo whose origin/main matches the initial commit."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for key, value in [("user.name", "Test"), ("user.email", "test@example.org")]:
        subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    root = tmp_path / "datasets"
    root.mkdir()
    (root / "small.json").write_text("{}\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "Base"], cwd=tmp_path, check=True)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=tmp_path, check=True)
    monkeypatch.setattr(precommit, "ROOT_DIR", tmp_path)
    allowlist = tmp_path / "infra" / "large-data-allowlist.json"
    allowlist.parent.mkdir()
    monkeypatch.setattr(precommit, "LARGE_DATA_ALLOWLIST_PATH", allowlist)
    return tmp_path


def _write_data(repo: Path, relative: str, kilobytes: int, binary: bool = False) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    unit = bytes(range(256))
    payload = (
        (unit * (kilobytes * 1024 // len(unit) + 1))[: kilobytes * 1024] if binary else b"x" * (1024 * kilobytes + 40)
    )
    path.write_bytes(payload)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return path


def _allowlist(repo: Path, entries: dict[str, int]) -> None:
    (repo / "infra" / "large-data-allowlist.json").write_text(json.dumps(entries, sort_keys=True, indent=2) + "\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)


def test_new_large_data_file_exceeding_limit_fails(repo: Path):
    path = _write_data(repo, "datasets/dump/results.json", 150)
    assert precommit.check_large_data([path], fix=False) == 1


def test_directory_total_counts_committed_binaries_and_gzip(repo: Path):
    _write_data(repo, "datasets/shards/part-000.json.gz", 90, binary=True)
    _write_data(repo, "datasets/shards/part-001.tar", 90, binary=True)
    _git("commit", "--quiet", "-m", "Compressed shards", cwd=repo)
    third = _write_data(repo, "datasets/shards/part-002.zst", 90, binary=True)
    assert precommit.check_large_data([third], fix=False) == 1


def test_allowlisted_file_at_its_recorded_size_passes(repo: Path):
    path = _write_data(repo, "datasets/legacy/results.json", 150)
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    assert precommit.check_large_data([path], fix=False) == 0


def test_allowlisted_file_that_grows_fails(repo: Path):
    path = _write_data(repo, "datasets/legacy/results.json", 150)
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    path.write_bytes(b"x" * (200 * 1024))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert precommit.check_large_data([path], fix=False) == 1


def test_allowlisted_file_may_shrink(repo: Path):
    path = _write_data(repo, "datasets/legacy/results.json", 150)
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    path.write_bytes(b"x" * (50 * 1024))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    # The entry may be lowered to the new size in the same change.
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    assert precommit.check_large_data([path], fix=False) == 0


def test_raising_an_allowlisted_size_fails_once_origin_main_has_the_allowlist(repo: Path):
    path = _write_data(repo, "datasets/legacy/results.json", 150)
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    _git("commit", "--quiet", "-m", "Adopt the allowlist", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
    grown = _write_data(repo, "datasets/legacy/results.json", 200)
    _allowlist(repo, {"datasets/legacy/results.json": grown.stat().st_size})
    assert precommit.check_large_data([path, precommit.LARGE_DATA_ALLOWLIST_PATH], fix=False) == 1


def test_missing_origin_main_fails_the_guard_with_a_fetch_hint(repo: Path):
    path = _write_data(repo, "datasets/legacy/results.json", 150)
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    _git("update-ref", "-d", "refs/remotes/origin/main", cwd=repo)
    with pytest.raises(RuntimeError, match="git fetch origin main"):
        precommit.check_large_data([path, precommit.LARGE_DATA_ALLOWLIST_PATH], fix=False)


def test_allowlisting_data_origin_main_never_had_fails_after_adoption(repo: Path):
    _allowlist(repo, {})
    _git("commit", "--quiet", "-m", "Adopt the allowlist", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
    dump = _write_data(repo, "datasets/fresh/dump.json", 150)
    _allowlist(repo, {"datasets/fresh/dump.json": dump.stat().st_size})
    assert precommit.check_large_data([dump, precommit.LARGE_DATA_ALLOWLIST_PATH], fix=False) == 1


def test_adoption_mode_accepts_entries_for_data_origin_main_lacks(repo: Path):
    # The branch itself introduced the data; the first allowlist commit adopts it.
    path = _write_data(repo, "datasets/legacy/results.json", 150)
    _allowlist(repo, {"datasets/legacy/results.json": path.stat().st_size})
    assert precommit.check_large_data([path, precommit.LARGE_DATA_ALLOWLIST_PATH], fix=False) == 0


def test_non_data_and_config_files_are_ignored(repo: Path):
    source = repo / "code.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert precommit.check_large_data([source], fix=False) == 0


def test_tracked_directory_symlink_keeps_large_data_checks_active(repo: Path):
    target = repo / "configurations"
    target.mkdir()
    link = repo / "config-link"
    link.symlink_to(target, target_is_directory=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert precommit.check_large_data([link], fix=False) == 0
    oversized = _write_data(repo, "datasets/dump/results.json", 150)
    assert precommit.check_large_data([link, oversized], fix=False) == 1
