# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from infra.ci.run_tests import default_base_ref, worktree_diff


def test_default_base_ref_refreshes_main_before_selecting_changed_files(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    repo.mkdir()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("remote", "add", "origin", str(origin))

    (repo / "pyproject.toml").write_text("base\n")
    git("add", ".")
    git("commit", "-qm", "base")
    old_main = git("rev-parse", "HEAD")
    git("push", "-q", "origin", "HEAD:main")

    (repo / "pyproject.toml").write_text("shellbox\n")
    git("commit", "-qam", "shellbox")
    git("push", "-q", "origin", "HEAD:main")
    (repo / "cache.py").write_text("cache\n")
    git("add", ".")
    git("commit", "-qm", "cache")
    git("update-ref", "refs/remotes/origin/main", old_main)

    assert set(worktree_diff("origin/main", repo).changed_files) == {"cache.py", "pyproject.toml"}

    base_ref = default_base_ref(repo)

    assert worktree_diff(base_ref, repo).changed_files == ("cache.py",)
