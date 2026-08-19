# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import stat
from pathlib import Path

import pytest

from infra.loom.materialize_home_files import apply_home_files


def test_apply_home_files_reconciles_managed_files_without_touching_user_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    staging = tmp_path / "staging"
    state = tmp_path / "state"
    home.mkdir()
    staging.mkdir()
    state.mkdir()
    kube = home / ".kube"
    kube.mkdir()
    (kube / "old-context").write_text("old")
    (kube / "user-created").write_text("keep")
    (state / "managed-paths.json").write_text('[".kube/old-context"]')
    (staging / "0").write_text("apiVersion: v1\n")
    plan = staging / "plan.json"
    plan.write_text(json.dumps([{"path": ".kube/coreweave-iris", "source": "0", "mode": "0600"}]))

    apply_home_files(plan, home, staging, state)

    target = kube / "coreweave-iris"
    assert target.read_text() == "apiVersion: v1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not (kube / "old-context").exists()
    assert (kube / "user-created").read_text() == "keep"
    assert json.loads((state / "managed-paths.json").read_text()) == [".kube/coreweave-iris"]
    assert not (home / "managed-paths.json").exists()


def test_apply_home_files_rejects_symlinked_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    staging = tmp_path / "staging"
    state = tmp_path / "state"
    outside = tmp_path / "outside"
    home.mkdir()
    staging.mkdir()
    state.mkdir()
    outside.mkdir()
    (home / ".kube").symlink_to(outside, target_is_directory=True)
    (staging / "0").write_text("secret")
    plan = staging / "plan.json"
    plan.write_text(json.dumps([{"path": ".kube/config", "source": "0", "mode": "0600"}]))

    with pytest.raises(OSError):
        apply_home_files(plan, home, staging, state)
    assert not (outside / "config").exists()


def test_apply_home_files_transitions_between_file_and_directory_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    staging = tmp_path / "staging"
    state = tmp_path / "state"
    home.mkdir()
    staging.mkdir()
    state.mkdir()
    (staging / "0").write_text("parent")
    (staging / "1").write_text("child")
    plan = staging / "plan.json"

    plan.write_text(json.dumps([{"path": "credential", "source": "0", "mode": "0600"}]))
    apply_home_files(plan, home, staging, state)
    assert (home / "credential").read_text() == "parent"

    plan.write_text(json.dumps([{"path": "credential/config", "source": "1", "mode": "0600"}]))
    apply_home_files(plan, home, staging, state)
    assert (home / "credential/config").read_text() == "child"

    unmanaged = home / "credential/user-created"
    unmanaged.write_text("keep")
    plan.write_text(json.dumps([{"path": "credential", "source": "0", "mode": "0600"}]))
    with pytest.raises(ValueError, match="not a regular file"):
        apply_home_files(plan, home, staging, state)
    assert unmanaged.read_text() == "keep"

    unmanaged.unlink()
    apply_home_files(plan, home, staging, state)
    assert (home / "credential").read_text() == "parent"
