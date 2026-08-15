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
    home.mkdir()
    staging.mkdir()
    kube = home / ".kube"
    kube.mkdir()
    (kube / "old-context").write_text("old")
    (kube / "user-created").write_text("keep")
    (home / ".loom-managed-home-files.json").write_text('[".kube/old-context"]')
    (staging / "0").write_text("apiVersion: v1\n")
    plan = staging / "plan.json"
    plan.write_text(json.dumps([{"path": ".kube/coreweave-iris", "source": "0", "mode": "0600"}]))

    apply_home_files(plan, home, staging)

    target = kube / "coreweave-iris"
    assert target.read_text() == "apiVersion: v1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not (kube / "old-context").exists()
    assert (kube / "user-created").read_text() == "keep"
    assert json.loads((home / ".loom-managed-home-files.json").read_text()) == [".kube/coreweave-iris"]


def test_apply_home_files_rejects_symlinked_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    home.mkdir()
    staging.mkdir()
    outside.mkdir()
    (home / ".kube").symlink_to(outside, target_is_directory=True)
    (staging / "0").write_text("secret")
    plan = staging / "plan.json"
    plan.write_text(json.dumps([{"path": ".kube/config", "source": "0", "mode": "0600"}]))

    with pytest.raises(OSError):
        apply_home_files(plan, home, staging)
    assert not (outside / "config").exists()
