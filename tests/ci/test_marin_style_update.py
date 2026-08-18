# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.ci.marin_style_consumers import LEGACY_MANAGED_FILES, LockMode, MarinStyleConsumer
from scripts.ci.marin_style_update import (
    MARIN_STYLE_MANIFEST,
    GeneratedManifest,
    ManifestMode,
    generate_marin_style_update,
)

OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def _consumer_repository(tmp_path: Path) -> tuple[Path, MarinStyleConsumer]:
    repository = tmp_path / "consumer"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    pin = f"git+https://github.com/marin-community/marin-style@{OLD_REVISION}\n"
    precommit = repository / "infra/pre-commit.py"
    precommit.parent.mkdir()
    precommit.write_text(pin)
    workflow = repository / ".github/workflows/marin-ci.yaml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(f"MARIN_STYLE_REV: {OLD_REVISION}\n")
    (repository / "README.md").write_text("consumer\n")
    _git(repository, "add", "infra/pre-commit.py", ".github/workflows/marin-ci.yaml", "README.md")
    _git(repository, "commit", "-m", "initial")
    consumer = MarinStyleConsumer(
        name="test",
        repository="marin-community/test",
        base_branch="main",
        pin_files=("infra/pre-commit.py", ".github/workflows/marin-ci.yaml"),
        required_checks=("test",),
        lock_mode=LockMode.NONE,
    )
    return repository, consumer


def _install_fake_uvx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "format": 1,
        "revision": NEW_REVISION,
        "files": {path: f"sha256:{'0' * 64}" for path in sorted(LEGACY_MANAGED_FILES)},
    }
    manifest_file = tmp_path / "target-manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    uvx = binary_dir / "uvx"
    uvx.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import pathlib\n"
        "import shutil\n"
        "import sys\n"
        "manifest = pathlib.Path(os.environ['FAKE_MANIFEST'])\n"
        "if sys.argv[-1] == 'managed-files':\n"
        "    print(manifest.read_text(), end='')\n"
        "else:\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('--repo-root') + 1])\n"
        "    target = root / '.agents/marin-style/manifest.json'\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(manifest, target)\n"
    )
    uvx.chmod(0o755)
    monkeypatch.setenv("FAKE_MANIFEST", str(manifest_file))
    monkeypatch.setenv("PATH", f"{binary_dir}:{os.environ['PATH']}")


def test_generate_bootstrap_updates_only_registered_and_managed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, consumer = _consumer_repository(tmp_path)
    _install_fake_uvx(tmp_path, monkeypatch)

    update = generate_marin_style_update(
        repo_root=repository,
        consumer=consumer,
        revision=NEW_REVISION,
        manifest_mode=ManifestMode.BOOTSTRAP,
    )

    assert update.old_revision == OLD_REVISION
    assert update.new_revision == NEW_REVISION
    assert update.changed_files == (
        ".agents/marin-style/manifest.json",
        ".github/workflows/marin-ci.yaml",
        "infra/pre-commit.py",
    )
    assert NEW_REVISION in (repository / "infra/pre-commit.py").read_text()
    assert NEW_REVISION in (repository / ".github/workflows/marin-ci.yaml").read_text()
    assert (repository / "README.md").read_text() == "consumer\n"


def test_generate_rejects_unregistered_worktree_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, consumer = _consumer_repository(tmp_path)
    _install_fake_uvx(tmp_path, monkeypatch)
    (repository / "README.md").write_text("unrelated\n")

    with pytest.raises(ValueError):
        generate_marin_style_update(
            repo_root=repository,
            consumer=consumer,
            revision=NEW_REVISION,
            manifest_mode=ManifestMode.BOOTSTRAP,
        )


def test_manifest_rejects_consumer_owned_agent_path() -> None:
    manifest = json.dumps(
        {
            "format": 1,
            "revision": NEW_REVISION,
            "files": {".agents/ops/runbook.md": f"sha256:{'0' * 64}"},
        }
    )

    with pytest.raises(ValueError):
        GeneratedManifest.from_text(manifest, expected_revision=NEW_REVISION)
