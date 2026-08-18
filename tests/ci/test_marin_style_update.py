# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.ci.marin_style_update import (
    LEGACY_MANAGED_FILES,
    GeneratedManifest,
    ManifestMode,
    generate_marin_style_update,
    installed_consumer_matrix_json,
)

OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def _consumer_repository(tmp_path: Path) -> Path:
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
    generated = repository / ".agents/skills/consult-echo/scripts/echo.py"
    generated.parent.mkdir(parents=True)
    generated.write_text(f"# marin-style@{OLD_REVISION}\n")
    (repository / "README.md").write_text("consumer\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "initial")
    return repository


def _install_fake_uvx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    revision: str = NEW_REVISION,
) -> None:
    manifest = {
        "format": 1,
        "revision": revision,
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
        "    echo = root / '.agents/skills/consult-echo/scripts/echo.py'\n"
        f"    echo.write_text('# marin-style@{revision}\\n')\n"
    )
    uvx.chmod(0o755)
    monkeypatch.setenv("FAKE_MANIFEST", str(manifest_file))
    monkeypatch.setenv("PATH", f"{binary_dir}:{os.environ['PATH']}")


def test_installed_consumer_matrix_uses_app_repository_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    gh = binary_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "print('marin-community/marin\\tmarin\\tmain')\n"
        "print('marin-community/MarinSkyRL\\tMarinSkyRL\\tmain')\n"
        "print('marin-community/axolotl\\taxolotl\\tmain')\n"
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary_dir}:{os.environ['PATH']}")

    matrix = json.loads(installed_consumer_matrix_json())
    selected = json.loads(installed_consumer_matrix_json("axolotl"))

    assert [row["repository"] for row in matrix["include"]] == [
        "marin-community/MarinSkyRL",
        "marin-community/axolotl",
    ]
    assert selected["include"] == [
        {
            "base_branch": "main",
            "name": "axolotl",
            "repository": "marin-community/axolotl",
            "repository_name": "axolotl",
        }
    ]


def test_bootstrap_generates_manifest_when_consumer_already_pins_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _consumer_repository(tmp_path)
    _install_fake_uvx(tmp_path, monkeypatch, revision=OLD_REVISION)

    update = generate_marin_style_update(
        repo_root=repository,
        base_branch="main",
        revision=OLD_REVISION,
        manifest_mode=ManifestMode.BOOTSTRAP,
    )

    assert update.changed_files == (".agents/marin-style/manifest.json",)


def test_generate_bootstrap_discovers_pins_and_managed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _consumer_repository(tmp_path)
    _install_fake_uvx(tmp_path, monkeypatch)

    update = generate_marin_style_update(
        repo_root=repository,
        base_branch="main",
        revision=NEW_REVISION,
        manifest_mode=ManifestMode.BOOTSTRAP,
    )

    assert update.changed_files == (
        ".agents/marin-style/manifest.json",
        ".agents/skills/consult-echo/scripts/echo.py",
        ".github/workflows/marin-ci.yaml",
        "infra/pre-commit.py",
    )
    assert NEW_REVISION in (repository / "infra/pre-commit.py").read_text()
    assert NEW_REVISION in (repository / ".github/workflows/marin-ci.yaml").read_text()
    assert (repository / "README.md").read_text() == "consumer\n"


def test_generate_rejects_unowned_worktree_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _consumer_repository(tmp_path)
    _install_fake_uvx(tmp_path, monkeypatch)
    (repository / "README.md").write_text("unrelated\n")

    with pytest.raises(ValueError):
        generate_marin_style_update(
            repo_root=repository,
            base_branch="main",
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
