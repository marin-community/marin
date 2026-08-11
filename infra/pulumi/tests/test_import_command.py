# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from iac import import_command


def test_import_manifest_must_stay_outside_repository(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(import_command, "REPOSITORY_ROOT", repository)

    with pytest.raises(ValueError, match="outside the repository"):
        import_command._require_local_manifest(repository / "import.json", must_exist=False)


def test_reviewed_import_manifest_must_be_owner_only(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = tmp_path / "import.json"
    manifest.write_text(json.dumps({"resources": []}))
    manifest.chmod(0o644)
    monkeypatch.setattr(import_command, "REPOSITORY_ROOT", repository)

    with pytest.raises(ValueError, match="mode 0600"):
        import_command._require_local_manifest(manifest, must_exist=True)
