# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BundleCreator."""

import io
import zipfile
from unittest.mock import patch

import pytest

from iris.cluster.client.bundle import BundleCreator


@pytest.fixture
def workspace(tmp_path):
    """Create a test workspace."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"cached")
    return tmp_path


def test_bundle_creator_uses_fallback_when_git_unavailable(workspace):
    with patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=None):
        creator = BundleCreator(workspace)
        bundle_bytes = creator.create_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/main.py" in names
        assert not any("__pycache__" in n for n in names)


def test_bundle_creator_uses_git_files_when_available(workspace):
    git_files = {workspace / "pyproject.toml", workspace / "src" / "main.py"}
    with patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=git_files):
        creator = BundleCreator(workspace)
        bundle_bytes = creator.create_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/main.py" in names
        assert not any("__pycache__" in n for n in names)
