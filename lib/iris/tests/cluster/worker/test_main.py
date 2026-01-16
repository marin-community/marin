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

"""Tests for the Iris worker CLI."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from iris.cluster.worker.main import cli


def test_cleanup_removes_cache_directory():
    """Test cleanup command removes cache directory."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "iris-cache"
        cache_dir.mkdir()
        test_file = cache_dir / "test.txt"
        test_file.write_text("test data")

        assert cache_dir.exists()

        result = runner.invoke(cli, ["cleanup", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "Removed cache directory" in result.output
        assert not cache_dir.exists()


def test_cleanup_handles_missing_directory():
    """Test cleanup command handles missing cache directory."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "nonexistent"
        assert not cache_dir.exists()

        result = runner.invoke(cli, ["cleanup", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "does not exist" in result.output
