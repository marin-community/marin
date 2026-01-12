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

"""Tests for the Fluster worker CLI."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from fluster.cluster.worker.main import cli


def test_cli_help():
    """Test CLI help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Fluster Worker" in result.output
    assert "serve" in result.output
    assert "cleanup" in result.output


def test_serve_help():
    """Test serve command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the Fluster worker service" in result.output
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--cache-dir" in result.output
    assert "--registry" in result.output


def test_cleanup_help():
    """Test cleanup command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["cleanup", "--help"])
    assert result.exit_code == 0
    assert "Clean up cached bundles" in result.output
    assert "--cache-dir" in result.output


def test_cleanup_removes_cache_directory():
    """Test cleanup command removes cache directory."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "fluster-cache"
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


def test_serve_requires_registry():
    """Test serve command requires --registry argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve"])
    assert result.exit_code != 0
    assert "registry" in result.output.lower() or "required" in result.output.lower()
