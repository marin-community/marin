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
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_serve_starts_server():
    """Test serve command starts server with correct configuration."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        uv_cache_dir = Path(tmpdir) / "uv"

        # Mock all the components
        with (
            patch("fluster.cluster.worker.main.BundleCache"),
            patch("fluster.cluster.worker.main.VenvCache") as mock_venv_cache,
            patch("fluster.cluster.worker.main.ImageBuilder"),
            patch("fluster.cluster.worker.main.DockerRuntime"),
            patch("fluster.cluster.worker.main.PortAllocator"),
            patch("fluster.cluster.worker.main.JobManager"),
            patch("fluster.cluster.worker.main.WorkerServiceImpl"),
            patch("fluster.cluster.worker.main.WorkerServer"),
            patch("fluster.cluster.worker.main.asyncio.run") as mock_asyncio_run,
        ):
            # Setup mocks
            mock_venv_instance = MagicMock()
            mock_venv_instance.ensure_permissions = MagicMock()
            mock_venv_cache.return_value = mock_venv_instance

            # Run serve command
            runner.invoke(
                cli,
                [
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9999",
                    "--cache-dir",
                    str(cache_dir),
                    "--uv-cache-dir",
                    str(uv_cache_dir),
                    "--registry",
                    "gcr.io/test",
                    "--max-concurrent-jobs",
                    "5",
                    "--port-range",
                    "40000-50000",
                    "--max-bundles",
                    "50",
                    "--max-venvs",
                    "10",
                    "--max-images",
                    "25",
                ],
                catch_exceptions=False,
            )

            # asyncio.run should have been called
            assert mock_asyncio_run.called


def test_serve_default_parameters():
    """Test serve command uses correct defaults."""
    runner = CliRunner()

    def run_coro_sync(coro):
        """Execute coroutine synchronously for testing."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with tempfile.TemporaryDirectory():
        with (
            patch("fluster.cluster.worker.main.BundleCache") as mock_bundle_cache,
            patch("fluster.cluster.worker.main.VenvCache") as mock_venv_cache,
            patch("fluster.cluster.worker.main.ImageBuilder") as mock_image_builder,
            patch("fluster.cluster.worker.main.DockerRuntime"),
            patch("fluster.cluster.worker.main.PortAllocator") as mock_port_allocator,
            patch("fluster.cluster.worker.main.JobManager") as mock_job_manager,
            patch("fluster.cluster.worker.main.WorkerServiceImpl"),
            patch("fluster.cluster.worker.main.WorkerServer") as mock_server,
            patch("fluster.cluster.worker.main.asyncio.run", side_effect=run_coro_sync),
        ):
            # Setup mocks
            mock_venv_instance = MagicMock()
            mock_venv_instance.ensure_permissions = MagicMock()
            mock_venv_cache.return_value = mock_venv_instance

            mock_server_instance = MagicMock()
            mock_server_instance.run_async = AsyncMock()
            mock_server.return_value = mock_server_instance

            # Run with minimal arguments
            runner.invoke(
                cli,
                [
                    "serve",
                    "--registry",
                    "gcr.io/test",
                ],
                catch_exceptions=False,
            )

            # Verify default parameters
            bundle_args = mock_bundle_cache.call_args
            assert bundle_args.kwargs["max_bundles"] == 100

            venv_args = mock_venv_cache.call_args
            assert venv_args.kwargs["max_entries"] == 20

            image_args = mock_image_builder.call_args
            assert image_args.kwargs["max_images"] == 50

            port_args = mock_port_allocator.call_args
            assert port_args[0][0] == (30000, 40000)

            manager_args = mock_job_manager.call_args
            assert manager_args.kwargs["max_concurrent_jobs"] == 10

            server_args = mock_server.call_args
            assert server_args[0][1] == "0.0.0.0"
            assert server_args[0][2] == 8080


def test_serve_shows_startup_messages():
    """Test serve command shows helpful startup messages."""
    runner = CliRunner()

    def run_coro_sync(coro):
        """Execute coroutine synchronously for testing."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"

        with (
            patch("fluster.cluster.worker.main.BundleCache"),
            patch("fluster.cluster.worker.main.VenvCache") as mock_venv_cache,
            patch("fluster.cluster.worker.main.ImageBuilder"),
            patch("fluster.cluster.worker.main.DockerRuntime"),
            patch("fluster.cluster.worker.main.PortAllocator"),
            patch("fluster.cluster.worker.main.JobManager"),
            patch("fluster.cluster.worker.main.WorkerServiceImpl"),
            patch("fluster.cluster.worker.main.WorkerServer") as mock_server,
            patch("fluster.cluster.worker.main.asyncio.run", side_effect=run_coro_sync),
        ):
            mock_venv_instance = MagicMock()
            mock_venv_instance.ensure_permissions = MagicMock()
            mock_venv_cache.return_value = mock_venv_instance

            mock_server_instance = MagicMock()
            mock_server_instance.run_async = AsyncMock()
            mock_server.return_value = mock_server_instance

            result = runner.invoke(
                cli,
                [
                    "serve",
                    "--cache-dir",
                    str(cache_dir),
                    "--registry",
                    "gcr.io/myproject",
                    "--max-concurrent-jobs",
                    "15",
                ],
                catch_exceptions=False,
            )

            # Verify startup messages
            assert "Starting Fluster worker" in result.output
            assert "gcr.io/myproject" in result.output
            assert str(cache_dir) in result.output
            assert "15" in result.output
