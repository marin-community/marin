# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import time

import pytest
from fray.v1.isolated_env import TemporaryVenv


def test_venv_creation():
    """Test basic venv creation and cleanup."""
    venv_path = None
    with TemporaryVenv() as venv:
        venv_path = venv.venv_path
        assert os.path.exists(venv_path)
        assert os.path.exists(venv.python_path)
        assert os.path.exists(venv.bin_path)
        assert venv.python_path == os.path.join(venv_path, "bin", "python")
        assert venv.bin_path == os.path.join(venv_path, "bin")
    assert not os.path.exists(venv_path)


def test_environment_with_base():
    with TemporaryVenv() as venv:
        base_env = {"CUSTOM_VAR": "value", "PATH": "/custom/path"}
        env = venv.get_env(base_env=base_env)
        assert env["VIRTUAL_ENV"] == venv.venv_path
        assert env["CUSTOM_VAR"] == "value"
        assert venv.bin_path in env["PATH"]
        assert "/custom/path" in env["PATH"]


def test_run_blocking():
    with TemporaryVenv() as venv:
        result = venv.run([venv.python_path, "--version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Python" in result.stdout


def test_run_blocking_with_error():
    with TemporaryVenv() as venv:
        with pytest.raises(subprocess.CalledProcessError):
            venv.run([venv.python_path, "-c", "import sys; sys.exit(1)"])


def test_run_blocking_no_check():
    with TemporaryVenv() as venv:
        result = venv.run([venv.python_path, "-c", "import sys; sys.exit(1)"], check=False)
        assert result.returncode == 1


def test_package_installation():
    with TemporaryVenv(pip_install_args=["six"]) as venv:
        result = venv.run(
            [venv.python_path, "-c", "import six; print(six.__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0


def test_multiple_processes_cleanup():
    venv = TemporaryVenv()
    processes = []
    with venv:
        for _ in range(3):
            proc = venv.run_async([venv.python_path, "-c", "import time; time.sleep(100)"])
            processes.append(proc)

    # wait for processes to terminate.
    time.sleep(0.5)

    for proc in processes:
        print(proc, proc.poll())
        assert proc.poll() is not None


def test_workspace_copy_basic(tmp_path):
    """Test workspace copying without pyproject.toml."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test.py").write_text('print("hello")')
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "nested.py").write_text('print("nested")')

    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)

    with TemporaryVenv(workspace=str(workspace)) as venv:
        assert os.path.exists(os.path.join(venv.workspace_path, "test.py"))
        assert os.path.exists(os.path.join(venv.workspace_path, "subdir", "nested.py"))
        assert os.path.exists(os.path.join(venv.workspace_path, ".venv"))
        assert venv.venv_path == os.path.join(venv.workspace_path, ".venv")
        result = venv.run(["python", "test.py"], capture_output=True, text=True)
        assert "hello" in result.stdout


def test_workspace_with_extras(tmp_path):
    """Test workspace extras installation."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    (workspace / "pyproject.toml").write_text(
        """
[project]
name = "testpkg"
version = "0.1.0"
dependencies = []

[project.optional-dependencies]
dev = ["six"]
"""
    )

    pkg_dir = workspace / "testpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('__version__ = "0.1.0"')
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)

    with TemporaryVenv(workspace=str(workspace), extras=["dev"], pip_install_args=["regex"]) as venv:
        result = venv.run(
            ["python", "-c", "import testpkg; print(testpkg.__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

        result = venv.run(
            ["python", "-c", "import six; print(six.__version__)"],
        )
        assert result.returncode == 0

        result = venv.run(
            ["python", "-c", "import regex; print(regex.__version__)"],
        )
        assert result.returncode == 0
