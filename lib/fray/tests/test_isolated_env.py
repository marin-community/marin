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

import os
import subprocess
import time

import pytest
from fray.isolated_env import TemporaryVenv


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
