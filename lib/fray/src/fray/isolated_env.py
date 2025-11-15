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

"""Utilities for managing isolated Python virtual environments."""

import ctypes
import ctypes.util
import logging
import os
import signal
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)


def _set_pdeathsig_preexec():
    """Use prctl(PR_SET_PDEATHSIG, SIGKILL) to kill subprocess if parent dies."""
    PR_SET_PDEATHSIG = 1
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            errno = ctypes.get_errno()
            logger.warning(f"Failed to set parent death signal: errno {errno}")
    except Exception as e:
        logger.debug(f"Could not set parent death signal: {e}")


def _signal_process(process: subprocess.Popen, sig: int) -> None:
    if sys.platform == "win32":
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()
    else:
        os.killpg(process.pid, sig)


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        _signal_process(process, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


class TemporaryVenv:
    """Context manager for temporary virtual environments with automatic cleanup.

    Creates an isolated Python virtual environment using uv, installs packages,
    and provides methods to run commands within that environment. Automatically
    tracks and terminates all spawned processes on exit.

    Example:
        with TemporaryVenv(
            pip_install_args=["--prerelease=allow", "vllm-tpu"]
        ) as venv:
            venv.run(["vllm", "--help"])
            server = venv.run_async(["vllm", "serve", "model"])
    """

    def __init__(
        self,
        *,
        pip_install_args: list[str] | None = None,
        venv_args: list[str] | None = None,
        prefix: str = "temp_venv_",
    ):
        """Initialize temporary venv context manager.

        Args:
            pip_install_args: Arguments to pass to `uv pip install` (e.g.,
                ["--prerelease=allow", "vllm-tpu", "package==1.0.0"]).
                These are passed directly to `uv pip install --python <venv_python> <args...>`
            prefix: Prefix for temporary directory name
        """
        self._pip_install_args = pip_install_args
        self._prefix = prefix
        self._venv_args = venv_args

        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._processes: list[subprocess.Popen] = []

    @property
    def venv_path(self) -> str:
        """Path to the virtual environment directory."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before accessing venv_path")
        return self._temp_dir.name

    @property
    def python_path(self) -> str:
        """Path to the Python binary in the venv (e.g., venv/bin/python)."""
        return os.path.join(self.venv_path, "bin", "python")

    @property
    def bin_path(self) -> str:
        """Path to the bin directory (e.g., venv/bin)."""
        return os.path.join(self.venv_path, "bin")

    def __enter__(self) -> "TemporaryVenv":
        """Create venv, install packages, return self."""
        if self._temp_dir:
            raise RuntimeError("TemporaryVenv cannot be entered twice")

        self._temp_dir = tempfile.TemporaryDirectory(prefix=self._prefix)

        py_version = sys.version_info
        python_spec = f"{py_version.major}.{py_version.minor}"

        logger.info(f"Creating temporary venv at {self.venv_path} with Python {python_spec}")

        venv_args = []
        if self._venv_args:
            venv_args = self._venv_args

        subprocess.check_call(
            [
                "uv",
                "venv",
                f"--python={python_spec}",
                "--python-preference=only-managed",
                "--no-project",
                "--clear",
                *venv_args,
                self.venv_path,
            ]
        )

        if self._pip_install_args:
            logger.info(f"Installing packages: {' '.join(self._pip_install_args)}")
            venv_env = self.get_env()
            subprocess.check_call(
                ["uv", "pip", "install", "--python", self.python_path, *self._pip_install_args],
                env=venv_env,
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Kill all tracked processes and clean up venv directory."""
        try:
            self._temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")

        for process in self._processes:
            if process.poll() is not None:
                continue

            try:
                _terminate_process(process)
            except (ProcessLookupError, OSError):
                logger.debug(f"Process {process.pid} already terminated")

    def get_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Get environment dict with venv activation."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before calling get_env()")

        env = (base_env or os.environ).copy()
        env["VIRTUAL_ENV"] = self.venv_path
        env["PATH"] = f"{self.bin_path}:{env['PATH']}"

        bad_env_vars = ("TPU_LIBRARY_PATH", "PYTHONPATH")
        for var in bad_env_vars:
            env.pop(var, None)

        for var in list(env.keys()):
            if var.startswith("RAY_") or var.startswith("CONDA_"):
                env.pop(var, None)

        env["PYTHONNOUSERSITE"] = "1"

        # Isolate library paths to prevent system libraries (like libtpu) from leaking in
        # Prepend venv lib to LD_LIBRARY_PATH to ensure venv packages take precedence
        venv_lib = os.path.join(self.venv_path, "lib")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{venv_lib}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = venv_lib
        return env

    def run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Run a command within the venv and wait for completion."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before calling run()")

        if env is None:
            env = self.get_env()

        logger.info("Running %s", " ".join(cmd))
        return subprocess.run(cmd, env=env, check=check, **kwargs)

    def run_async(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> subprocess.Popen:
        """Start a command within the venv without waiting."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before calling run_async()")

        if env is None:
            env = self.get_env()

        # Unix: use process groups + parent death signal for cleanup
        if sys.platform != "win32":
            if "start_new_session" not in kwargs:
                kwargs["start_new_session"] = True
            if "preexec_fn" not in kwargs:
                kwargs["preexec_fn"] = _set_pdeathsig_preexec

        logger.info("Running %s", " ".join(cmd))
        process = subprocess.Popen(cmd, env=env, **kwargs)
        self._processes.append(process)
        return process
