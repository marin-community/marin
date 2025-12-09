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
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Helper functions for terminating processes.
# On unix, we use process groups to collect and terminate subprocess + all children.
# On linux, we use PR_SET_PDEATHSIG to automatically terminate, on OSX + windows, best effort.


def _set_pdeathsig_preexec():
    """Use prctl(PR_SET_PDEATHSIG, SIGKILL) to kill subprocess if parent dies."""
    PR_SET_PDEATHSIG = 1
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            errno = ctypes.get_errno()
            logger.warning(f"Failed to set parent death signal: errno {errno}")
    except Exception as e:
        logger.info(f"Could not set parent death signal: {e}")


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except Exception as e:
        logger.info(f"Failed to terminate process {process.pid} -- already terminated? {e}")


class JobGroup:
    """Manages a group of processes with automatic cleanup.
    with JobGroup() as jobs:
        proc = jobs.run_async(["python", "worker.py"])
        jobs.run(["python", "setup.py"])
    """

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        """Initialize job group.

        Args:
            env: Base environment dict for commands. If None, uses os.environ.
            cwd: Working directory for commands. If None, uses current directory.
        """
        self._base_env = env
        self._cwd = cwd
        self._processes: list[subprocess.Popen] = []
        self._entered = False

    def _check_entered(self) -> None:
        if not self._entered:
            raise RuntimeError("JobGroup must be entered before calling run methods")

    def __enter__(self) -> "JobGroup":
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for process in self._processes:
            if process.poll() is not None:
                continue
            _terminate_process(process)

    def run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Run a command and wait for completion."""
        self._check_entered()

        if "cwd" not in kwargs and self._cwd:
            kwargs["cwd"] = self._cwd

        logger.info("Running %s", " ".join(cmd))
        return subprocess.run(cmd, env=env, check=check, **kwargs)

    def run_async(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> subprocess.Popen:
        """Start a command without waiting, track for cleanup."""
        self._check_entered()

        if "cwd" not in kwargs and self._cwd:
            kwargs["cwd"] = self._cwd

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
        workspace: str | None = None,
        pip_install_args: list[str] | None = None,
        extras: list[str] | None = None,
        venv_args: list[str] | None = None,
        prefix: str = "temp_venv_",
    ):
        """Initialize temporary venv context manager.

        Args:
            workspace: Path to workspace to copy (git-tracked files). If provided,
                workspace files are copied to the temp directory root, and a venv
                is created in .venv/ subdirectory.
            pip_install_args: Arguments to pass to `uv pip install` (e.g.,
                ["--prerelease=allow", "vllm-tpu", "package==1.0.0"]).
                These are passed directly to `uv pip install --python <venv_python> <args...>`
            extras: Workspace extras to install (e.g., ['tpu', 'eval']). Only used
                when workspace has a pyproject.toml.
            venv_args: Additional arguments to pass to `uv venv`
            prefix: Prefix for temporary directory name
        """
        self._workspace = workspace
        self._pip_install_args = pip_install_args
        self._extras = extras or []
        self._prefix = prefix
        self._venv_args = venv_args

        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._job_group: JobGroup | None = None

    @property
    def workspace_path(self) -> str:
        """Path to the workspace root directory (temp dir root)."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before accessing workspace_path")
        return self._temp_dir.name

    @property
    def venv_path(self) -> str:
        """Path to the virtual environment directory (.venv/ if workspace, else temp root)."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before accessing venv_path")
        if self._workspace:
            return os.path.join(self.workspace_path, ".venv")
        else:
            return self.workspace_path

    @property
    def python_path(self) -> str:
        """Path to the Python binary in the venv (e.g., venv/bin/python)."""
        return os.path.join(self.venv_path, "bin", "python")

    @property
    def bin_path(self) -> str:
        """Path to the bin directory (e.g., venv/bin)."""
        return os.path.join(self.venv_path, "bin")

    def _copy_workspace_files(self) -> None:
        """Copy git-tracked files from workspace to temp directory root."""
        workspace = Path(self._workspace)
        dest = Path(self.workspace_path)

        logger.info(f"Copying workspace {workspace} to {dest}")

        try:
            # Use git ls-files (respects .gitignore)
            git_files = subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=workspace,
                text=True,
            ).splitlines()

            for rel_path in git_files:
                src = workspace / rel_path
                dst = dest / rel_path

                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git failed, falling back to copytree: {e}")
            shutil.copytree(workspace, dest, dirs_exist_ok=True)

    def _install_packages(self) -> None:
        """Install workspace (editable + extras) and pip packages."""
        sync_args = ["--all-packages"]

        if self._extras:
            sync_args.extend(["--extra=" + ",".join(self._extras)])

        pyproject = Path(self.workspace_path) / "pyproject.toml"
        if self._workspace and pyproject.exists():
            subprocess.check_call(
                ["uv", "sync", "--python", self.python_path, *sync_args],
                cwd=self.workspace_path,
                env=self.get_env(),
            )

        install_args = []
        if self._pip_install_args:
            install_args.extend(self._pip_install_args)

        if install_args:
            logger.info(f"Installing packages: {' '.join(install_args)}")
            subprocess.check_call(
                ["uv", "pip", "install", "--python", self.python_path, *install_args],
                cwd=self.workspace_path,
                env=self.get_env(),
            )

    def __enter__(self) -> "TemporaryVenv":
        if self._temp_dir:
            raise RuntimeError("TemporaryVenv cannot be entered twice")

        self._temp_dir = tempfile.TemporaryDirectory(prefix=self._prefix)
        self._job_group = JobGroup(cwd=self._temp_dir.name)
        self._job_group.__enter__()

        if self._workspace:
            self._copy_workspace_files()

        py_version = sys.version_info
        python_spec = f"{py_version.major}.{py_version.minor}"

        logger.info(f"Creating temporary venv at {self.venv_path} with Python {python_spec}")

        venv_args = self._venv_args or []
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

        # Install packages (workspace + explicit packages)
        if self._workspace or self._pip_install_args:
            self._install_packages()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Kill all tracked processes and clean up venv directory."""
        self._job_group.__exit__(exc_type, exc_val, exc_tb)
        try:
            self._temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")

    def get_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Get environment dict with venv activation."""
        if not self._temp_dir:
            raise RuntimeError("TemporaryVenv must be entered before calling get_env()")

        env = (base_env or os.environ).copy()
        env["VIRTUAL_ENV"] = self.venv_path
        env["PATH"] = f"{self.bin_path}:{env['PATH']}"

        # todo, how to handle torch backend?
        env["TORCH_BACKEND"] = "cpu"

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
        if env is None:
            env = self.get_env()
        return self._job_group.run(cmd, check=check, env=env, **kwargs)

    def run_async(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> subprocess.Popen:
        """Start a command within the venv without waiting."""
        if env is None:
            env = self.get_env()
        return self._job_group.run_async(cmd, env=env, **kwargs)
