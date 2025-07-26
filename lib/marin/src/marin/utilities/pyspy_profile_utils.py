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

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import signal
import subprocess
from collections.abc import Iterator

logger = logging.getLogger(__name__)


# NOTE: from https://github.com/benfred/py-spy/issues/531#issue-1423197590
@contextmanager
def pyspy_profiler(
    output_file: str | Path,
    *,
    sudo: bool = True,
    output_format: str = "chrometrace",
    duration_seconds: int | None = None,
    rate_per_second: int = 100,
    native: bool = False,
    subprocesses: bool = False,
    block_output: bool = True,
) -> Iterator[None]:
    """
    Runs py-spy to profile the current process, writing output to `output_file`.
    Requires uvx to be installed.

    Args:
        output_file: Path to write the py-spy output file.
        sudo: Whether to run py-spy with sudo.
        output_format: Format of the output file.
        duration_seconds: Duration to run py-spy for. If None, runs until the context is exited.
        rate_per_second: Sampling rate per second.
        native: Whether to include native frames (not supported on macOS)
        subprocesses: Whether to profile subprocesses as well.
        block_output: Whether to block py-spy output (both stdout/stderr).
    """
    args = ["uvx", "py-spy", "record", "-o", str(output_file), "-f", output_format, "--pid", str(os.getpid())]
    if native:
        args.append("--native")
    if subprocesses:
        args.append("--subprocesses")
    args += ["--rate", str(rate_per_second)]
    if duration_seconds is not None:
        args += ["--duration", str(duration_seconds)]
    if sudo:
        args = ["sudo", *args]
    p = subprocess.Popen(
        args, stdout=subprocess.DEVNULL if block_output else None, stderr=subprocess.DEVNULL if block_output else None
    )
    yield
    p.send_signal(signal.SIGINT)
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        logger.warning(f"py-spy process pid={p.pid} did not terminate in time after SIGINT, killing it")
        p.kill()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"py-spy process pid={p.pid} did not terminate in time after kill, giving up ...")
