# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private userspace networking for admission containers on host-network Iris pods."""

import os
import select
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def private_network(container: str, logs: Path):
    """Connect an existing network-none Docker container without changing host routes."""
    pid = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Pid}}", container], text=True).strip()
    if os.stat(f"/proc/{pid}/ns/net").st_ino == os.stat("/proc/self/ns/net").st_ino:
        raise RuntimeError("Admission task must not share the worker's network namespace")
    reader, writer = os.pipe()
    with logs.open("wb") as stream:
        process = subprocess.Popen(
            [
                "slirp4netns",
                "--configure",
                "--mtu=65520",
                "--disable-host-loopback",
                f"--ready-fd={writer}",
                pid,
                "tap0",
            ],
            pass_fds=(writer,),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        os.close(writer)
        try:
            if not select.select([reader], [], [], 30)[0] or os.read(reader, 1) != b"1":
                raise RuntimeError(f"Private admission network did not become ready; see {logs}")
            subprocess.run(
                ["docker", "exec", container, "sh", "-c", "printf 'nameserver 10.0.2.3\\n' > /etc/resolv.conf"],
                check=True,
            )
            yield
        finally:
            os.close(reader)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
