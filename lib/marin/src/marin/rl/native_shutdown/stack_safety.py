# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact safety definitions extracted from qualified0f2 controls; standalone observer only."""
import dataclasses
import os
import signal
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Identity:
    pid: int
    starttime: int
    pgrp: int


def parse_stat(text):
    left, right = text.find("("), text.rfind(")")
    if left < 1 or right <= left:
        raise ValueError("Invalid proc stat")
    pid = int(text[:left].strip())
    fields = text[right + 1 :].split()
    return Identity(pid, int(fields[19]), int(fields[2]))


def identity(pid):
    return parse_stat(Path(f"/proc/{pid}/stat").read_text())


def proc_status(pid):
    selected = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"State", "TracerPid"}:
            selected[key] = value.strip()
    return selected


def direct_children():
    """Only this standalone control supervisor's children, across its threads."""
    children = set()
    for task in Path("/proc/self/task").iterdir():
        try:
            children.update(int(x) for x in (task / "children").read_text().split())
        except FileNotFoundError:
            continue
    return children


def reap_debugger_descendants(baseline_children, target_pid, deadline):
    """Subreaper recovers debugger helpers even if LLDB-server changed session.

    Only valid in the standalone, sequential CPU-control supervisor. It owns
    every newly created child; never call from a Ray worker/shared supervisor.
    """
    reaped = []
    while True:
        owned = direct_children() - baseline_children - {target_pid}
        if not owned:
            return reaped
        for pid in sorted(owned):
            try:
                expected = identity(pid)
                descriptor = os.pidfd_open(pid, 0)
            except ProcessLookupError:
                continue
            except FileNotFoundError:
                continue
            try:
                if identity(pid) != expected:
                    raise RuntimeError("Debugger descendant PID binding changed")
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(descriptor)
            try:
                waited, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if waited:
                reaped.append({"pid": waited, "status": status, "starttime": expected.starttime})
        if time.monotonic() >= deadline:
            raise RuntimeError("Owned debugger descendants remain after cleanup budget")
        time.sleep(0.002)


class LinuxBinding:
    """Pin signals to the original process, and guard every numeric-PID attach."""

    def __init__(self, expected):
        self.expected = expected
        if identity(expected.pid) != expected:
            raise RuntimeError("Target PID/starttime binding changed before pidfd_open")
        self.fd = os.pidfd_open(expected.pid, 0)
        try:
            self.verify()
        except BaseException:
            os.close(self.fd)
            raise

    def verify(self):
        if identity(self.expected.pid) != self.expected:
            raise RuntimeError("Target PID/starttime binding changed")
        return proc_status(self.expected.pid)

    def resume(self):
        # Iris LocalProfileDispatch resumes in finally; pidfd avoids PID reuse.
        try:
            signal.pidfd_send_signal(self.fd, signal.SIGCONT)
            return "sent_to_bound_pidfd"
        except ProcessLookupError:
            return "target_exited"

    def kill(self):
        # Do not use Popen.kill here: its pre-signal poll can reap the reserved
        # PID while a sampler is still preparing numeric debugger attach.
        try:
            signal.pidfd_send_signal(self.fd, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def close(self):
        os.close(self.fd)
