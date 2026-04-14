# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`rigging.log_setup`, focused on fault-handler installation.

Uses subprocess isolation for the SIGSEGV test so the crash doesn't take the
test runner with it, and because ``faulthandler.enable()`` is inherently
process-global.
"""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

from rigging import log_setup


def test_install_fault_handler_is_idempotent():
    # First call wins; subsequent calls should just return True.
    assert log_setup.install_fault_handler() is True
    assert log_setup.install_fault_handler() is True


def test_install_fault_handler_respects_opt_out(monkeypatch):
    # Reset module-level state so the env-var opt-out can actually take effect.
    monkeypatch.setattr(log_setup, "_fault_handler_installed", False)
    monkeypatch.setenv(log_setup.FAULTHANDLER_DISABLE_ENV, "1")
    assert log_setup.install_fault_handler() is False


def test_configure_logging_installs_fault_handler(monkeypatch):
    monkeypatch.setattr(log_setup, "_fault_handler_installed", False)
    # Ensure the opt-out is unset so configure_logging installs.
    monkeypatch.delenv(log_setup.FAULTHANDLER_DISABLE_ENV, raising=False)
    log_setup.configure_logging()
    assert log_setup._fault_handler_installed is True


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSEGV semantics differ on Windows")
def test_fault_handler_dumps_traceback_on_sigsegv(tmp_path):
    """End-to-end: a child that calls configure_logging then segfaults must
    leave a Python traceback on stderr. Without faulthandler the exit would
    be silent (return code -11 / 139 with no diagnostic)."""
    script = textwrap.dedent(
        """
        import os, signal
        from rigging.log_setup import configure_logging
        configure_logging()
        # Deliver SIGSEGV with default behavior. faulthandler dumps first.
        os.kill(os.getpid(), signal.SIGSEGV)
        """
    )
    script_path = tmp_path / "crash.py"
    script_path.write_text(script)

    env = os.environ.copy()
    env.pop(log_setup.FAULTHANDLER_DISABLE_ENV, None)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        env=env,
        timeout=30,
    )

    # Exit via signal → negative returncode on POSIX.
    assert proc.returncode != 0
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert "Fatal Python error" in stderr or "Segmentation fault" in stderr, (
        f"expected faulthandler traceback in stderr, got:\n{stderr}"
    )


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 unavailable")
def test_fault_handler_sigusr1_dump(tmp_path):
    """SIGUSR1 should trigger an on-demand thread dump without killing the
    process."""
    script = textwrap.dedent(
        """
        import os, signal, sys, time
        from rigging.log_setup import configure_logging
        configure_logging()
        os.kill(os.getpid(), signal.SIGUSR1)
        # Give the handler a beat to print, then exit cleanly.
        sys.stderr.flush()
        time.sleep(0.2)
        """
    )
    script_path = tmp_path / "dump.py"
    script_path.write_text(script)

    env = os.environ.copy()
    env.pop(log_setup.FAULTHANDLER_DISABLE_ENV, None)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        env=env,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    # faulthandler.register's dump output starts with this banner.
    assert "Current thread" in stderr or "Stack" in stderr, (
        f"expected SIGUSR1 thread dump in stderr, got:\n{stderr}"
    )
