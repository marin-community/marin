# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import faulthandler
import os
import signal
import sys
import threading
from collections.abc import Generator

import pytest
from pytest_timeout import Settings, is_debugging

HARD_KILL_DELAY = 5.0
HARD_KILL_MESSAGE = "pytest-timeout signal handler did not stop the test; hard-killing process\n"
HARD_KILL_TIMER = pytest.StashKey[threading.Timer]()


def _hard_kill(item: pytest.Item, settings: Settings) -> None:
    if not settings.disable_debugger_detection and is_debugging():
        return

    try:
        capture_manager = item.config.pluginmanager.getplugin("capturemanager")
        if capture_manager is not None:
            capture_manager.suspend_global_capture(item)
        os.write(sys.stderr.fileno(), HARD_KILL_MESSAGE.encode())
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    finally:
        os._exit(1)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: Settings) -> Generator[None, None, None]:
    if settings.method == "signal" and hasattr(signal, "SIGALRM"):
        timer = threading.Timer(settings.timeout + HARD_KILL_DELAY, _hard_kill, (item, settings))
        timer.name = f"pytest hard timeout {item.nodeid}"
        timer.daemon = True
        item.stash[HARD_KILL_TIMER] = timer
        timer.start()

    yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_timeout_cancel_timer(item: pytest.Item) -> Generator[None, None, None]:
    try:
        yield
    finally:
        timer = item.stash.get(HARD_KILL_TIMER, None)
        if timer is not None:
            timer.cancel()
            timer.join()
