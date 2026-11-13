# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CLOCK_CONFTEST = """\
import signal
import threading
import time

import pytest


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.signal_deadline = None
        self.timers = []

    def setitimer(self, which, seconds, interval=0):
        self.signal_deadline = self.now + seconds if seconds else None
        return (0.0, 0.0)

    def advance(self, seconds):
        target = self.now + seconds
        while True:
            deadlines = [timer.deadline for timer in self.timers if timer.deadline is not None]
            if self.signal_deadline is not None:
                deadlines.append(self.signal_deadline)
            if not deadlines or min(deadlines) > target:
                self.now = target
                return
            self.now = min(deadlines)
            if self.signal_deadline == self.now:
                self.signal_deadline = None
                signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
            for timer in self.timers:
                if timer.deadline == self.now:
                    timer.deadline = None
                    timer.function(*timer.args, **timer.kwargs)


_clock = Clock()


class Timer:
    def __init__(self, interval, function, args=(), kwargs=None):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs or {}
        self.deadline = None
        _clock.timers.append(self)

    def start(self):
        self.deadline = _clock.now + self.interval

    def cancel(self):
        self.deadline = None

    def join(self):
        pass


# Replace only clock/timer I/O; pytest-timeout and the shared guard run normally.
signal.setitimer = _clock.setitimer
threading.Timer = Timer
time.time = lambda: _clock.now


@pytest.fixture
def clock():
    return _clock
"""


@pytest.fixture
def timeout_workspace(tmp_path: Path) -> Path:
    for relative in (
        "pyproject.toml",
        "lib/levanter/pyproject.toml",
        "infra/ci/__init__.py",
        "infra/ci/pytest_timeout_guard.py",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    (tmp_path / "sentinels").mkdir()
    return tmp_path


def run_probe(workspace: Path, configuration: str, body: str, cwd: str = "root") -> subprocess.CompletedProcess[str]:
    test_dir = workspace / ("tests" if configuration == "root" else "lib/levanter/tests")
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "conftest.py").write_text(CLOCK_CONFTEST)
    test_file = test_dir / "test_timeout_probe.py"
    test_file.write_text(
        "from pathlib import Path\nimport pytest\n"
        f"OUTPUT = Path({str(workspace / 'sentinels')!r})\n\n" + textwrap.dedent(body)
    )
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        environment.pop(name, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_timeout",
            "-p",
            "xdist.plugin",
            "-n",
            "0",
            "-q",
            f"--junitxml={workspace / 'probe.xml'}",
            str(test_file),
        ],
        cwd=workspace if cwd == "root" else workspace / "lib/levanter",
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("configuration", ["root", "levanter"])
@pytest.mark.parametrize("cwd", ["root", "package"])
def test_safe_selection_does_not_execute_excluded_tests(timeout_workspace: Path, configuration: str, cwd: str) -> None:
    markers = ["slow", "integration", "data_integration", "cluster", "requires_cluster", "docker", "manual"]
    if configuration == "levanter":
        markers.append("torch")
    body = "def test_safe():\n    (OUTPUT / 'safe').touch()\n"
    for marker in markers:
        body += f"\n@pytest.mark.{marker}\ndef test_{marker}():\n    (OUTPUT / {marker!r}).touch()\n"

    result = run_probe(timeout_workspace, configuration, body, cwd)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in (timeout_workspace / "sentinels").iterdir()) == ["safe"]


@pytest.mark.parametrize("configuration", ["root", "levanter"])
def test_default_timeout_interrupts_test_before_its_side_effect(timeout_workspace: Path, configuration: str) -> None:
    result = run_probe(
        timeout_workspace,
        configuration,
        """\
        def test_too_long(clock):
            clock.advance(61)
            (OUTPUT / "escaped").touch()
        """,
    )

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    testcase = ET.parse(timeout_workspace / "probe.xml").find(".//testcase")
    assert testcase is not None and testcase.get("name") == "test_too_long"
    assert testcase.find("failure") is not None
    assert not (timeout_workspace / "sentinels/escaped").exists()


@pytest.mark.parametrize("configuration", ["root", "levanter"])
def test_long_test_exception_can_finish_after_default_deadline(timeout_workspace: Path, configuration: str) -> None:
    result = run_probe(
        timeout_workspace,
        configuration,
        """\
        @pytest.mark.timeout(120)
        def test_long_but_allowed(clock):
            clock.advance(90)
            (OUTPUT / "completed").touch()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (timeout_workspace / "sentinels/completed").exists()


@pytest.mark.parametrize("configuration", ["root", "levanter"])
def test_suppressed_signal_timeout_still_exits_process(timeout_workspace: Path, configuration: str) -> None:
    result = run_probe(
        timeout_workspace,
        configuration,
        """\
        def test_suppressed_timeout(clock):
            try:
                clock.advance(60)
            except pytest.fail.Exception:
                (OUTPUT / "signal-suppressed").touch()
            clock.advance(5)
            (OUTPUT / "escaped").touch()
        """,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert (timeout_workspace / "sentinels/signal-suppressed").exists()
    assert not (timeout_workspace / "sentinels/escaped").exists()


@pytest.mark.parametrize("configuration", ["root", "levanter"])
def test_session_budget_stops_before_starting_another_test(timeout_workspace: Path, configuration: str) -> None:
    result = run_probe(
        timeout_workspace,
        configuration,
        """\
        @pytest.mark.timeout(180)
        @pytest.mark.parametrize("index", range(4))
        def test_uses_budget(clock, index):
            clock.advance(151)
            (OUTPUT / str(index)).touch()

        def test_after_budget():
            (OUTPUT / "escaped").touch()
        """,
    )

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert sorted(path.name for path in (timeout_workspace / "sentinels").iterdir()) == ["0", "1", "2", "3"]
