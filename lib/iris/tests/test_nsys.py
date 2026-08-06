# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.hooks.nsys_main — the user-invoked Nsight Systems launch wrapper. iris the
scheduler knows nothing about it; the GPU image bakes `nsys` in. None of this runs nsys."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from iris.cluster.client.job_info import set_job_info
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.hooks.nsys import NsysHook
from iris.hooks.nsys_main import (
    build_nsys_argv,
    default_output_uri,
    report_path,
    resolve_nsys_bin,
    run,
    selection_index,
    should_profile,
)
from iris.hooks.nsys_main import main as nsys_main

CMD = ["python", "train.py", "--steps", "10"]
OUT = "s3://bucket/tmp/ttl=30d/nsys"


@pytest.fixture(autouse=True)
def clear_job_info_cache() -> Iterator[None]:
    """Drop the memoized JobInfo (a ContextVar) around every test in this module."""
    set_job_info(None)
    yield
    set_job_info(None)


@pytest.mark.parametrize(
    ("tasks", "index", "selected"),
    [
        ("first", 0, True),
        ("first", 1, False),
        ("all", 127, True),
        ("0,7", 7, True),
        ("0,7", 6, False),
    ],
)
def test_selector_picks_units(tasks: str, index: int, selected: bool) -> None:
    assert should_profile(tasks, index) is selected


def test_unparseable_spec_raises() -> None:
    with pytest.raises(ValueError, match="comma-separated list"):
        should_profile("every-other", 0)


def test_selection_index_uses_task_index_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # No multigpu env → node scope: the whole task is the unit, keyed on its task index.
    monkeypatch.delenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, raising=False)
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/3")
    monkeypatch.setenv("IRIS_NUM_TASKS", "8")
    assert selection_index() == 3


def test_selection_index_uses_process_index_under_multigpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # A multigpu child (process scope) selects on its own global rank, not the task index.
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/3")
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "5")
    assert selection_index() == 5


def test_selection_index_requires_a_task_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, raising=False)
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    with pytest.raises(RuntimeError, match="no iris job context"):
        selection_index()


def test_default_output_uri_keys_on_the_job_and_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolved from the task's own MARIN_PREFIX (right cluster even under federation) and
    keyed on the job so a run's reports are findable and self-expiring."""
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("IRIS_TASK_ID", "/rav/train-42/0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "1")
    assert default_output_uri() == "s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/rav/train-42"


def test_build_nsys_argv_matches_what_the_container_allows() -> None:
    # perf_event_paranoid=4 in a task pod blocks sampling and context switches.
    out = Path("/app/nsys/r00000-h")
    plain = build_nsys_argv("/n/nsys", out, "cuda,nvtx", capture_range=False)
    assert {"--sample=none", "--cpuctxsw=none"} <= set(plain)
    assert not any(a.startswith("--capture-range") for a in plain)
    ranged = build_nsys_argv("/n/nsys", out, "cuda,nvtx", capture_range=True)
    assert {"--capture-range=cudaProfilerApi", "--capture-range-end=stop"} <= set(ranged)


def test_report_path_carries_the_selection_index() -> None:
    assert report_path(Path("/app/nsys"), 7).name.startswith("r00007-")


def test_resolve_nsys_bin_prefers_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iris.hooks.nsys_main.shutil.which", lambda _: "/usr/local/bin/nsys")
    assert resolve_nsys_bin() == "/usr/local/bin/nsys"


def test_resolve_nsys_bin_requires_a_gpu_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iris.hooks.nsys_main.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="no nsys on PATH"):
        resolve_nsys_bin()


def _child(exit_code: int = 0, marker: Path | None = None) -> list[str]:
    marker_write = ""
    if marker is not None:
        marker_write = f"pathlib.Path({str(marker)!r}).write_text('ran');"
    return [sys.executable, "-c", f"import pathlib,sys;{marker_write}sys.exit({exit_code})"]


class _Execed(Exception):
    """Stands in for exec replacing the process, which never returns."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv


@pytest.fixture
def fake_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    def _exec(file: str, args: Sequence[str]) -> NoReturn:
        raise _Execed(list(args))

    monkeypatch.setattr("os.execvp", _exec)


def test_unselected_unit_execs_command_unwrapped(monkeypatch: pytest.MonkeyPatch, fake_exec: None) -> None:
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/1")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    with pytest.raises(_Execed) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=OUT, argv=CMD)
    assert excinfo.value.argv == CMD


@pytest.fixture
def selected_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Put this process at index 0 and return its workdir."""
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_WORKDIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_nsys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Install a deterministic external nsys executable on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "nsys"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "if signal_number := os.environ.get('FAKE_NSYS_SIGNAL'):\n"
        "    os.kill(os.getpid(), int(signal_number))\n"
        "output_index = sys.argv.index('-o') + 1\n"
        "command_index = output_index + 1\n"
        "while command_index < len(sys.argv) and sys.argv[command_index].startswith('--capture-range'):\n"
        "    command_index += 1\n"
        "returncode = subprocess.call(sys.argv[command_index:])\n"
        "if os.environ.get('FAKE_NSYS_WRITE_REPORT') == '1':\n"
        "    pathlib.Path(sys.argv[output_index] + '.nsys-rep').write_bytes(b'fake report')\n"
        "sys.exit(returncode)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return executable


def test_main_argv_runs_profiled_command(monkeypatch: pytest.MonkeyPatch, selected_task: Path, fake_nsys: Path) -> None:
    destination = selected_task / "main-uploads"
    marker = selected_task / "main-child-ran"
    monkeypatch.setenv("FAKE_NSYS_WRITE_REPORT", "1")

    with pytest.raises(SystemExit) as excinfo:
        nsys_main(
            [
                "--tasks",
                "0,7",
                "--trace",
                "cuda,nvtx",
                "--output-uri",
                f"file://{destination}",
                "--capture-range",
                "--",
                *_child(marker=marker),
            ]
        )

    assert excinfo.value.code == 0
    assert marker.read_text() == "ran"
    assert len(list(destination.glob("*.nsys-rep"))) == 1


def test_selected_unit_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_task: Path, fake_nsys: Path) -> None:
    monkeypatch.setenv("FAKE_NSYS_WRITE_REPORT", "1")
    destination = selected_task / "uploads"
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=_child())
    assert excinfo.value.code == 0
    uploaded = list(destination.iterdir())
    assert len(uploaded) == 1
    assert uploaded[0].name.startswith("r00000-") and uploaded[0].name.endswith(".nsys-rep")
    assert uploaded[0].read_bytes() == b"fake report"
    assert os.environ["TMPDIR"] == str(selected_task / "nsys")  # /tmp is noexec


def test_run_uploads_to_the_default_when_output_uri_unset(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path, fake_nsys: Path
) -> None:
    monkeypatch.setenv("MARIN_PREFIX", f"file://{selected_task / 'cluster'}/marin")
    monkeypatch.setenv("FAKE_NSYS_WRITE_REPORT", "1")
    destination = Path(default_output_uri().removeprefix("file://"))
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=None, argv=_child())
    assert excinfo.value.code == 0
    assert len(list(destination.iterdir())) == 1


def test_failing_command_still_uploads_its_report(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path, fake_nsys: Path
) -> None:
    """A crash is exactly when the profile is worth keeping."""
    monkeypatch.setenv("FAKE_NSYS_WRITE_REPORT", "1")
    destination = selected_task / "uploads"
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=_child(7))
    assert excinfo.value.code == 7
    assert len(list(destination.iterdir())) == 1


def test_profiled_nsys_signal_is_normalized(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path, fake_nsys: Path
) -> None:
    monkeypatch.setenv("FAKE_NSYS_SIGNAL", str(signal.SIGTERM))
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{selected_task / 'up'}", argv=_child())
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_missing_report_surfaces_the_command_exit_code(selected_task: Path, fake_nsys: Path) -> None:
    destination = selected_task / "uploads"
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=_child(3))
    assert excinfo.value.code == 3
    assert not destination.exists()


def test_missing_report_fails_even_when_the_command_succeeded(selected_task: Path, fake_nsys: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{selected_task / 'up'}", argv=_child())
    assert excinfo.value.code != 0


def test_hook_wrap_command_runs_through_entry_point(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path, fake_nsys: Path
) -> None:
    destination = selected_task / "hook-uploads"
    marker = selected_task / "hook-child-ran"
    monkeypatch.setenv("FAKE_NSYS_WRITE_REPORT", "1")
    wrapped = NsysHook(
        output_uri=f"file://{destination}",
        tasks="0,7",
        trace="cuda,nvtx",
        capture_range=True,
    ).wrap(_child(marker=marker))

    result = subprocess.run(wrapped, env=os.environ.copy(), check=False)

    assert result.returncode == 0
    assert marker.read_text() == "ran"
    assert len(list(destination.glob("*.nsys-rep"))) == 1


def test_hook_omits_output_uri_when_unset() -> None:
    """Unset output drops the flag, so the wrapper defaults it from the task's env."""
    assert "--output-uri" not in NsysHook().wrap(["python", "x.py"])
