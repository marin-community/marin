# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.hooks.nsys_main — the user-invoked Nsight Systems launch wrapper. iris the
scheduler knows nothing about it; the GPU image bakes `nsys` in. None of this runs nsys."""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from iris.cluster.client.job_info import set_job_info
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.hooks.nsys import NsysHook
from iris.hooks.nsys_main import (
    _supervise,
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
# Positional signature of iris.hooks.nsys_main.run, as main() calls it.
_RUN_PARAMS = ("tasks", "trace", "capture_range", "output_uri", "argv")
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


def test_main_argv_round_trips_into_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user (or multigpu --wrap) builds this argv and main() parses it into run()."""
    seen: dict[str, object] = {}
    monkeypatch.setattr("iris.hooks.nsys_main.run", lambda *a: seen.update(zip(_RUN_PARAMS, a, strict=True)))
    nsys_main(["--tasks", "0,7", "--trace", "cuda,nvtx", "--output-uri", OUT, "--capture-range", "--", "python", "x.py"])
    assert seen == {
        "tasks": "0,7",
        "trace": "cuda,nvtx",
        "capture_range": True,
        "output_uri": OUT,
        "argv": ["python", "x.py"],
    }


class _Execed(Exception):
    """Stands in for exec replacing the process, which never returns."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv


class _FakePopen:
    def __init__(self, returncode: int) -> None:
        self._returncode = returncode

    def send_signal(self, signum: int) -> None:
        raise AssertionError("nothing should signal an already-exited child")

    def wait(self) -> int:
        return self._returncode


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
    """Put this process at index 0 with an nsys on PATH, and return the workdir."""
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_WORKDIR", str(tmp_path))
    monkeypatch.setattr("iris.hooks.nsys_main.shutil.which", lambda _: "/usr/local/bin/nsys")
    return tmp_path


def _fake_supervise(returncode: int, write_report: bool):
    """Stand in for nsys, which writes <output>.nsys-rep once the child exits."""

    def _run(nsys_argv: Sequence[str], command: Sequence[str]) -> int:
        assert nsys_argv[1] == "profile"
        assert list(command) == CMD
        output = Path(nsys_argv[nsys_argv.index("-o") + 1])
        if write_report:
            output.with_name(output.name + ".nsys-rep").write_bytes(b"fake report")
        return returncode

    return _run


def test_selected_unit_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    monkeypatch.setattr("iris.hooks.nsys_main._supervise", _fake_supervise(0, write_report=True))
    destination = selected_task / "uploads"
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)
    assert excinfo.value.code == 0
    uploaded = list(destination.iterdir())
    assert len(uploaded) == 1
    assert uploaded[0].name.startswith("r00000-") and uploaded[0].name.endswith(".nsys-rep")
    assert uploaded[0].read_bytes() == b"fake report"
    assert os.environ["TMPDIR"] == str(selected_task / "nsys")  # /tmp is noexec


def test_run_uploads_to_the_default_when_output_uri_unset(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    destination = selected_task / "default-dest"
    monkeypatch.setattr("iris.hooks.nsys_main.default_output_uri", lambda: f"file://{destination}")
    monkeypatch.setattr("iris.hooks.nsys_main._supervise", _fake_supervise(0, write_report=True))
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=None, argv=CMD)
    assert excinfo.value.code == 0
    assert len(list(destination.iterdir())) == 1


def test_failing_command_still_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    """A crash is exactly when the profile is worth keeping."""
    monkeypatch.setattr("iris.hooks.nsys_main._supervise", _fake_supervise(7, write_report=True))
    destination = selected_task / "uploads"
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)
    assert excinfo.value.code == 7
    assert len(list(destination.iterdir())) == 1


def test_supervise_normalizes_a_signalled_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Popen.wait reports a SIGTERM'd child as -15; 128 + signum is the convention."""
    monkeypatch.setattr("subprocess.Popen", lambda argv: _FakePopen(-signal.SIGTERM))
    assert _supervise(["nsys", "profile"], ["true"]) == 143


def test_missing_report_surfaces_the_command_exit_code(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    monkeypatch.setattr("iris.hooks.nsys_main._supervise", _fake_supervise(3, write_report=False))
    destination = selected_task / "uploads"
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)
    assert excinfo.value.code == 3
    assert not destination.exists()


def test_missing_report_fails_even_when_the_command_succeeded(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path
) -> None:
    monkeypatch.setattr("iris.hooks.nsys_main._supervise", _fake_supervise(0, write_report=False))
    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{selected_task / 'up'}", argv=CMD)
    assert excinfo.value.code != 0


def test_hook_wrap_builds_the_command_the_entry_point_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The programmatic wrap emits exactly what nsys_main's parser accepts."""
    wrapped = NsysHook(output_uri=OUT, tasks="0,7", trace="cuda,nvtx", capture_range=True).wrap(["python", "x.py"])
    assert wrapped[:3] == ["python", "-m", "iris.hooks.nsys_main"]
    seen: dict[str, object] = {}
    monkeypatch.setattr("iris.hooks.nsys_main.run", lambda *a: seen.update(zip(_RUN_PARAMS, a, strict=True)))
    nsys_main(wrapped[3:])
    assert seen == {
        "tasks": "0,7",
        "trace": "cuda,nvtx",
        "capture_range": True,
        "output_uri": OUT,
        "argv": ["python", "x.py"],
    }


def test_hook_omits_output_uri_when_unset() -> None:
    """Unset output drops the flag, so the wrapper defaults it from the task's env."""
    assert "--output-uri" not in NsysHook().wrap(["python", "x.py"])
