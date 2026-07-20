# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.cluster.hooks.nsys_main launch wrapper, the client-side entrypoint
wrapping that drives it, and the setup script that installs the profiler. None of this runs nsys."""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator, Sequence
from glob import glob
from pathlib import Path
from typing import NoReturn

import pytest
from iris.client.client import collect_hooks
from iris.cluster.client.job_info import set_job_info
from iris.cluster.hooks.nsys import NSYS_INSTALL_DIR, NSYS_VERSION, NsysHook, nsys_bin_glob
from iris.cluster.hooks.nsys_main import (
    _supervise,
    build_nsys_argv,
    default_output_uri,
    report_path,
    resolve_nsys_bin,
    run,
    should_profile,
    task_index_from_env,
    workdir,
)
from iris.cluster.hooks.nsys_main import (
    main as nsys_main,
)
from iris.cluster.types import EnvironmentSpec, ResourceSpec, gpu_device

CMD = ["python", "train.py", "--steps", "10"]
# Positional signature of iris.cluster.hooks.nsys_main.run, as main() calls it.
_RUN_PARAMS = ("tasks", "trace", "capture_range", "output_uri", "argv")
OUT = "s3://bucket/tmp/ttl=30d/nsys"


@pytest.fixture(autouse=True)
def clear_job_info_cache() -> Iterator[None]:
    """Drop the memoized JobInfo around every test in this module.

    ``get_job_info`` caches its env parse in a ContextVar, which outlives the
    ``monkeypatch.setenv`` that produced it. Clearing only on the way in would leave
    the last test's identity cached for whatever else shares this process, so a leaked
    JobInfo would silently reassign the nsys task selector's index.
    """
    set_job_info(None)
    yield
    set_job_info(None)


def _gpu_resources(count: int) -> ResourceSpec:
    return ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=gpu_device("H100", count))


@pytest.mark.parametrize(
    ("tasks", "task_index", "selected"),
    [
        ("first", 0, True),
        ("first", 1, False),
        ("all", 127, True),
        ("0,7", 7, True),
        ("0,7", 6, False),
    ],
)
def test_selector_picks_tasks(tasks: str, task_index: int, selected: bool) -> None:
    assert should_profile(tasks, task_index) is selected


def test_unparseable_task_spec_raises() -> None:
    with pytest.raises(ValueError, match="comma-separated task list"):
        should_profile("every-other", 0)


def test_task_index_from_env_reads_the_task_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/3")
    monkeypatch.setenv("IRIS_NUM_TASKS", "8")
    assert task_index_from_env() == 3


def test_task_index_from_env_requires_a_task_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    with pytest.raises(RuntimeError, match="no iris job context"):
        task_index_from_env()


def test_hook_argv_is_accepted_by_the_runtime_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nsys hook builds this argv and iris.cluster.hooks.nsys_main parses it. Nothing else
    binds the two, so a flag renamed on one side has to fail here rather than on a GPU."""
    hook = NsysHook(output_uri=OUT, tasks="0,7", trace="cuda,nvtx", capture_range=True)
    wrapped = hook.wrap(["python", "train.py"])
    assert wrapped[:3] == ["python", "-m", "iris.cluster.hooks.nsys_main"]

    seen: dict[str, object] = {}
    monkeypatch.setattr("iris.cluster.hooks.nsys_main.run", lambda *a: seen.update(zip(_RUN_PARAMS, a, strict=True)))
    nsys_main(wrapped[3:])

    assert seen == {
        "tasks": "0,7",
        "trace": "cuda,nvtx",
        "capture_range": True,
        "output_uri": OUT,
        "argv": ["python", "train.py"],
    }


def test_collect_hooks_warns_but_proceeds_without_gpu(caplog: pytest.LogCaptureFixture) -> None:
    """nsys on a GPU-less job is best-effort: logged, not rejected, and the hook still runs."""
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    env = EnvironmentSpec(profile=NsysHook(output_uri=OUT))
    with caplog.at_level("ERROR"):
        hooks = collect_hooks(env, cpu_only, processes_per_task=1)
    assert len(hooks) == 1
    assert "no GPU device" in caplog.text


def test_collect_hooks_orders_nsys_outside_the_multigpu_supervisor() -> None:
    """collect_hooks returns [multigpu, nsys]; folded in order, nsys ends up outermost, so
    its child-tracing sweeps every rank the supervisor spawns into one report. Uses a
    default (unset) output_uri, which collect_hooks must accept — the task resolves it."""
    env = EnvironmentSpec(profile=NsysHook())
    command = ["python", "train.py"]
    for hook in collect_hooks(env, _gpu_resources(8), processes_per_task=8):
        command = hook.wrap(command)
    assert command.index("iris.cluster.hooks.nsys_main") < command.index("iris.cluster.hooks.multigpu_main")


def test_hook_omits_output_uri_when_unset() -> None:
    """An unset output_uri drops the flag entirely; the wrapper then defaults it in-task."""
    assert "--output-uri" not in NsysHook().wrap(["python", "x.py"])


def test_default_output_uri_keys_on_the_job_and_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is resolved from the task's own MARIN_PREFIX (so it matches the cluster
    the task runs on) and keyed on the job so a run's reports are findable and self-expiring."""
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("IRIS_TASK_ID", "/rav/train-42/0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "1")
    assert default_output_uri() == "s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/rav/train-42"


def test_build_nsys_argv_matches_what_the_container_allows() -> None:
    # perf_event_paranoid=4 in a task pod blocks sampling and context switches; asking
    # for either fails the run. Capture range is opt-in and brackets cuProfilerStart/Stop.
    out = Path("/app/nsys/task00000-h")
    plain = build_nsys_argv("/n/nsys", out, "cuda,nvtx", capture_range=False)
    assert {"--sample=none", "--cpuctxsw=none"} <= set(plain)
    assert not any(a.startswith("--capture-range") for a in plain)

    ranged = build_nsys_argv("/n/nsys", out, "cuda,nvtx", capture_range=True)
    assert {"--capture-range=cudaProfilerApi", "--capture-range-end=stop"} <= set(ranged)


def test_report_path_carries_the_task_index() -> None:
    # All tasks upload into one directory, so the filename carries the task index.
    assert report_path(Path("/app/nsys"), 7).name.startswith("task00007-")


def test_resolve_nsys_bin_reports_missing_install(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="was the nsight setup script run"):
        resolve_nsys_bin(tmp_path / "nowhere")


def _install_fake_nsys(root: Path) -> Path:
    """Lay out a fake extracted deb the way the arm64 package does."""
    target = root / "opt/nvidia/nsight-systems" / NSYS_VERSION / "target-linux-sbsa-armv8"
    target.mkdir(parents=True)
    (target / "nsys").touch()
    return target / "nsys"


def test_resolve_nsys_bin_finds_the_extracted_binary(tmp_path: Path) -> None:
    # The deb's target dir is arch-specific, so the wrapper resolves it by glob.
    nsys_bin = _install_fake_nsys(tmp_path)
    assert resolve_nsys_bin(tmp_path) == str(nsys_bin)


def test_setup_script_and_wrapper_agree_on_the_install_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The setup script writes where the wrapper looks. They share nsys_bin_glob, but
    the script interpolates a shell $IRIS_WORKDIR while the wrapper resolves it first."""
    monkeypatch.setenv("IRIS_WORKDIR", str(tmp_path))
    nsys_bin = _install_fake_nsys(tmp_path / NSYS_INSTALL_DIR)
    script_glob = nsys_bin_glob(f"$IRIS_WORKDIR/{NSYS_INSTALL_DIR}")
    assert resolve_nsys_bin(workdir() / NSYS_INSTALL_DIR) == str(nsys_bin)
    # What bash resolves from the script must be the same file.
    assert glob(os.path.expandvars(script_glob)) == [str(nsys_bin)]


def test_environment_spec_appends_profile_setup_only_when_requested() -> None:
    without = EnvironmentSpec(extras=["gpu"]).to_proto()
    assert not any("nsight-systems" in s for s in without.setup_scripts)
    with_profile = EnvironmentSpec(extras=["gpu"], profile=NsysHook(output_uri=OUT)).to_proto()
    assert any("nsight-systems" in s for s in with_profile.setup_scripts)


def test_inherited_setup_still_installs_profiler() -> None:
    """A child job that reuses its parent's setup scripts still needs the profiler installed:
    its entrypoint is already wrapped, and an unwrapped install would fail at launch."""
    inherited = EnvironmentSpec(setup_scripts=["echo parent setup"], profile=NsysHook(output_uri=OUT)).to_proto()
    assert any("nsight-systems" in s for s in inherited.setup_scripts)
    assert inherited.setup_scripts[0] == "echo parent setup"


def test_no_setup_plus_profile_is_rejected() -> None:
    """`setup_scripts=[]` runs no install, and the wrapper looks nowhere else — so the
    combination can only fail on a GPU. It has to fail at submit instead."""
    with pytest.raises(ValueError, match="setup_scripts=\\[\\]"):
        EnvironmentSpec(setup_scripts=[], profile=NsysHook(output_uri=OUT)).to_proto()
    # Without a profile, no-setup is still the bring-your-own-image path.
    assert EnvironmentSpec(setup_scripts=[]).to_proto().setup_scripts == []


def test_caller_output_uri_is_passed_through_verbatim() -> None:
    """No scheme validation: even a workdir-relative URI is used as the caller asked."""
    (hook,) = collect_hooks(EnvironmentSpec(profile=NsysHook(output_uri="reports")), _gpu_resources(1), 1)
    wrapped = hook.wrap(["python", "x.py"])
    assert wrapped[wrapped.index("--output-uri") + 1] == "reports"


class _Execed(Exception):
    """Stands in for exec replacing the process, which never returns."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv


class _FakePopen:
    """A child that has already exited with `returncode` (negative if signalled)."""

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


def test_unselected_task_execs_command_unwrapped(
    monkeypatch: pytest.MonkeyPatch, fake_exec: None, tmp_path: Path
) -> None:
    """An unselected task runs the real command, and never needs an nsys install."""
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/1")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")

    with pytest.raises(_Execed) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=OUT, argv=CMD)
    assert excinfo.value.argv == CMD


@pytest.fixture
def selected_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Put this process at task index 0 with an nsys install, and return the workdir."""
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_WORKDIR", str(tmp_path))
    _install_fake_nsys(tmp_path / NSYS_INSTALL_DIR)
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


def test_selected_task_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    monkeypatch.setattr("iris.cluster.hooks.nsys_main._supervise", _fake_supervise(0, write_report=True))
    destination = selected_task / "uploads"

    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)

    assert excinfo.value.code == 0
    uploaded = list(destination.iterdir())
    assert len(uploaded) == 1
    assert uploaded[0].name.startswith("task00000-") and uploaded[0].name.endswith(".nsys-rep")
    assert uploaded[0].read_bytes() == b"fake report"
    # /tmp is noexec, so nsys must stage its injection libraries elsewhere.
    assert os.environ["TMPDIR"] == str(selected_task / "nsys")


def test_run_uploads_to_the_default_when_output_uri_is_unset(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path
) -> None:
    """output_uri=None routes the upload through default_output_uri (resolved in-task)."""
    destination = selected_task / "default-dest"
    monkeypatch.setattr("iris.cluster.hooks.nsys_main.default_output_uri", lambda: f"file://{destination}")
    monkeypatch.setattr("iris.cluster.hooks.nsys_main._supervise", _fake_supervise(0, write_report=True))

    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=None, argv=CMD)

    assert excinfo.value.code == 0
    assert len(list(destination.iterdir())) == 1


def test_failing_command_still_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    """A crash is exactly when the profile is worth keeping."""
    monkeypatch.setattr("iris.cluster.hooks.nsys_main._supervise", _fake_supervise(7, write_report=True))
    destination = selected_task / "uploads"

    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)

    assert excinfo.value.code == 7
    assert len(list(destination.iterdir())) == 1


def test_supervise_normalizes_a_signalled_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Popen.wait reports a SIGTERM'd child as -15; sys.exit would wrap that to 241 and
    read as an application failure. 128 + signum is the convention (multigpu agrees)."""
    monkeypatch.setattr("subprocess.Popen", lambda argv: _FakePopen(-signal.SIGTERM))
    assert _supervise(["nsys", "profile"], ["true"]) == 143


def test_missing_report_surfaces_the_command_exit_code(monkeypatch: pytest.MonkeyPatch, selected_task: Path) -> None:
    """If nsys wrote nothing, the command's own failure is the useful signal, not ours."""
    monkeypatch.setattr("iris.cluster.hooks.nsys_main._supervise", _fake_supervise(3, write_report=False))
    destination = selected_task / "uploads"

    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)

    assert excinfo.value.code == 3
    assert not destination.exists()


def test_missing_report_fails_even_when_the_command_succeeded(
    monkeypatch: pytest.MonkeyPatch, selected_task: Path
) -> None:
    """Exiting 0 here would record a green task that produced no profile and then drop
    the workdir — the one artifact the run was for."""
    monkeypatch.setattr("iris.cluster.hooks.nsys_main._supervise", _fake_supervise(0, write_report=False))

    with pytest.raises(SystemExit) as excinfo:
        run(tasks="first", trace="cuda", capture_range=False, output_uri=f"file://{selected_task / 'uploads'}", argv=CMD)

    assert excinfo.value.code != 0
