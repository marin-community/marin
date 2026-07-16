# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.runtime.nsys launch wrapper, the client-side entrypoint wrapping
that drives it, and the setup script that installs the profiler. None of this runs nsys."""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator, Sequence
from glob import glob
from pathlib import Path
from typing import NoReturn

import pytest
from iris.client.client import _wrap_entrypoint_for_multiprocess, _wrap_entrypoint_for_nsys
from iris.cluster.client.job_info import set_job_info
from iris.cluster.setup_scripts import NSYS_INSTALL_DIR, NSYS_VERSION, nsys_bin_glob
from iris.cluster.types import Entrypoint, EnvironmentSpec, NsysScope, NsysSpec, ResourceSpec, gpu_device
from iris.runtime.nsys import (
    Rank,
    _supervise,
    build_nsys_argv,
    report_path,
    resolve_nsys_bin,
    run,
    should_profile,
    validate_selector,
    workdir,
)
from iris.runtime.nsys import (
    main as nsys_main,
)

CMD = ["python", "train.py", "--steps", "10"]
# Positional signature of iris.runtime.nsys.run, as main() calls it.
_RUN_PARAMS = ("ranks", "scope", "trace", "capture_range", "output_uri", "argv")
OUT = "s3://bucket/tmp/ttl=30d/nsys"


@pytest.fixture(autouse=True)
def clear_job_info_cache() -> Iterator[None]:
    """Drop the memoized JobInfo around every test in this module.

    ``get_job_info`` caches its env parse in a ContextVar, which outlives the
    ``monkeypatch.setenv`` that produced it. Clearing only on the way in would leave
    the last test's identity cached for whatever else shares this process: the
    multigpu supervisor derives child ranks from ``task_index``, so a leaked
    JobInfo silently renumbers its ranks.
    """
    set_job_info(None)
    yield
    set_job_info(None)


def _gpu_resources(count: int) -> ResourceSpec:
    return ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=gpu_device("H100", count))


@pytest.mark.parametrize(
    ("ranks", "rank", "selected"),
    [
        ("first", Rank(global_rank=0, local_rank=0), True),
        ("first", Rank(global_rank=1, local_rank=1), False),
        # A node leader that is not rank 0 is still not the first rank.
        ("first", Rank(global_rank=4, local_rank=0), False),
        ("per-node", Rank(global_rank=4, local_rank=0), True),
        ("per-node", Rank(global_rank=5, local_rank=1), False),
        ("all", Rank(global_rank=127, local_rank=3), True),
        ("0,7", Rank(global_rank=7, local_rank=3), True),
        ("0,7", Rank(global_rank=6, local_rank=2), False),
    ],
)
def test_selector_picks_units(ranks: str, rank: Rank, selected: bool) -> None:
    assert should_profile(ranks, rank) is selected


def test_unparseable_rank_spec_raises() -> None:
    with pytest.raises(ValueError, match="comma-separated rank list"):
        should_profile("every-other", Rank(global_rank=0, local_rank=0))


def test_rank_from_env_uses_task_index_without_multigpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # processes_per_task=1 stamps no rank env, so the task is the rank and is its own leader.
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_INDEX", raising=False)
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/3")
    monkeypatch.setenv("IRIS_NUM_TASKS", "8")
    rank = Rank.from_env()
    assert (rank.global_rank, rank.local_rank) == (3, 0)


def test_rank_from_env_derives_local_rank_under_multigpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # 8 processes over 2 tasks: global rank 5 is local rank 1 on the second node.
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/1")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "5")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "8")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    rank = Rank.from_env()
    assert (rank.global_rank, rank.local_rank) == (5, 1)


def test_rank_from_env_requires_a_task_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_INDEX", raising=False)
    with pytest.raises(RuntimeError, match="no iris job context"):
        Rank.from_env()


def test_wrapper_argv_is_accepted_by_the_runtime_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client builds this argv and iris.runtime.nsys parses it. Nothing else binds
    the two, so a flag renamed on one side has to fail here rather than on a GPU."""
    wrapped = _wrap_entrypoint_for_nsys(
        Entrypoint.from_command("python", "train.py"),
        _gpu_resources(8),
        NsysSpec(output_uri=OUT, scope=NsysScope.NODE, ranks="0,7", trace="cuda,nvtx", capture_range=True),
    )
    assert wrapped.command[:3] == ["python", "-m", "iris.runtime.nsys"]

    seen: dict[str, object] = {}
    monkeypatch.setattr("iris.runtime.nsys.run", lambda *a: seen.update(zip(_RUN_PARAMS, a, strict=True)))
    nsys_main(wrapped.command[3:])

    assert seen == {
        "ranks": "0,7",
        "scope": NsysScope.NODE,
        "trace": "cuda,nvtx",
        "capture_range": True,
        "output_uri": OUT,
        "argv": ["python", "train.py"],
    }


def test_wrap_entrypoint_argv_needs_no_shell_expansion() -> None:
    """The wrapper argv is exec'd, not run through a shell, so a '$VAR' would arrive
    literally. The install path is therefore resolved in-task, never passed as text."""
    wrapped = _wrap_entrypoint_for_nsys(
        Entrypoint.from_command("python", "x.py"), _gpu_resources(1), NsysSpec(output_uri=OUT)
    )
    assert not any("$" in arg for arg in wrapped.command)


def test_wrap_entrypoint_requires_gpu() -> None:
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    with pytest.raises(ValueError, match="requires a GPU device"):
        _wrap_entrypoint_for_nsys(Entrypoint.from_command("python", "x.py"), cpu_only, NsysSpec(output_uri=OUT))


def test_process_scope_wraps_inside_the_multigpu_supervisor() -> None:
    """Process scope needs nsys in each child: that is what lets a subset of a node's
    ranks be traced, and rank selection reads the per-child rank env the supervisor stamps."""
    entry = Entrypoint.from_command("python", "train.py")
    wrapped = _wrap_entrypoint_for_nsys(entry, _gpu_resources(8), NsysSpec(output_uri=OUT))
    wrapped = _wrap_entrypoint_for_multiprocess(wrapped, _gpu_resources(8), processes_per_task=8)
    assert wrapped.command.index("iris.runtime.multigpu") < wrapped.command.index("iris.runtime.nsys")


def test_node_scope_wraps_around_the_multigpu_supervisor() -> None:
    """Node scope needs nsys outside the supervisor, so its child-tracing sweeps every
    rank on the node into one report."""
    entry = Entrypoint.from_command("python", "train.py")
    wrapped = _wrap_entrypoint_for_multiprocess(entry, _gpu_resources(8), processes_per_task=8)
    wrapped = _wrap_entrypoint_for_nsys(wrapped, _gpu_resources(8), NsysSpec(output_uri=OUT, scope=NsysScope.NODE))
    assert wrapped.command.index("iris.runtime.nsys") < wrapped.command.index("iris.runtime.multigpu")


def test_per_node_selector_is_rejected_under_node_scope() -> None:
    """It would silently be a synonym for 'all' — a node report already covers the node."""
    with pytest.raises(ValueError, match="meaningless"):
        validate_selector("per-node", NsysScope.NODE)
    validate_selector("per-node", NsysScope.PROCESS)
    validate_selector("all", NsysScope.NODE)


def test_build_nsys_argv_matches_what_the_container_allows() -> None:
    # perf_event_paranoid=4 in a task pod blocks sampling and context switches; asking
    # for either fails the run. Capture range is opt-in and brackets cuProfilerStart/Stop.
    out = Path("/app/nsys/rank00000-h")
    plain = build_nsys_argv("/n/nsys", out, "cuda,nvtx", capture_range=False)
    assert {"--sample=none", "--cpuctxsw=none"} <= set(plain)
    assert not any(a.startswith("--capture-range") for a in plain)

    ranged = build_nsys_argv("/n/nsys", out, "cuda,nvtx", capture_range=True)
    assert {"--capture-range=cudaProfilerApi", "--capture-range-end=stop"} <= set(ranged)


def test_report_path_identifies_its_unit() -> None:
    # Every unit uploads into one directory, so the name has to carry rank/node identity.
    out = Path("/app/nsys")
    assert report_path(out, Rank(0, 0), NsysScope.PROCESS) != report_path(out, Rank(1, 1), NsysScope.PROCESS)
    assert report_path(out, Rank(7, 3), NsysScope.PROCESS).name.startswith("rank00007-")
    assert report_path(out, Rank(3, 0), NsysScope.NODE).name.startswith("node00003-")


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


def test_environment_spec_appends_nsys_setup_only_when_requested() -> None:
    without = EnvironmentSpec(extras=["gpu"]).to_proto()
    assert not any("nsight-systems" in s for s in without.setup_scripts)
    with_nsys = EnvironmentSpec(extras=["gpu"], nsys=NsysSpec(output_uri=OUT)).to_proto()
    assert any("nsight-systems" in s for s in with_nsys.setup_scripts)


def test_inherited_setup_still_installs_nsys() -> None:
    """A child job that reuses its parent's setup scripts still needs Nsight installed:
    its entrypoint is already wrapped, and an unwrapped install would fail at launch."""
    inherited = EnvironmentSpec(setup_scripts=["echo parent setup"], nsys=NsysSpec(output_uri=OUT)).to_proto()
    assert any("nsight-systems" in s for s in inherited.setup_scripts)
    assert inherited.setup_scripts[0] == "echo parent setup"


def test_no_setup_plus_nsys_is_rejected() -> None:
    """`setup_scripts=[]` runs no install, and the wrapper looks nowhere else — so the
    combination can only fail on a GPU. It has to fail at submit instead."""
    with pytest.raises(ValueError, match="setup_scripts=\\[\\]"):
        EnvironmentSpec(setup_scripts=[], nsys=NsysSpec(output_uri=OUT)).to_proto()
    # Without nsys, no-setup is still the bring-your-own-image path.
    assert EnvironmentSpec(setup_scripts=[]).to_proto().setup_scripts == []


@pytest.mark.parametrize("uri", ["reports", "/app/reports"])
def test_scheme_less_output_uri_is_rejected(uri: str) -> None:
    """Such a URI resolves inside the task workdir, which the pod destroys — the wrapper
    would log a successful upload to storage that no longer exists."""
    with pytest.raises(ValueError, match="needs a scheme"):
        _wrap_entrypoint_for_nsys(Entrypoint.from_command("python", "x.py"), _gpu_resources(1), NsysSpec(output_uri=uri))


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


def test_unselected_rank_execs_command_unwrapped(
    monkeypatch: pytest.MonkeyPatch, fake_exec: None, tmp_path: Path
) -> None:
    """An unselected rank runs the real command, and never needs an nsys install."""
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_INDEX", raising=False)
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/1")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")

    with pytest.raises(_Execed) as excinfo:
        run(ranks="first", scope=NsysScope.PROCESS, trace="cuda", capture_range=False, output_uri=OUT, argv=CMD)
    assert excinfo.value.argv == CMD


@pytest.fixture
def selected_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Put this process at global rank 0 with an nsys install, and return the workdir."""
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_INDEX", raising=False)
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


def test_selected_rank_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_rank: Path) -> None:
    monkeypatch.setattr("iris.runtime.nsys._supervise", _fake_supervise(0, write_report=True))
    destination = selected_rank / "uploads"

    with pytest.raises(SystemExit) as excinfo:
        run(
            ranks="first",
            scope=NsysScope.PROCESS,
            trace="cuda",
            capture_range=False,
            output_uri=f"file://{destination}",
            argv=CMD,
        )

    assert excinfo.value.code == 0
    uploaded = list(destination.iterdir())
    assert len(uploaded) == 1
    assert uploaded[0].name.startswith("rank00000-") and uploaded[0].name.endswith(".nsys-rep")
    assert uploaded[0].read_bytes() == b"fake report"
    # /tmp is noexec, so nsys must stage its injection libraries elsewhere.
    assert os.environ["TMPDIR"] == str(selected_rank / "nsys")


def test_failing_command_still_uploads_its_report(monkeypatch: pytest.MonkeyPatch, selected_rank: Path) -> None:
    """A crash is exactly when the profile is worth keeping."""
    monkeypatch.setattr("iris.runtime.nsys._supervise", _fake_supervise(7, write_report=True))
    destination = selected_rank / "uploads"

    with pytest.raises(SystemExit) as excinfo:
        run(
            ranks="first",
            scope=NsysScope.PROCESS,
            trace="cuda",
            capture_range=False,
            output_uri=f"file://{destination}",
            argv=CMD,
        )

    assert excinfo.value.code == 7
    assert len(list(destination.iterdir())) == 1


def test_supervise_normalizes_a_signalled_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Popen.wait reports a SIGTERM'd child as -15; sys.exit would wrap that to 241 and
    read as an application failure. 128 + signum is the convention (multigpu agrees)."""
    monkeypatch.setattr("subprocess.Popen", lambda argv: _FakePopen(-signal.SIGTERM))
    assert _supervise(["nsys", "profile"], ["true"]) == 143


def test_supervise_passes_through_a_normal_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("subprocess.Popen", lambda argv: _FakePopen(7))
    assert _supervise(["nsys", "profile"], ["false"]) == 7


def test_missing_report_surfaces_the_command_exit_code(monkeypatch: pytest.MonkeyPatch, selected_rank: Path) -> None:
    """If nsys wrote nothing, the command's own failure is the useful signal, not ours."""
    monkeypatch.setattr("iris.runtime.nsys._supervise", _fake_supervise(3, write_report=False))
    destination = selected_rank / "uploads"

    with pytest.raises(SystemExit) as excinfo:
        run(
            ranks="first",
            scope=NsysScope.PROCESS,
            trace="cuda",
            capture_range=False,
            output_uri=f"file://{destination}",
            argv=CMD,
        )

    assert excinfo.value.code == 3
    assert not destination.exists()


def test_missing_report_fails_even_when_the_command_succeeded(
    monkeypatch: pytest.MonkeyPatch, selected_rank: Path
) -> None:
    """Exiting 0 here would record a green task that produced no profile and then drop
    the workdir — the one artifact the run was for."""
    monkeypatch.setattr("iris.runtime.nsys._supervise", _fake_supervise(0, write_report=False))

    with pytest.raises(SystemExit) as excinfo:
        run(
            ranks="first",
            scope=NsysScope.PROCESS,
            trace="cuda",
            capture_range=False,
            output_uri=f"file://{selected_rank / 'uploads'}",
            argv=CMD,
        )

    assert excinfo.value.code != 0
