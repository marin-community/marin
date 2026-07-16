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
from iris.cluster.setup_scripts import NSYS_INSTALL_DIR, NSYS_VERSION, nsys_bin_glob, nsys_setup_script
from iris.cluster.types import Entrypoint, EnvironmentSpec, NsysSpec, ResourceSpec, gpu_device
from iris.runtime.nsys import (
    Rank,
    _supervise,
    build_nsys_argv,
    report_path,
    resolve_nsys_bin,
    run,
    should_profile,
    workdir,
)

CMD = ["python", "train.py", "--steps", "10"]
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


def test_first_selects_only_global_rank_zero() -> None:
    assert should_profile("first", Rank(global_rank=0, local_rank=0))
    assert not should_profile("first", Rank(global_rank=1, local_rank=1))
    # A node leader that is not rank 0 is still not selected.
    assert not should_profile("first", Rank(global_rank=4, local_rank=0))


def test_per_node_selects_every_node_leader() -> None:
    assert should_profile("per-node", Rank(global_rank=0, local_rank=0))
    assert should_profile("per-node", Rank(global_rank=4, local_rank=0))
    assert not should_profile("per-node", Rank(global_rank=5, local_rank=1))


def test_all_selects_every_rank() -> None:
    assert should_profile("all", Rank(global_rank=0, local_rank=0))
    assert should_profile("all", Rank(global_rank=127, local_rank=3))


def test_explicit_list_selects_named_global_ranks() -> None:
    assert should_profile("0,7", Rank(global_rank=7, local_rank=3))
    assert not should_profile("0,7", Rank(global_rank=6, local_rank=2))


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


def test_wrap_entrypoint_prepends_nsys_wrapper() -> None:
    wrapped = _wrap_entrypoint_for_nsys(
        Entrypoint.from_command("python", "train.py", "--steps", "10"),
        _gpu_resources(8),
        NsysSpec(output_uri=OUT, ranks="per-node", trace="cuda,nvtx"),
    )
    assert wrapped.command == [
        "python",
        "-m",
        "iris.runtime.nsys",
        "--ranks",
        "per-node",
        "--trace",
        "cuda,nvtx",
        "--output-uri",
        OUT,
        "--",
        "python",
        "train.py",
        "--steps",
        "10",
    ]


def test_wrap_entrypoint_argv_needs_no_shell_expansion() -> None:
    """The wrapper argv is exec'd, not run through a shell, so a '$VAR' would arrive
    literally. The install path is therefore resolved in-task, never passed as text."""
    wrapped = _wrap_entrypoint_for_nsys(
        Entrypoint.from_command("python", "x.py"), _gpu_resources(1), NsysSpec(output_uri=OUT)
    )
    assert not any("$" in arg for arg in wrapped.command)


def test_wrap_entrypoint_passes_capture_range() -> None:
    wrapped = _wrap_entrypoint_for_nsys(
        Entrypoint.from_command("python", "train.py"),
        _gpu_resources(1),
        NsysSpec(output_uri=OUT, capture_range=True),
    )
    assert "--capture-range" in wrapped.command
    assert wrapped.command.index("--capture-range") < wrapped.command.index("--")


def test_wrap_entrypoint_requires_gpu() -> None:
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    with pytest.raises(ValueError, match="requires a GPU device"):
        _wrap_entrypoint_for_nsys(Entrypoint.from_command("python", "x.py"), cpu_only, NsysSpec(output_uri=OUT))


def test_nsys_wraps_inside_the_multigpu_supervisor() -> None:
    """The supervisor must spawn nsys, not the reverse: each child needs its own report,
    and rank selection reads the per-child rank env the supervisor stamps."""
    entry = Entrypoint.from_command("python", "train.py")
    wrapped = _wrap_entrypoint_for_nsys(entry, _gpu_resources(8), NsysSpec(output_uri=OUT))
    wrapped = _wrap_entrypoint_for_multiprocess(wrapped, _gpu_resources(8), processes_per_task=8)
    assert wrapped.command.index("iris.runtime.multigpu") < wrapped.command.index("iris.runtime.nsys")


def test_build_nsys_argv_disables_unavailable_collection() -> None:
    # perf_event_paranoid=4 in a task container blocks both; leaving them on fails the run.
    argv = build_nsys_argv("/n/nsys", Path("/app/nsys/rank00000-h"), "cuda,nvtx", capture_range=False)
    assert "--sample=none" in argv
    assert "--cpuctxsw=none" in argv
    assert not any(a.startswith("--capture-range") for a in argv)


def test_build_nsys_argv_capture_range_stops_without_killing_the_process() -> None:
    argv = build_nsys_argv("/n/nsys", Path("/app/nsys/rank00000-h"), "cuda", capture_range=True)
    assert "--capture-range=cudaProfilerApi" in argv
    assert "--capture-range-end=stop" in argv


def test_report_path_is_unique_per_rank() -> None:
    out = Path("/app/nsys")
    assert report_path(out, Rank(0, 0)) != report_path(out, Rank(1, 1))
    assert report_path(out, Rank(7, 3)).name.startswith("rank00007-")


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


def test_setup_script_extracts_rather_than_installs() -> None:
    # apt would drag in the Qt/GUI chain; the target CLI binary is self-contained.
    script = nsys_setup_script()
    assert "dpkg-deb -x" in script
    assert "apt-get" not in script


def test_setup_script_selects_arch_specific_package() -> None:
    script = nsys_setup_script()
    assert "sbsa" in script and "x86_64" in script


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


def test_environment_spec_no_setup_stays_empty() -> None:
    # `setup_scripts=[]` is bring-your-own-image; iris adds nothing to it.
    assert EnvironmentSpec(setup_scripts=[], nsys=NsysSpec(output_uri=OUT)).to_proto().setup_scripts == []


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
        run(ranks="first", trace="cuda", capture_range=False, output_uri=OUT, argv=CMD)
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
        run(ranks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)

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
        run(ranks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)

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
        run(ranks="first", trace="cuda", capture_range=False, output_uri=f"file://{destination}", argv=CMD)

    assert excinfo.value.code == 3
    assert not destination.exists()
