# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native vLLM server subprocess: log routing and startup retry.

``_LogPump`` forwards the subprocess's stdout/stderr to the parent's fds and to the on-disk logs,
routing by severity and flushing/draining on teardown. A second group covers
``_start_vllm_native_server``'s bounded retry around a transient Run:ai streamer read fault: it
retries that fault, fails fast on anything else, and shares one deadline across all attempts.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from marin.inference.config import VllmCompilationCacheMode
from marin.inference.vllm_cache import VllmCompilationCache, VllmCompileIdentity
from marin.inference.vllm_server import (
    TransientStartupError,
    VllmServerHandle,
    _engine_kwargs_to_cli_args,
    _LogPump,
    _native_logs_tail,
    _start_vllm_native_server,
)
from rigging.timing import ExponentialBackoff


def test_engine_kwargs_forward_dtype_to_vllm_command() -> None:
    assert _engine_kwargs_to_cli_args({"dtype": "float16"}) == ["--dtype", "float16"]


def _spawn(script: str, *, start_new_session: bool = False) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=start_new_session,
    )


def test_log_pump_forwards_to_parent_fds_and_persists(tmp_path, capsys):
    # The child writes an INFO line on each of its streams plus an ERROR line. Severity, not the
    # source stream, picks the parent fd: INFO -> parent stdout, ERROR -> parent stderr.
    script = (
        "import sys\n"
        "print('INFO worker throughput: 42 tokens/s')\n"
        "sys.stdout.flush()\n"
        "print('INFO 07-17 gen throughput: 100.0 tokens/s', file=sys.stderr)\n"
        "print('ERROR 07-17 EngineCore boom', file=sys.stderr)\n"
        "sys.stderr.flush()\n"
    )
    proc = _spawn(script)
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    pump = _LogPump(proc, str(stdout_log), str(stderr_log))
    pump.start()
    assert proc.wait(timeout=10) == 0
    pump.join(timeout=5)
    pump.close()

    # On-disk logs keep stdout/stderr provenance (they back diagnostics() and the failure tail).
    assert "worker throughput: 42 tokens/s" in stdout_log.read_text()
    stderr_text = stderr_log.read_text()
    assert "gen throughput: 100.0 tokens/s" in stderr_text
    assert "EngineCore boom" in stderr_text

    # Both INFO lines (including the one the child wrote to its stderr) go to the parent's stdout;
    # only the ERROR line goes to stderr.
    captured = capsys.readouterr()
    assert "worker throughput: 42 tokens/s" in captured.out
    assert "gen throughput: 100.0 tokens/s" in captured.out
    assert "EngineCore boom" not in captured.out
    assert "EngineCore boom" in captured.err


def test_native_logs_tail_sees_final_lines_after_join(tmp_path):
    # The startup-failure path joins the pump before building its diagnostic, so the tail must
    # include the child's final lines once join() returns.
    script = "import sys; print('LAST_STDOUT_LINE'); print('LAST_STDERR_LINE', file=sys.stderr)"
    proc = _spawn(script)
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    proc.wait(timeout=10)
    pump.join(timeout=5)

    tail = _native_logs_tail(str(tmp_path))
    assert "LAST_STDOUT_LINE" in tail
    assert "LAST_STDERR_LINE" in tail
    pump.close()


def test_native_logs_tail_includes_unterminated_final_fragment(tmp_path):
    # A child that crashes mid-line leaves a final fragment with no trailing newline. The pump
    # flushes on EOF so the startup-failure tail — read right after join(), before close() — sees
    # it; without that flush the line-buffered file would hold the newline-less fragment.
    proc = _spawn("import sys; sys.stderr.write('FATAL partial line no newline'); sys.stderr.flush()")
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    proc.wait(timeout=10)
    pump.join(timeout=5)

    assert "FATAL partial line no newline" in _native_logs_tail(str(tmp_path))
    pump.close()


def test_handle_stop_terminates_drains_and_is_idempotent(tmp_path, monkeypatch):
    # The child logs a line, then blocks; stop() must terminate it, drain that line to the
    # on-disk log, and be safe to call again.
    proc = _spawn("import sys, time; print('SERVE_READY'); sys.stdout.flush(); time.sleep(30)", start_new_session=True)
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    try:
        process_group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        process_group_id = None
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "marin"))
    compilation_cache = VllmCompilationCache.prepare(
        launcher_identity="test",
        compile_identity=VllmCompileIdentity(model_name_or_path="test/model", extra_cli_args=()),
        environment={},
        mode=VllmCompilationCacheMode.MANAGED,
    )
    compilation_cache_root = Path(compilation_cache.environment()["JAX_COMPILATION_CACHE_DIR"]).parent
    handle = VllmServerHandle(
        server_url="http://127.0.0.1:0/v1",
        port=0,
        process=proc,
        process_group_id=process_group_id,
        log_dir=str(tmp_path),
        log_pump=pump,
        compilation_cache=compilation_cache,
    )

    # Wait until the child has started Python and its line is pumped to disk, so teardown below
    # is deterministic rather than racing the child's startup.
    deadline = time.monotonic() + 10
    while "SERVE_READY" not in _native_logs_tail(str(tmp_path)):
        if time.monotonic() > deadline:
            raise AssertionError("child never logged SERVE_READY")
        time.sleep(0.05)

    handle.stop(timeout_seconds=5)
    assert proc.poll() is not None  # terminated
    # Teardown flushed and closed the on-disk logs, so the tail still reads the child's output.
    assert "SERVE_READY" in _native_logs_tail(str(tmp_path))
    assert not compilation_cache_root.exists()

    handle.stop(timeout_seconds=5)  # second call must not raise


# --- bounded retry around a transient Run:ai streamer read fault during startup ---

_FAKE_VLLM_SERVER = str(Path(__file__).parent / "fake_vllm_server.py")
_FAST_BACKOFF = ExponentialBackoff(initial=0.001, maximum=0.01, factor=2.0, jitter=0.0)


class _FakeLauncher:
    """Runs fake_vllm_server.py (see its modes) in place of the real ``vllm serve`` child."""

    def __init__(self, *mode_args: str) -> None:
        self._mode_args = mode_args

    def command(self) -> list[str]:
        return [sys.executable, _FAKE_VLLM_SERVER, *self._mode_args]

    def env(self) -> dict[str, str]:
        return {}

    def cache_identity(self) -> str:
        return "fake"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(launcher: _FakeLauncher, *, timeout_seconds: float = 30) -> VllmServerHandle:
    return _start_vllm_native_server(
        model_name_or_path="fake-model",
        port=_free_port(),
        timeout_seconds=timeout_seconds,
        launcher=launcher,
        # Managed mode would restore/publish a remote cache archive on every attempt.
        compilation_cache_mode=VllmCompilationCacheMode.CALLER_MANAGED,
        max_attempts=3,
        poll_interval_seconds=0.05,
        backoff=_FAST_BACKOFF,
    )


def test_serves_when_startup_succeeds():
    # A returned handle means the fake server answered /v1/models with 200.
    handle = _start(_FakeLauncher("serve"))
    handle.stop()


def test_retries_streamer_fault_then_serves(tmp_path):
    counter = tmp_path / "starts"
    handle = _start(_FakeLauncher("fail", str(counter), "2"))
    try:
        assert counter.read_text() == "3"  # two streamer faults, served on the third start
    finally:
        handle.stop()


def test_fails_fast_on_non_streamer_error(tmp_path):
    counter = tmp_path / "starts"
    with pytest.raises(RuntimeError) as excinfo:
        _start(_FakeLauncher("fail", str(counter), "99", "RuntimeError: CUDA out of memory"))
    assert not isinstance(excinfo.value, TransientStartupError)
    assert "CUDA out of memory" in str(excinfo.value)
    assert counter.read_text() == "1"  # a non-streamer failure is not retried


def test_exhausts_retries_then_raises_with_diagnostics(tmp_path):
    counter = tmp_path / "starts"
    with pytest.raises(TransientStartupError) as excinfo:
        _start(_FakeLauncher("fail", str(counter), "99"))
    assert counter.read_text() == "3"  # retried up to the attempt budget
    err = excinfo.value
    assert "Could not receive runai_response from libstreamer" in str(err)  # streamer fault preserved
    assert "fake-model" in str(err)  # attempted command preserved
    assert err.__notes__


def test_hang_times_out_without_retry(tmp_path):
    # A live-but-never-ready server raises TimeoutError, not a streamer fault, so it is not retried.
    counter = tmp_path / "starts"
    with pytest.raises(TimeoutError):
        _start(_FakeLauncher("hang", str(counter)), timeout_seconds=0.5)
    assert counter.read_text() == "1"
