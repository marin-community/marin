# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for native vLLM server log routing.

``_LogPump`` forwards the subprocess's stdout/stderr to the parent's fds and to the on-disk logs,
routing by severity and flushing/draining on teardown; these exercise that.
"""

import dataclasses
import os
import subprocess
import sys
import time

from marin.inference.vllm_server import (
    InstalledPackageFingerprint,
    VllmRuntimeFingerprint,
    VllmServerHandle,
    _engine_kwargs_to_cli_args,
    _LogPump,
    _native_logs_tail,
)


def test_engine_kwargs_forward_dtype_to_vllm_command() -> None:
    assert _engine_kwargs_to_cli_args({"dtype": "float16"}) == ["--dtype", "float16"]


def _runtime_fingerprint(*packages: InstalledPackageFingerprint) -> VllmRuntimeFingerprint:
    return VllmRuntimeFingerprint(
        packages=packages,
        environment=(("LIBTPU_INIT_ARGS", "--xla_tpu_scoped_vmem_limit_kib=98304"),),
        engine_args=("serve", "model"),
        launcher_args=("uvx", "vllm"),
        python_version="3.12.0",
        platform="Linux",
        libc=("glibc", "2.36"),
        os_release=(("ID", "debian"),),
        cpu_affinity_count=160,
        isolation="container",
    )


def test_runtime_fingerprint_uses_vcs_revision_instead_of_nondeterministic_record() -> None:
    vllm = InstalledPackageFingerprint(
        name="vllm",
        version="1.0",
        direct_url=(
            '{"url":"https://example.com/vllm.git","vcs_info":'
            '{"vcs":"git","commit_id":"abc","requested_revision":"abc"}}'
        ),
        record_sha256="first-build",
    )
    fingerprint = _runtime_fingerprint(vllm)

    rebuilt = _runtime_fingerprint(dataclasses.replace(vllm, record_sha256="second-build"))
    moved_revision = _runtime_fingerprint(
        dataclasses.replace(vllm, direct_url=vllm.direct_url.replace('"abc"', '"def"'))
    )

    assert rebuilt.digest() == fingerprint.digest()
    assert moved_revision.digest() != fingerprint.digest()


def test_runtime_fingerprint_keeps_registry_wheel_record_and_version() -> None:
    jax = InstalledPackageFingerprint(
        name="jax",
        version="0.10.1",
        direct_url="",
        record_sha256="wheel-a",
    )
    fingerprint = _runtime_fingerprint(jax)

    assert _runtime_fingerprint(dataclasses.replace(jax, record_sha256="wheel-b")).digest() != fingerprint.digest()
    assert _runtime_fingerprint(dataclasses.replace(jax, version="0.10.2")).digest() != fingerprint.digest()


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


def test_handle_stop_terminates_drains_and_is_idempotent(tmp_path):
    # The child logs a line, then blocks; stop() must terminate it, drain that line to the
    # on-disk log, and be safe to call again.
    proc = _spawn("import sys, time; print('SERVE_READY'); sys.stdout.flush(); time.sleep(30)", start_new_session=True)
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    try:
        process_group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        process_group_id = None
    handle = VllmServerHandle(
        server_url="http://127.0.0.1:0/v1",
        port=0,
        process=proc,
        process_group_id=process_group_id,
        log_dir=str(tmp_path),
        log_pump=pump,
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

    handle.stop(timeout_seconds=5)  # second call must not raise
