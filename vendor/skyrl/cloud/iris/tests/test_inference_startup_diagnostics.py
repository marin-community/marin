"""Readiness failure must retain native stacks before actor teardown."""

import json
from dataclasses import asdict
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from marinskyrl import inference_startup_diagnostics as diagnostics
from marinskyrl.environment_contract import DEBUG_MODE_ENV, DEBUG_ARTIFACT_DIR_ENV, DebugMode
from skyrl_train.inference_engines import ray_wrapped_inference_engine as startup


def test_timeout_dump_captures_native_frames_and_records_unwind_failure(tmp_path, monkeypatch):
    profiler = tmp_path / "py-spy"
    profiler.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "pid = sys.argv[sys.argv.index('--pid') + 1]\n"
        "if pid == '2147483647' and '--native' in sys.argv:\n"
        "    sys.stderr.write('native unwinder unavailable'); sys.exit(1)\n"
        "print('ncclAllGather native frame' if '--native' in sys.argv else 'wait_for_engine Python frame')\n"
    )
    profiler.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(
        diagnostics.psutil,
        "process_iter",
        lambda attrs: [
            SimpleNamespace(pid=2147483646, info={"name": "EngineCore_DP0", "cmdline": []}),
            SimpleNamespace(pid=2147483647, info={"name": "ray::AsyncVLLMInferenceEngine", "cmdline": []}),
            SimpleNamespace(pid=2147483645, info={"name": "unrelated", "cmdline": []}),
        ],
    )
    receipts = diagnostics.capture_node_stacks(str(tmp_path))
    assert len(receipts) == 2
    native, fallback = sorted(receipts, key=lambda receipt: receipt.pid)
    assert native.native and "ncclAllGather native frame" in Path(native.stack_path).read_text()
    assert not fallback.native and "native unwinder unavailable" in fallback.native_error
    assert "wait_for_engine Python frame" in Path(fallback.stack_path).read_text()
    persisted = [json.loads(path.read_text()) for path in (tmp_path / "stacks").glob("*.json")]
    assert sorted(persisted, key=lambda receipt: receipt["pid"]) == [asdict(native), asdict(fallback)]


@pytest.mark.parametrize("host_lost", [False, True])
def test_readiness_timeout_preserves_stacks_before_cleanup_and_survives_lost_host(tmp_path, monkeypatch, host_lost):
    profiler = tmp_path / "py-spy"
    profiler.write_text(f"#!{sys.executable}\nprint('ncclAllGather native frame')\n")
    profiler.chmod(0o755)
    monkeypatch.setenv(DEBUG_MODE_ENV, DebugMode.DISTRIBUTED)
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path))
    clock = [0.0]
    killed = []

    def wait(pending, **kwargs):
        clock[0] += kwargs["timeout"]
        return [], pending

    class HostProfiler:
        def options(self, **kwargs):
            return self

        def remote(self, root):
            if host_lost:
                raise RuntimeError("diagnostic host lost")
            return diagnostics.capture_process_stacks(2147483646, root, str(profiler))

    def cleanup(actor):
        killed.append(actor)
        if host_lost:
            return
        receipts = [json.loads(path.read_text()) for path in (tmp_path / "stacks").glob("*.json")]
        assert len(receipts) == 1 and receipts[0]["native"]
        assert "ncclAllGather" in Path(receipts[0]["stack_path"]).read_text()

    monkeypatch.setattr(startup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(startup.ray, "wait", wait)
    monkeypatch.setattr(startup.ray, "get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(startup.ray, "kill", cleanup)
    monkeypatch.setattr(diagnostics.ray, "nodes", lambda: [{"NodeID": "a" * 56, "Alive": True}])
    monkeypatch.setattr(diagnostics, "remote_capture_node_stacks", HostProfiler())
    with pytest.raises(TimeoutError):
        startup.wait_for_inference_engine_startup(["pending"], ["actor"], timeout_seconds=1)
    assert killed == ["actor"]
