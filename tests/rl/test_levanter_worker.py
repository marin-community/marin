# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real subprocess update and checkpoint continuation at the learner boundary."""

import concurrent.futures
import copy
import gzip
import hashlib
import json
import os
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax import Axis
from jax.sharding import AxisType, Mesh
from levanter.field_dtype_overrides import with_field_dtype_overrides
from levanter.grad_accum import microbatched
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig
from levanter.policy_learner import _slice_batch
from levanter.policy_model import policy_model_loss, response_logprobs
from levanter.testing.devices import use_test_devices
from levanter.trainer import WrappedLossFunction
from levanter.utils.mesh import MeshConfig
from marin.rl.levanter_worker_protocol import LearnerProcess
from marin.rl.levanter_worker_runtime import NativePolicyRuntime, _state_fingerprint
from marin.rl.native_shutdown.stack_safety import direct_children, identity
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize(
    "dtype,expected_sha,expected_layout,expected_bytes",
    [
        (
            "float32",
            "aa8f1209084d0ef305e8ea9bda7d14f863453aa512df256724fe306999c57b7b",
            "253fe9f21f5495c92547974e6b12d212dc93345b6a3715ae177ab42e66e76e0c",
            4194322,
        ),
        (
            "bfloat16",
            "710da197f2a2341a63310f6b453d1b75f8d18019f2e0a62a25a1612e912f7476",
            "7f4ee246ef231b7164efe179ef7e6fa524de0b0957ce69ed07511a88831a1455",
            14,
        ),
    ],
)
def test_fingerprint_reports_nonfinite_without_changing_legacy_bytes(
    dtype, expected_sha, expected_layout, expected_bytes
):
    # Literal digests were produced by the published d3fecc6 byte encoder.
    if dtype == "float32":
        weights = np.zeros((1 << 20) + 3, np.float32)
        weights.view(np.uint32)[0] = 0x7FC01234
        weights[(1 << 20) - 1] = np.inf
        weights[1 << 20] = -np.inf
    else:
        weights = np.array([0, np.nan, np.inf, -np.inf], dtype=jnp.bfloat16)
    original = weights.tobytes()
    proof = _state_fingerprint({"weights": weights, "counter": np.array(7, np.int32), "flag": np.array([True, False])})
    assert proof["sha256"] == expected_sha and proof["layout_sha256"] == expected_layout
    assert proof["addressable_bytes"] == expected_bytes
    assert weights.tobytes() == original
    assert proof["nonfinite_elements"] == 3
    assert proof["elements"] == weights.size + 3
    assert proof["finite_elements"] == weights.size


_FINITE_FAILURE_WORKER = """
import json, sys
from types import SimpleNamespace
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from marin.rl.levanter_worker_runtime import NativePolicyRuntime
rank, address, scenario = int(sys.argv[1]), sys.argv[2], sys.argv[3]
jax.distributed.initialize(coordinator_address=address, num_processes=2, process_id=rank)
try:
    mesh = Mesh(np.array(jax.devices()), ('data',))
    source = np.arange(4, dtype=np.float32)
    if rank == 1 and scenario in ('poison', 'poison_io'):
        source[2] = np.nan
    weight = jax.make_array_from_callback((4,), NamedSharding(mesh, PartitionSpec('data')), lambda i: source[i])
    runtime = NativePolicyRuntime()
    runtime.settings = {'distributed': {'num_processes': 2, 'process_id': rank, 'local_device_count': 1}, 'runtime': {}}
    runtime.source_identity = {'path': 'finite-test', 'revision': None}
    runtime.version = 1
    runtime.state = SimpleNamespace(model={'weight': weight}, opt_state={'mu': jnp.zeros(4)},
        training_key=jax.random.PRNGKey(41), step=jnp.array(1), model_averaging=None)
    print(json.dumps({'ready': rank}), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command['method'] == 'plain':
            print(json.dumps(runtime.state_fingerprints()), flush=True)
            continue
        if rank == 1 and scenario == 'scan_error':
            weight.delete()
        request = {'phase': 'step1', 'step': 1, 'version': 1}
        if rank == 1 and scenario == 'phase':
            request = {'phase': 'step2', 'step': 2, 'version': 2}
        try:
            runtime.validate_state_finite(request)
        except RuntimeError as error:
            print(json.dumps({'rank': rank, 'error': str(error)}), flush=True)
        else:
            print(json.dumps({'rank': rank, 'error': None}), flush=True)
        break
finally:
    jax.distributed.shutdown()
"""


@pytest.mark.multidevice
@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "scenario,kind",
    [
        ("poison", "nonfinite_state"),
        ("scan_error", "state_scan_failed"),
        ("phase", "validation_mismatch"),
        ("io", "proof_persistence_failed"),
        ("poison_io", "nonfinite_state"),
    ],
)
def test_finite_gate_rejects_all_hosts_without_making_plain_hash_collective(tmp_path, scenario, kind):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    children, logs = [], []
    try:
        for rank in range(2):
            directory = tmp_path / f"rank-{rank}"
            if rank == 1 and scenario in ("io", "poison_io"):
                directory.write_text("A real file prevents creation of the required proof directory.")
            env = {
                **os.environ,
                "JAX_PLATFORMS": "cpu",
                "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
                "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                "IRIS_OUTPUT_DIR": str(directory),
            }
            env.pop("SKYRL_DEBUG_ARTIFACT_DIR", None)
            log = (tmp_path / f"child-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [sys.executable, "-u", "-c", _FINITE_FAILURE_WORKER, str(rank), address, scenario],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                )
            )
        for rank, child in enumerate(children):
            assert json.loads(child.stdout.readline()) == {"ready": rank}
        # Rank1 is idle on stdin: a plain rank0 fingerprint must return alone.
        children[0].stdin.write(json.dumps({"method": "plain"}) + "\n")
        children[0].stdin.flush()
        plain = json.loads(children[0].stdout.readline())
        assert plain["components"]["model"]["nonfinite_elements"] == 0
        for child in children:
            child.stdin.write(json.dumps({"method": "validate"}) + "\n")
            child.stdin.flush()
        errors = [json.loads(child.stdout.readline())["error"] for child in children]
        assert all(kind in error for error in errors)
        assert all(child.wait(timeout=10) == 0 for child in children)
        verdicts = list((tmp_path / "rank-0").rglob("*-verdict.json"))
        assert len(verdicts) == 1
        verdict = json.loads(verdicts[0].read_text())
        assert verdict["status"] == "failed" and verdict["failure_kind"] == kind
        assert sorted(peer["process_id"] for peer in verdict["peers"]) == [0, 1]
        if scenario == "poison":
            assert verdict["peers"][0]["fingerprints"]["components"]["model"]["nonfinite_elements"] == 0
            assert verdict["peers"][1]["fingerprints"]["components"]["model"]["nonfinite_elements"] == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdin.close()
            child.stdout.close()
        for log in logs:
            log.close()


@pytest.mark.multidevice
def test_native_fixed_projection_authorizes_full_512_and_microbatch_32_graphs():
    with use_test_devices(32) as devices:
        mesh = Mesh(
            np.asarray(devices).reshape(1, 1, 4, 8, 1, 1),
            ("replica_dcn", "replica", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 6,
        )
        with jax.set_mesh(mesh):
            config = SnowballConfig(
                vocab_size=64,
                hidden_dim=32,
                intermediate_dim=64,
                shared_expert_intermediate_dim=32,
                num_layers=1,
                num_experts=32,
                num_experts_per_token=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                max_seq_len=6,
                sliding_window=4,
                attention_implementation="reference",
            )
            model = SnowballLMHeadModel.init(Axis("vocab", 64), config, key=jax.random.key(71))
            overrides = {"router": jnp.dtype("float32"), "router_bias": jnp.dtype("float32")}
            runtime = NativePolicyRuntime()
            runtime.state = SimpleNamespace(model=model, step=jnp.array(0))
            runtime.source_identity = {"path": "tiny-native-source", "revision": None}
            runtime.settings = {
                "runtime": {"projection_logits_dtype": "bfloat16", "projection_block_size": 16},
                "mini_batch_size": 512,
                "optimizer": {"lr": 1e-6},
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
            }
            mapping = {"batch": ["replica_dcn", "data", "expert"], "position": "context"}
            native_mapping = MeshConfig(compute_mapping=mapping, param_mapping={})
            runtime.trainer = SimpleNamespace(
                device_mesh=mesh,
                compute_axis_mapping=native_mapping.resolved_compute_mapping,
                parameter_axis_mapping=native_mapping.resolved_param_mapping,
                mp=with_field_dtype_overrides(jmp.get_policy("p=f32,c=bf16"), overrides),
                config=SimpleNamespace(microbatch_size=32, field_dtype_overrides=overrides),
                hooks=SimpleNamespace(jit_hooks=[]),
            )
            arrays = {
                "tokens": np.ones((512, 6), dtype=np.int32),
                "attention_mask": np.ones((512, 6), dtype=np.bool_),
                "position_ids": np.tile(np.arange(6, dtype=np.int32), (512, 1)),
                "response_mask": np.ones((512, 2), dtype=np.bool_),
            }
            payload = {"arrays": arrays, "response_length": 2, "global_step": 1}
            plan = runtime._prepare_projection(payload, "forward")
            assert {signature.x_shape for signature in plan.signatures} == {(2, 32), (32, 32)}
            payload["global_step"] = 2
            runtime.version = 1
            assert runtime._prepare_projection(payload, "forward") == plan
            batch = runtime._batch(payload)
            compute = eqx.filter_eval_shape(runtime.trainer.mp.cast_to_compute, model)

            def shape(batch):
                with hax.axis_mapping(mapping):
                    return eqx.filter_eval_shape(
                        response_logprobs, compute, batch, block_size=16, logits_dtype=jnp.bfloat16, projection_plan=plan
                    )

            assert shape(batch).array.shape == (512, 2)
            micro = jax.tree.map(
                lambda value: _slice_batch(value, start=0, size=32),
                batch,
                is_leaf=lambda value: isinstance(value, hax.NamedArray),
            )
            assert shape(micro).array.shape == (32, 2)
            loss = WrappedLossFunction(
                partial(
                    policy_model_loss,
                    config=ClippedPolicyConfig(0.2, 0.2, 0.0),
                    accumulation_steps=16,
                    block_size=16,
                    logits_dtype=jnp.bfloat16,
                ),
                runtime.trainer.mp,
                mapping,
            )
            gradients = microbatched(
                partial(eqx.filter_value_and_grad(loss, has_aux=True), model),
                Axis("batch", 512),
                32,
                native_mapping.resolved_param_mapping,
                mapping,
            )
            with hax.axis_mapping(mapping):
                (_, _), gradient_shape = eqx.filter_eval_shape(
                    gradients, batch, key=jax.random.key(72), projection_plan=plan
                )
            assert jax.tree.structure(gradient_shape) == jax.tree.structure(model)
            unexpected = jax.tree.map(
                lambda value: _slice_batch(value, start=0, size=64),
                batch,
                is_leaf=lambda value: isinstance(value, hax.NamedArray),
            )
            with pytest.raises(ValueError, match="unapproved projection signature"):
                shape(unexpected)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_stall_diagnostics_capture_only_active_main_thread(enabled):
    script = """
import json, os, threading, time
from marin.rl import levanter_worker_runtime as worker
worker._DIAGNOSTIC_STALL_SECONDS = 0.15
runtime = worker.NativePolicyRuntime()
runtime.settings = {"runtime": {"validate_checkpoint_roundtrip": ENABLED}}
def wait_for_input():
    read_fd, write_fd = os.pipe()
    def release():
        os.write(write_fd, b"x")
    writer = threading.Timer(0.35, release)
    writer.start()
    try:
        os.read(read_fd, 1)
    finally:
        writer.join()
        os.close(read_fd)
        os.close(write_fd)
with runtime.request_diagnostics("train"):
    wait_for_input()
with runtime.request_diagnostics("model_fingerprint"):
    pass
time.sleep(0.25)  # A cancelled watchdog must not report the subsequent idle frame.
""".replace(
        "ENABLED", repr(enabled)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line.split(" ", 1)[1])
        for line in result.stderr.splitlines()
        if line.startswith("native_phase_receipt ")
    ]
    if enabled:
        stalled = [row for row in rows if row["event"] == "request_stalled"]
        assert len(stalled) == 1
        assert stalled[0]["method"] == "train"
        assert stalled[0]["thread_id"] != stalled[0]["main_thread_id"]
        assert len(stalled[0]["main_thread_stack"]) <= 40
        assert any("wait_for_input" in frame for frame in stalled[0]["main_thread_stack"])
        assert rows[-1]["event"] == "request_end" and rows[-1]["method"] == "model_fingerprint"
    else:
        assert rows == []


def test_native_child_import_time_stdout_does_not_corrupt_response(tmp_path, capfd):
    # sitecustomize runs before the runtime's imports or Python stdout redirect.
    (tmp_path / "sitecustomize.py").write_text("import os\nos.write(1, b'native startup stdout\\n')\n")
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(tmp_path)}
    child = LearnerProcess(sys.executable, environment=environment, timeout_seconds=30)
    try:
        assert child.request("shutdown", {}) is None
        assert child.process.returncode == 0
    finally:
        child.close()
        child.close()
    assert "native startup stdout" in capfd.readouterr().out


@pytest.mark.parametrize("blocked_sink", [False, True])
def test_native_shutdown_phase_artifact_preserves_real_cpu_natural_exit(tmp_path, blocked_sink):
    (tmp_path / "phase_sink_runtime.py").write_text(
        """
import jax
import numpy as np
from marin.rl.levanter_worker_runtime import NativePolicyRuntime, main

def runtime_factory():
    runtime = NativePolicyRuntime()
    runtime.settings = {'runtime': {'validate_checkpoint_roundtrip': True,
                                   'diagnostic_shutdown_observer': False},
                        'distributed': {'process_id': 0}}
    runtime.state = {'owned_buffer': jax.device_put(np.ones((2, 2), dtype=np.float32))}
    return runtime

main(runtime_factory)
"""
    )
    destination = tmp_path / "artifacts"
    if blocked_sink:
        destination.write_text("A file prevents creation of the diagnostic directory.")
    child = LearnerProcess(
        sys.executable,
        environment={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(tmp_path),
            "SKYRL_DEBUG_ARTIFACT_DIR": str(destination),
            "IRIS_OUTPUT_DIR": str(tmp_path / "iris-output"),
        },
        timeout_seconds=30,
        runtime_module="phase_sink_runtime",
    )
    try:
        assert child.request("shutdown", {}) is None
        # This exercises the real protocol's unchanged five-second natural wait.
        assert child.process.returncode == 0
        if blocked_sink:
            assert destination.is_file()
            assert not (tmp_path / "iris-output").exists()
        else:
            path = destination / "native-finalization" / str(child.process.pid) / "shutdown-phases.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert rows[-1]["event"] == "process_exit_begin" and rows[-1]["reason"] == "shutdown"
            assert {row["phase"] for row in rows if row["event"] == "cleanup_phase_end"} == {
                "synchronize",
                "trainer_exit",
                "release_owned_resources",
                "clear_haliax_caches",
                "clear_jax_caches",
                "collect_python_resources",
                "release_backend_clients",
                "distributed_shutdown",
            }
            assert all(row["pid"] == child.process.pid and row["process_id"] == 0 for row in rows)
            assert all(
                row["event"] in {"cleanup_phase_begin", "cleanup_phase_end", "process_exit_begin"} for row in rows
            )
            assert [row["monotonic_ns"] for row in rows] == sorted(row["monotonic_ns"] for row in rows)
    finally:
        child.close()


def test_native_shutdown_phase_artifact_is_bounded_and_excludes_payloads(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYRL_DEBUG_ARTIFACT_DIR", raising=False)
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stderr", None)
    runtime = NativePolicyRuntime()
    runtime.settings = {"runtime": {"validate_checkpoint_roundtrip": True}, "distributed": {"process_id": 3}}
    for _ in range(100):
        runtime.shutdown_receipt(
            "cleanup_phase_error",
            phase="synchronize",
            error_type="RuntimeError",
            model_payload=object(),
            message="This diagnostic field must never enter the artifact.",
            pid="Caller fields must not replace the real process identity.",
        )
    path = tmp_path / "native-finalization" / str(os.getpid()) / "shutdown-phases.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 64 and path.stat().st_size <= 32 << 10
    assert all(
        set(row) == {"event", "phase", "error_type", "pid", "process_id", "thread_id", "time_ns", "monotonic_ns"}
        for row in rows
    )
    assert all(row["pid"] == os.getpid() and row["process_id"] == 3 for row in rows)
    assert all(row["error_type"] == "RuntimeError" and row["phase"] == "synchronize" for row in rows)


def test_native_shutdown_phase_artifact_does_no_io_for_post_ack_or_timer_events(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYRL_DEBUG_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stderr", None)
    runtime = NativePolicyRuntime()
    runtime.settings = {"runtime": {"validate_checkpoint_roundtrip": True}, "distributed": {"process_id": 0}}
    runtime.shutdown_receipt("process_exit_begin", reason="shutdown")
    path = tmp_path / "native-finalization" / str(os.getpid()) / "shutdown-phases.jsonl"
    before = path.read_bytes()
    attempted_paths = []

    def unavailable_file(path, *args, **kwargs):
        attempted_paths.append(path)
        raise OSError("Unexpected filesystem work after ACK or on the timer")

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", unavailable_file)
        for event in ("shutdown_ack_written", "python_atexit_begin", "request_stalled"):
            runtime.shutdown_receipt(event, phase="synchronize", error_type="RuntimeError")
    assert attempted_paths == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("watched_writable", [True, False])
def test_native_finalization_watchdog_survives_atexit_without_forcing_exit(tmp_path, watched_writable):
    if not watched_writable:
        (tmp_path / "watched").write_text("A real file prevents the diagnostic directory from being created.")
    script = """
import atexit
import threading
from marin.rl.levanter_worker_runtime import NativePolicyRuntime

def delayed_cleanup():
    # Reproduce a finalizer still running after the shutdown acknowledgement.
    # The native watchdog must sample it without changing the eventual exit0.
    threading.Event().wait(3.5)

atexit.register(delayed_cleanup)
runtime = NativePolicyRuntime()
runtime.settings = {
    'runtime': {'validate_checkpoint_roundtrip': True},
    'distributed': {'process_id': 7},
}
runtime.close()
runtime.close()
runtime.prepare_process_exit('shutdown')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "JAX_PLATFORMS": "must-not-initialize-during-cleanup",
            "SKYRL_DEBUG_ARTIFACT_DIR": str(tmp_path / "watched"),
            "IRIS_OUTPUT_DIR": str(tmp_path / "iris-fallback"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    receipts = [
        json.loads(line.split(" ", 1)[1])
        for line in result.stderr.splitlines()
        if line.startswith("native_shutdown_receipt ")
    ]
    final = next(row for row in receipts if row["event"] == "process_exit_begin")
    assert final["process_id"] == 7
    if watched_writable:
        assert final["trace_scope"] == "watched_debug_artifacts"
        traces = list((tmp_path / "watched").glob("native-finalization/*/atexit-stack.txt"))
        assert len(traces) == 1
        assert str(traces[0]) == final["trace_path"]
        assert "delayed_cleanup" in traces[0].read_text()
    else:
        assert final["trace_scope"] == "stderr_fallback"
        assert final["trace_path"] == "stderr"
        assert "delayed_cleanup" in result.stderr
    assert not (tmp_path / "iris-fallback").exists()


@pytest.mark.parametrize("stderr_kind", ["closed", "unsupported", "missing"])
@pytest.mark.parametrize("deleted_buffer", [False, True])
def test_native_cleanup_without_diagnostic_sinks_preserves_resource_result(tmp_path, stderr_kind, deleted_buffer):
    blocked = tmp_path / "blocked-artifacts"
    blocked.write_text("This file prevents creation of the finalization artifact directory.")
    script = """
import gc
import io
import json
import sys
import weakref
import jax
from jax.extend import backend
import numpy as np
from marin.rl.levanter_worker_runtime import NativePolicyRuntime

runtime = NativePolicyRuntime()
runtime.settings = {'runtime': {'validate_checkpoint_roundtrip': True}}
runtime.state = {'owned_buffer': jax.device_put(np.ones((2, 2), dtype=np.float32))}
buffer = weakref.ref(runtime.state['owned_buffer'])
client = weakref.ref(backend.get_backend('cpu'))
if DELETED_BUFFER:
    runtime.state['owned_buffer'].delete()
original_stderr = sys.stderr
sys.stderr = None if STDERR_KIND == 'missing' else io.StringIO()
if STDERR_KIND == 'closed':
    sys.stderr.close()
try:
    try:
        runtime.dispatch('shutdown', {})
        status = 'success'
    except ExceptionGroup:
        status = 'cleanup_error'
    runtime.close()
    runtime.prepare_process_exit('shutdown')
finally:
    sys.stderr = original_stderr
gc.collect()
print(json.dumps({'status': status, 'owned_buffer_released': buffer() is None, 'backend_released': client() is None}))
""".replace(
        "DELETED_BUFFER", repr(deleted_buffer)
    ).replace(
        "STDERR_KIND", repr(stderr_kind)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "SKYRL_DEBUG_ARTIFACT_DIR": str(blocked)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    backend_released = observed.pop("backend_released")
    if not deleted_buffer:
        assert backend_released
    assert observed == {
        "status": "cleanup_error" if deleted_buffer else "success",
        "owned_buffer_released": True,
    }


def test_native_shutdown_releases_named_jit_mesh_and_backend_before_returning():
    # Haliax's compilation cache holds output shardings independently of JAX's
    # caches. Exercise that real ownership path, with no external device owners.
    script = """
import json
import weakref
import haliax as hax
import jax
import numpy as np
from jax.extend import backend
from jax.sharding import Mesh
from marin.rl.levanter_worker_runtime import NativePolicyRuntime

def computation(value):
    return value + 1

def initialized_runtime():
    runtime = NativePolicyRuntime()
    mesh = Mesh(np.array(jax.devices()[:1]), ('data',))
    with jax.set_mesh(mesh):
        runtime.state = hax.named_jit(computation, axis_resources={'batch': 'data'})(
            hax.named(jax.device_put(np.ones(2)), ('batch',)))
    return runtime, weakref.ref(mesh), weakref.ref(backend.get_backend('cpu'))

runtime, mesh, client = initialized_runtime()
runtime.dispatch('shutdown', {})
print(json.dumps({'mesh_released': mesh() is None, 'backend_released': client() is None}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"mesh_released": True, "backend_released": True}


@pytest.mark.parametrize("require_accelerator", [True, False])
@pytest.mark.parametrize("allocator,preallocate,numpy_policy", [("bfc", True, None), ("vmm", False, False)])
def test_native_child_backend_intent_overrides_inherited_coordinator_platform(
    require_accelerator, allocator, preallocate, numpy_policy, monkeypatch
):
    host_module = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu" if require_accelerator else "cuda,cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    monkeypatch.setenv("JAX_CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.1")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    monkeypatch.setenv("TF_FORCE_UNIFIED_MEMORY", "true")
    monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", "1")
    environment = host_module._child_environment(
        sys.executable,
        0.9,
        require_accelerator=require_accelerator,
        preallocate=preallocate,
        allocator=allocator,
        numpy_madvise_hugepage=numpy_policy,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
import jax
import numpy as np
from jaxlib.xla_client import generate_pjrt_gpu_plugin_options
from marin.rl.levanter_worker_runtime import _initialize_distributed
try:
    count = jax.local_device_count()
    _initialize_distributed({
        "distributed": {"num_processes": 1, "local_device_count": count},
        "require_accelerator": sys.argv[1] == "true",
    })
    backend = jax.default_backend()
except (RuntimeError, ValueError) as error:
    receipt = {"initialized": False, "error": str(error)}
else:
    receipt = {"initialized": True, "backend": backend}
receipt["gpu_visibility"] = os.environ.get("CUDA_VISIBLE_DEVICES")
receipt["jax_platforms"] = os.environ.get("JAX_PLATFORMS")
options = generate_pjrt_gpu_plugin_options()
receipt["allocation_policy"] = {key: options[key] for key in ("allocator", "memory_fraction", "preallocate")}
receipt["unified_memory"] = os.environ.get("TF_FORCE_UNIFIED_MEMORY")
receipt["numpy_environment"] = os.environ.get("NUMPY_MADVISE_HUGEPAGE")
receipt["numpy_version"] = np.__version__
receipt["numpy_madvise_hugepage"] = bool(np._core.multiarray._get_madvise_hugepage())
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0" if receipt["numpy_environment"] == "1" else "1"
receipt["numpy_after_late_environment_change"] = bool(np._core.multiarray._get_madvise_hugepage())
print(json.dumps(receipt))
""",
            str(require_accelerator).lower(),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["gpu_visibility"] == "0,1,2,3,4,5,6,7"
    assert receipt["jax_platforms"] == ("cuda,cpu" if require_accelerator else "cpu")
    assert receipt["allocation_policy"] == {"allocator": allocator, "memory_fraction": 0.9, "preallocate": preallocate}
    assert receipt["unified_memory"] == "false"
    assert receipt["numpy_environment"] == ("1" if numpy_policy is None else "0")
    assert receipt["numpy_version"] == np.__version__
    assert receipt["numpy_after_late_environment_change"] == receipt["numpy_madvise_hugepage"]
    assert receipt["numpy_madvise_hugepage"] == (sys.platform == "linux" and numpy_policy is None)
    assert os.environ["NUMPY_MADVISE_HUGEPAGE"] == "1"
    if require_accelerator:
        if receipt["initialized"]:
            assert receipt["backend"] == "gpu"
        else:
            assert "cuda" in receipt["error"].lower()
    else:
        assert receipt["initialized"] and receipt["backend"] == "cpu"


@pytest.mark.parametrize("inherited", [None, "--xla_gpu_deterministic_ops=true"])
@pytest.mark.parametrize("compiler_diagnostics", [False, True])
def test_native_deterministic_profile_reaches_child_and_reports_actual_environment(
    inherited, compiler_diagnostics, monkeypatch
):
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    if inherited is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", inherited)
    monkeypatch.setenv("UNRELATED_NATIVE_ENVIRONMENT", "not-part-of-the-receipt")
    environment = host._child_environment(
        sys.executable,
        0.9,
        require_accelerator=False,
        compiler_diagnostics=compiler_diagnostics,
        deterministic_gpu_ops=True,
    )
    flags = shlex.split(environment["XLA_FLAGS"])
    dump = flags[-3].split("=", 1)[1] if compiler_diagnostics else None
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json, sys
from types import SimpleNamespace
# The child environment deliberately drops PYTHONPATH; select this checkout.
sys.path.insert(0, sys.argv[2])
import jax.numpy as jnp
from marin.rl.levanter_worker_runtime import NativePolicyRuntime
runtime = NativePolicyRuntime()
# A tiny real array suffices for this identity-only query; no model is loaded.
runtime.state = SimpleNamespace(model={'weight': jnp.ones(3)})
runtime.settings = {'runtime': {'deterministic_gpu_ops': False}}
if sys.argv[1]:
    runtime.settings['compiler_dump_dir'] = sys.argv[1]
print(json.dumps(runtime.dispatch('runtime_identity', {})))
""",
                dump or "",
                str(Path(__file__).resolve().parents[2] / "lib/marin/src"),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        receipt = json.loads(result.stdout)
        expected = ["--xla_gpu_deterministic_ops=true"]
        normalized = expected.copy()
        if compiler_diagnostics:
            assert Path(dump).is_dir()
            expected += [f"--xla_dump_to={dump}", "--xla_dump_hlo_as_text", "--xla_dump_hlo_module_re=train_step"]
            normalized += [
                "--xla_dump_to=<native-compiler-diagnostics>",
                "--xla_dump_hlo_as_text",
                "--xla_dump_hlo_module_re=train_step",
            ]
        assert flags == expected
        assert receipt["compiler_environment"] == {
            "XLA_FLAGS": shlex.join(expected),
            "normalized_xla_flags": normalized,
            "JAX_PLATFORMS": "cpu",
        }
        assert receipt["global_model_parameters"] == 3 and receipt["backend"] == "cpu"
        assert "not-part-of-the-receipt" not in result.stdout
        # The child reports its actual environment even when intent differs.
        assert os.environ.get("XLA_FLAGS") == inherited
    finally:
        if dump is not None:
            shutil.rmtree(dump)


@pytest.mark.parametrize(
    "inherited",
    [
        "--xla_gpu_deterministic_ops=false",
        "--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=true",
        "--xla_gpu_deterministic_ops=true --xla_gpu_enable_latency_hiding_scheduler=false",
        "--xla_dump_to=/tmp/unowned",
    ],
)
def test_native_deterministic_profile_rejects_unreviewed_inherited_compiler_settings(inherited, monkeypatch):
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    monkeypatch.setenv("XLA_FLAGS", inherited)
    with pytest.raises(ValueError, match="Native deterministic GPU profile"):
        host._child_environment(sys.executable, 0.9, require_accelerator=True, deterministic_gpu_ops=True)
    # Legacy runs still receive exactly their existing ambient settings.
    legacy = host._child_environment(sys.executable, 0.9, require_accelerator=False)
    result = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ['XLA_FLAGS'], end='')"],
        env=legacy,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout == inherited


@pytest.mark.parametrize(
    "preallocate,flags,unified,expected,allocator",
    [
        (True, "--xla_gpu_command_buffer_update_mode=SKIP_TEMP", "false", "requires.*ALWAYS_UPDATE", "bfc"),
        (False, "--xla_gpu_command_buffer_update_mode SKIP_TEMP", "false", "requires.*ALWAYS_UPDATE", "bfc"),
        (True, "", "true", "requires disabled TF_FORCE_UNIFIED_MEMORY", "bfc"),
        (False, "", "invalid", "requires disabled TF_FORCE_UNIFIED_MEMORY", "bfc"),
        (True, "--xla_gpu_temp_buffer_use_separate_color=true", "false", "requires disabled.*separate_color", "bfc"),
        (False, "--xla_gpu_temp_buffer_use_separate_color true", "false", "requires disabled.*separate_color", "bfc"),
        (True, "--xla_gpu_temp_buffer_use_separate_color", "false", "requires disabled.*separate_color", "bfc"),
        (False, "--xla_gpu_temp_buffer_use_separate_color=invalid", "false", "requires disabled.*separate_color", "bfc"),
        (True, "", "false", "requires bfc, or vmm with preallocate=false", "vmm"),
    ],
)
def test_native_allocation_overrides_reject_before_backend_or_model(preallocate, flags, unified, expected, allocator):
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    environment = host._child_environment(sys.executable, 0.9, require_accelerator=False, preallocate=preallocate)
    # If policy validation stops preceding initialization, this invalid backend
    # produces a different error before any model can be read.
    environment.update(JAX_PLATFORMS="allocation-policy-test-invalid-backend", TF_FORCE_UNIFIED_MEMORY=unified)
    environment["XLA_PYTHON_CLIENT_ALLOCATOR"] = allocator
    environment["XLA_FLAGS"] = environment.get("XLA_FLAGS", "") + " " + flags
    child = LearnerProcess(sys.executable, environment=environment, timeout_seconds=30)
    try:
        with pytest.raises(RuntimeError, match=expected):
            child.request(
                "initialize",
                {
                    "runtime": {"allocator": allocator, "preallocate": preallocate, "device_memory_fraction": 0.9},
                    "distributed": {"num_processes": 1, "local_device_count": 1},
                    "require_accelerator": False,
                },
            )
        assert child.process.wait(timeout=5) != 0
    finally:
        child.close()


def test_numpy_import_policy_mismatch_rejects_before_backend_or_model():
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    environment = host._child_environment(sys.executable, 0.9, require_accelerator=False, numpy_madvise_hugepage=False)
    environment["JAX_PLATFORMS"] = "numpy-policy-test-invalid-backend"
    child = LearnerProcess(sys.executable, environment=environment, timeout_seconds=30)
    try:
        with pytest.raises(RuntimeError, match="Requested NumPy madvise-hugepage policy is not effective"):
            child.request(
                "initialize",
                {
                    "runtime": {
                        "numpy_madvise_hugepage": True,
                        "allocator": "bfc",
                        "preallocate": False,
                        "device_memory_fraction": 0.9,
                    },
                    "distributed": {"num_processes": 1, "local_device_count": 1},
                    "require_accelerator": False,
                },
            )
        assert child.process.wait(timeout=5) != 0
    finally:
        child.close()


def _hf_initializer(path):
    vocab = {"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>")),
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    config = SnowballConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", 64), config, key=jax.random.key(31))
        config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).save_pretrained(
            model,
            str(path),
            save_tokenizer=False,
            save_reference_code=False,
            dtype=jnp.float32,
        )


def _request_all(children, method, payloads):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(children)) as pool:
        futures = [
            pool.submit(child.request, method, payload) for child, payload in zip(children, payloads, strict=True)
        ]
        done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        error = next((future.exception() for future in done if future.exception() is not None), None)
        if error is not None:
            for child in children:
                child.close()
            raise error
        return [future.result() for future in futures]


def _same_request(children, method, payload):
    return _request_all(children, method, [payload] * len(children))


@contextmanager
def _learner_processes(config, process_count, fast_bwd_override_rank=None):
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo"}
    numpy_policy = config["runtime"].get("numpy_madvise_hugepage")
    if numpy_policy is not None:
        environment["NUMPY_MADVISE_HUGEPAGE"] = str(int(numpy_policy))
    environment.pop("XLA_CLIENT_MEM_FRACTION", None)
    environment["XLA_PYTHON_CLIENT_ALLOCATOR"] = config["runtime"]["allocator"]
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config["runtime"]["device_memory_fraction"])
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(config["runtime"]["preallocate"]).lower()
    environment["TF_FORCE_UNIFIED_MEMORY"] = "false"
    if config["runtime"]["preallocate"]:
        environment["XLA_FLAGS"] = environment.get("XLA_FLAGS", "") + " --xla_gpu_temp_buffer_use_separate_color=false"
    environment.pop("LEVANTER_CE_XLA_FAST_BWD", None)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    children = []
    child_environments = []
    try:
        for rank in range(process_count):
            child_environment = dict(environment)
            if config["runtime"].get("validate_checkpoint_roundtrip", False):
                host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
                child_environment = host._child_environment(
                    sys.executable,
                    0.9,
                    require_accelerator=False,
                    compiler_diagnostics=True,
                    preallocate=config["runtime"]["preallocate"],
                    allocator=config["runtime"]["allocator"],
                    numpy_madvise_hugepage=numpy_policy,
                )
                child_environment["JAX_CPU_COLLECTIVES_IMPLEMENTATION"] = "gloo"
            if rank == fast_bwd_override_rank:
                child_environment["LEVANTER_CE_XLA_FAST_BWD"] = "1"
            child_environments.append(child_environment)
            children.append(LearnerProcess(sys.executable, environment=child_environment, timeout_seconds=60))
        configs = []
        for rank in range(process_count):
            local = copy.deepcopy(config)
            local["distributed"].update(process_id=rank, coordinator_address=address)
            if config["runtime"].get("validate_checkpoint_roundtrip", False):
                # Each native child has its own actual compiler output directory.
                local["compiler_dump_dir"] = shlex.split(child_environments[rank]["XLA_FLAGS"])[-3].split("=", 1)[1]
            configs.append(local)
        receipts = _request_all(children, "initialize", configs)
        yield children, receipts, configs
    finally:
        for child in children:
            child.close()


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "process_count,host_restart,projection_dtype,fast_bwd_override_rank,compiler_failure,preallocate,allocator",
    [
        pytest.param(1, False, "bfloat16", None, False, False, "bfc", id="bf16-single"),
        pytest.param(
            2, False, "bfloat16", None, False, False, "bfc", marks=pytest.mark.multidevice, id="bf16-multihost"
        ),
        pytest.param(1, True, "bfloat16", None, False, False, "bfc", id="bf16-host-restart"),
        pytest.param(
            2, True, "bfloat16", None, False, False, "bfc", marks=pytest.mark.multidevice, id="bf16-multihost-restart"
        ),
        pytest.param(1, False, "float32", None, False, False, "bfc", id="default-fp32"),
        pytest.param(
            4, False, "bfloat16", 3, False, False, "bfc", marks=pytest.mark.multidevice, id="bf16-override-rejected"
        ),
        pytest.param(1, False, "bfloat16", None, True, False, "bfc", id="compiler-evidence-failure"),
        pytest.param(1, False, "bfloat16", None, False, True, "bfc", id="preallocated-native-continuation"),
        pytest.param(1, True, "bfloat16", None, False, False, "vmm", id="vmm-native-continuation"),
    ],
)
def test_native_learner_subprocess_updates_and_resumes_exactly(
    tmp_path,
    process_count,
    host_restart,
    projection_dtype,
    fast_bwd_override_rank,
    compiler_failure,
    preallocate,
    allocator,
    monkeypatch,
):
    host_module = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker") if host_restart else None
    initializer = tmp_path / "initializer"
    initializer.mkdir()
    _hf_initializer(initializer)
    local_count = jax.local_device_count(backend="cpu")
    count = local_count * process_count
    batch_size = count * 2
    numpy_policy = False if allocator == "vmm" else None
    if numpy_policy is not None:
        monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", "1")
    config = {
        "runtime": {
            "numpy_madvise_hugepage": numpy_policy,
            "allocator": allocator,
            "preallocate": preallocate,
            "device_memory_fraction": 0.9,
            "validate_checkpoint_roundtrip": host_restart or compiler_failure,
            "mesh_axes": {"replica_dcn": 1, "replica": 1, "data": count, "expert": 1, "model": 1, "context": 1},
            "checkpoint_dir": str(tmp_path),
            "projection_block_size": 16,
            "projection_logits_dtype": projection_dtype,
            "attention_implementation": "reference",
            "moe_implementation": "ring_gather_combine",
            "router_dot_fp32": True,
            "moe_capacity_factor": 8.0,
            "fp32_activations": False,
            "head_gate_fp32": True,
            "xsa_fp32": True,
        },
        "distributed": {
            "num_processes": process_count,
            "process_id": 0,
            "local_device_count": local_count,
            "coordinator_address": None,
        },
        "initialization_timeout_seconds": 60,
        "run_id": "subprocess-policy-regression",
        "seed": 41,
        "bf16": True,
        "num_training_steps": 3,
        "mini_batch_size": batch_size,
        "micro_batch_size_per_device": 1,
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "require_accelerator": False,
        "model": {"path": str(initializer), "revision": None},
        "optimizer": {
            "max_grad_norm": 0.5,
            "lr": 0.001,
            "adam_betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
    }
    output_dir = tmp_path / "attempt-output"
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(output_dir))
    if host_restart and process_count == 2:
        monkeypatch.delenv("SKYRL_DEBUG_ARTIFACT_DIR", raising=False)
    if compiler_failure:
        blocked = tmp_path / "blocked-upload"
        blocked.write_text("This regular file prevents the real uploader from creating its destination.")
        config["runtime"]["checkpoint_dir"] = str(blocked / "checkpoint")
    tokens = np.tile(np.array([0, 2, 3, 4, 5, 6], dtype=np.int32), (batch_size, 1))
    arrays = {
        "tokens": tokens,
        "attention_mask": tokens != 0,
        "position_ids": np.tile(np.array([0, 0, 1, 2, 3, 4], dtype=np.int32), (batch_size, 1)),
        "response_mask": np.ones((batch_size, 2), dtype=np.bool_),
    }
    payload = {"arrays": arrays, "response_length": 2, "global_step": 1}
    checkpoint = {"ckpt_dir": str(tmp_path / "checkpoint"), "include_fingerprints": True}
    with _learner_processes(config, process_count, fast_bwd_override_rank) as (children, receipts, configs):
        assert all(row["global_devices"] == count and row["microbatches"] == 2 for row in receipts)
        assert all(row["backend"] == "cpu" and row["pid"] > 0 and row["global_model_parameters"] > 0 for row in receipts)
        for row in receipts:
            policy = row["numpy_allocation_policy"]
            assert policy["requested_madvise_hugepage"] is numpy_policy
            assert policy["numpy_version"] == np.__version__
            if numpy_policy is False:
                assert policy["observed_madvise_hugepage"] is False and policy["environment_value"] == "0"
        assert all(
            row["allocation_policy"]
            == {"allocator": allocator, "memory_fraction": 0.9, "preallocate": preallocate, "unified_memory": False}
            for row in receipts
        )
        initial_finite = _same_request(children, "validate_state_finite", {"phase": "initial", "step": 0, "version": 0})
        for proof in initial_finite:
            for component in proof["components"].values():
                assert component["nonfinite_elements"] == 0
                assert component["finite_elements"] == component["elements"]
        if fast_bwd_override_rank is not None:
            # Keep every parent alive until all four real children report the
            # collective rejection; FIRST_EXCEPTION teardown would hide stragglers.
            with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as pool:
                futures = [pool.submit(child.request, "forward", payload) for child in children]
                errors = [future.exception() for future in futures]
            assert all(isinstance(error, RuntimeError) and "FAST_BWD to be unset" in str(error) for error in errors)
            assert all("[(3," in str(error) for error in errors)
            return
        scores = _same_request(children, "forward", payload)[0]["output"]
        arrays.update(
            {
                "old_logprobs": scores.copy(),
                "advantages": np.ones((batch_size, 2), dtype=np.float32),
                "policy_weights": np.full((batch_size, 2), 1 / (batch_size * 2), dtype=np.float32),
                "kl_weights": np.zeros((batch_size, 2), dtype=np.float32),
            }
        )
        if compiler_failure:
            arrays["advantages"].fill(np.nan)
            host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
            observer = host.LevanterHostWorker(
                host.OmegaConf.create(
                    {
                        "trainer": {
                            "placement": {"policy_num_gpus_per_node": local_count},
                            "policy": {
                                "levanter_config": {
                                    "validate_checkpoint_roundtrip": True,
                                    "deterministic_gpu_ops": False,
                                }
                            },
                        }
                    }
                ),
                0,
            )
            observer.child = children[0]
            observer.initialization_config = config
            observer._record_finite_state("initial", 0, 0, initial_finite[0])
            observer.sender = host.LevanterExpertBlockSender(observer.publication)
            observer.expert_block_rpc("inventory")
            # This real parent-side filesystem failure must not stop publication
            # or replace the native learner's subsequent genuine NaN failure.
            monkeypatch.setenv("IRIS_OUTPUT_DIR", str(blocked / "attempt-output"))
            observer.publication("begin", {"version": 0})
            assert observer.publication("next", {})["version"] == 0
            observer.publication("end", {})
            with pytest.raises(RuntimeError, match="Loss is NaN"):
                observer.train(payload)
            stages = list((output_dir / "native-compile").iterdir())
            assert len(stages) == 1
            stage = stages[0]
            before = json.loads((stage / "before-update.json").read_text())
            failed = json.loads((stage / "failure-final.json").read_text())
            assert before["compiled_memory"]["alias_size_in_bytes"] > 0
            assert "error_type" not in before
            assert failed["error_type"] == "RuntimeError"
            assert (
                failed["before_update_sha256"] == hashlib.sha256((stage / "before-update.json").read_bytes()).hexdigest()
            )
            for phase in ("before-update", "failure-final"):
                upload = json.loads((stage / f"upload-{phase}.json").read_text())
                assert upload["uploaded"] is False
                assert upload["uploader"]["backend"] == "cpu"
            with gzip.open(stage / "train.optimized.hlo.txt.gz", "rt") as hlo:
                assert "is_scheduled=true" in hlo.readline()
            assert any("buffer-assignment" in path.name for path in stage.iterdir())
            assert any("memory-usage-report" in path.name for path in stage.iterdir())
            return
        if host_restart:
            output_dir = tmp_path / "attempt-output"
            monkeypatch.setenv("IRIS_OUTPUT_DIR", str(output_dir))
            cfg = host_module.OmegaConf.create(
                {
                    "trainer": {
                        "placement": {"policy_num_gpus_per_node": local_count},
                        "policy": {
                            "levanter_config": {
                                "python_executable": sys.executable,
                                "device_memory_fraction": 0.9,
                                "preallocate": preallocate,
                                "allocator": allocator,
                                "numpy_madvise_hugepage": numpy_policy,
                                "command_timeout_seconds": 60,
                                "initialization_timeout_seconds": 60,
                                "validate_checkpoint_roundtrip": True,
                                "deterministic_gpu_ops": False,
                            }
                        },
                    },
                }
            )
            workers = [host_module.LevanterHostWorker(cfg, rank) for rank in range(process_count)]
            for worker, child, local in zip(workers, children, configs, strict=True):
                worker.child = child
                worker.initialization_config = local
                worker.sender = host_module.LevanterExpertBlockSender(worker.publication)
            with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as pool:
                list(pool.map(lambda worker: worker._set_pad_token_id(0), workers))
            if process_count == 2:
                # Block ONLY rank1's supplemental parent mirror. The required
                # child native-finite subtree remains writable on both hosts.
                (output_dir / "native-state-finite-step1-process-1.json").mkdir()
        before_models = [row["components"]["model"] for row in initial_finite]
        updates = _same_request(children, "train", payload)
        assert all(update["optimizer_step"] == 1 and update["router_bias_unchanged"] == 1 for update in updates)
        after = _same_request(children, "forward", payload)[0]["output"]
        assert np.max(np.abs(after - scores)) > 1e-5
        saved = _same_request(children, "save_checkpoint", checkpoint)
        for row in saved:
            for name in ("model", "opt_state", "training_key", "step"):
                assert row["state_fingerprints"]["components"][name]["addressable_bytes"] > 0
        if host_restart and process_count == 1:
            # A fresh, diagnostics-off learner receives the same frozen batch.
            # Exact state equality covers model, Adam, RNG, step and averaging.
            baseline = copy.deepcopy(config)
            baseline["runtime"]["validate_checkpoint_roundtrip"] = False
            baseline["runtime"]["numpy_madvise_hugepage"] = None
            with _learner_processes(baseline, 1) as (control, _, _):
                _same_request(control, "train", payload)
                actual = _same_request(control, "state_fingerprints", {})[0]
            assert actual == saved[0]["state_fingerprints"]
            stages = list((output_dir / "native-compile").iterdir())
            assert len(stages) == 1
            compile_receipt = json.loads((stages[0] / "before-update.json").read_text())
            assert compile_receipt["compiled_memory"]["alias_size_in_bytes"] > 0
            assert compile_receipt["selected_reports"]
            upload = json.loads((stages[0] / "upload-before-update.json").read_text())
            assert upload["uploaded"] and upload["uploader"]["backend"] == "cpu"
        if host_restart:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as pool:
                    futures = [
                        pool.submit(worker.validate_checkpoint_roundtrip, before, status)
                        for worker, before, status in zip(workers, before_models, updates, strict=True)
                    ]
                    roundtrips = [future.result() for future in futures]
            finally:
                children[:] = [worker.child for worker in workers if worker.child is not None]
            for rank, row in enumerate(roundtrips):
                assert row["previous_runtime"]["pid"] != row["restored_runtime"]["pid"]
                assert row["previous_runtime"]["backend"] == row["restored_runtime"]["backend"] == "cpu"
                assert row["restored_runtime"]["global_model_parameters"] > 0
                assert row["old_exit_code"] == 0
                assert row["model_change"]["global_model_changed"]
                assert row["saved"] == row["restored"]
                assert row["saved"]["state_fingerprints"] == saved[rank]["state_fingerprints"]
                if process_count == 2 and rank == 1:
                    mirror = row["finite_state_phases_at_restore"]["step1"]["parent_mirror"]
                    assert mirror["persisted"] is False and mirror["paths"] == []
                    assert "IsADirectoryError" in mirror["error_types"]
                assert workers[rank].publication("inventory", {})["version"] == 1
                assert len(workers[rank].expert_block_rpc("inventory")) == local_count
                artifact = output_dir / f"native-checkpoint-roundtrip-process-{rank}.json"
                assert host_module.json.loads(artifact.read_text()) == row
            for _ in range(2):
                with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as pool:
                    begun = [pool.submit(worker.publication, "begin", {"version": 1}) for worker in workers]
                    assert all(future.result()["version"] == 1 for future in begun)
                    payloads = [pool.submit(worker.publication, "next", {}) for worker in workers]
                    assert all(future.result()["version"] == 1 for future in payloads)
                    ended = [pool.submit(worker.publication, "end", {}) for worker in workers]
                    assert all(future.result()["version"] == 1 for future in ended)
                current_memory = [
                    (output_dir / f"native-memory-before-first-publication-next-process-{rank}.json").read_bytes()
                    for rank in range(process_count)
                ]
                if _ == 0:
                    first_memory = current_memory
                else:
                    assert current_memory == first_memory
        payload["global_step"] = 2
        if host_restart:
            with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as pool:
                futures = [pool.submit(worker.train, payload) for worker in workers]
                expected = [future.result() for future in futures]
            for rank in range(process_count):
                for phase in ("before-first-publication-next", "before-first-train"):
                    memory = json.loads((output_dir / f"native-memory-{phase}-process-{rank}.json").read_text())
                    assert memory["phase"] == phase
                    assert memory["native"]["optimizer_step"] == memory["native"]["version"] == 1
                    assert memory["native"]["allocation_policy"]["allocator"] == allocator
                    assert memory["native"]["devices"]
                    assert all(
                        row["device_type"] == "cpu" and "driver_global_free_bytes" not in row
                        for row in memory["parent_torch"]
                    )
        else:
            expected = _same_request(children, "train", payload)
        expected_scores = _same_request(children, "forward", payload)[0]["output"]
        expected_state = _same_request(children, "state_fingerprints", {})
        if host_restart:
            for rank in range(process_count):
                for phase in ("initial", "step1", "restored", "step2"):
                    path = output_dir / f"native-state-finite-{phase}-process-{rank}.json"
                    if process_count == 2 and rank == 1 and phase == "step1":
                        assert path.is_dir()
                    else:
                        finite = json.loads(path.read_text())
                        assert finite["status"] == "passed" and finite["parent_mirror"]["persisted"]
                    candidates = list(
                        (output_dir / "native-finite").glob(f"process-{rank}-pid-*/{phase}-candidate.json")
                    )
                    verdicts = list((output_dir / "native-finite").glob(f"process-{rank}-pid-*/{phase}-verdict.json"))
                    assert len(candidates) == len(verdicts) == 1
                    finite = json.loads(candidates[0].read_text())
                    verdict = json.loads(verdicts[0].read_text())
                    assert finite["ok"] and finite["complete"] and verdict["status"] == "passed"
                    assert all(peer["persistence"]["persisted"] for peer in verdict["peers"])
                    assert all(c["nonfinite_elements"] == 0 for c in finite["fingerprints"]["components"].values())
        else:
            assert (
                _same_request(children, "validate_state_finite", {"phase": "step2", "step": 2, "version": 2})
                == expected_state
            )
        if projection_dtype == "float32":
            # Coordinator EOF must finish owned resources and exit naturally too.
            for child in children:
                child.process.stdin.close()
            assert all(child.process.wait(timeout=5) == 0 for child in children)
        else:
            _same_request(children, "shutdown", {})
        assert all(child.process.returncode == 0 for child in children)

    with _learner_processes(config, process_count) as (children, _, _):
        restored = _same_request(children, "load_checkpoint", checkpoint)
        for expected_saved, actual_saved in zip(saved, restored, strict=True):
            assert actual_saved["step"] == actual_saved["version"] == 1
            assert actual_saved["state_fingerprints"] == expected_saved["state_fingerprints"]
        np.testing.assert_array_equal(_same_request(children, "forward", payload)[0]["output"], after)
        assert _same_request(children, "train", payload) == expected
        np.testing.assert_array_equal(_same_request(children, "forward", payload)[0]["output"], expected_scores)
        assert _same_request(children, "state_fingerprints", {}) == expected_state
        np.testing.assert_array_equal(arrays["old_logprobs"], scores)
        _same_request(children, "shutdown", {})
        assert all(child.process.returncode == 0 for child in children)


def test_shutdown_ack_marker_is_atomic_immutable_and_framework_free(tmp_path):
    marker = {"pid": os.getpid(), "starttime": 1, "nonce": "a" * 32, "ack_path": str(tmp_path / "ack.json")}
    script = """
import json, sys
from marin.rl.levanter_shutdown_observer import publish_ack_marker
assert 'jax' not in sys.modules and 'torch' not in sys.modules
marker = json.loads(sys.argv[1])
publish_ack_marker(marker, process_id=0)
first = open(marker['ack_path'], 'rb').read()
try:
    publish_ack_marker(marker, process_id=0)
except FileExistsError:
    pass
else:
    raise AssertionError('An existing acknowledgement was overwritten')
assert open(marker['ack_path'], 'rb').read() == first
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(marker)],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "JAX_PLATFORMS": "must-not-create-a-backend"},
    )
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path / "ack.json").read_text())
    assert row["event"] == "shutdown_ack_written" and row["nonce"] == marker["nonce"]
    assert row["pid"] == marker["pid"] and row["monotonic_ns"] > 0
    assert not (tmp_path / "ack.pending").exists()


@pytest.mark.parametrize("enabled,blocked", [(False, False), (True, False), (True, True)])
def test_real_protocol_shutdown_publishes_only_opt_in_ack_marker(tmp_path, capfd, enabled, blocked):
    if enabled and sys.platform != "linux":
        pytest.skip("Enabled marker validates the actual Linux /proc process identity")
    module = tmp_path / "native_marker_fixture.py"
    module.write_text(
        """
from marin.rl.levanter_worker_runtime import NativePolicyRuntime, main

def factory():
    runtime = NativePolicyRuntime()
    runtime.settings = {'runtime': {'diagnostic_shutdown_observer': True}, 'distributed': {'process_id': 0}}
    return runtime

if __name__ == '__main__':
    main(factory)
"""
    )
    marker_dir = tmp_path / "marker"
    marker_dir.mkdir(mode=0o700)
    if blocked:
        (marker_dir / "ack.pending").write_text("occupied")
    child = LearnerProcess(
        sys.executable,
        environment={
            **os.environ,
            "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "JAX_PLATFORMS": "must-not-create-a-backend",
        },
        timeout_seconds=30,
        runtime_module="native_marker_fixture",
    )
    try:
        payload = (
            {}
            if not enabled
            else {
                "diagnostic_shutdown_observer": {
                    "pid": child.process.pid,
                    "starttime": identity(child.process.pid).starttime,
                    "nonce": "b" * 32,
                    "ack_path": str(marker_dir / "ack.json"),
                }
            }
        )
        assert child.request("shutdown", payload) is None
        assert child.process.returncode == 0
        if blocked:
            assert not (marker_dir / "ack.json").exists()
            assert (marker_dir / "ack.pending").read_text() == "occupied"
            rows = [
                json.loads(line.split(" ", 1)[1])
                for line in capfd.readouterr().err.splitlines()
                if line.startswith("native_shutdown_receipt ")
            ]
            assert rows[-1]["observer_marker_error_type"] == "FileExistsError"
        elif enabled:
            row = json.loads((marker_dir / "ack.json").read_text())
            assert row["pid"] == child.process.pid and row["nonce"] == "b" * 32
            assert row["event"] == "shutdown_ack_written"
        else:
            assert list(marker_dir.iterdir()) == []
    finally:
        child.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires actual Linux pidfds and /proc")
def test_guardian_owner_gone_before_pidfd_open_terminates_only_bound_target(tmp_path):
    # Deliberately omit PDEATHSIG in this target: the guardian must enforce the
    # ownership contract itself, rather than accidentally passing via the kernel.
    waiting = "import sys; sys.stdin.buffer.read()"
    owner = subprocess.Popen([sys.executable, "-c", waiting], stdin=subprocess.PIPE)
    target = subprocess.Popen([sys.executable, "-c", waiting], stdin=subprocess.PIPE)
    sentinel = subprocess.Popen([sys.executable, "-c", waiting], stdin=subprocess.PIPE)
    try:
        expected_owner = identity(owner.pid)
        expected_target = identity(target.pid)
        owner.communicate(timeout=5)
        assert not Path(f"/proc/{owner.pid}").exists()
        config = {
            "pid": expected_target.pid,
            "starttime": expected_target.starttime,
            "pgrp": expected_target.pgrp,
            "owner_pid": expected_owner.pid,
            "owner_starttime": expected_owner.starttime,
            "nonce": "d" * 32,
            "debugger_log_path": str(tmp_path / "debugger.log"),
            "guardian_receipt_path": str(tmp_path / "guardian.json"),
            "lldb_receipt_path": str(tmp_path / "absent-lldb.json"),
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        guardian = Path(__file__).parents[2] / "lib/marin/src/marin/rl/native_shutdown/observer_guardian.py"
        result = subprocess.run(
            [sys.executable, "-I", str(guardian), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        row = json.loads((tmp_path / "guardian.json").read_text())
        assert result.returncode != 0 and row["status"] == "error"
        assert row["debugger_started"] is False
        # This returns only after the guardian's pidfd reported target exit. The
        # parent alone subsequently reaps its native child; the guardian does not.
        assert row["orphan_target_kill_sent"] and row["orphan_target_exit_observed"]
        exited = os.waitid(os.P_PID, target.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert exited is not None and exited.si_status == signal.SIGKILL
        assert target.wait(timeout=2) == -signal.SIGKILL
        assert sentinel.poll() is None
    finally:
        for child in (owner, target, sentinel):
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


@pytest.mark.integration
@pytest.mark.parametrize("process_id", [0, 2])
@pytest.mark.skipif(sys.platform != "linux", reason="Requires actual Linux pidfds, LLDB and /proc")
def test_public_worker_retry_keeps_original_child_after_missing_observer_receipt(tmp_path, monkeypatch, process_id):
    if not Path("/usr/bin/lldb").is_file() or shutil.which("timeout") is None:
        pytest.skip("Requires the Linux observer image's LLDB and GNU timeout")
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    config = host.OmegaConf.load(Path(host.__file__).parents[2] / "config/ppo_base_config.yaml")
    config.trainer.placement.colocate_all = False
    config.trainer.algorithm.use_kl_loss = False
    config.trainer.policy.levanter_config.diagnostic_shutdown_observer = True
    # A subprocess-boundary sentinel makes an unintended replacement observable
    # without running a second learner or touching any accelerator.
    created = tmp_path / "replacement-created"
    replacement = tmp_path / "replacement-python"
    replacement.write_text("#!/bin/sh\nprintf created > " + shlex.quote(str(created)) + "\nexit 70\n")
    replacement.chmod(0o700)
    config.trainer.policy.levanter_config.python_executable = str(replacement)
    worker = host.LevanterHostWorker(config, process_id)
    worker.initialization_config = host._runtime_config(
        config,
        model_path="unused-local-path",
        num_training_steps=1,
        process_id=process_id,
        coordinator_address="unused-local-address",
    )
    # An uninitialized actual protocol child is an ownership control only. This
    # neither initializes a CPU model nor changes HostWorker's accelerator policy.
    child = LearnerProcess(
        sys.executable, environment={**os.environ, "JAX_PLATFORMS": "must-not-create-a-backend"}, timeout_seconds=30
    )
    worker.child = child
    monkeypatch.setenv("SKYRL_DEBUG_ARTIFACT_DIR", str(tmp_path))
    receipt_path = None
    saved_receipt = None
    try:
        worker._prepare_shutdown_observer()
        observer = worker._shutdown_observer
        Path(observer.config["cancel_path"]).touch()
        observer.guardian.wait(timeout=3)
        receipt_path = Path(observer.config["guardian_receipt_path"])
        saved_receipt = receipt_path.read_bytes()
        assert json.loads(saved_receipt)["safe_cleanup"] is True
        # Actual filesystem loss reproduces unknown cleanup in the public worker;
        # the tracer itself is already safely retired, so this control can recover.
        receipt_path.unlink()
        with pytest.raises(FileNotFoundError):
            worker._retire_shutdown_observer(cancel=True)
        # Let the original process exit on real EOF, but preserve its unreaped
        # numeric PID while ownership is unknown. A hidden poll/close would reap
        # this zombie and is observable through waitid(WNOWAIT), without mocks.
        descriptor = os.pidfd_open(child.process.pid, 0)
        try:
            child.process.stdin.close()
            assert select.select([descriptor], [], [], 30)[0]
        finally:
            os.close(descriptor)
        before = os.waitid(os.P_PID, child.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert before is not None and before.si_status == 0
        owned_before_retry = direct_children()
        with pytest.raises(RuntimeError):
            worker.init_model("unused-local-path", 1)
        with pytest.raises(RuntimeError):
            worker._start_child()
        with pytest.raises(RuntimeError):
            worker.close_telemetry()
        assert not created.exists()
        assert direct_children() == owned_before_retry
        assert worker.child is child and child.process.returncode is None
        after = os.waitid(os.P_PID, child.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert after is not None and after.si_pid == before.si_pid and after.si_status == 0
    finally:
        if saved_receipt is not None:
            receipt_path.write_bytes(saved_receipt)
            worker._retire_shutdown_observer(cancel=True)
        # Recover only after the real safe teardown proof has been restored.
        if worker._shutdown_observer is None:
            child.close()
        else:
            raise RuntimeError("Unknown observer cleanup; isolated integration task teardown is required")


@pytest.mark.parametrize("process_id", [0, 1, 2, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_shutdown_observation_covers_each_opted_in_native_worker(tmp_path, monkeypatch, process_id, enabled):
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    config = host.OmegaConf.load(Path(host.__file__).parents[2] / "config/ppo_base_config.yaml")
    config.trainer.policy.levanter_config.diagnostic_shutdown_observer = enabled
    worker = host.LevanterHostWorker(config, process_id)
    worker.child = SimpleNamespace(process=SimpleNamespace(pid=7321))
    monkeypatch.setenv("SKYRL_DEBUG_ARTIFACT_DIR", str(tmp_path))

    class GuardianProcessBoundary:
        # Portable selection coverage stops at the real Linux subprocess
        # boundary; pidfd/ptrace lifetime qualification remains a Linux control.
        def __init__(self, process, destination, native_rank):
            self.destination = destination
            self.target = {"pid": process.pid, "native_rank": native_rank}

        def wait_ready(self):
            (self.destination / "guardian-ready.json").write_text(json.dumps(self.target))

    monkeypatch.setattr(host, "ShutdownObserver", GuardianProcessBoundary)
    worker._prepare_shutdown_observer()
    ready = tmp_path / "guardian-ready.json"
    if enabled:
        assert json.loads(ready.read_text()) == {"pid": 7321, "native_rank": process_id}
    else:
        assert not ready.exists()
