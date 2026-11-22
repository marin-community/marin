# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real subprocess update and checkpoint continuation at the learner boundary."""

import concurrent.futures
import copy
import gzip
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
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
from marin.rl.levanter_worker_runtime import NativePolicyRuntime
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


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


@pytest.mark.parametrize("require_accelerator", [True, False])
def test_native_child_backend_intent_overrides_inherited_coordinator_platform(require_accelerator, monkeypatch):
    host_module = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu" if require_accelerator else "cuda,cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    monkeypatch.setenv("JAX_CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    environment = host_module._child_environment(sys.executable, 0.9, require_accelerator=require_accelerator)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
import jax
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
    if require_accelerator:
        if receipt["initialized"]:
            assert receipt["backend"] == "gpu"
        else:
            assert "cuda" in receipt["error"].lower()
    else:
        assert receipt["initialized"] and receipt["backend"] == "cpu"


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
                    sys.executable, 0.9, require_accelerator=False, compiler_diagnostics=True
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
    "process_count,host_restart,projection_dtype,fast_bwd_override_rank,compiler_failure",
    [
        pytest.param(1, False, "bfloat16", None, False, id="bf16-single"),
        pytest.param(2, False, "bfloat16", None, False, marks=pytest.mark.multidevice, id="bf16-multihost"),
        pytest.param(1, True, "bfloat16", None, False, id="bf16-host-restart"),
        pytest.param(2, True, "bfloat16", None, False, marks=pytest.mark.multidevice, id="bf16-multihost-restart"),
        pytest.param(1, False, "float32", None, False, id="default-fp32"),
        pytest.param(4, False, "bfloat16", 3, False, marks=pytest.mark.multidevice, id="bf16-override-rejected"),
        pytest.param(1, False, "bfloat16", None, True, id="compiler-evidence-failure"),
    ],
)
def test_native_learner_subprocess_updates_and_resumes_exactly(
    tmp_path, process_count, host_restart, projection_dtype, fast_bwd_override_rank, compiler_failure, monkeypatch
):
    host_module = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker") if host_restart else None
    initializer = tmp_path / "initializer"
    initializer.mkdir()
    _hf_initializer(initializer)
    local_count = jax.local_device_count(backend="cpu")
    count = local_count * process_count
    batch_size = count * 2
    config = {
        "runtime": {
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
    if host_restart or compiler_failure:
        output_dir = tmp_path / "attempt-output"
        monkeypatch.setenv("IRIS_OUTPUT_DIR", str(output_dir))
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
            with pytest.raises(RuntimeError, match="Loss is NaN"):
                children[0].request("train", payload)
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
        before_models = _same_request(children, "model_fingerprint", {})
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
                                "command_timeout_seconds": 60,
                                "initialization_timeout_seconds": 60,
                                "validate_checkpoint_roundtrip": True,
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
                assert workers[rank].publication("inventory", {})["version"] == 1
                assert len(workers[rank].expert_block_rpc("inventory")) == local_count
                artifact = output_dir / f"native-checkpoint-roundtrip-process-{rank}.json"
                assert host_module.json.loads(artifact.read_text()) == row
        payload["global_step"] = 2
        expected = _same_request(children, "train", payload)
        expected_scores = _same_request(children, "forward", payload)[0]["output"]
        expected_state = _same_request(children, "state_fingerprints", {})
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
