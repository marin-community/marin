# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The LR0 job binds the reviewed callbacks, gate inputs and receipt bytes; it reads no real data here."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import cloudpickle
import pytest
from levanter.models.snowball import SnowballConfig
from levanter.tracker import NoopConfig

from experiments.post_training.identity_context_distill import lr0_probe
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
from experiments.post_training.identity_context_distill.test_runtime_recipe import inputs

RUN_ROOT = "s3://marin-rno2a/fixture/run"
T310_LOCAL = Path(os.environ.get("T310_LOCAL", Path.home() / "Dropbox/marin-notes/snowball-identity/t310"))
REVIEWED_RECEIPTS = {
    "runtime-recipe-4188e87c-combine-off.json": (
        "dcddbf4d59d0863a773e11a97002567dbea622fab52506d9c716fa4b4a30b7d2",
        False,
    ),
    "runtime-recipe-4188e87c-combine-on.json": (
        "d123ddbdbeed0c3cc843cd247e23d74cb2b21e7538056150da8139843c81fddd",
        True,
    ),
}
RECEIPT_INPUTS = T310_LOCAL / "receipt-inputs"
"""Local copies of the pinned source ``config.json`` / ``generation_config.json`` and the routing ``routing.json``."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(
    monkeypatch,
    router_combine_fp32: bool = False,
    fp32_activations: bool = False,
    attention_implementation: str = "torch_order",
    xla_precision: str = "default",
):
    """Bind synthetic source bytes and the process's recorded XLA flags."""
    config, args = inputs()
    assert isinstance(config.model, SnowballConfig)
    model = replace(
        config.model,
        router_combine_fp32=router_combine_fp32,
        fp32_activations=fp32_activations,
        attention_implementation=attention_implementation,
    )
    config = replace(config, model=model)
    _, receipt = configure_precision_runtime(config, **dict(args, xla_precision=xla_precision))
    for name, value in lr0_probe.xla_precision_env(xla_precision).items():
        monkeypatch.setenv(name, value)
    if xla_precision == "default":
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    receipt_bytes = json.dumps(receipt).encode()
    files = {
        "config.json": args["config_bytes"],
        "generation_config.json": args["generation_bytes"],
        "routing.json": args["rounding_bytes"],
    }

    class Path:
        def __init__(self, path):
            self.path = str(path)

        def __str__(self):
            return self.path

        def __truediv__(self, name):
            return Path(f"{self.path.rstrip('/')}/{name}")

        def read_bytes(self):
            return files[self.path.rsplit("/", 1)[-1]]

    monkeypatch.setattr(lr0_probe, "StoragePath", Path)
    monkeypatch.setattr(lr0_probe, "check_pod_storage", lambda: None)
    return receipt_bytes


def binding_worker(monkeypatch, receipt_bytes, receipt_sha256, route_policy):
    calls: dict = {"writer": 0}
    monkeypatch.setattr(lr0_probe, "make_pre_update_gate", lambda inputs: calls.setdefault("gate", inputs))

    def writer(action):
        calls["writer"] += 1
        return action()

    def probe(config, **kwargs):
        calls["config"], calls["kwargs"] = config, kwargs
        return {"completed_updates": 1}

    monkeypatch.setattr(lr0_probe, "run_on_export_writer", writer)
    monkeypatch.setattr(
        lr0_probe, "write_export_json", lambda path, value: calls.setdefault("result", (path.path, value))
    )
    monkeypatch.setattr(lr0_probe, "run_precision_probe", probe)
    lr0_probe.lr0_worker(RUN_ROOT, "t310-lr0", receipt_bytes, receipt_sha256 or sha(receipt_bytes), route_policy)
    callbacks = calls["kwargs"]
    audits = []
    monkeypatch.setattr(lr0_probe, "audit_precision_source", lambda *args: audits.append(("source", *args)))
    monkeypatch.setattr(lr0_probe, "audit_initialized_masters", lambda *args: audits.append(("master", *args)))
    monkeypatch.setattr(
        lr0_probe, "audit_precision_export", lambda *args, receipt_path: audits.append(("export", *args, receipt_path))
    )

    class State:
        class model:
            @staticmethod
            def to_state_dict():
                return {"w": 1}

    callbacks.pop("source_audit")("SRC")
    master = callbacks.pop("master_audit")
    master(State, 0)
    master(State, 1)
    with pytest.raises(ValueError, match="master audit step"):
        master(State, 2)
    callbacks.pop("export_audit")("SRC", "EXP")
    receipts = {}
    monkeypatch.setattr(lr0_probe, "write_json", lambda path, value: receipts.__setitem__(path, value))
    rows = [{"process": process, "local_device": 0, "peak_bytes_in_use": 7} for process in (0, 1)]
    callbacks.pop("post_update_receipt")(rows)
    calls.update(audits=audits, rank_receipts=receipts)
    return calls


def run_worker(receipt_bytes, receipt_sha256=None, route_policy="G1a"):
    """Run callback-binding checks in a fresh process with the real pre-backend guard.

    Source/pod storage, gate/probe and source/master/export callbacks, and
    export/rank receipt writers are fakes. The real guard precedes the
    rank callback's first JAX query; test_lr0_startup supplies the real
    distributed lifecycle.
    """
    payload = (receipt_bytes, receipt_sha256, route_policy, lr0_probe.StoragePath, lr0_probe.check_pod_storage)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.pkl"
        child = subprocess.run(
            [sys.executable, __file__, "--binding-worker", str(output)],
            input=cloudpickle.dumps(payload),
            capture_output=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
            timeout=60,
        )
        assert child.returncode == 0, child.stderr.decode()[-4000:]
        ok, result = cloudpickle.loads(output.read_bytes())
    if not ok:
        raise result
    return result


def test_worker_binds_the_measured_recipe_and_reviewed_callbacks(monkeypatch):
    receipt_bytes = fixture(monkeypatch)
    receipt = json.loads(receipt_bytes)
    calls = run_worker(receipt_bytes)
    config, kwargs = calls["config"], calls["kwargs"]
    assert config.model.router_dot_fp32 is True
    assert (config.model.moe_capacity_factor, config.model.router_combine_fp32) == (4.0, False)
    assert config.trainer.compute_dtype_overrides == {"router_bias": "float32"}
    assert isinstance(config.trainer.tracker, NoopConfig)
    assert config.hf_save_path == f"{RUN_ROOT}/hf"
    # One JAX slice per H100 host: context crosses hosts, as in the #302 SFT.
    assert dict(config.trainer.mesh.dcn_axes) == {"replica_dcn": 1, "context": 4}
    assert list(config.trainer.mesh.compute_mapping["batch"]) == ["replica_dcn", "data", "expert"]
    assert list(config.trainer.mesh.param_mapping["embed"]) == ["data", "context"]
    assert lr0_probe.mesh_layout(config.trainer.mesh)["expected_num_slices"] == 4
    assert kwargs["expected_runtime_receipt"] == receipt
    assert kwargs["expected_source_receipt"] == receipt["source_metadata"]
    # The launch's route policy reaches the gate.
    assert calls["gate"].route_policy == "G1a" and kwargs["pre_update_gate"] is calls["gate"]
    # The result write is collective: the worker calls it on every rank, not behind a rank-0 guard.
    assert calls["writer"] == 1
    assert calls["result"] == (f"{RUN_ROOT}/lr0-result.json", {"completed_updates": 1})

    source_receipt = f"{RUN_ROOT}/audits/source-tensor-hashes.json"
    assert calls["audits"] == [
        ("source", "SRC", lr0_probe.SOURCE_FILE_HASHES, source_receipt),
        ("master", {"w": 1}, source_receipt, f"{RUN_ROOT}/audits/master-step-0.json"),
        ("master", {"w": 1}, source_receipt, f"{RUN_ROOT}/audits/master-step-1.json"),
        ("export", "SRC", "EXP", lr0_probe.SOURCE_FILE_HASHES, "EXP/precision-audit.json"),
    ]
    # After the update, each rank writes its own devices' memory into its gate rank receipt.
    rows = [{"process": process, "local_device": 0, "peak_bytes_in_use": 7} for process in (0, 1)]
    assert calls["rank_receipts"] == {
        f"{RUN_ROOT}/precision-gate/ranks/0.json": {
            "process": 0,
            "stage": "updated",
            "device_memory_after_update": rows[:1],
        }
    }


def test_worker_takes_the_runtime_settings_from_the_bound_receipt(monkeypatch):
    receipt_bytes = fixture(
        monkeypatch,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        xla_precision="no_excess_precision",
    )
    model = run_worker(receipt_bytes)["config"].model
    assert (model.moe_capacity_factor, model.router_combine_fp32, model.fp32_activations) == (4.0, True, True)
    assert model.attention_implementation == "torch_order"


def test_worker_refuses_a_process_without_the_receipt_xla_precision(monkeypatch):
    receipt_bytes = fixture(monkeypatch, xla_precision="no_excess_precision")
    monkeypatch.delenv("XLA_FLAGS")
    with pytest.raises(ValueError, match="XLA precision"):
        run_worker(receipt_bytes)


def test_base_config_refuses_a_receipt_without_the_runtime_settings():
    _, args = inputs()
    names = ("moe_capacity_factor", "router_combine_fp32", "fp32_activations", "attention_implementation")
    for name in names:
        runtime: dict = {"moe_capacity_factor": 4.0, "router_combine_fp32": True, "fp32_activations": True}
        runtime["attention_implementation"] = "torch_order"
        receipt = {"schema_version": 6, "runtime_model_config": runtime}
        del runtime[name]
        with pytest.raises(ValueError, match=name):
            lr0_probe.base_config(RUN_ROOT, "t310-lr0", args["config_bytes"], args["generation_bytes"], receipt)
    receipt = {"schema_version": 5, "runtime_model_config": dict.fromkeys(names)}
    with pytest.raises(ValueError, match="schema 5"):
        lr0_probe.base_config(RUN_ROOT, "t310-lr0", args["config_bytes"], args["generation_bytes"], receipt)


def test_approved_receipt_checks_the_passed_sha():
    data = json.dumps({"schema_version": 4}).encode()
    assert lr0_probe.approved_receipt(data, sha(data)) == {"schema_version": 4}
    with pytest.raises(ValueError, match="differs from the reviewed receipt"):
        lr0_probe.approved_receipt(data, "0" * 64)


@pytest.mark.skipif(
    not all((T310_LOCAL / name).exists() for name in REVIEWED_RECEIPTS) or not RECEIPT_INPUTS.exists(),
    reason="local t310 copies",
)
@pytest.mark.parametrize("name", sorted(REVIEWED_RECEIPTS))
def test_schema4_reviewed_receipts_are_refused(name):
    """The reviewed schema-4 receipts predate ``fp32_activations`` and the attention and XLA settings: they still
    pass ``approved_receipt`` at their sha256, but ``base_config`` refuses them, so a run needs a schema-6 receipt."""
    receipt_sha256, combine = REVIEWED_RECEIPTS[name]
    receipt = lr0_probe.approved_receipt((T310_LOCAL / name).read_bytes(), receipt_sha256)
    assert receipt["schema_version"] == 4 and receipt["router_combine_fp32"] is combine
    config_bytes = (RECEIPT_INPUTS / "config.json").read_bytes()
    generation_bytes = (RECEIPT_INPUTS / "generation_config.json").read_bytes()
    with pytest.raises(ValueError, match="schema 4"):
        lr0_probe.base_config(RUN_ROOT, "t310-lr0", config_bytes, generation_bytes, receipt)


def test_worker_refuses_unreviewed_or_mismatched_receipts(monkeypatch):
    receipt_bytes = fixture(monkeypatch)
    with pytest.raises(ValueError, match="Runtime receipt sha256"):
        run_worker(receipt_bytes + b" ", sha(receipt_bytes))
    changed = json.loads(receipt_bytes)
    changed["trigger"]["bias_only"] = True
    with pytest.raises(ValueError, match="Derived runtime recipe differs"):
        run_worker(json.dumps(changed).encode())


def test_gate_inputs_are_the_frozen_regional_inputs():
    routing = "s3://marin-rno2a/marin/experiments/snowball-identity/t310/torch-ref-queue-r1/results/r1/routing.json"
    gate = lr0_probe.gate_inputs(RUN_ROOT, routing, "G1")
    assert gate.output == f"{RUN_ROOT}/precision-gate"
    assert gate.torch_routing == routing.rsplit("/", 1)[0]
    assert all(path.startswith("s3://marin-rno2a/") for path in (gate.positions, gate.vllm, gate.identity))
    assert gate.floor_receipt == f"{lr0_probe.T310}/diag-torch-floor-r1/receipt.json"
    assert gate.torch_top6 == f"{lr0_probe.T310}/diag-torch-floor-r1/batch1/top6.pt"
    assert gate.torch_batched_top6 == f"{lr0_probe.T310}/diag-torch-floor-r1/batched/top6.pt"
    assert (gate.batch, gate.pad_multiple, gate.deadline_seconds) == (8, 64, 1800)
    assert (gate.route_policy, gate.vllm_capture) == ("G1", f"{lr0_probe.T310}/vllm-route-capture-r1")
    with pytest.raises(ValueError, match="route_policy must be one of"):
        lr0_probe.gate_inputs(RUN_ROOT, routing, "G2")


def test_plan_has_no_retries_and_a_bounded_ceiling():
    config, _ = inputs()
    job_plan = lr0_probe.plan(
        "job",
        RUN_ROOT,
        7200,
        "s3://marin-rno2a/x/routing.json",
        "interactive",
        config.trainer.mesh,
        "ab" * 32,
        "no_excess_precision",
        "G1a",
    )
    assert job_plan["runtime_receipt_sha256"] == "ab" * 32
    assert job_plan["xla_precision"] == "no_excess_precision"
    assert job_plan["env_vars"] == {"XLA_FLAGS": "--xla_allow_excess_precision=false"}
    assert job_plan["route_policy"] == job_plan["gate"]["route_policy"] == "G1a"
    assert (job_plan["replicas"], job_plan["gpus"], job_plan["region"]) == (4, 32, "cw-rno2a")
    assert job_plan["resources"] == {"cpu": 64, "ram": "768GB", "disk": "1TB"}
    assert job_plan["priority"] == "interactive"
    assert job_plan["mesh"]["expected_num_slices"] == 4
    assert job_plan["retries_failure"] == job_plan["retries_preemption"] == 0
    assert job_plan["h100_hour_ceiling"] == 64


def test_pod_storage_check_names_every_missing_setting(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("FSSPEC_S3", "{}")
    monkeypatch.setattr(lr0_probe, "s3_credential_options", lambda: {"key": "k"})
    with pytest.raises(RuntimeError, match=r"AWS_ENDPOINT_URL.*AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"):
        lr0_probe.check_pod_storage()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://endpoint")
    monkeypatch.setattr(lr0_probe, "s3_credential_options", lambda: {"key": "k", "secret": "s"})
    lr0_probe.check_pod_storage()


def test_worker_checks_pod_storage_before_any_read(monkeypatch):
    receipt_bytes = fixture(monkeypatch)

    def missing():
        raise RuntimeError("Pod storage settings missing")

    monkeypatch.setattr(lr0_probe, "check_pod_storage", missing)
    monkeypatch.setattr(lr0_probe, "StoragePath", lambda path: pytest.fail("read before the storage check"))
    with pytest.raises(RuntimeError, match="Pod storage"):
        run_worker(receipt_bytes)


def test_io_smoke_runs_every_step_and_cleans_only_its_own_objects(monkeypatch, tmp_path):
    scratch = str(tmp_path / "scratch")
    config_bytes = b'{"config": true}'
    store = {
        f"{lr0_probe.BASE_MODEL}config.json": config_bytes,
        f"{lr0_probe.BASE_MODEL}model.safetensors.index.json": (
            json.dumps({"weight_map": {"a": "model-00002.safetensors", "b": "model-00001.safetensors"}}).encode()
        ),
    }
    opened = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            return b"x" * count

    def open_stream(url):
        opened.append(url)
        return Stream()

    routing = {"path": "s3://marin-rno2a/x/routing.json", "sha256": hashlib.sha256(b"routing").hexdigest()}
    monkeypatch.setitem(lr0_probe.SOURCE_FILE_HASHES, "config.json", hashlib.sha256(config_bytes).hexdigest())
    monkeypatch.setattr(lr0_probe, "check_pod_storage", lambda: None)
    monkeypatch.setattr(lr0_probe, "StoragePath", lambda path: _Local(str(path), store))
    monkeypatch.setattr(lr0_probe.precision_source_audit, "open_audit_stream", open_stream, raising=False)
    monkeypatch.setattr(
        lr0_probe, "write_export_json", lambda path, value: store.__setitem__(path.path, json.dumps(value).encode())
    )
    monkeypatch.setattr(lr0_probe, "write_json", lambda path, value: store.__setitem__(path, json.dumps(value).encode()))
    monkeypatch.setattr(lr0_probe, "save_positions", lambda records, path: store.__setitem__(path, b"pt"))
    monkeypatch.setattr(lr0_probe, "read", lambda path: store[path] if path in store else b"routing")
    monkeypatch.setattr(lr0_probe, "stored_sequences", lambda path: [1] * 465)
    monkeypatch.setattr(lr0_probe, "load_tokenizer", lambda root: object())

    class FS:
        def exists(self, path):
            return path in store

        def rm_file(self, path):
            del store[path]

    monkeypatch.setattr(lr0_probe, "filesystem_for", lambda path: (FS(), path))
    store["s3://marin-rno2a/keep"] = b"not ours"

    result = lr0_probe.io_smoke(scratch, routing)
    assert result["ok"], result
    assert list(result["steps"]) == [
        "pod_storage",
        "source_config",
        "source_shard_first_mb",
        "receipt_writer",
        "gate_writers",
        "gate_readers",
        "cleanup",
    ]
    assert opened == [f"{lr0_probe.BASE_MODEL}model-00001.safetensors"]
    assert result["steps"]["source_shard_first_mb"]["bytes"] == lr0_probe.SMOKE_BYTES
    assert result["steps"]["gate_readers"]["stored_sequences"] == 465
    assert not [path for path in store if path.startswith(scratch)]
    assert store["s3://marin-rno2a/keep"] == b"not ours"


def test_io_smoke_reports_a_failed_step_and_still_runs_the_rest(monkeypatch, tmp_path):
    def missing():
        raise RuntimeError("Pod storage settings missing")

    monkeypatch.setattr(lr0_probe, "check_pod_storage", missing)

    def unreachable(path):
        raise OSError(f"unreachable: {path}")

    monkeypatch.setattr(lr0_probe, "StoragePath", unreachable)
    monkeypatch.setattr(lr0_probe, "read", unreachable)
    monkeypatch.setattr(lr0_probe, "filesystem_for", unreachable)
    monkeypatch.setattr(lr0_probe, "write_json", lambda path, value: unreachable(path))
    result = lr0_probe.io_smoke(str(tmp_path), {"path": "p", "sha256": "s"})
    assert not result["ok"]
    assert result["steps"]["pod_storage"] == {"ok": False, "error": "RuntimeError('Pod storage settings missing')"}
    assert len(result["steps"]) == 7 and not any(step["ok"] for step in result["steps"].values())


class _Local:
    def __init__(self, path, store):
        self.path, self._store = path, store

    def __truediv__(self, name):
        return _Local(f"{self.path.rstrip('/')}/{name}", self._store)

    def __str__(self):
        return self.path

    def read_bytes(self):
        return self._store[self.path]


if __name__ == "__main__":
    receipt_bytes, receipt_sha, route_policy, storage, pod_storage = cloudpickle.loads(sys.stdin.buffer.read())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lr0_probe, "StoragePath", storage)
        patch.setattr(lr0_probe, "check_pod_storage", pod_storage)
        try:
            result = (True, binding_worker(patch, receipt_bytes, receipt_sha, route_policy))
        except Exception as error:
            result = (False, error)
    Path(sys.argv[2]).write_bytes(cloudpickle.dumps(result))
