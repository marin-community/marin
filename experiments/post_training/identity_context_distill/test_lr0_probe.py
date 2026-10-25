# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The LR0 job binds the reviewed callbacks, gate inputs and receipt bytes; it reads no real data here."""

import hashlib
import json

import pytest
from levanter.tracker import NoopConfig

from experiments.post_training.identity_context_distill import lr0_probe
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
from experiments.post_training.identity_context_distill.test_runtime_recipe import inputs

RUN_ROOT = "s3://marin-rno2a/fixture/run"


def fixture(monkeypatch):
    """Fixture source/routing bytes behind a fake StoragePath, and the receipt they derive, as approved bytes."""
    config, args = inputs()
    _, receipt = configure_precision_runtime(config, **args)
    receipt_bytes = json.dumps(receipt).encode()
    files = {
        "config.json": args["config_bytes"],
        "generation_config.json": args["generation_bytes"],
        "routing.json": args["rounding_bytes"],
    }

    class Path:
        def __init__(self, path):
            self.path = str(path)

        def __truediv__(self, name):
            return Path(f"{self.path.rstrip('/')}/{name}")

        def read_bytes(self):
            return files[self.path.rsplit("/", 1)[-1]]

    monkeypatch.setattr(lr0_probe, "StoragePath", Path)
    monkeypatch.setattr(lr0_probe, "check_pod_storage", lambda: None)
    monkeypatch.setattr(lr0_probe, "RUNTIME_RECEIPT_SHA256", hashlib.sha256(receipt_bytes).hexdigest())
    return receipt_bytes


def run_worker(monkeypatch, receipt_bytes):
    calls: dict = {"writer": 0}

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
    lr0_probe.lr0_worker(RUN_ROOT, "t310-lr0", receipt_bytes)
    return calls


def test_worker_binds_the_measured_recipe_and_reviewed_callbacks(monkeypatch):
    receipt_bytes = fixture(monkeypatch)
    receipt = json.loads(receipt_bytes)
    calls = run_worker(monkeypatch, receipt_bytes)
    config, kwargs = calls["config"], calls["kwargs"]
    assert config.model.router_dot_fp32 is True
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
    # The result write is collective: the worker calls it on every rank, not behind a rank-0 guard.
    assert calls["writer"] == 1
    assert calls["result"] == (f"{RUN_ROOT}/lr0-result.json", {"completed_updates": 1})

    audits = []
    monkeypatch.setattr(lr0_probe, "audit_precision_source", lambda *args: audits.append(("source", *args)))
    monkeypatch.setattr(lr0_probe, "audit_initialized_masters", lambda *args: audits.append(("master", *args)))
    monkeypatch.setattr(lr0_probe, "audit_precision_export", lambda *args: audits.append(("export", *args)))

    class State:
        class model:
            @staticmethod
            def to_state_dict():
                return {"w": 1}

    kwargs["source_audit"]("SRC")
    kwargs["master_audit"](State, 0)
    kwargs["master_audit"](State, 1)
    kwargs["export_audit"]("SRC", "EXP")
    source_receipt = f"{RUN_ROOT}/audits/source-tensor-hashes.json"
    assert audits == [
        ("source", "SRC", lr0_probe.SOURCE_FILE_HASHES, source_receipt),
        ("master", {"w": 1}, source_receipt, f"{RUN_ROOT}/audits/master-step-0.json"),
        ("master", {"w": 1}, source_receipt, f"{RUN_ROOT}/audits/master-step-1.json"),
        ("export", "SRC", "EXP", lr0_probe.SOURCE_FILE_HASHES),
    ]
    with pytest.raises(ValueError, match="master audit step"):
        kwargs["master_audit"](State, 2)


def test_worker_refuses_unreviewed_or_mismatched_receipts(monkeypatch):
    receipt_bytes = fixture(monkeypatch)
    with pytest.raises(ValueError, match="reviewed derived receipt"):
        run_worker(monkeypatch, receipt_bytes + b" ")
    changed = json.loads(receipt_bytes)
    changed["trigger"]["bias_only"] = True
    changed_bytes = json.dumps(changed).encode()
    monkeypatch.setattr(lr0_probe, "RUNTIME_RECEIPT_SHA256", hashlib.sha256(changed_bytes).hexdigest())
    with pytest.raises(ValueError, match="differs from the reviewed receipt"):
        run_worker(monkeypatch, changed_bytes)


def test_gate_inputs_are_the_frozen_regional_inputs():
    routing = "s3://marin-rno2a/marin/experiments/snowball-identity/t310/torch-ref-queue-r1/results/r1/routing.json"
    gate = lr0_probe.gate_inputs(RUN_ROOT, routing)
    assert gate.output == f"{RUN_ROOT}/precision-gate"
    assert gate.torch_routing == routing.rsplit("/", 1)[0]
    assert all(path.startswith("s3://marin-rno2a/") for path in (gate.positions, gate.vllm, gate.identity))
    assert (gate.batch, gate.pad_multiple, gate.deadline_seconds) == (8, 64, 1800)


def test_plan_has_no_retries_and_a_bounded_ceiling():
    config, _ = inputs()
    job_plan = lr0_probe.plan(
        "job", RUN_ROOT, 7200, "s3://marin-rno2a/x/routing.json", "interactive", config.trainer.mesh
    )
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
        run_worker(monkeypatch, receipt_bytes)


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
