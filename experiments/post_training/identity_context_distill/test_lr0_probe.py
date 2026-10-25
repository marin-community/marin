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
