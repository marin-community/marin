# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "snowball_zero_kl.py"
_SPEC = importlib.util.spec_from_file_location("snowball_zero_kl_benchmark", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.timeout(180)
def test_snowball_zero_kl_cpu_executes_teacher_prefill_and_cached_decode(tmp_path, monkeypatch):
    output = tmp_path / "cpu-reference.json"
    monkeypatch.chdir(tmp_path)
    _MODULE.main(
        [
            "--backend",
            "cpu",
            "--warmup",
            "0",
            "--measured",
            "1",
            "--source-sha",
            _MODULE.BASE_OID,
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())

    assert report["status"] == "success"
    assert report["metadata"]["source_sha"] == _MODULE.BASE_OID
    assert len(report["rows"]) == 20
    assert report["tpu_qualification"]["status"] == "unexecuted"
    assert report["parameter_identity"]["all_equal_bytes"]
    assert len({row["row_key"] for row in report["rows"]}) == len(report["rows"])
    assert {row["status"] for row in report["rows"]} == {"success"}
    assert all(row["input_cache_unchanged"] for row in report["rows"] if row["phase"] == "cached_decode")
    for arm in ("cpu_native_reference", "cpu_batch_invariant_norm_reference"):
        accuracy = report["accuracy"][arm]
        assert np.isfinite(accuracy["teacher_weighted_next_token_loss"])
        assert accuracy["gradient_finite"]
        assert len(accuracy["serving"]) == 2
        for serving in accuracy["serving"]:
            assert {decode["position"] for decode in serving["decode"]} == {3, 4, 7}
            # BF16 KV storage changes values; this is a reference-control error bound,
            # not an exact-bit promise for the separately unexecuted TPU arms.
            assert serving["teacher_vs_full_prefill"]["max_abs_error"] < 1e-4
            for decode in serving["decode"]:
                assert decode["teacher_vs_cached_decode"]["max_abs_error"] < 1e-4
                assert decode["prefill_vs_cached_decode"]["max_abs_error"] < 1e-4
    assert report["accuracy"]["cross_arm"]["gradient"]["finite"]
    assert report["accuracy"]["cross_arm"]["gradient"]["max_abs_error"] < 1e-4
    assert report["accuracy"]["cross_arm"]["teacher"]["max_abs_error"] < 1e-4
    assert report["accuracy"]["cross_arm"]["loss_abs_error"] < 1e-4
    json.dumps(report)


def test_snowball_zero_kl_metrics_distinguish_bits_and_detect_wrong_serving_logits():
    teacher = np.asarray([[0.0, 1.0, -2.0]], dtype=np.float32)
    signed_zero = teacher.copy()
    signed_zero[0, 0] = -0.0
    metrics = _MODULE._logit_metrics(teacher, signed_zero)
    assert not metrics["bitwise_fp32"]
    assert metrics["bit_mismatch_count"] == 1
    assert metrics["numerical_mismatch_count"] == 0

    wrong = teacher.copy()
    wrong[0, 2] += 0.25
    metrics = _MODULE._logit_metrics(teacher, wrong)
    assert metrics["max_abs_error"] == 0.25
    assert metrics["teacher_kl"] > 0.0


@pytest.mark.timeout(180)
def test_snowball_zero_kl_preserves_partial_report_when_public_loss_fails(tmp_path):
    mesh = _MODULE.compact_grug_mesh(expert_axis_size=1)
    config = dataclasses.replace(_MODULE.qualified_config(implementation="native", backend="cpu"), max_seq_len=8)
    with jax.set_mesh(mesh):
        model = _MODULE.SnowballLMHeadModel.init(hax.Axis("vocab", _MODULE.VOCAB_SIZE), config, key=jax.random.key(4))
    fixture = _MODULE.make_fixture(sequence_length=8, page_size=4)
    # A stale batch's weights must surface the real public-loss shape failure,
    # retaining the completed teacher row rather than losing the entire receipt.
    stale_weights = hax.named(jnp.ones((4, 7), dtype=jnp.float32), ("batch", "position"))
    fixture = dataclasses.replace(fixture, example=dataclasses.replace(fixture.example, loss_weight=stale_weights))
    output = tmp_path / "failed.json"
    args = _MODULE.BenchmarkArgs(warmup=0, measured=1, backend="cpu", output=output)
    with pytest.raises(ValueError):
        _MODULE._run_benchmark(
            args,
            backend="cpu",
            fixture=fixture,
            mesh=mesh,
            models=(("left", model), ("right", model)),
        )
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["accuracy_status"] == "incomplete"
    assert report["rows"][0]["phase"] == "teacher_forward"
    assert report["rows"][0]["status"] == "success"
    assert report["rows"][-1]["phase"] == "teacher_loss_grad"
    assert report["rows"][-1]["status"] == "failed"


@pytest.mark.timeout(180)
def test_snowball_zero_kl_accuracy_reporting_on_four_explicit_cpu_devices(tmp_path):
    output = tmp_path / "four-device-reference.json"
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "4",
        "PYTHONPATH": os.pathsep.join(
            (
                str(_MODULE_PATH.parents[1] / "src"),
                str(_MODULE_PATH.parents[2] / "haliax" / "src"),
            )
        ),
    }
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(_MODULE_PATH),
            "--backend",
            "cpu",
            "--warmup",
            "0",
            "--measured",
            "1",
            "--source-sha",
            _MODULE.BASE_OID,
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=150,
    )
    report = json.loads(output.read_text())
    assert report["metadata"]["local_device_count"] == 4
    assert report["metadata"]["mesh"]["data"] == 4
    assert report["status"] == "success"
    assert report["accuracy_status"] == "observed"
    for arm in ("cpu_native_reference", "cpu_batch_invariant_norm_reference"):
        for serving in report["accuracy"][arm]["serving"]:
            assert len(serving["decode"]) == 3
            assert all(case["teacher_vs_cached_decode"]["finite"] for case in serving["decode"])
