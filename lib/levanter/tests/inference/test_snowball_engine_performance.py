# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parents[4]
_BENCHMARK = _ROOT / "lib/levanter/benchmarks/snowball_engine_performance.py"


def _run_cli(tmp_path: Path, backend: str) -> tuple[subprocess.CompletedProcess, dict]:
    output = tmp_path / "engine.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    # Preserve the lexical checkout even when the existing venv has editable
    # packages pointing to a different worktree. No historical Git is needed.
    roots = [
        str(_ROOT / "lib" / name / "src")
        for name in ("levanter", "haliax", "marin", "iris", "fray", "zephyr", "rigging")
    ]
    env["PYTHONPATH"] = os.pathsep.join(roots + [env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, str(_BENCHMARK), "--backend", backend, "--source-sha", "0" * 40, "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=160,
    )
    assert output.exists(), result.stdout + result.stderr
    return result, json.loads(output.read_text())


@pytest.mark.timeout(180)
def test_real_cpu_cli_preserves_work_and_outputs_across_round_limits(tmp_path):
    result, report = _run_cli(tmp_path, "cpu")
    assert result.returncode == 0, result.stdout + result.stderr + str(report.get("error"))
    assert report["status"] == "passed"
    assert report["actual_backend"] == "cpu"
    assert report["provenance"]["source_sha_origin"] == "caller"
    configs = report["engine_configs"]
    one, many = dict(configs["1"]), dict(configs["32"])
    assert one.pop("max_rounds") == 1
    assert many.pop("max_rounds") == 32
    assert one == many
    assert report["provenance"]["benchmark_sha256"] == hashlib.sha256(_BENCHMARK.read_bytes()).hexdigest()
    engine_source = _ROOT / "lib/levanter/src/levanter/inference/engine.py"
    assert (
        report["provenance"]["source_file_sha256"]["src/levanter/inference/engine.py"]
        == hashlib.sha256(engine_source.read_bytes()).hexdigest()
    )
    workloads = {row["name"]: row for row in report["workloads"]}
    assert set(workloads) == {"homogeneous_burst", "heterogeneous_pending_reuse", "seeded_first_token_stop"}
    mixed = workloads["heterogeneous_pending_reuse"]
    assert len(mixed["requests"]) > configs["1"]["max_seqs"]
    assert len({len(request["prompt_tokens"]) for request in mixed["requests"]}) > 1
    assert len({request["max_num_tokens"] for request in mixed["requests"]}) > 1
    for workload in workloads.values():
        requests = workload["requests"]
        baseline = workload["calls"][0]["outputs"]
        assert len(baseline) == len(requests)
        assert workload["output_invariance_exact"]
        for call in workload["calls"]:
            assert call["status"] == "complete"
            assert call["outputs"] == baseline
            assert call["comparison_to_max_rounds1_first_call"]["exact"]
            assert call["input_tokens"] == sum(len(request["prompt_tokens"]) for request in requests)
            assert call["generated_tokens"] == sum(len(output["tokens"]) for output in baseline)
            assert call["generated_tokens_per_second"] == pytest.approx(
                call["generated_tokens"] / call["completed_batch_latency"]
            )
            assert call["completed_batch_latency"] > 0
            for request, output in zip(requests, call["outputs"], strict=True):
                expected_length = (
                    1
                    if workload["name"] == "seeded_first_token_stop"
                    else request["max_num_tokens"] - len(request["prompt_tokens"])
                )
                assert output["request_id"] == request["request_id"]
                assert len(output["tokens"]) == expected_length
                assert len(output["sampled_logprobs"]) == expected_length
                assert np.isfinite(output["sampled_logprobs"]).all()
                assert output["finish_reason"] == ("STOP" if request["stop_token"] is not None else "LENGTH")
        for repetition in range(8):
            calls = [call for call in workload["calls"] if call["repetition"] == repetition]
            assert [call["max_rounds"] for call in calls] == ([1, 32] if repetition % 2 == 0 else [32, 1])
        for rounds in (1, 32):
            arm = workload["arms"][str(rounds)]
            samples = [
                call for call in workload["calls"] if call["max_rounds"] == rounds and call["phase"] == "measured"
            ]
            assert len(samples) >= 5
            assert arm["raw_measured_latencies"] == [call["completed_batch_latency"] for call in samples]
            assert arm["aggregate_generated_tokens_per_second"] == pytest.approx(
                sum(call["generated_tokens"] for call in samples)
                / sum(call["completed_batch_latency"] for call in samples)
            )
    first = workloads["homogeneous_burst"]["calls"][0]["outputs"]
    stopped = workloads["seeded_first_token_stop"]["calls"][0]["outputs"]
    assert [output["tokens"] for output in stopped] == [[output["tokens"][0]] for output in first]


@pytest.mark.timeout(30)
def test_tpu_cli_rejects_cpu_fallback_and_writes_actual_failure(tmp_path):
    result, report = _run_cli(tmp_path, "tpu")
    assert result.returncode != 0
    assert report["status"] == "failed"
    assert report["requested_backend"] == "tpu"
    assert report["actual_backend"] == "cpu"
    assert report["progress"]["phase"] == "runtime_verification"
    assert report["error"]["type"] == "RuntimeError"
    assert report["workloads"] == []
