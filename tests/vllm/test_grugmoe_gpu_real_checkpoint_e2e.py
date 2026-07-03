# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-checkpoint GrugMoE e2e for GPU vLLM and Levanter/JAX.

This is the CoreWeave/CUDA analogue of the TPU e2e. It validates the trained
checkpoint through the new vLLM PyTorch implementation and the JAX/Levanter
reference on GPUs.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

GPU_LOCK_PATH = "/tmp/marin-grugmoe-gpu-e2e.lock"
BACKEND_PATH = Path(__file__).with_name("grugmoe_gpu_real_checkpoint_backend.py").resolve()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_backend_module():
    spec = importlib.util.spec_from_file_location("grugmoe_gpu_real_checkpoint_backend", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GrugMoE GPU e2e backend from {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backend = _load_backend_module()

pytestmark = [
    pytest.mark.gpu_ci,
    pytest.mark.integration,
    pytest.mark.timeout(7200),
]


def _require_no_active_xdist(request: pytest.FixtureRequest) -> None:
    xdist_env = {key: value for key, value in os.environ.items() if key.startswith("PYTEST_XDIST_WORKER")}
    workerinput = getattr(request.config, "workerinput", None)
    numprocesses = getattr(request.config.option, "numprocesses", None)
    if xdist_env or workerinput is not None or numprocesses not in (None, 0, "0"):
        raise RuntimeError(
            "GrugMoE GPU real-checkpoint e2e cannot run under active pytest-xdist parallelism. "
            f"Detected env={xdist_env!r}, workerinput={workerinput is not None}, numprocesses={numprocesses!r}. "
            "Run this test with -n 0 or without xdist."
        )


def _nvidia_smi_diagnostic() -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"checked": False}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("GrugMoE GPU e2e requires nvidia-smi on PATH")
    result = subprocess.run([nvidia_smi, "-L"], text=True, capture_output=True, check=False, timeout=30)
    diagnostic.update(
        {
            "checked": True,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"nvidia-smi did not report GPUs for GrugMoE GPU e2e: {diagnostic!r}")
    gpu_lines = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
    diagnostic["gpu_count"] = len(gpu_lines)
    diagnostic["expected_gpu_count"] = backend.EXPECTED_GPU_COUNT
    diagnostic["h100_count"] = sum("H100" in line for line in gpu_lines)
    if diagnostic["gpu_count"] < backend.EXPECTED_GPU_COUNT or diagnostic["h100_count"] < backend.EXPECTED_GPU_COUNT:
        raise RuntimeError(f"Expected {backend.EXPECTED_GPU_COUNT} visible H100 GPUs; got {diagnostic!r}")
    return diagnostic


def _repo_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        return f"unavailable:{exc!r}"


@pytest.fixture(scope="module")
def no_active_xdist(request: pytest.FixtureRequest) -> None:
    _require_no_active_xdist(request)


@pytest.fixture(scope="module")
def gpu_lock(no_active_xdist: None):
    del no_active_xdist
    with open(GPU_LOCK_PATH, "w") as lock_file:
        try:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another Marin GPU e2e appears to hold {GPU_LOCK_PATH}") from exc
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(json.dumps({"pid": os.getpid(), "test": __name__, "started": time.time()}) + "\n")
            lock_file.flush()
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


@pytest.fixture(scope="module")
def e2e_paths(gpu_lock: None) -> backend.E2EPaths:
    del gpu_lock
    backend._require_runtime_region()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    output_dir = backend._join_path(backend.OUTPUT_ROOT, run_id)
    paths = backend.E2EPaths(
        output_dir=output_dir,
        cache_dir=backend._join_path(backend.CACHE_ROOT, run_id),
        artifact_dir=backend._join_path(output_dir, "artifact"),
        export_result_path=backend._join_path(output_dir, "export-result.json"),
        vllm_result_path=backend._join_path(output_dir, "vllm-result.json"),
        levanter_result_path=backend._join_path(output_dir, "levanter-result.json"),
        summary_result_path=backend._join_path(output_dir, "result.json"),
    )
    backend._require_constants_are_coreweave(paths)
    return paths


def _run_backend(phase: str, paths: backend.E2EPaths, result_path: str) -> dict[str, Any]:
    backend._require_coreweave_path("result_path", result_path)
    command = [
        sys.executable,
        str(BACKEND_PATH),
        "--backend",
        phase,
        "--checkpoint-path",
        backend.CHECKPOINT_PATH,
        "--tokenizer-path",
        backend.TOKENIZER_PATH,
        "--output-dir",
        paths.output_dir,
        "--artifact-dir",
        paths.artifact_dir,
        "--cache-dir",
        paths.cache_dir,
        "--result-path",
        result_path,
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["CUDA_VISIBLE_DEVICES"] = backend.VISIBLE_CUDA_DEVICES
    env.setdefault("MARIN_GIT_SHA", _repo_git_sha())
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if value)
    print("grugmoe_gpu_real_checkpoint_e2e_command=" + json.dumps(command), flush=True)
    completed = subprocess.run(command, check=False, env=env, cwd=REPO_ROOT)
    if completed.returncode == 0:
        return backend._read_json(result_path)
    if backend._exists(result_path):
        result = backend._read_json(result_path)
        result["backend_returncode"] = completed.returncode
        return result
    completed.check_returncode()
    raise AssertionError("unreachable")


@pytest.fixture(scope="module")
def export_result(e2e_paths: backend.E2EPaths) -> dict[str, Any]:
    diagnostic = _nvidia_smi_diagnostic()
    print("grugmoe_gpu_real_checkpoint_export_gpu_preflight=" + json.dumps(diagnostic, sort_keys=True), flush=True)
    return _run_backend("export", e2e_paths, e2e_paths.export_result_path)


@pytest.fixture(scope="module")
def vllm_result(e2e_paths: backend.E2EPaths, export_result: dict[str, Any]) -> dict[str, Any]:
    assert export_result["artifact_dir"] == e2e_paths.artifact_dir
    diagnostic = _nvidia_smi_diagnostic()
    print("grugmoe_gpu_real_checkpoint_vllm_gpu_preflight=" + json.dumps(diagnostic, sort_keys=True), flush=True)
    return _run_backend("vllm", e2e_paths, e2e_paths.vllm_result_path)


@pytest.fixture(scope="module")
def levanter_result(e2e_paths: backend.E2EPaths) -> dict[str, Any]:
    diagnostic = _nvidia_smi_diagnostic()
    print("grugmoe_gpu_real_checkpoint_levanter_gpu_preflight=" + json.dumps(diagnostic, sort_keys=True), flush=True)
    return _run_backend("levanter", e2e_paths, e2e_paths.levanter_result_path)


def _write_summary_update(
    e2e_paths: backend.E2EPaths,
    *,
    export_result: dict[str, Any] | None = None,
    backend_result: dict[str, Any] | None = None,
) -> None:
    if backend._exists(e2e_paths.summary_result_path):
        summary = backend._read_json(e2e_paths.summary_result_path)
    else:
        summary = {
            "checkpoint_path": backend.CHECKPOINT_PATH,
            "tokenizer_path": backend.TOKENIZER_PATH,
            "region": backend.REGION,
            "coreweave_signing_region": backend.COREWEAVE_SIGNING_REGION,
            "gpu_node_type": backend.GPU_NODE_TYPE,
            "gpu_nodepool": backend.GPU_NODEPOOL,
            "expected_gpu_count": backend.EXPECTED_GPU_COUNT,
            "visible_cuda_devices": backend.VISIBLE_CUDA_DEVICES,
            "vllm_tensor_parallel_size": backend.VLLM_TENSOR_PARALLEL_SIZE,
            "vllm_data_parallel_size": backend.VLLM_DATA_PARALLEL_SIZE,
            "vllm_expert_parallel_size": backend.VLLM_EXPERT_PARALLEL_SIZE,
            "vllm_attention_backend": backend.VLLM_ATTENTION_BACKEND,
            "vllm_default_attention_backend": backend.VLLM_DEFAULT_ATTENTION_BACKEND,
            "levanter_moe_capacity_factor": backend.LEVANTER_MOE_CAPACITY_FACTOR,
            "levanter_decode_use_active_prefix": backend.LEVANTER_DECODE_USE_ACTIVE_PREFIX,
            "prompt_batch_size": backend.PROMPT_BATCH_SIZE,
            "prompt": backend.PROMPT,
            "expected_continuation": backend.EXPECTED_CONTINUATION,
            "result_paths": {
                "export": e2e_paths.export_result_path,
                "vllm": e2e_paths.vllm_result_path,
                "levanter": e2e_paths.levanter_result_path,
                "summary": e2e_paths.summary_result_path,
            },
            "runtime": backend._runtime_snapshot(include_grugmoe_spec=True),
            "backend_results": {},
            "caveat": (
                "This e2e validates real trained-checkpoint serving correctness through GPU vLLM and "
                "GPU Levanter/JAX on one 8xH100 node. vLLM is launched with tensor_parallel_size=1, "
                "data_parallel_size=8, and expert parallelism enabled; the JAX/Levanter reference uses "
                "an expert_axis_size=8 mesh. It does not validate broad context windows, logprob parity, "
                "or performance."
            ),
        }

    if export_result is not None:
        summary["export_result"] = export_result
    if backend_result is not None:
        phase = str(backend_result["phase"])
        summary.setdefault("backend_results", {})[phase] = backend_result
        summary[f"actual_{phase}_output"] = backend_result.get("completion")

    backend_results = summary.get("backend_results", {})
    summary["completed_backend_phases"] = sorted(backend_results)
    if all(phase in backend_results for phase in ("vllm", "levanter")):
        vllm_completion = backend_results["vllm"].get("completion")
        levanter_completion = backend_results["levanter"].get("completion")
        vllm_completions = backend_results["vllm"].get("completions")
        levanter_completions = backend_results["levanter"].get("completions")
        summary["vllm_levanter_match"] = vllm_completion == levanter_completion
        summary["vllm_levanter_batch_match"] = vllm_completions == levanter_completions
        summary["vllm_observed_gpu_count"] = backend_results["vllm"].get("torch_runtime", {}).get("device_count")
        summary["vllm_observed_tensor_parallel_size"] = backend_results["vllm"].get("vllm_tensor_parallel_size")
        summary["vllm_observed_data_parallel_size"] = backend_results["vllm"].get("vllm_data_parallel_size")
        summary["vllm_observed_expert_parallel_size"] = backend_results["vllm"].get("vllm_expert_parallel_size")
        summary["vllm_worker_ep_summary"] = backend_results["vllm"].get("worker_ep_summary")
        summary["vllm_routed_expert_owner_ranks"] = backend_results["vllm"].get("routed_expert_owner_ranks")
        summary["vllm_routed_expert_owner_rank_coverage"] = backend_results["vllm"].get(
            "routed_expert_owner_rank_coverage"
        )
        summary["vllm_requested_data_parallel_ranks"] = backend_results["vllm"].get("requested_data_parallel_ranks")
        levanter_jax_runtime = backend_results["levanter"].get("jax_runtime", {})
        levanter_jax_mesh = backend_results["levanter"].get("jax_mesh", {})
        summary["levanter_jax_gpu_device_count"] = levanter_jax_runtime.get("gpu_device_count")
        summary["levanter_jax_mesh_device_count"] = levanter_jax_mesh.get("device_count")
        summary["levanter_jax_mesh_shape"] = levanter_jax_mesh.get("shape")
        summary["levanter_jax_reference_uses_expected_gpu_count"] = (
            levanter_jax_runtime.get("uses_expected_gpu_count") is True
            and levanter_jax_mesh.get("uses_expected_gpu_count") is True
        )
    summary["passed"] = all(backend_results.get(phase, {}).get("passed") is True for phase in ("vllm", "levanter"))
    if "vllm_levanter_match" in summary:
        summary["passed"] = (
            summary["passed"]
            and summary["vllm_levanter_match"]
            and summary["vllm_levanter_batch_match"]
            and summary["vllm_routed_expert_owner_rank_coverage"] is True
            and backend_results["vllm"].get("worker_ep_summary", {}).get("ep_rank_coverage") is True
            and backend_results["vllm"].get("worker_ep_summary", {}).get("local_expert_coverage") is True
        )
    backend._write_json(e2e_paths.summary_result_path, summary)
    print("grugmoe_gpu_real_checkpoint_e2e_result=" + json.dumps(summary, sort_keys=True), flush=True)


def test_grugmoe_gpu_real_checkpoint_e2e_static_preconditions() -> None:
    backend._require_constants_are_coreweave()
    assert backend.REGION == "cw-us-east-02a"
    assert backend.COREWEAVE_SIGNING_REGION == "US-EAST-02A"
    assert backend.GPU_NODE_TYPE == "H100x8"
    assert backend.GPU_NODEPOOL == "cw-use02a-h100-8x"
    assert backend.EXPECTED_GPU_COUNT == 8
    assert backend.VISIBLE_CUDA_DEVICES == "0,1,2,3,4,5,6,7"
    assert backend.VLLM_TENSOR_PARALLEL_SIZE == 1
    assert backend.VLLM_DATA_PARALLEL_SIZE == 8
    assert backend.VLLM_EXPERT_PARALLEL_SIZE == 8
    assert backend.VLLM_MAX_NUM_SEQS >= backend.PROMPT_BATCH_SIZE
    assert backend.VLLM_DEFAULT_ATTENTION_BACKEND == "TRITON_ATTN"
    assert backend.VLLM_ATTENTION_BACKEND in backend.VLLM_ATTENTION_BACKENDS_UNDER_TEST
    assert backend.VLLM_ATTENTION_BACKEND_ENV == "MARIN_GRUGMOE_VLLM_ATTENTION_BACKEND"
    assert backend.LEVANTER_MOE_CAPACITY_FACTOR == float(backend.EXPECTED_GPU_COUNT)
    assert backend.LEVANTER_DECODE_USE_ACTIVE_PREFIX is True
    assert backend.CHECKPOINT_PATH.startswith(backend.COREWEAVE_S3_PREFIX)
    assert backend.TOKENIZER_PATH.startswith(backend.COREWEAVE_S3_PREFIX)


def test_grugmoe_worker_extension_reports_structured_ep_state() -> None:
    class FakeParallelConfig:
        use_ep = True
        tp_size = 1
        tp_rank = 0
        dp_size = 8
        dp_rank = 3
        ep_size = 8
        ep_rank = 3
        all2all_backend = "allgather_reducescatter"

    class FakeMoeConfig:
        moe_parallel_config = FakeParallelConfig()
        num_experts = 256
        num_logical_experts = 256
        num_local_experts = 32
        experts_per_token = 2

    class FakeExpertMapManager:
        def get_local_expert_ids(self) -> list[int]:
            return list(range(96, 128))

    class FakeRunner:
        moe_config = FakeMoeConfig()
        expert_map_manager = FakeExpertMapManager()
        expert_placement_strategy = "linear"

    class GrugMoeMLP:
        experts = FakeRunner()

    class FakeModel:
        def named_modules(self):
            yield "model.layers.3.mlp", GrugMoeMLP()

    worker = SimpleNamespace(
        rank=3,
        local_rank=3,
        model_runner=SimpleNamespace(model=FakeModel()),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=True),
        ),
    )

    state = backend.GrugMoeDiagnosticsWorkerExtension.grugmoe_ep_state(worker)

    assert state == {
        "found": True,
        "worker_rank": 3,
        "local_rank": 3,
        "module_name": "model.layers.3.mlp",
        "use_ep": True,
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 8,
        "dp_rank": 3,
        "ep_size": 8,
        "ep_rank": 3,
        "global_num_experts": 256,
        "logical_num_experts": 256,
        "local_num_experts": 32,
        "local_expert_ids": list(range(96, 128)),
        "local_expert_ownership": "[96..127]",
        "top_k": 2,
        "expert_placement_strategy": "linear",
        "all2all_backend": "allgather_reducescatter",
        "routed_experts_capture_enabled": True,
    }


def test_grugmoe_collective_rpc_collects_each_data_parallel_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"

        def __init__(self, ep_rank: int):
            self.ep_rank = ep_rank

        def json(self) -> dict[str, Any]:
            first_expert = self.ep_rank * 32
            return {
                "results": [
                    {
                        "found": True,
                        "worker_rank": self.ep_rank,
                        "local_rank": self.ep_rank,
                        "module_name": "model.layers.0.mlp",
                        "use_ep": True,
                        "tp_size": 1,
                        "tp_rank": 0,
                        "dp_size": 8,
                        "dp_rank": self.ep_rank,
                        "ep_size": 8,
                        "ep_rank": self.ep_rank,
                        "global_num_experts": 256,
                        "logical_num_experts": 256,
                        "local_num_experts": 32,
                        "local_expert_ids": list(range(first_expert, first_expert + 32)),
                        "local_expert_ownership": f"[{first_expert}..{first_expert + 31}]",
                        "top_k": 4,
                        "expert_placement_strategy": "linear",
                        "all2all_backend": "allgather_reducescatter",
                        "routed_experts_capture_enabled": True,
                    }
                ]
            }

        def raise_for_status(self) -> None:
            raise AssertionError("raise_for_status should not be called for successful fake response")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        rank = int(kwargs["headers"]["X-data-parallel-rank"])
        return FakeResponse(rank)

    monkeypatch.setattr(backend.requests, "post", fake_post)
    env = SimpleNamespace(server_url="http://127.0.0.1:8000/v1")

    states = backend._collect_grug_moe_worker_ep_states(env)
    summary = backend._assert_grug_moe_worker_ep_states(states, num_experts=256)

    assert [call["headers"]["X-data-parallel-rank"] for call in calls] == [
        str(rank) for rank in range(backend.VLLM_DATA_PARALLEL_SIZE)
    ]
    assert {call["url"] for call in calls} == {"http://127.0.0.1:8000/collective_rpc"}
    assert [state["ep_rank"] for state in states] == list(range(backend.VLLM_EXPERT_PARALLEL_SIZE))
    assert summary["worker_count"] == backend.EXPECTED_GPU_COUNT
    assert summary["ep_rank_coverage"] is True
    assert summary["local_expert_coverage"] is True


@pytest.mark.gpu_ci
@pytest.mark.slow
@pytest.mark.data_integration
def test_grugmoe_gpu_real_checkpoint_vllm_output(
    e2e_paths: backend.E2EPaths,
    export_result: dict[str, Any],
    vllm_result: dict[str, Any],
) -> None:
    _write_summary_update(e2e_paths, export_result=export_result, backend_result=vllm_result)
    assert vllm_result["phase"] == "vllm"
    assert vllm_result["vllm_tensor_parallel_size"] == backend.VLLM_TENSOR_PARALLEL_SIZE
    assert vllm_result["vllm_data_parallel_size"] == backend.VLLM_DATA_PARALLEL_SIZE
    assert vllm_result["vllm_expert_parallel_size"] == backend.VLLM_EXPERT_PARALLEL_SIZE
    assert vllm_result["vllm_args"][vllm_result["vllm_args"].index("--tensor-parallel-size") + 1] == "1"
    assert vllm_result["vllm_args"][vllm_result["vllm_args"].index("--data-parallel-size") + 1] == "8"
    assert "--enable-expert-parallel" in vllm_result["vllm_args"]
    assert "--enable-return-routed-experts" in vllm_result["vllm_args"]
    assert "--worker-extension-cls" in vllm_result["vllm_args"]
    assert vllm_result["vllm_args"][vllm_result["vllm_args"].index("--max-num-seqs") + 1] == "16"
    assert vllm_result["torch_runtime"]["device_count"] >= backend.EXPECTED_GPU_COUNT
    assert vllm_result["torch_runtime"]["cuda_visible_devices"] == backend.VISIBLE_CUDA_DEVICES
    assert vllm_result["prompt_batch_size"] == backend.PROMPT_BATCH_SIZE
    assert vllm_result["single_prompt_completion"] == backend.EXPECTED_CONTINUATION
    assert len(vllm_result["completions"]) == backend.PROMPT_BATCH_SIZE
    assert all(completion == backend.EXPECTED_CONTINUATION for completion in vllm_result["completions"])
    assert vllm_result["requested_data_parallel_ranks"] == list(range(backend.VLLM_DATA_PARALLEL_SIZE))
    assert vllm_result["routed_expert_owner_ranks"] == list(range(backend.VLLM_EXPERT_PARALLEL_SIZE))
    assert vllm_result["routed_expert_owner_rank_coverage"] is True
    assert vllm_result["worker_ep_summary"]["worker_count"] == backend.EXPECTED_GPU_COUNT
    assert vllm_result["worker_ep_summary"]["ep_ranks"] == list(range(backend.VLLM_EXPERT_PARALLEL_SIZE))
    assert vllm_result["worker_ep_summary"]["ep_rank_coverage"] is True
    assert vllm_result["worker_ep_summary"]["local_expert_coverage"] is True
    assert {state["ep_rank"] for state in vllm_result["worker_ep_states"]} == set(
        range(backend.VLLM_EXPERT_PARALLEL_SIZE)
    )
    assert vllm_result["passed"] is True


@pytest.mark.gpu_ci
@pytest.mark.slow
@pytest.mark.data_integration
def test_grugmoe_gpu_real_checkpoint_levanter_output(
    e2e_paths: backend.E2EPaths,
    export_result: dict[str, Any],
    vllm_result: dict[str, Any],
    levanter_result: dict[str, Any],
) -> None:
    _write_summary_update(e2e_paths, export_result=export_result, backend_result=vllm_result)
    _write_summary_update(e2e_paths, backend_result=levanter_result)
    assert vllm_result["checkpoint_path"] == levanter_result["checkpoint_path"] == backend.CHECKPOINT_PATH
    assert vllm_result["prompt"] == levanter_result["prompt"] == backend.PROMPT
    assert vllm_result["completion"] == levanter_result["completion"]
    assert vllm_result["completions"] == levanter_result["completions"]
    assert levanter_result["phase"] == "levanter"
    assert levanter_result["jax_runtime"]["gpu_device_count"] >= backend.EXPECTED_GPU_COUNT
    assert levanter_result["jax_runtime"]["cuda_visible_devices"] == backend.VISIBLE_CUDA_DEVICES
    assert levanter_result["jax_mesh"]["device_count"] >= backend.EXPECTED_GPU_COUNT
    assert levanter_result["jax_mesh"]["uses_expected_gpu_count"] is True
    assert levanter_result["jax_mesh"]["shape"]["expert"] == backend.EXPECTED_GPU_COUNT
    assert levanter_result["jax_mesh"]["shape"]["model"] == 1
    assert levanter_result["jax_mesh"]["shape"]["data"] == 1
    assert len(levanter_result["completions"]) == backend.PROMPT_BATCH_SIZE
    assert all(completion == backend.EXPECTED_CONTINUATION for completion in levanter_result["completions"])
    assert levanter_result["passed"] is True
