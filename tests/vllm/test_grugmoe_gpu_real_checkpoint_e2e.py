# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-checkpoint GrugMoE e2e for GPU vLLM and Levanter/JAX.

This is the CoreWeave/CUDA analogue of the TPU e2e. It validates the trained
checkpoint through the new vLLM PyTorch implementation and the JAX/Levanter
reference on GPUs.
"""

from __future__ import annotations

import base64
import fcntl
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
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


def _resolve_run_id() -> str:
    run_id = os.environ.get(backend.RUN_ID_ENV)
    if run_id:
        if "/" in run_id or run_id in {".", ".."}:
            raise ValueError(f"{backend.RUN_ID_ENV} must be a path segment, got {run_id!r}")
        return run_id
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


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
    run_id = _resolve_run_id()
    output_dir = os.environ.get(backend.OUTPUT_DIR_ENV) or backend._join_path(backend.OUTPUT_ROOT, run_id)
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
            "vllm_dtype": backend.VLLM_DTYPE,
            "vllm_moe_compute": backend.VLLM_MOE_COMPUTE,
            "vllm_route_diagnostics": backend.VLLM_ROUTE_DIAGNOSTICS,
            "levanter_reference_mode": backend.LEVANTER_REFERENCE_MODE,
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
                "install_report": os.environ.get(backend.INSTALL_REPORT_PATH_ENV),
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
        summary["vllm_observed_dtype"] = backend_results["vllm"].get("vllm_dtype")
        summary["vllm_observed_moe_compute"] = backend_results["vllm"].get("vllm_moe_compute")
        summary["vllm_observed_route_diagnostics"] = backend_results["vllm"].get("vllm_route_diagnostics")
        summary["vllm_worker_ep_summary"] = backend_results["vllm"].get("worker_ep_summary")
        summary["vllm_routed_expert_owner_ranks"] = backend_results["vllm"].get("routed_expert_owner_ranks")
        summary["vllm_routed_expert_owner_rank_coverage"] = backend_results["vllm"].get(
            "routed_expert_owner_rank_coverage"
        )
        summary["vllm_observed_worker_data_parallel_ranks"] = backend_results["vllm"].get(
            "observed_worker_data_parallel_ranks"
        )
        summary["vllm_requested_data_parallel_ranks"] = backend_results["vllm"].get("requested_data_parallel_ranks")
        levanter_jax_runtime = backend_results["levanter"].get("jax_runtime", {})
        levanter_jax_mesh = backend_results["levanter"].get("jax_mesh", {})
        summary["levanter_observed_reference_mode"] = backend_results["levanter"].get("levanter_reference_mode")
        summary["levanter_reference_policy"] = backend_results["levanter"].get("levanter_reference_policy")
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
            and backend_results["vllm"].get("worker_ep_summary", {}).get("dp_rank_coverage") is True
            and backend_results["vllm"].get("worker_ep_summary", {}).get("ep_rank_coverage") is True
            and backend_results["vllm"].get("worker_ep_summary", {}).get("local_expert_coverage") is True
        )
    backend._write_json(e2e_paths.summary_result_path, summary)
    print("grugmoe_gpu_real_checkpoint_e2e_result=" + json.dumps(summary, sort_keys=True), flush=True)


def test_grugmoe_gpu_real_checkpoint_e2e_static_preconditions() -> None:
    backend._require_constants_are_coreweave()
    assert backend.VLLM_MAX_NUM_SEQS >= backend.PROMPT_BATCH_SIZE
    assert backend.WORKER_EXTENSION_CLS == f"{backend.WORKER_EXTENSION_MODULE}.{backend.WORKER_EXTENSION_CLASS}"
    assert backend.VLLM_ATTENTION_BACKEND in backend.VLLM_ATTENTION_BACKENDS_UNDER_TEST
    assert backend.VLLM_DTYPE in backend.VLLM_DTYPE_CHOICES
    assert backend.VLLM_MOE_COMPUTE in backend.VLLM_MOE_COMPUTE_CHOICES
    assert backend.LEVANTER_REFERENCE_MODE in backend.LEVANTER_REFERENCE_MODE_CHOICES
    assert backend.LEVANTER_MOE_CAPACITY_FACTOR == float(backend.EXPECTED_GPU_COUNT)
    assert backend.LEVANTER_DECODE_USE_ACTIVE_PREFIX is True
    assert backend.CHECKPOINT_PATH.startswith(backend.COREWEAVE_S3_PREFIX)
    assert backend.TOKENIZER_PATH.startswith(backend.COREWEAVE_S3_PREFIX)


def test_grugmoe_gpu_e2e_run_id_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(backend.RUN_ID_ENV, "manual-coherent-install-triton")
    assert _resolve_run_id() == "manual-coherent-install-triton"

    monkeypatch.setenv(backend.RUN_ID_ENV, "bad/name")
    with pytest.raises(ValueError, match=backend.RUN_ID_ENV):
        _resolve_run_id()


def test_vllm_gpu_env_enables_debug_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    snapshot = backend._configure_vllm_gpu_env()
    assert os.environ["VLLM_LOGGING_LEVEL"] == "DEBUG"
    assert snapshot["vllm_logging_level"] == "DEBUG"
    assert snapshot["vllm_moe_compute"] == backend.VLLM_MOE_COMPUTE
    assert os.environ[backend.VLLM_GRUGMOE_MOE_COMPUTE_ENV] == backend.VLLM_MOE_COMPUTE
    assert snapshot["vllm_route_diagnostics"] == backend.VLLM_ROUTE_DIAGNOSTICS
    assert snapshot["worker_extension_module"] == backend.WORKER_EXTENSION_MODULE
    assert snapshot["worker_extension_cls"] == backend.WORKER_EXTENSION_CLS
    assert snapshot["worker_extension_path"] == str(BACKEND_PATH.parent)
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == snapshot["worker_extension_path"]


def test_vllm_server_logs_are_copied_to_output_prefix(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stdout.log").write_text("stdout full log\n")
    (log_dir / "stderr.log").write_text("stderr full log\n")

    artifacts = backend._copy_vllm_server_logs(str(log_dir), str(tmp_path / "result-prefix"))

    assert artifacts["copied"] is True
    assert artifacts["artifact_dir"] == str(tmp_path / "result-prefix" / "vllm-server-logs")
    assert {item["name"] for item in artifacts["files"]} == {"stdout.log", "stderr.log"}
    assert (tmp_path / "result-prefix" / "vllm-server-logs" / "stdout.log").read_text() == "stdout full log\n"
    assert (tmp_path / "result-prefix" / "vllm-server-logs" / "stderr.log").read_text() == "stderr full log\n"


def _encoded_routed_experts(expert_ids: list[int]) -> str:
    payload = io.BytesIO()
    np.save(payload, np.array([[expert_ids]], dtype=np.int64))
    return base64.b64encode(payload.getvalue()).decode()


def test_vllm_completion_diagnostics_compare_rank4_and_rank0(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            raise AssertionError("raise_for_status should not be called for successful fake response")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        if url.endswith("/collective_rpc"):
            return FakeResponse(
                {
                    "results": [
                        {
                            "worker_rank": rank,
                            "route_diagnostics_enabled": True,
                            "layers": [],
                        }
                        for rank in range(backend.EXPECTED_GPU_COUNT)
                    ]
                }
            )
        rank = int(kwargs["headers"]["X-data-parallel-rank"])
        choices = [
            {
                "text": f"rank-{rank}-choice-{index}",
                "finish_reason": "length",
                "routed_experts": _encoded_routed_experts([rank * 32, 255]),
                "logprobs": {
                    "tokens": [f"rank-{rank}"],
                    "token_logprobs": [-0.1],
                    "top_logprobs": [{f"rank-{rank}": -0.1}],
                },
            }
            for index, _ in enumerate(kwargs["json"]["prompt"])
        ]
        return FakeResponse(
            {
                "choices": choices,
                "usage": {"completion_tokens": len(choices)},
            }
        )

    worker_ep_states = [
        {
            "ep_rank": ep_rank,
            "local_expert_ids": list(range(ep_rank * 32, ep_rank * 32 + 32)),
        }
        for ep_rank in range(backend.VLLM_EXPERT_PARALLEL_SIZE)
    ]
    monkeypatch.setattr(backend.requests, "post", fake_post)
    env = SimpleNamespace(server_url="http://127.0.0.1:8000/v1", model_id="grugmoe")

    diagnostics = backend._run_vllm_completion_diagnostics(env, worker_ep_states=worker_ep_states)

    assert diagnostics["compare_ranks"] == [4, 0]
    assert diagnostics["repeated_attempts"] == 3
    completion_calls = [call for call in calls if call["url"].endswith("/completions")]
    rpc_calls = [call for call in calls if call["url"].endswith("/collective_rpc")]
    assert len(completion_calls) == 12
    assert len(rpc_calls) == 2
    assert calls[0]["headers"] == {
        "X-data-parallel-rank": "4",
        "X-Request-Id": "grugmoe-diagnostic-rank4-first-token-repeated-bs2",
    }
    assert {request["data_parallel_rank"] for request in diagnostics["requests"]} == {0, 4}
    assert [
        request["attempt"] for request in diagnostics["requests"] if request["name"].startswith("rank4-repeated-bs2")
    ] == [0, 1, 2]
    rank4_non_repeated = next(
        request for request in diagnostics["requests"] if request["name"] == "rank4-non-repeated-bs2"
    )
    assert rank4_non_repeated["prompt_kind"] == "non_repeated"
    assert rank4_non_repeated["choice_count"] == 2
    assert rank4_non_repeated["choices"][0]["routed_experts_shape"] == [1, 1, 2]
    assert rank4_non_repeated["choices"][0]["routed_owner_ranks"] == [4, 7]
    rank4_first_token = next(
        request for request in diagnostics["requests"] if request["name"] == "rank4-first-token-repeated-bs2"
    )
    assert rank4_first_token["max_tokens"] == 1
    assert rank4_first_token["logprobs"] == backend.DIAGNOSTIC_LOGPROBS
    assert len(rank4_first_token["route_diagnostics"]) == backend.EXPECTED_GPU_COUNT


def test_grugmoe_worker_extension_reports_structured_ep_state(monkeypatch: pytest.MonkeyPatch) -> None:
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

    class FakeAttentionBackend:
        @staticmethod
        def get_name() -> str:
            return "TRITON_ATTN"

    class GrugMoeMLP:
        experts = FakeRunner()

    class FakeLayer:
        mlp = GrugMoeMLP()
        self_attn = SimpleNamespace(attn=SimpleNamespace(attn_backend=FakeAttentionBackend()))

    class FakeModel:
        def __init__(self) -> None:
            self.layers = [FakeLayer()]

        def named_modules(self):
            yield "model.layers.3.mlp", self.layers[0].mlp

    fake_grugmoe_module = ModuleType("vllm.model_executor.models.grugmoe")

    def fake_get_grug_moe_runtime_info(vllm_config: Any, model: Any) -> dict[str, Any]:
        layer = model.layers[0]
        runner = layer.mlp.experts
        moe_config = runner.moe_config
        moe_parallel_config = moe_config.moe_parallel_config
        local_expert_ids = runner.expert_map_manager.get_local_expert_ids()
        return {
            "use_ep": bool(moe_parallel_config.use_ep),
            "tp_size": int(moe_parallel_config.tp_size),
            "tp_rank": int(moe_parallel_config.tp_rank),
            "dp_size": int(moe_parallel_config.dp_size),
            "dp_rank": int(moe_parallel_config.dp_rank),
            "ep_size": int(moe_parallel_config.ep_size),
            "ep_rank": int(moe_parallel_config.ep_rank),
            "num_experts": int(moe_config.num_experts),
            "num_logical_experts": int(moe_config.num_logical_experts),
            "num_local_experts": int(moe_config.num_local_experts),
            "local_expert_ids": local_expert_ids,
            "local_expert_ownership": "[96..127]",
            "top_k": int(moe_config.experts_per_token),
            "expert_placement_strategy": str(runner.expert_placement_strategy),
            "all2all_backend": str(moe_parallel_config.all2all_backend),
            "attention_backend": layer.self_attn.attn.attn_backend.get_name(),
            "routed_experts_capture_enabled": bool(
                getattr(vllm_config.model_config, "enable_return_routed_experts", False)
            ),
        }

    fake_grugmoe_module.get_grug_moe_runtime_info = fake_get_grug_moe_runtime_info  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.grugmoe", fake_grugmoe_module)

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
        "num_experts": 256,
        "num_logical_experts": 256,
        "num_local_experts": 32,
        "local_expert_ids": list(range(96, 128)),
        "local_expert_ownership": "[96..127]",
        "top_k": 2,
        "expert_placement_strategy": "linear",
        "all2all_backend": "allgather_reducescatter",
        "attention_backend": "TRITON_ATTN",
        "routed_experts_capture_enabled": True,
    }


def test_grugmoe_collective_rpc_collects_all_data_parallel_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, Any]:
            results: list[dict[str, Any]] = []
            for ep_rank in range(backend.VLLM_EXPERT_PARALLEL_SIZE):
                first_expert = ep_rank * 32
                results.append(
                    {
                        "found": True,
                        "worker_rank": ep_rank,
                        "local_rank": ep_rank,
                        "module_name": "model.layers.0.mlp",
                        "use_ep": True,
                        "tp_size": 1,
                        "tp_rank": 0,
                        "dp_size": 8,
                        "dp_rank": ep_rank,
                        "ep_size": 8,
                        "ep_rank": ep_rank,
                        "num_experts": 256,
                        "num_logical_experts": 256,
                        "num_local_experts": 32,
                        "local_expert_ids": list(range(first_expert, first_expert + 32)),
                        "local_expert_ownership": f"[{first_expert}..{first_expert + 31}]",
                        "top_k": 4,
                        "expert_placement_strategy": "linear",
                        "all2all_backend": "allgather_reducescatter",
                        "attention_backend": "TRITON_ATTN",
                        "routed_experts_capture_enabled": True,
                    }
                )
            return {
                "results": results,
            }

        def raise_for_status(self) -> None:
            raise AssertionError("raise_for_status should not be called for successful fake response")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    env = SimpleNamespace(server_url="http://127.0.0.1:8000/v1")

    states = backend._collect_grug_moe_worker_ep_states(env)
    summary = backend._assert_grug_moe_worker_ep_states(states, num_experts=256)

    assert len(calls) == 1
    assert {call["url"] for call in calls} == {"http://127.0.0.1:8000/collective_rpc"}
    assert calls[0]["headers"] == {}
    assert calls[0]["json"] == {"method": "grugmoe_ep_state", "timeout": 300}
    assert [state["dp_rank"] for state in states] == list(range(backend.VLLM_DATA_PARALLEL_SIZE))
    assert [state["ep_rank"] for state in states] == list(range(backend.VLLM_EXPERT_PARALLEL_SIZE))
    assert summary["worker_count"] == backend.EXPECTED_GPU_COUNT
    assert summary["dp_rank_coverage"] is True
    assert summary["ep_rank_coverage"] is True
    assert summary["local_expert_coverage"] is True
    assert backend._owners_for_worker_expert_placement(
        [[0, 31], [96, 127], [224, 255]],
        worker_ep_states=states,
    ) == [0, 3, 7]


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
    assert vllm_result["vllm_max_num_seqs"] >= backend.PROMPT_BATCH_SIZE
    assert vllm_result["torch_runtime"]["device_count"] >= backend.EXPECTED_GPU_COUNT
    assert vllm_result["torch_runtime"]["cuda_visible_devices"] == backend.VISIBLE_CUDA_DEVICES
    assert vllm_result["prompt_batch_size"] == backend.PROMPT_BATCH_SIZE
    assert vllm_result["single_prompt_completion"] == backend.EXPECTED_CONTINUATION
    assert len(vllm_result["completions"]) == backend.PROMPT_BATCH_SIZE
    assert all(completion == backend.EXPECTED_CONTINUATION for completion in vllm_result["completions"])
    assert vllm_result["observed_worker_data_parallel_ranks"] == list(range(backend.VLLM_DATA_PARALLEL_SIZE))
    assert vllm_result["requested_data_parallel_ranks"] == list(range(backend.VLLM_DATA_PARALLEL_SIZE))
    assert vllm_result["routed_expert_owner_ranks"] == list(range(backend.VLLM_EXPERT_PARALLEL_SIZE))
    assert vllm_result["routed_expert_owner_rank_coverage"] is True
    assert vllm_result["worker_ep_summary"]["worker_count"] == backend.EXPECTED_GPU_COUNT
    assert vllm_result["worker_ep_summary"]["dp_ranks"] == list(range(backend.VLLM_DATA_PARALLEL_SIZE))
    assert vllm_result["worker_ep_summary"]["dp_rank_coverage"] is True
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
