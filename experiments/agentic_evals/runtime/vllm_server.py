"""vLLM server lifecycle management on Ray clusters.

Extracted from OT-Agent ``hpc/vllm_utils.py``. Provides a context manager for
managing the vLLM server lifecycle (start, health-check, warmup, stop).

The ``RayCluster`` dependency is replaced with a local ``RayClusterProtocol``
so this module has no ``hpc.*`` imports. ``build_vllm_cli_args`` is imported
from ``..serve.vllm_args``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class RayClusterProtocol(Protocol):
    """Minimal protocol the VLLMServer needs from a Ray cluster handle."""

    @property
    def head_ip(self) -> str: ...

    @property
    def address(self) -> str: ...

    # Nested config with gpus_per_node
    @property
    def config(self) -> Any: ...


from ..serve.vllm_args import build_vllm_cli_args


@dataclass
class VLLMConfig:
    """Configuration for a vLLM server."""

    model_path: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    api_port: int = 8000
    endpoint_json_path: Optional[str] = None
    custom_model_name: Optional[str] = None

    health_max_attempts: int = 480
    health_retry_delay: int = 15
    health_path: str = "v1/models"

    controller_script: str = "scripts/vllm/start_vllm_ray_controller.py"
    wait_for_endpoint_script: str = "scripts/vllm/wait_for_endpoint.py"

    server_config: dict = field(default_factory=dict)


@dataclass
class VLLMServer:
    """Context manager for vLLM server lifecycle."""

    config: VLLMConfig
    ray_cluster: RayClusterProtocol
    log_path: Optional[Path] = None
    extra_env_vars: Optional[Dict[str, str]] = None
    _process: Optional[subprocess.Popen] = None
    _log_file: Optional[object] = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}/v1"

    @property
    def base_url(self) -> str:
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}"

    @property
    def metrics_endpoint(self) -> str:
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}/metrics"

    def start(self) -> str:
        """Start the vLLM server. Returns the API endpoint URL."""
        if self._process is not None:
            print(f"vLLM server already started at {self.endpoint}")
            return self.endpoint

        if self.config.endpoint_json_path and os.path.exists(self.config.endpoint_json_path):
            print(f"Removing stale endpoint JSON: {self.config.endpoint_json_path}")
            try:
                os.remove(self.config.endpoint_json_path)
            except OSError as e:
                print(f"  Warning: could not remove stale endpoint file: {e}")

        print(f"=== Starting vLLM Server ===")
        print(f"  Model: {self.config.model_path}")
        print(f"  TP/PP/DP: {self.config.tensor_parallel_size}/{self.config.pipeline_parallel_size}/{self.config.data_parallel_size}")
        print(f"  Host: {self.ray_cluster.head_ip}")
        print(f"  Port: {self.config.api_port}")
        print(f"  Ray Address: {self.ray_cluster.address}")
        print(f"============================")

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w", buffering=1)
            stdout_dest = self._log_file
            stderr_dest = subprocess.STDOUT
        else:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL

        cmd = [
            sys.executable,
            self.config.controller_script,
            "--ray-address",
            self.ray_cluster.address,
            "--host",
            self.ray_cluster.head_ip,
            "--port",
            str(self.config.api_port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.config.pipeline_parallel_size),
            "--data-parallel-size",
            str(self.config.data_parallel_size),
        ]

        if self.config.endpoint_json_path:
            cmd.extend(["--endpoint-json", self.config.endpoint_json_path])

        if self.config.custom_model_name:
            cmd.extend(["--served-model-name", self.config.custom_model_name])

        extra_env_vars: dict[str, str] = {}
        if self.config.server_config:
            extra_cli_args, extra_env_vars = build_vllm_cli_args(self.config.server_config)
            cmd.extend(extra_cli_args)
            if extra_cli_args:
                print(f"  Extra vLLM args: {' '.join(extra_cli_args[:10])}{'...' if len(extra_cli_args) > 10 else ''}")

        if self.config.data_parallel_size > 1 and "--data-parallel-address" not in cmd:
            cmd.extend(["--data-parallel-address", self.ray_cluster.head_ip])

        if self.config.data_parallel_size > 1 and "--data-parallel-size-local" not in cmd:
            gpus_per_node = self.ray_cluster.config.gpus_per_node
            dp_per_node = max(1, gpus_per_node // self.config.tensor_parallel_size)
            dp_per_node = min(dp_per_node, self.config.data_parallel_size)
            cmd.extend(["--data-parallel-size-local", str(dp_per_node)])

        if self.config.data_parallel_size > 1 and "--data-parallel-backend" not in cmd:
            cmd.extend(["--data-parallel-backend", "ray"])

        env = os.environ.copy()
        env["VLLM_MODEL_PATH"] = self.config.model_path
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env_vars:
            env.update(extra_env_vars)
            print(f"  Extra vLLM env: {', '.join(f'{k}={v}' for k, v in extra_env_vars.items())}")
        env.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7200")
        env.setdefault("VLLM_RINGBUFFER_WARNING_INTERVAL", "600")
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if self.config.data_parallel_size > 1:
            tp_pp = self.config.tensor_parallel_size * self.config.pipeline_parallel_size
            if tp_pp > self.ray_cluster.config.gpus_per_node:
                env.setdefault("VLLM_RAY_DP_PACK_STRATEGY", "span")
            else:
                env.setdefault("VLLM_RAY_DP_PACK_STRATEGY", "strict")
        env["VLLM_HOST_IP"] = self.ray_cluster.head_ip
        if self.config.server_config:
            env.update(extra_env_vars)
        if self.extra_env_vars:
            env.update(self.extra_env_vars)

        _saved_affinity = None
        try:
            _saved_affinity = os.sched_getaffinity(0)
            all_cpus = set(range(os.cpu_count() or 1))
            if _saved_affinity != all_cpus:
                os.sched_setaffinity(0, all_cpus)
                print(f"  Reset CPU affinity: {len(_saved_affinity)} -> {len(all_cpus)} CPUs")
        except (OSError, AttributeError):
            pass

        self._process = subprocess.Popen(cmd, stdout=stdout_dest, stderr=stderr_dest, env=env)

        if _saved_affinity is not None:
            try:
                os.sched_setaffinity(0, _saved_affinity)
            except (OSError, AttributeError):
                pass

        print(f"  Started vLLM controller (PID: {self._process.pid})")
        if self.log_path:
            print(f"  Log file: {self.log_path}")

        self._wait_for_healthy()

        try:
            self._warmup_serving()
        except Exception as e:
            print(f"  [warmup] non-fatal exception during warmup: {e!r}")

        print(f"=== vLLM Server Ready ===")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Metrics: {self.metrics_endpoint}")
        print(f"=========================")

        return self.endpoint

    def _warmup_serving(self) -> None:
        """Pre-JIT vLLM-native Triton kernels to avoid mid-inference deadlocks."""
        import urllib.request
        import urllib.error
        import json as _json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=10) as r:
                model_name = _json.loads(r.read().decode())["data"][0]["id"]
        except Exception as e:
            print(f"  [warmup] could not fetch /v1/models ({e!r}); skipping")
            return

        def _fire(prompt: str, max_tokens: int, label: str) -> tuple[str, float, bool, str]:
            body = _json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "top_k": 20,
                "top_p": 0.95,
                "temperature": 0.7,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    _ = r.read()
                return (label, time.time() - t0, True, "")
            except Exception as e:
                return (label, time.time() - t0, False, repr(e))

        seq_prompts = [
            "Hello world.",
            "Write a short Python function that adds two numbers.",
            "Once upon a time, in a faraway land, there lived a curious cat. " * 8,
            "Solve: x^2 + 3x - 4 = 0. Walk through it step by step.",
            "Describe a sunset in 2 sentences.",
            "What is the capital of France?",
            "Compute the integral of x dx from 0 to 1.",
            "List 10 random English words: " + " ".join(f"word{i}" for i in range(48)),
        ]
        print(f"  [warmup] phase 1: {len(seq_prompts)} sequential requests (max_tokens=32)...")
        for i, prompt in enumerate(seq_prompts, 1):
            label, dt, success, err = _fire(prompt, 32, f"seq {i}/{len(seq_prompts)}")
            if success:
                print(f"  [warmup] {label} OK ({dt:.1f}s)")
            else:
                print(f"  [warmup] {label} FAILED: {err}")

        concurrent_n = 16
        long_prompt = (
            "You are an experienced software engineer. " * 20
        )
        batch_prompts = list(seq_prompts) + [long_prompt]
        batch_n_prompts = [batch_prompts[i % len(batch_prompts)] for i in range(concurrent_n)]
        print(f"  [warmup] phase 2: {concurrent_n} concurrent requests (max_tokens=512)...")
        batch_ok = 0
        with ThreadPoolExecutor(max_workers=concurrent_n) as ex:
            futures = [ex.submit(_fire, p, 512, f"par {i+1}/{concurrent_n}") for i, p in enumerate(batch_n_prompts)]
            for f in as_completed(futures):
                label, dt, success, err = f.result()
                if success:
                    batch_ok += 1
                    print(f"  [warmup] {label} OK ({dt:.1f}s)")
                else:
                    print(f"  [warmup] {label} FAILED: {err}")
        print(f"  [warmup] complete (phase 2: {batch_ok}/{concurrent_n})")

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return
        print("Stopping vLLM server...")
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
            print("  vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("  vLLM server not responding, killing...")
            self._process.kill()
            self._process.wait()
        self._process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _wait_for_healthy(self) -> None:
        if self.config.endpoint_json_path:
            self._wait_for_endpoint_json()
        script_path = Path(self.config.wait_for_endpoint_script)
        if script_path.exists():
            self._wait_with_script()
        else:
            self._wait_with_http()

    def _wait_for_endpoint_json(self, timeout: int = 600) -> None:
        if not self.config.endpoint_json_path:
            return
        print(f"  Waiting for endpoint JSON: {self.config.endpoint_json_path}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM controller exited early (code {self._process.returncode}). "
                    f"Check logs at {self.log_path}"
                )
            if os.path.exists(self.config.endpoint_json_path):
                print(f"  Endpoint JSON found after {time.time() - start_time:.1f}s")
                return
            time.sleep(5)
        raise TimeoutError(
            f"Endpoint JSON not created at {self.config.endpoint_json_path} after {timeout}s"
        )

    def _wait_with_script(self) -> None:
        cmd = [
            sys.executable,
            self.config.wait_for_endpoint_script,
            "--max-attempts", str(self.config.health_max_attempts),
            "--retry-delay", str(self.config.health_retry_delay),
            "--health-path", self.config.health_path,
        ]
        if self.config.endpoint_json_path:
            cmd.extend(["--endpoint-json", self.config.endpoint_json_path])
        else:
            cmd.extend(["--endpoint", self.base_url])
        print(f"  Running health check (max {self.config.health_max_attempts} attempts)...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"vLLM health check failed after {self.config.health_max_attempts} attempts"
            ) from e

    def _wait_with_http(self) -> None:
        import urllib.request
        import urllib.error
        health_url = f"{self.base_url}/{self.config.health_path}"
        print(f"  Waiting for health endpoint: {health_url}")
        for attempt in range(1, self.config.health_max_attempts + 1):
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM controller exited early (code {self._process.returncode})"
                )
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        print(f"  Health check passed on attempt {attempt}")
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass
            if attempt < self.config.health_max_attempts:
                time.sleep(self.config.health_retry_delay)
        raise RuntimeError(
            f"vLLM health check failed after {self.config.health_max_attempts} attempts"
        )

    def get_endpoint_info(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "metrics_endpoint": self.metrics_endpoint,
            "model": self.config.model_path,
            "host": self.ray_cluster.head_ip,
            "port": self.config.api_port,
        }

    def write_endpoint_json(self, path: Optional[str] = None) -> str:
        output_path = path or self.config.endpoint_json_path
        if not output_path:
            raise ValueError("No endpoint JSON path specified")
        info = self.get_endpoint_info()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)
        return output_path

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def run_endpoint_health_check(
    endpoint_json: Path,
    max_attempts: int,
    retry_delay: int,
    repo_root: Optional[Path] = None,
) -> None:
    """Run the vLLM endpoint health check script."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "vllm" / "wait_for_endpoint.py"),
        "--endpoint-json",
        str(endpoint_json),
        "--max-attempts",
        str(max_attempts),
        "--retry-delay",
        str(retry_delay),
        "--health-path",
        "v1/models",
    ]
    subprocess.run(cmd, check=True)


__all__ = [
    "RayClusterProtocol",
    "VLLMConfig",
    "VLLMServer",
    "run_endpoint_health_check",
]
