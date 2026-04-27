# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe native vLLM startup for Llama 70B on an Iris TPU worker.

This is intentionally smaller than the IFBench rollout runner: it starts
`vllm serve`, waits for `/v1/models`, sends one prompt if ready, and always
copies startup logs to durable storage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import subprocess
import time
from typing import Any

import fsspec
import httpx
import requests

from marin.inference.vllm_server import _vllm_env
from marin.utils import _hacky_remove_tpu_lockfile

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = "gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_CACHE_DIR = "/app/.hf_cache"
POLL_INTERVAL_SECONDS = 5
HEARTBEAT_SECONDS = 60


def write_text(path: str, text: str) -> None:
    with fsspec.open(path, "wt") as f:
        f.write(text)


def copy_file_to_uri(local_path: pathlib.Path, output_uri: str) -> None:
    if not local_path.exists():
        write_text(output_uri, f"<missing {local_path}>\n")
        return
    with local_path.open("rb") as src, fsspec.open(output_uri, "wb") as dst:
        shutil.copyfileobj(src, dst)


def tail(path: pathlib.Path, n_bytes: int = 200_000) -> str:
    if not path.exists():
        return f"<missing {path}>"
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_bytes))
        return f.read().decode(errors="replace")


def configure_model_cache(env: dict[str, str], cache_dir: str) -> None:
    cache_path = pathlib.Path(cache_dir)
    hub_path = cache_path / "hub"
    for path in (cache_path, hub_path, cache_path / "transformers", cache_path / "xdg", cache_path / "vllm-assets"):
        path.mkdir(parents=True, exist_ok=True)

    # Iris defaults HF_HOME to /root/.cache/huggingface, which lives on the
    # small container overlay. 70B weights need the disk-backed workdir volume.
    env["HF_HOME"] = str(cache_path)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_path)
    env["HF_HUB_CACHE"] = str(hub_path)
    env["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    env["XDG_CACHE_HOME"] = str(cache_path / "xdg")
    env["VLLM_ASSETS_CACHE"] = str(cache_path / "vllm-assets")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    return f"$ {' '.join(command)}\nexit={result.returncode}\n{output}"


def storage_snapshot(cache_dir: str) -> dict[str, str]:
    paths = ["/", "/app", "/tmp", cache_dir]
    return {
        "df": command_output(["df", "-h", *paths]),
        "mount": command_output(["mount"]),
        "cache_du": command_output(["du", "-sh", cache_dir]),
    }


def write_status(output_dir: str, status: dict[str, Any], stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    write_text(f"{output_dir}/status.json", json.dumps(status, indent=2, ensure_ascii=False))
    copy_file_to_uri(stdout_path, f"{output_dir}/stdout.log")
    copy_file_to_uri(stderr_path, f"{output_dir}/stderr.log")
    write_text(
        f"{output_dir}/tail.txt",
        "--- stdout tail ---\n" f"{tail(stdout_path)}\n" "--- stderr tail ---\n" f"{tail(stderr_path)}\n",
    )


def readiness_loop(server_url: str, process: subprocess.Popen[str], timeout: int) -> None:
    models_url = f"{server_url}/models"
    start = time.time()
    last_heartbeat = start
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM exited before readiness with code {return_code}")
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM ready: %s", response.text[:500])
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            logger.info("Still waiting for vLLM startup at %s after %.1fs", models_url, now - start)
            last_heartbeat = now
        if now - start > timeout:
            raise TimeoutError(f"vLLM did not become ready within {timeout}s")
        time.sleep(POLL_INTERVAL_SECONDS)


def one_completion(server_url: str, model_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{server_url}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                "temperature": 0,
                "max_tokens": 8,
            },
        )
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-precompile", action="store_true")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_dir = pathlib.Path("/tmp/ifbench_vllm_70b_probe")
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    status: dict[str, Any] = {
        "model_id": args.model_id,
        "output_dir": args.output_dir,
        "started_at": time.time(),
        "ready": False,
        "completion_ok": False,
        "command": None,
        "return_code": None,
        "cache_dir": args.cache_dir,
    }

    vllm_bin = shutil.which("vllm") or "vllm"
    command = [
        vllm_bin,
        "serve",
        args.model_id,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    status["command"] = command
    env = _vllm_env()
    configure_model_cache(env, args.cache_dir)
    if args.skip_precompile:
        env["VLLM_TPU_SKIP_PRECOMPILE"] = "1"
    status["storage_before"] = storage_snapshot(args.cache_dir)
    status["env_subset"] = {
        key: env.get(key)
        for key in (
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "XDG_CACHE_HOME",
            "VLLM_ASSETS_CACHE",
            "MODEL_IMPL_TYPE",
            "JAX_COMPILATION_CACHE_DIR",
            "VLLM_XLA_CACHE_PATH",
            "VLLM_TPU_SKIP_PRECOMPILE",
            "TPU_MIN_LOG_LEVEL",
            "TPU_STDERR_LOG_LEVEL",
            "TPU_VISIBLE_DEVICES",
            "TPU_ACCELERATOR_TYPE",
        )
    }

    logger.info("Removing stale TPU lockfile before vLLM startup")
    _hacky_remove_tpu_lockfile()
    logger.info("Starting command: %s", command)
    process: subprocess.Popen[str] | None = None
    with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
        try:
            process = subprocess.Popen(command, stdout=stdout_f, stderr=stderr_f, text=True, env=env)
            server_url = f"http://127.0.0.1:{args.port}/v1"
            readiness_loop(server_url, process, args.timeout)
            status["ready"] = True
            completion = one_completion(server_url, args.model_id)
            status["completion_ok"] = True
            status["completion"] = completion
        except Exception as exc:
            logger.exception("Probe failed")
            status["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if process is not None:
                status["return_code"] = process.poll()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                status["final_return_code"] = process.poll()
            status["finished_at"] = time.time()
            status["storage_after"] = storage_snapshot(args.cache_dir)
            write_status(args.output_dir, status, stdout_path, stderr_path)
            logger.info("Wrote probe diagnostics to %s", args.output_dir)


if __name__ == "__main__":
    main()
