#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scratch-only direct vLLM prompt-logprob baseline runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "A B"
DEFAULT_MAX_MODEL_LEN = 1024
DEFAULT_MAX_NUM_BATCHED_TOKENS = 1024
DEFAULT_ARTIFACT_DIR = "tmp/iris_pressure_test_20260504/prompt_logprob_baseline_remote_artifacts"

VLLM_ENV_DEFAULTS = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    "HF_ALLOW_CODE_EVAL": "1",
    "WANDB_MODE": "offline",
}


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for key, value in VLLM_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    server_url = f"http://{args.host}:{args.port}/v1"

    print_marker(
        "PROMPT_LOGPROB_RUN_CONFIG",
        {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "tokenizer": args.tokenizer,
            "prompt": args.prompt,
            "artifact_dir": str(artifact_dir),
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "server_url": server_url,
            "vllm_env": {key: os.environ.get(key) for key in sorted(VLLM_ENV_DEFAULTS)},
        },
    )

    http_artifact: dict[str, object] | None = None
    process: subprocess.Popen[str] | None = None
    runner_error: str | None = None

    try:
        process = start_vllm_native_server(args, artifact_dir)
        wait_for_models(server_url, process, args.server_timeout, args.http_timeout)
        models_status_code, models_text = get_text(f"{server_url}/models", timeout=args.http_timeout)
        models_payload = json.loads(models_text)
        model_id = str(models_payload["data"][0]["id"])
        completion_payload = {
            "model": model_id,
            "prompt": args.prompt,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": 1,
            "seed": 1234,
            "echo": True,
        }
        completion_status_code, completion_text = post_json(
            f"{server_url}/completions",
            completion_payload,
            timeout=args.http_timeout,
        )
        http_artifact = {
            "server_url": server_url,
            "models_status_code": models_status_code,
            "models_response": models_payload,
            "selected_model_id": model_id,
            "completion_request": completion_payload,
            "completion_status_code": completion_status_code,
            "completion_response_text": completion_text,
        }
        write_json(artifact_dir / "http_response.json", http_artifact)
        print_marker("PROMPT_LOGPROB_HTTP_RESPONSE", http_artifact)

    except Exception as exc:
        runner_error = f"{type(exc).__name__}: {exc}"
        write_text(artifact_dir / "runner_error.txt", runner_error)
        print_marker("PROMPT_LOGPROB_RUNNER_ERROR", {"error": runner_error})
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        time.sleep(args.log_flush_delay)
        preserve_vllm_logs(artifact_dir)

    print_marker(
        "PROMPT_LOGPROB_RUN_SUMMARY",
        {
            "runner_error": runner_error,
            "had_http_artifact": http_artifact is not None,
            "artifact_dir": str(artifact_dir),
        },
    )

    # A vLLM HTTP 500 is a valid captured baseline. Fail only when the runner
    # could not complete the direct HTTP exchange.
    return 0 if http_artifact is not None else 1


def start_vllm_native_server(args: argparse.Namespace, artifact_dir: Path) -> subprocess.Popen[str]:
    cmd = [
        shutil.which("vllm") or "vllm",
        "serve",
        args.model_path,
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    write_text(artifact_dir / "vllm_command.txt", " ".join(cmd))
    print_marker("PROMPT_LOGPROB_VLLM_COMMAND", cmd)

    stdout_f = open(artifact_dir / "vllm_stdout.log", "w")
    stderr_f = open(artifact_dir / "vllm_stderr.log", "w")
    try:
        return subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True, env=vllm_env())
    finally:
        stdout_f.close()
        stderr_f.close()


def wait_for_models(server_url: str, process: subprocess.Popen[str], timeout: int, http_timeout: int) -> None:
    models_url = f"{server_url}/models"
    start = time.monotonic()
    while time.monotonic() - start <= timeout:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with exit code {process.returncode}")
        try:
            status_code, _ = get_text(models_url, timeout=http_timeout)
            if status_code == 200:
                return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server at {models_url} did not become ready within {timeout} seconds")


def get_text(url: str, *, timeout: int) -> tuple[int, str]:
    with urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return int(response.status), body


def post_json(url: str, payload: object, *, timeout: int) -> tuple[int, str]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), body


def vllm_env() -> dict[str, str]:
    env = dict(os.environ)
    cache_dir = env.get("JAX_COMPILATION_CACHE_DIR", f"{env.get('MARIN_PREFIX', '/tmp/marin')}/compilation-cache")
    defaults = {
        "MODEL_IMPL_TYPE": "vllm",
        "TPU_MIN_LOG_LEVEL": "3",
        "TPU_STDERR_LOG_LEVEL": "3",
        "JAX_ENABLE_COMPILATION_CACHE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "JAX_COMPILATION_CACHE_DIR": cache_dir,
        "VLLM_XLA_CACHE_PATH": env.get("VLLM_XLA_CACHE_PATH", cache_dir),
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "2",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def preserve_vllm_logs(artifact_dir: Path) -> None:
    for name in ("stdout.log", "stderr.log"):
        source = artifact_dir / f"vllm_{name}"
        if source.exists():
            text = source.read_text(errors="replace")
        else:
            text = f"<missing {source}>"
        marker_name = f"VLLM_{name.replace('.', '_').upper()}"
        print_text_marker(marker_name, text)


def print_marker(name: str, payload: object) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print_text_marker(name, text)


def print_text_marker(name: str, text: str) -> None:
    print(f"=== {name}_BEGIN ===")
    print(text)
    if text and not text.endswith("\n"):
        print()
    print(f"=== {name}_END ===")
    sys.stdout.flush()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text if text.endswith("\n") else f"{text}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="qwen3-0_6b-direct-prompt-logprob")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--server-timeout", type=int, default=1800)
    parser.add_argument("--http-timeout", type=int, default=120)
    parser.add_argument("--log-flush-delay", type=float, default=2.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
