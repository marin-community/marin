# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse

import requests

from marin.evaluation.utils import kill_process_on_port


def _is_object_store_path(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"gs", "s3"}


def _wait_for_vllm(process: subprocess.Popen, host: str, port: int, *, timeout_seconds: int) -> str:
    server_url = f"http://{host}:{port}/v1"
    start_time = time.time()
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM process exited early with code {return_code}. See logs above.")

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"vLLM did not become ready within {timeout_seconds}s")
        try:
            response = requests.get(f"{server_url}/models", timeout=10)
            if response.status_code == 200:
                return server_url
        except requests.ConnectionError:
            pass
        time.sleep(5)


def _first_model_id(server_url: str) -> str:
    response = requests.get(f"{server_url}/models", timeout=10)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"vLLM server returned no models: {payload}")
    model_id = data[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"Unexpected /models response: {payload}")
    return model_id


def _smoke_completion(server_url: str, *, prompt: str, max_tokens: int) -> dict[str, Any]:
    model_id = _first_model_id(server_url)
    response = requests.post(
        f"{server_url}/completions",
        json={
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _preflight_imports() -> None:
    """Fail fast on common environment issues (e.g., broken torch/torchvision)."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys\n"
            "print('python', sys.version)\n"
            "print('executable', sys.executable)\n"
            "import importlib\n"
            "import torch\n"
            "print('torch', torch.__version__)\n"
            "try:\n"
            "    import torchvision\n"
            "    print('torchvision', torchvision.__version__)\n"
            "    importlib.import_module('torchvision.ops')\n"
            "except Exception as e:\n"
            "    raise SystemExit(f'torchvision import failed: {e!r}')\n"
            "try:\n"
            "    import jax\n"
            "    print('jax', getattr(jax, '__version__', 'unknown'))\n"
            "    print('jax backend', jax.default_backend())\n"
            "    print('jax devices', len(jax.devices()))\n"
            "except Exception as e:\n"
            "    print(f'jax import failed: {e!r}')\n"
            "try:\n"
            "    import vllm\n"
            "    print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
            "except Exception as e:\n"
            "    raise SystemExit(f'vllm import failed: {e!r}')\n"
        ),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoketest vLLM on TPU nodes.")
    parser.add_argument("--mode", choices=["python", "server"], default="python")
    parser.add_argument(
        "--model",
        required=True,
        help='Model identifier or path (e.g. "meta-llama/Meta-Llama-3.1-8B" or "gs://...").',
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument(
        "--model-impl-type",
        choices=["auto", "vllm", "flax_nnx"],
        default=os.environ.get("MODEL_IMPL_TYPE", "vllm"),
        help=(
            "Value for MODEL_IMPL_TYPE (tpu_inference backend). "
            "Default is vllm because flax_nnx currently fails without an auto mesh context."
        ),
    )
    parser.add_argument(
        "--load-format",
        default=None,
        help='vLLM load format (e.g. "runai_streamer_sharded"). If unset, auto-enables for gs:// or s3:// paths.',
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip import/version checks (useful if vLLM takes a long time to import).",
    )
    parser.epilog = (
        "Example:\n"
        "  uv run python scripts/smoke/vllm_tpu_smoke.py --model gs://BUCKET/path/to/model\n"
        "  uv run python scripts/smoke/vllm_tpu_smoke.py --model meta-llama/Meta-Llama-3.1-8B\n"
    )
    args, extra = parser.parse_known_args()

    model = args.model

    os.environ["MODEL_IMPL_TYPE"] = args.model_impl_type

    if not args.skip_preflight:
        _preflight_imports()

    load_format = args.load_format
    if load_format is None and _is_object_store_path(model):
        # Prefer the non-sharded streamer by default; it supports standard HF-style
        # checkpoints stored in object storage.
        load_format = "runai_streamer"

    if args.mode == "python":
        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "trust_remote_code": True,
            "max_model_len": args.max_model_len,
        }
        if args.tensor_parallel_size is not None:
            llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        if load_format is not None:
            llm_kwargs["load_format"] = load_format

        print(f"Constructing vLLM LLM: {llm_kwargs}", flush=True)
        llm = LLM(**llm_kwargs)
        outputs = llm.generate(args.prompt, SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
        text = outputs[0].outputs[0].text
        print("Completion OK", flush=True)
        print(text, flush=True)
        return

    command = [
        "vllm",
        "serve",
        model,
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--distributed-executor-backend",
        "ray",
        "--max-model-len",
        str(args.max_model_len),
        *extra,
    ]

    if load_format is not None:
        command.extend(["--load-format", load_format])

    print(f"Starting vLLM: {' '.join(command)}", flush=True)

    process = subprocess.Popen(command, env=os.environ.copy())
    try:
        server_url = _wait_for_vllm(process, args.host, args.port, timeout_seconds=args.timeout_seconds)
        print(f"vLLM ready at {server_url}", flush=True)

        result = _smoke_completion(server_url, prompt=args.prompt, max_tokens=args.max_tokens)
        text = result.get("choices", [{}])[0].get("text", "")
        print("Completion OK", flush=True)
        print(text, flush=True)
    finally:
        try:
            process.terminate()
            process.wait(timeout=30)
        except Exception:
            process.kill()
        kill_process_on_port(args.port)


if __name__ == "__main__":
    main()
