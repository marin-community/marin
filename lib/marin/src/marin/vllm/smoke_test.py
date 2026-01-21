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

import argparse
import os
import sys
import time
import traceback
from typing import Literal
from urllib.parse import urlparse

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES, VllmEnvironment, resolve_vllm_mode
from marin.utils import remove_tpu_lockfile_on_exit


def run_one_query(
    *,
    model_name_or_path: str,
    prompt: str,
    load_format: str | None,
    max_model_len: int | None,
    mode: Literal["docker", "native"] | None,
    docker_image: str | None,
) -> str:
    parsed = urlparse(model_name_or_path)
    is_object_store = parsed.scheme in {"gs", "s3"}
    engine_kwargs: dict = {}
    if load_format is not None:
        engine_kwargs["load_format"] = load_format
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len

    if is_object_store:
        model = ModelConfig(name="smoke-test-model", path=model_name_or_path, engine_kwargs=engine_kwargs)
    else:
        model = ModelConfig(name=model_name_or_path, path=None, engine_kwargs=engine_kwargs)

    env = VllmEnvironment(
        model=model,
        host="127.0.0.1",
        port=None,
        timeout_seconds=3600,
        mode=mode,
        docker_image=docker_image,
        use_server=True,
    )
    try:
        with env:
            if env.model_id is None:
                raise RuntimeError("Expected vLLM server to expose a model id.")
            model_id = env.model_id
            response = requests.post(
                f"{env.server_url}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 128,
                },
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
    except Exception as exc:
        print("Smoke test failed with exception:", exc)
        print("Environment snapshot:", env.debug_snapshot())
        if env.vllm_server is not None and env.vllm_server.mode == "docker":
            try:
                if env.vllm_server.docker_run_cmd:
                    print("vLLM Docker run command (redacted):", env.vllm_server.docker_run_cmd)
                print("vLLM Docker logs (tail):")
                print(env.vllm_server.logs_tail())
                print("vLLM Docker inspect:")
                print(env.vllm_server.inspect())
            except Exception as diag_exc:
                print("Failed to collect Docker diagnostics:", diag_exc)
        traceback.print_exc()
        raise
    finally:
        model.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM TPU Docker sidecar via OpenAI-compatible HTTP API.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier: HF repo id (e.g. meta-llama/Llama-3.3-8B-Instruct) or object-store path (gs://... or s3://...).",
    )
    parser.add_argument(
        "--load-format",
        choices=["runai_streamer", "runai_streamer_sharded"],
        default=None,
        help="Optional vLLM load format (recommended for gs:// or s3://). Defaults to evaluator auto-selection.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max model sequence length to configure vLLM with (default: 8192).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to start/stop vLLM and run the query (default: 1). Useful for cache validation.",
    )
    parser.add_argument(
        "--local-cache-dir",
        default=None,
        help=(
            "Optional stable local compilation cache dir (e.g. /tmp/marin-jax-compilation-cache). "
            "When set, exports JAX_COMPILATION_CACHE_DIR and VLLM_XLA_CACHE_PATH."
        ),
    )
    parser.add_argument("--prompt", default="Write a short haiku about TPUs.", help="Prompt to send.")
    parser.add_argument(
        "--mode",
        choices=["docker", "native"],
        default=None,
        help="Override MARIN_VLLM_MODE (default: use env; docker if unset).",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Override MARIN_VLLM_DOCKER_IMAGE (required in docker mode if env var unset).",
    )
    parser.add_argument(
        "--tpu-type",
        default="v5p-8",
        help="TPU type to request when launching via Ray/Fray (default: v5p-8).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in the current process instead of launching a Ray/Fray job.",
    )
    args = parser.parse_args(argv)

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    if args.local:
        if args.local_cache_dir is not None:
            os.environ["JAX_COMPILATION_CACHE_DIR"] = args.local_cache_dir
            os.environ["VLLM_XLA_CACHE_PATH"] = args.local_cache_dir

        for i in range(args.repeat):
            start = time.time()
            output = run_one_query(
                model_name_or_path=args.model,
                prompt=args.prompt,
                load_format=args.load_format,
                max_model_len=args.max_model_len,
                mode=args.mode,
                docker_image=args.docker_image,
            )
            elapsed = time.time() - start
            print(f"[run {i + 1}/{args.repeat}] {elapsed:.1f}s")
            print(output)
        return 0

    mode_str = resolve_vllm_mode(args.mode)

    env_vars: dict[str, str] = {}
    if args.mode is not None:
        env_vars["MARIN_VLLM_MODE"] = args.mode
    if args.docker_image is not None:
        env_vars["MARIN_VLLM_DOCKER_IMAGE"] = args.docker_image
    if args.local_cache_dir is not None:
        env_vars["JAX_COMPILATION_CACHE_DIR"] = args.local_cache_dir
        env_vars["VLLM_XLA_CACHE_PATH"] = args.local_cache_dir

    def _run() -> None:
        with remove_tpu_lockfile_on_exit():
            for i in range(args.repeat):
                start = time.time()
                try:
                    output = run_one_query(
                        model_name_or_path=args.model,
                        prompt=args.prompt,
                        load_format=args.load_format,
                        max_model_len=args.max_model_len,
                        mode=args.mode,
                        docker_image=args.docker_image,
                    )
                except Exception:
                    traceback.print_exc()
                    raise
                elapsed = time.time() - start
                print(f"[run {i + 1}/{args.repeat}] {elapsed:.1f}s")
                print(output)

    cluster = current_cluster()
    resources = ResourceConfig.with_tpu(args.tpu_type)
    job_request = JobRequest(
        name=f"vllm-smoke:{args.tpu_type}",
        entrypoint=Entrypoint.from_callable(_run),
        resources=resources,
        environment=EnvironmentConfig.create(
            extras=["eval", "tpu"],
            pip_packages=VLLM_NATIVE_PIP_PACKAGES if mode_str == "native" else (),
            env_vars=env_vars or None,
        ),
    )
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
