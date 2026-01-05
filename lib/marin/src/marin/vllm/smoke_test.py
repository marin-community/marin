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
import json
import os
import sys
from typing import Literal

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator, VllmServerHandle
from marin.utils import remove_tpu_lockfile_on_exit


def _get_first_model_id(server_url: str) -> str:
    response = requests.get(f"{server_url}/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"No models returned from {server_url}/models: {json.dumps(payload)[:2000]}")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"Missing model id in {server_url}/models response: {json.dumps(payload)[:2000]}")
    return str(model_id)


def run_one_query(
    *,
    model_name_or_path: str,
    prompt: str,
    mode: Literal["docker", "native"] | None,
    docker_image: str | None,
) -> str:
    model = ModelConfig(name=model_name_or_path, path=None, engine_kwargs={})
    vllm_server: VllmServerHandle | None = None
    try:
        vllm_server = VllmTpuEvaluator.start_vllm_server_in_background(
            model=model,
            host="127.0.0.1",
            port=None,
            timeout_seconds=3600,
            extra_args=None,
            mode=mode,
            docker_image=docker_image,
        )

        model_id = _get_first_model_id(vllm_server.server_url)
        response = requests.post(
            f"{vllm_server.server_url}/chat/completions",
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
    finally:
        VllmTpuEvaluator.cleanup(model, vllm_server=vllm_server)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM TPU Docker sidecar via OpenAI-compatible HTTP API.")
    parser.add_argument(
        "--model",
        default="/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-1B-Instruct",
        help="Model path or HF repo id (default: Llama 3.3 1B path under /opt/gcsfuse_mount).",
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

    if args.local:
        output = run_one_query(
            model_name_or_path=args.model,
            prompt=args.prompt,
            mode=args.mode,
            docker_image=args.docker_image,
        )
        print(output)
        return 0

    mode_str = (args.mode if args.mode is not None else os.environ.get("MARIN_VLLM_MODE", "docker")).lower()
    if mode_str not in ("native", "docker"):
        raise ValueError(f"Unknown mode={mode_str!r}; expected 'native' or 'docker'.")

    env_vars: dict[str, str] = {}
    if args.mode is not None:
        env_vars["MARIN_VLLM_MODE"] = args.mode
    if args.docker_image is not None:
        env_vars["MARIN_VLLM_DOCKER_IMAGE"] = args.docker_image

    def _run() -> None:
        with remove_tpu_lockfile_on_exit():
            output = run_one_query(
                model_name_or_path=args.model,
                prompt=args.prompt,
                mode=args.mode,
                docker_image=args.docker_image,
            )
            print(output)

    cluster = current_cluster()
    resources = ResourceConfig.with_tpu(args.tpu_type)
    job_request = JobRequest(
        name=f"vllm-smoke:{args.tpu_type}",
        entrypoint=Entrypoint.from_callable(_run),
        resources=resources,
        environment=EnvironmentConfig.create(
            extras=["eval", "tpu"],
            pip_packages=("vllm-tpu",) if mode_str == "native" else (),
            env_vars=env_vars or None,
        ),
    )
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
