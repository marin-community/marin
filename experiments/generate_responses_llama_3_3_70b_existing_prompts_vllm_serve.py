# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run the same succeeded prompts artifact through Marin's `VllmEnvironment`.

This is a side-by-side comparison target for the direct `llm.generate(...)`
path used by `generate_responses.py`. Unlike that path, this experiment starts
`vllm serve` via `VllmEnvironment` and sends OpenAI-compatible
`/v1/chat/completions` requests against the local server.

The timing artifact written by this script is scoped to work done inside the
remote worker only. It intentionally excludes Iris scheduler wait time.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name generate-responses-llama-3-3-70b-vllm-serve-gcs-us-central1 \
        --extra marin:tpu \
        --tpu v5p-8 \
        --region us-central1 \
        --zone us-central1-a \
        -- python experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve.py
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests
from iris.marin_fs import url_to_fs

from experiments.models import llama_3_3_70b_instruct
from marin.alignment.generate_prompts import load_sharded_jsonl_gz, write_sharded_jsonl_gz
from marin.alignment.generate_responses import _build_messages, _make_response_record
from marin.alignment.inference_config import VLLMConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.execution.remote import remote
from marin.inference.vllm_server import VllmEnvironment
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

PROMPTS_PATH = "gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d"
MODEL_STEP = llama_3_3_70b_instruct


@dataclass(frozen=True)
class VllmServeResponseGenConfig:
    prompts_path: str
    output_path: str
    model_path: str
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 512
    max_model_len: int = 4096
    tensor_parallel_size: int = 4
    client_workers: int = 76
    timeout_seconds: int = 3600


def _write_json(output_path: str, filename: str, payload: dict[str, Any]) -> None:
    fs, base_path = url_to_fs(output_path)
    fs.makedirs(base_path, exist_ok=True)
    with fs.open(f"{base_path}/{filename}", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(output_path: str, filename: str, content: str) -> None:
    fs, base_path = url_to_fs(output_path)
    fs.makedirs(base_path, exist_ok=True)
    with fs.open(f"{base_path}/{filename}", "w") as f:
        f.write(content)


def _request_one(
    *,
    prompt: dict[str, Any],
    server_url: str,
    model_id: str,
    model_path: str,
    temperature: float,
    max_tokens: int,
    n: int,
) -> dict[str, Any]:
    messages = _build_messages(prompt, behavior_statements=None)
    response = requests.post(
        f"{server_url}/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        },
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()
    responses = [
        {"content": choice["message"]["content"] or "", "index": idx} for idx, choice in enumerate(payload["choices"])
    ]
    return _make_response_record(prompt, model_path, responses)


def generate_responses_via_vllm_serve(config: VllmServeResponseGenConfig) -> None:
    timings: dict[str, float] = {}
    artifact_payload: dict[str, Any] = {
        "config": asdict(config),
        "prompt_count": 0,
        "client_workers": 0,
        "server_mode": None,
        "server_url": None,
        "model_id": None,
        "model_path": config.model_path,
    }

    overall_start = time.perf_counter()

    prompt_load_start = time.perf_counter()
    prompts = load_sharded_jsonl_gz(config.prompts_path)
    timings["prompt_load_seconds"] = time.perf_counter() - prompt_load_start
    artifact_payload["prompt_count"] = len(prompts)
    logger.info("Loaded %d prompts for vLLM serve experiment", len(prompts))

    model = ModelConfig(
        name="llama-3-3-70b-vllm-serve",
        path=config.model_path,
        engine_kwargs={
            "load_format": "runai_streamer",
            "max_model_len": config.max_model_len,
            "tensor_parallel_size": config.tensor_parallel_size,
        },
        apply_chat_template=True,
    )

    env: VllmEnvironment | None = None
    try:
        with remove_tpu_lockfile_on_exit():
            server_start = time.perf_counter()
            with VllmEnvironment(model=model, mode="native", timeout_seconds=config.timeout_seconds) as env:
                timings["vllm_server_start_seconds"] = time.perf_counter() - server_start
                if env.model_id is None:
                    raise RuntimeError("Expected vLLM server to expose a model id.")

                request_start = time.perf_counter()
                worker_count = min(config.client_workers, len(prompts))
                artifact_payload["client_workers"] = worker_count
                artifact_payload["server_mode"] = env.mode
                artifact_payload["server_url"] = env.server_url
                artifact_payload["model_id"] = env.model_id
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = [
                        pool.submit(
                            _request_one,
                            prompt=prompt,
                            server_url=env.server_url,
                            model_id=env.model_id,
                            model_path=config.model_path,
                            temperature=config.temperature,
                            max_tokens=config.max_tokens,
                            n=config.n,
                        )
                        for prompt in prompts
                    ]
                    results = [future.result() for future in futures]
                timings["vllm_request_seconds"] = time.perf_counter() - request_start
                timings["vllm_only_seconds"] = timings["vllm_server_start_seconds"] + timings["vllm_request_seconds"]

                write_start = time.perf_counter()
                write_sharded_jsonl_gz(results, config.output_path, shard_size=5000)
                timings["result_write_seconds"] = time.perf_counter() - write_start

                _write_json(
                    config.output_path,
                    "timing.json",
                    {
                        "timings": timings,
                        **artifact_payload,
                    },
                )
                _write_text(
                    config.output_path,
                    "vllm_server_logs_tail.txt",
                    env.logs_tail(max_lines=400),
                )
    except Exception:
        if env is not None and env.vllm_server is not None:
            try:
                diagnostics = env.diagnostics(max_lines=400)
                for label, value in diagnostics.items():
                    logger.error("%s:\n%s", label, value)
            except Exception:
                logger.exception("Failed to collect vLLM diagnostics after serve experiment failure")
        raise
    finally:
        timings["total_worker_seconds"] = time.perf_counter() - overall_start
        _write_json(
            config.output_path,
            "timing.json",
            {
                "timings": timings,
                **artifact_payload,
            },
        )


llama_3_3_70b_vllm = VLLMConfig(
    model=output_path_of(MODEL_STEP),
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="500g",
    ram="256g",
)

response_step = ExecutorStep(
    name="align/debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve/responses",
    description="Generate responses on succeeded prompts artifact via Marin VllmEnvironment and vllm serve",
    fn=remote(
        generate_responses_via_vllm_serve,
        resources=llama_3_3_70b_vllm.resources,
        env_vars={"MARIN_VLLM_MODE": "native"},
        pip_dependency_groups=["tpu", "vllm"],
    ),
    config=VllmServeResponseGenConfig(
        prompts_path=PROMPTS_PATH,
        output_path=this_output_path(),
        model_path=output_path_of(MODEL_STEP),
        n=1,
        temperature=0.7,
        max_tokens=512,
        max_model_len=4096,
        tensor_parallel_size=4,
        client_workers=76,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[response_step],
        description=(
            "Isolated vllm serve response generation on succeeded prompts with " "regional GCS Llama 3.3 70B Instruct"
        ),
    )
