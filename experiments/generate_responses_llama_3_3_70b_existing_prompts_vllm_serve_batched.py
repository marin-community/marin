# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run the same succeeded prompts artifact through Marin's `VllmEnvironment`,
but use a single batched OpenAI-compatible `/v1/completions` request.

This is a tighter comparison target for the direct `llm.generate(...)` path
used by `generate_responses.py`. Unlike the per-prompt chat-completions serve
experiment, this script:

- stages the tokenizer files from the staged `gs://...` model artifact
- renders the exact prompt strings locally via `apply_chat_template(...)`
- submits one batched `/v1/completions` request with `prompt=[...]`

The timing artifact written by this script is scoped to work done inside the
remote worker only. It intentionally excludes Iris scheduler wait time.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1 \
        --extra marin:tpu \
        --tpu v5p-8 \
        --region us-central1 \
        --zone us-central1-a \
        -- python experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests
from iris.marin_fs import url_to_fs
from transformers import AutoTokenizer

from experiments.models import llama_3_3_70b_instruct
from marin.alignment.generate_prompts import load_sharded_jsonl_gz, write_sharded_jsonl_gz
from marin.alignment.generate_responses import _build_messages, _make_response_record
from marin.alignment.inference_config import VLLMConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.execution.remote import remote
from marin.inference.vllm_server import VllmEnvironment
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

PROMPTS_PATH = "gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d"
MODEL_STEP = llama_3_3_70b_instruct


@dataclass(frozen=True)
class VllmServeBatchedResponseGenConfig:
    prompts_path: str
    output_path: str
    model_path: str
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 512
    max_model_len: int = 4096
    tensor_parallel_size: int = 4
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


def _group_completion_texts(choice_texts: list[str], *, prompt_count: int, n: int) -> list[list[dict[str, str | int]]]:
    expected_choice_count = prompt_count * n
    if len(choice_texts) != expected_choice_count:
        raise ValueError(
            f"Expected {expected_choice_count} completion choices for {prompt_count} prompts with n={n}, "
            f"got {len(choice_texts)}"
        )
    grouped: list[list[dict[str, str | int]]] = []
    for prompt_index in range(prompt_count):
        start = prompt_index * n
        prompt_texts = choice_texts[start : start + n]
        grouped.append([{"content": text, "index": choice_index} for choice_index, text in enumerate(prompt_texts)])
    return grouped


def generate_responses_via_vllm_serve_batched(config: VllmServeBatchedResponseGenConfig) -> None:
    timings: dict[str, float] = {}
    artifact_payload: dict[str, Any] = {
        "config": asdict(config),
        "prompt_count": 0,
        "server_mode": None,
        "server_url": None,
        "model_id": None,
        "model_path": config.model_path,
        "request_endpoint": "/completions",
        "request_mode": "single_batched_request",
    }

    overall_start = time.perf_counter()

    prompt_load_start = time.perf_counter()
    prompts = load_sharded_jsonl_gz(config.prompts_path)
    timings["prompt_load_seconds"] = time.perf_counter() - prompt_load_start
    artifact_payload["prompt_count"] = len(prompts)
    logger.info("Loaded %d prompts for batched vLLM serve experiment", len(prompts))

    render_start = time.perf_counter()
    with LMEvaluationHarnessEvaluator._stage_remote_tokenizer_dir(config.model_path) as tokenizer_dir:
        if tokenizer_dir is None:
            raise RuntimeError(f"Could not stage tokenizer files for {config.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        prompt_texts = []
        for prompt in prompts:
            messages = _build_messages(prompt, behavior_statements=None)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(text)
    timings["prompt_render_seconds"] = time.perf_counter() - render_start

    model = ModelConfig(
        name="llama-3-3-70b-vllm-serve-batched",
        path=config.model_path,
        engine_kwargs={
            "load_format": "runai_streamer",
            "max_model_len": config.max_model_len,
            "tensor_parallel_size": config.tensor_parallel_size,
        },
        apply_chat_template=False,
    )

    env: VllmEnvironment | None = None
    try:
        with remove_tpu_lockfile_on_exit():
            server_start = time.perf_counter()
            with VllmEnvironment(model=model, mode="native", timeout_seconds=config.timeout_seconds) as env:
                timings["vllm_server_start_seconds"] = time.perf_counter() - server_start
                if env.model_id is None:
                    raise RuntimeError("Expected vLLM server to expose a model id.")

                artifact_payload["server_mode"] = env.mode
                artifact_payload["server_url"] = env.server_url
                artifact_payload["model_id"] = env.model_id

                request_start = time.perf_counter()
                response = requests.post(
                    f"{env.server_url}/completions",
                    json={
                        "model": env.model_id,
                        "prompt": prompt_texts,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "n": config.n,
                    },
                    timeout=900,
                )
                response.raise_for_status()
                payload = response.json()
                timings["vllm_request_seconds"] = time.perf_counter() - request_start
                timings["vllm_only_seconds"] = timings["vllm_server_start_seconds"] + timings["vllm_request_seconds"]

                choice_texts = [choice["text"] for choice in payload["choices"]]
                grouped_responses = _group_completion_texts(choice_texts, prompt_count=len(prompts), n=config.n)

                results = [
                    _make_response_record(prompt, config.model_path, responses)
                    for prompt, responses in zip(prompts, grouped_responses, strict=True)
                ]

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
                logger.exception("Failed to collect vLLM diagnostics after batched serve experiment failure")
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
    disk="5g",
    ram="256g",
)

response_step = ExecutorStep(
    name="align/debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched_disk5g/responses",
    description=(
        "Generate responses on succeeded prompts artifact via Marin "
        "VllmEnvironment and one batched vllm serve completions request "
        "with disk=5g"
    ),
    fn=remote(
        generate_responses_via_vllm_serve_batched,
        resources=llama_3_3_70b_vllm.resources,
        env_vars={"MARIN_VLLM_MODE": "native"},
        pip_dependency_groups=["tpu", "vllm"],
    ),
    config=VllmServeBatchedResponseGenConfig(
        prompts_path=PROMPTS_PATH,
        output_path=this_output_path(),
        model_path=output_path_of(MODEL_STEP),
        n=1,
        temperature=0.7,
        max_tokens=512,
        max_model_len=4096,
        tensor_parallel_size=4,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[response_step],
        description=(
            "Isolated batched vllm serve response generation on succeeded prompts with regional GCS "
            "Llama 3.3 70B Instruct"
        ),
    )
