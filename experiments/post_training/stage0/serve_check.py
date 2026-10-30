# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke check that an Axolotl Stage 0 export serves in vLLM through its embedded template.

Reads the export's ``tokenizer_config.json`` (and ``chat_template.jinja`` when present) and
requires both to hold the recipe's template, then serves the export and sends a few
chat completions that stop only on the family's end-of-turn token. It fails when the template
is not embedded or a request errors; turn-end and think-close counts are reported, not gated,
since a 20-step smoke model is not expected to hold the protocol.
Native Qwen checks also require a parsed tool call through an explicit model configuration.

Run as a small CPU job; serving dispatches its own GPU job::

    python -m experiments.post_training.stage0.serve_check --family qwen3_5_moe \\
        --checkpoint s3://.../s1-wildchat-rewild-smoke20 --name q35-smoke --out s3://.../serve_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import requests
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig, load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from transformers import AutoTokenizer

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, FAMILY_TOKENIZER_REVISIONS, shared_template
from experiments.post_training.stage0.axolotl_stage0 import QWEN_NATIVE_TEMPLATES

logger = logging.getLogger(__name__)

PROMPTS = (
    "Write a haiku about rivers.",
    "What is 17 * 23? Think it through, then answer.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
)
MAX_TOKENS = 1024
GPUS = 4
REQUEST_TIMEOUT = 3600
TOOL_MAX_TOKENS = 32768


def embedded_templates(checkpoint: str) -> dict[str, str | None]:
    """The chat templates an export carries, keyed by the file that holds them."""
    with fsspec.open(f"{checkpoint}/tokenizer_config.json", "rt") as handle:
        templates = {"tokenizer_config.json": json.load(handle).get("chat_template")}
    fs, path = fsspec.core.url_to_fs(f"{checkpoint}/chat_template.jinja")
    if fs.exists(path):
        with fs.open(path, "rt") as handle:
            templates["chat_template.jinja"] = handle.read()
    return templates


def checkpoint_model(name: str, checkpoint: str) -> ModelConfig:
    """Serve an fp32-master Axolotl export in bf16 with its own template, text only."""
    return ModelConfig(
        name=name,
        location=checkpoint,
        tokenizer=checkpoint,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={"H100": GPUS}, memory="400g", disk="500g"),
        serve=ServeConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            max_model_len=8192,
            limit_mm_per_prompt='{"image":0,"video":0}',
            vllm_extra_args=("--dtype", "bfloat16"),
        ),
    )


def checked_templates(checkpoint: str, expected: str) -> dict[str, str | None]:
    """Verify every exported template, including the processor's higher-priority Jinja file."""
    templates = embedded_templates(checkpoint)
    mismatched = [name for name, template in templates.items() if template != expected]
    if mismatched:
        raise ValueError(f"{checkpoint}: {mismatched} do not hold the expected template")
    return templates


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_TOKEN_MAPS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="slash-free serving name")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--recipe",
        choices=("shared-curriculum", "qwen-native-s3", "qwen-native-curriculum"),
        default="shared-curriculum",
    )
    parser.add_argument("--model-config", type=Path, help="explicit serving configuration; required for native Qwen")
    args = parser.parse_args()

    tokens = FAMILY_TOKEN_MAPS[args.family]
    native = args.recipe != "shared-curriculum"
    expected = QWEN_NATIVE_TEMPLATES[args.family] if native else shared_template(tokens)
    templates = checked_templates(args.checkpoint, expected)
    model = load_model_config(args.model_config) if args.model_config else checkpoint_model(args.name, args.checkpoint)
    if native and args.model_config is None:
        parser.error("native Qwen smoke requires --model-config with the intended context and tool parser")
    if model.location != args.checkpoint or model.tokenizer != args.checkpoint:
        raise ValueError("serving model and tokenizer must both reload the export being checked")
    gpus = model.resource_hint.gpu["H100"]
    repo, revision = FAMILY_TOKENIZER_REVISIONS[args.family]
    end_of_turn = AutoTokenizer.from_pretrained(repo, revision=revision).convert_tokens_to_ids(tokens.turn_close)

    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=gpus, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        model,
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        request_timeout_seconds=REQUEST_TIMEOUT,
    )
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}

        def complete(prompt: str) -> dict:
            body = {
                "model": endpoint.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": MAX_TOKENS,
                "skip_special_tokens": False,
                "ignore_eos": True,
                "stop_token_ids": [end_of_turn],
            }
            response = requests.post(
                endpoint.url("chat/completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            text = choice["message"]["content"] or ""
            return {
                "prompt": prompt,
                "turn_ended": choice["finish_reason"] == "stop" and choice.get("stop_reason") == end_of_turn,
                "think_closed": tokens.think_open not in text or tokens.think_close in text,
                "text": text,
            }

        with ThreadPoolExecutor(len(PROMPTS)) as pool:
            completions = list(pool.map(complete, PROMPTS))
        tool_response = None
        if native:
            body = {
                "model": endpoint.model,
                "messages": [{"role": "user", "content": "Call get_weather for Paris. Do not guess the weather."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather in a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": TOOL_MAX_TOKENS,
            }
            if args.family == "qwen3_5_moe":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            response = requests.post(
                endpoint.url("chat/completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            tool_response = response.json()
    report = {
        "checkpoint": args.checkpoint,
        "templates_checked": sorted(templates),
        "turn_ended": sum(c["turn_ended"] for c in completions),
        "think_closed": sum(c["think_closed"] for c in completions),
        "completions": completions,
        "tool_call_parser": model.serve.tool_call_parser,
        "tool_response": tool_response,
    }
    with fsspec.open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    if tool_response is not None:
        calls = tool_response["choices"][0]["message"].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "get_weather":
            raise ValueError("native serving smoke did not return the requested parsed tool call; see saved report")
        arguments = json.loads(calls[0]["function"]["arguments"])
        if arguments.get("city", "").casefold() != "paris":
            raise ValueError(f"native serving smoke returned incorrect tool arguments: {arguments}")
    logger.info("served %d completions; %d ended the turn", len(completions), report["turn_ended"])


if __name__ == "__main__":
    main()
