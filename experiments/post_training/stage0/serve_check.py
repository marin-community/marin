# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke check that an Axolotl Stage 0 export serves in vLLM through its embedded template.

Reads the export's ``tokenizer_config.json`` (and ``chat_template.jinja`` when present) and
requires both to hold the family's shared template, then serves the export and sends a few
chat completions that stop only on the family's end-of-turn token. It fails when the template
is not embedded or a request errors; turn-end and think-close counts are reported, not gated,
since a 20-step smoke model is not expected to hold the protocol.

Run as a small CPU job; serving dispatches its own GPU job::

    python -m experiments.post_training.stage0.serve_check --family qwen3_5_moe \\
        --checkpoint s3://.../s1-wildchat-rewild-smoke20 --name q35-smoke --out s3://.../serve_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec
import requests
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from transformers import AutoTokenizer

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, FAMILY_TOKENIZER_REVISIONS, shared_template

logger = logging.getLogger(__name__)

PROMPTS = (
    "Write a haiku about rivers.",
    "What is 17 * 23? Think it through, then answer.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
)
MAX_TOKENS = 1024
GPUS = 4
REQUEST_TIMEOUT = 600


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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_TOKEN_MAPS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="slash-free serving name")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokens = FAMILY_TOKEN_MAPS[args.family]
    expected = shared_template(tokens)
    templates = embedded_templates(args.checkpoint)
    mismatched = [name for name, template in templates.items() if template != expected]
    if mismatched:
        raise ValueError(f"{args.checkpoint}: {mismatched} do not hold the shared {args.family} template")
    repo, revision = FAMILY_TOKENIZER_REVISIONS[args.family]
    end_of_turn = AutoTokenizer.from_pretrained(repo, revision=revision).convert_tokens_to_ids(tokens.turn_close)

    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=GPUS, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        checkpoint_model(args.name, args.checkpoint),
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
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
    report = {
        "checkpoint": args.checkpoint,
        "templates_checked": sorted(templates),
        "turn_ended": sum(c["turn_ended"] for c in completions),
        "think_closed": sum(c["think_closed"] for c in completions),
        "completions": completions,
    }
    with fsspec.open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    logger.info("served %d completions; %d ended the turn", len(completions), report["turn_ended"])


if __name__ == "__main__":
    main()
