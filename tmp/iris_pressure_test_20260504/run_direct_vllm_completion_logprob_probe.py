#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import traceback

import requests
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

VLLM_ENV_DEFAULTS = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    "WANDB_MODE": "offline",
}


def main() -> None:
    args = parse_args()
    for key, value in VLLM_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    model = ModelConfig(
        name=args.model_path,
        path=None,
        engine_kwargs={
            "tokenizer": args.tokenizer,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
    )
    env = VllmEnvironment(model=model, host="127.0.0.1", port=args.port, timeout_seconds=args.timeout_seconds)

    try:
        with env:
            if env.model_id is None:
                raise RuntimeError("vLLM server did not expose a model id")
            payload = {
                "model": env.model_id,
                "prompt": args.prompt,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": 1234,
                "echo": True,
            }
            print("direct_probe_request_start")
            print(json.dumps(payload, indent=2, sort_keys=True))
            print("direct_probe_request_end")
            response = requests.post(f"{env.server_url}/completions", json=payload, timeout=args.request_timeout_seconds)
            print(f"direct_probe_http_status={response.status_code}")
            print("direct_probe_response_start")
            print(response.text)
            print("direct_probe_response_end")
            response.raise_for_status()
            parsed = response.json()
            choice = parsed["choices"][0]
            logprobs = choice["logprobs"]
            print("direct_probe_summary_start")
            print(
                json.dumps(
                    {
                        "model_id": env.model_id,
                        "server_url": env.server_url,
                        "choice_keys": sorted(choice),
                        "logprobs_keys": sorted(logprobs),
                        "tokens_len": len(logprobs["tokens"]),
                        "token_logprobs_len": len(logprobs["token_logprobs"]),
                        "top_logprobs_len": len(logprobs["top_logprobs"]),
                        "text_offset_len": len(logprobs["text_offset"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print("direct_probe_summary_end")
    except Exception as exc:
        print("direct_probe_exception_start")
        print(repr(exc))
        traceback.print_exc()
        print("direct_probe_exception_end")
        print("direct_probe_env_snapshot_start")
        print(json.dumps(env.debug_snapshot(), indent=2, sort_keys=True))
        print("direct_probe_env_snapshot_end")
        if env.vllm_server is not None:
            print("direct_probe_diagnostics_start")
            for label, value in env.diagnostics(max_lines=400).items():
                print(f"--- {label} ---")
                print(value)
            print("direct_probe_diagnostics_end")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="A B")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main()
