# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Next-token reference log-probs from a vLLM serve of the same checkpoint.

Runs inside one Iris H100x8 task. vLLM is launched the way the Snowball parity job launches it
(Marin fork, TP=1 DP=8 with expert parallelism, prefix caching off). Each prompt's exact token IDs go
to the completions endpoint with ``max_tokens=1`` and the top 50 log-probs. All requests are pinned to
data-parallel rank 0, so every prompt sees the same rank.

Output JSON: ``{prompt_id: {"tokens": [...], "greedy": id, "top_logprobs": {id: logprob}}}``.
"""

import argparse
import json
import logging

import requests
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_vllm import iris_vllm_followers, iris_vllm_launch
from marin.inference.vllm_backend import VllmBackend
from rigging.filesystem.buckets import filesystem_for

from experiments.interp.snowball_identity.prompts import SNOWBALL_V01, hf_repo_revision, load_tokenizer, read_prompts

logger = logging.getLogger(__name__)

GPU_COUNT = 8
TOP_LOGPROBS = 50
MAX_MODEL_LEN = 8192


def next_token(completions_url: str, model: str, tokens: tuple[int, ...]) -> dict:
    response = requests.post(
        completions_url,
        headers={"X-data-parallel-rank": "0"},
        json={
            "model": model,
            "prompt": list(tokens),
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": TOP_LOGPROBS,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        },
        timeout=(30, 300),
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]
    if choice["prompt_token_ids"] != list(tokens):
        raise ValueError("vLLM changed the prompt token IDs")
    (greedy,) = choice["token_ids"]
    (top,) = choice["logprobs"]["top_logprobs"]
    return {
        "tokens": list(tokens),
        "greedy": int(greedy),
        "top_logprobs": {int(token.removeprefix("token_id:")): float(value) for token, value in top.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=SNOWBALL_V01)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True, help="JSON file path (s3:// or local)")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts, load_tokenizer(args.weights))
    repo, revision = hf_repo_revision(args.weights) or (args.weights, None)
    spec = ModelSpec(
        weights=repo,
        revision=revision,
        api_model="snowball-v0.1",
        num_chips=GPU_COUNT,
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_seqs=1,
            extra_args=(
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--max-logprobs",
                str(TOP_LOGPROBS),
                "--attention-backend",
                args.attention_backend,
                "--no-enable-prefix-caching",
            ),
        ),
    )
    results = {}
    with iris_vllm_launch(pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=GPU_COUNT) as launch:
        with backend.start(spec, extra_args=launch.extra_cli_args, subprocess_env=launch.subprocess_env) as server:
            server.wait_until_ready()
            with iris_vllm_followers(launch):
                for prompt in prompts:
                    results[prompt.id] = next_token(f"{server.server_url}/completions", spec.api_model, prompt.tokens)
                    logger.info(
                        "vLLM %s: %d tokens, greedy %d", prompt.id, len(prompt.tokens), results[prompt.id]["greedy"]
                    )
    fs, path = filesystem_for(args.output)
    fs.pipe_file(path, json.dumps(results).encode())
    logger.info("Wrote %d prompts to %s", len(results), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
