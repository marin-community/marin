# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-position reference log-probs from a vLLM serve of the same checkpoint.

Runs inside one Iris H100x8 task. vLLM is launched the way the Snowball parity job launches it
(Marin fork, TP=1 DP=8 with expert parallelism, prefix caching off). Each sequence from
``sequences.build_sequences`` goes to the completions endpoint as exact token IDs with ``max_tokens=1``
and ``prompt_logprobs=TOP_LOGPROBS``. All requests are pinned to data-parallel rank 0, so every sequence
sees the same rank.

Output JSON: ``{sequence_id: {"tokens": [...], "actual": [logprob of token i | None], "top": [{id: logprob} | None]}}``.
Entry ``i`` is the distribution over token ``i`` given tokens ``0..i-1``.
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

from experiments.interp.snowball_identity.prompts import SNOWBALL_V01, hf_repo_revision, load_tokenizer
from experiments.interp.snowball_identity.sequences import build_sequences

logger = logging.getLogger(__name__)

GPU_COUNT = 8
TOP_LOGPROBS = 20
MAX_MODEL_LEN = 8192


def _token_id(key: str) -> int:
    return int(key.removeprefix("token_id:"))


def _logprob(value) -> float:
    return float(value["logprob"] if isinstance(value, dict) else value)


def prompt_logprobs(completions_url: str, model: str, tokens: tuple[int, ...]) -> dict:
    response = requests.post(
        completions_url,
        headers={"X-data-parallel-rank": "0"},
        json={
            "model": model,
            "prompt": list(tokens),
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": TOP_LOGPROBS,
        },
        timeout=(30, 300),
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]
    per_position = choice["prompt_logprobs"]
    if len(per_position) != len(tokens):
        raise ValueError(f"vLLM returned {len(per_position)} prompt positions for {len(tokens)} tokens")
    actual, top = [], []
    for token, entry in zip(tokens, per_position, strict=True):
        if entry is None:
            actual.append(None)
            top.append(None)
            continue
        values = {_token_id(key): _logprob(value) for key, value in entry.items()}
        actual.append(values[token])
        top.append(dict(sorted(values.items(), key=lambda item: -item[1])[:TOP_LOGPROBS]))
    return {"tokens": list(tokens), "actual": actual, "top": top}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=SNOWBALL_V01)
    parser.add_argument("--identity", required=True, help="identity.jsonl with the contrastive rows (s3:// or local)")
    parser.add_argument("--output", required=True, help="JSON file path (s3:// or local)")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    args = parser.parse_args()

    sequences = build_sequences(load_tokenizer(args.weights), args.identity)
    hf = hf_repo_revision(args.weights)
    repo, revision = hf or (args.weights, None)
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
                # The distributed loader reads object storage only; vLLM rejects it for a Hugging Face repo.
                *(() if hf else ("--model-loader-extra-config", '{"distributed":true}')),
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
                for number, sequence in enumerate(sequences):
                    results[sequence.id] = prompt_logprobs(
                        f"{server.server_url}/completions", spec.api_model, sequence.tokens
                    )
                    if number % 50 == 0:
                        logger.info("vLLM %d/%d: %s", number, len(sequences), sequence.id)
    fs, path = filesystem_for(args.output)
    fs.pipe_file(path, json.dumps(results).encode())
    logger.info("Wrote %d sequences to %s", len(results), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
