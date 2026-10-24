# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""C0: generate the dev sequences the WP4 scoring passes consume.

32 AIME/HMMT prompts from the reserved-benchmark file and 32 X-task prompts each
get one continuation sampled from B with thinking on, rendered once through the
shared chat template. The output rows carry the full sequence's token ids (the
scorer's input; ``probe_hash`` derives from them) and the token-level bucket tags
from ``buckets.tag_conversation``. B's reply goes to the tagger verbatim as
``content``: the tagger reads its think markers, paired or not, so the rendered
tokens stay the ones B sampled.

    python -m experiments.post_training.mismatch_probe.generate_dev_sequences \
        --model-path s3://marin-rno2a/.../models/b-step92 \
        --reserved-file .../reserved_benchmarks.jsonl --x-tasks .../x_train.jsonl \
        --out s3://marin-rno2a/.../kl-dev/sequences.jsonl

Run inside a SkyRL GPU job: the engines are the vendored vLLM fork's Grug support
at tensor-parallel 1 with expert parallelism across the node's eight ranks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import fsspec
from transformers import AutoTokenizer

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template
from experiments.post_training.mismatch_probe.protocol import request_seed
from experiments.post_training.probe_sets.buckets import tag_conversation

FAMILY = "grug"
MATH_MAX_TOKENS = 8_192
X_MAX_TOKENS = 4_096
TEMPERATURE = 1.0
TOP_P = 1.0


def render_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """The generation prompt's token ids, encoded exactly as the tagger encodes."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        chat_template=shared_template(FAMILY_TOKEN_MAPS[FAMILY]),
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def select_prompts(reserved_rows: list[dict], x_rows: list[dict], *, count_per_source: int, seed: int) -> list[dict]:
    """``count_per_source`` AIME/HMMT prompts and X prompts, in a seeded order."""
    generator = random.Random(seed)
    math_rows = [row for row in reserved_rows if any(tag in str(row.get("source", "")) for tag in ("aime", "hmmt"))]
    if len(math_rows) < count_per_source or len(x_rows) < count_per_source:
        raise ValueError(f"need {count_per_source} prompts per source; have {len(math_rows)} math, {len(x_rows)} X")
    chosen = generator.sample(math_rows, count_per_source) + generator.sample(x_rows, count_per_source)
    return [
        {
            "sample_id": f"dev-math-{index}",
            "bucket_group": "P_think",
            "prompt": row["problem"],  # the reserved file keeps WP7's problem field
            "max_tokens": MATH_MAX_TOKENS,
        }
        for index, row in enumerate(chosen[:count_per_source])
    ] + [
        {"sample_id": f"dev-x-{index}", "bucket_group": "P_X", "prompt": row["prompt"], "max_tokens": X_MAX_TOKENS}
        for index, row in enumerate(chosen[count_per_source:])
    ]


async def generate(engines_client, tokenizer, requests: list[dict], *, master_seed: int) -> list[str]:
    """One sampled continuation per request, each under its own seed and token cap.

    The engine client shares one set of sampling parameters across a call, so each
    request is its own call; the calls run concurrently.
    """
    from skyrl_train.inference_engines.base import InferenceEngineInput  # noqa: PLC0415  # skyrl_train only in a runtime

    async def one(request: dict) -> list[int]:
        output = await engines_client.generate(
            InferenceEngineInput(
                prompt_token_ids=[render_prompt_ids(tokenizer, request["prompt"])],
                sampling_params={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": request["max_tokens"],
                    "seed": request_seed(master_seed, request["sample_id"], 0),
                },
            )
        )
        (response_ids,) = output["response_ids"]
        return response_ids

    started = time.monotonic()
    response_ids = await asyncio.gather(*(one(request) for request in requests))
    print(f"generation took {time.monotonic() - started:.0f} s")
    return [response_text(tokenizer, ids) for ids in response_ids]


def response_text(tokenizer, ids: list[int]) -> str:
    """The continuation's text with the think markers kept and the closing turn token dropped.

    The engines' own text skips special tokens, which would erase the think markers
    the tagger reads; the shared template re-adds the turn close.
    """
    stop_ids = {tokenizer.convert_tokens_to_ids(FAMILY_TOKEN_MAPS[FAMILY].turn_close), tokenizer.eos_token_id}
    while ids and ids[-1] in stop_ids:
        ids = ids[:-1]
    return tokenizer.decode(ids, skip_special_tokens=False)


def sequence_rows(requests: list[dict], responses: list[str], tokenizer) -> list[dict]:
    """Scored-sequence rows: token ids with the tagger's bucket tags."""
    rows = []
    for request, response in zip(requests, responses, strict=True):
        messages = [{"role": "user", "content": request["prompt"]}, {"role": "assistant", "content": response}]
        tagged = tag_conversation(tokenizer, FAMILY, messages)
        rows.append(
            {
                "sample_id": request["sample_id"],
                "bucket_group": request["bucket_group"],
                "token_ids": tagged.input_ids,
                "token_type_tags": [bucket.value for bucket in tagged.buckets],
                "position_tags": tagged.assistant_turn,
                "reasoning_decile": tagged.reasoning_decile,
            }
        )
    return rows


def read_jsonl(path: str) -> list[dict]:
    filesystem, path_in_fs = fsspec.core.url_to_fs(path)
    with filesystem.open(path_in_fs) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    filesystem, path_in_fs = fsspec.core.url_to_fs(path)
    with filesystem.open(path_in_fs, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reserved-file", required=True)
    parser.add_argument("--x-tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--num-per-source", type=int, default=32)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    # skyrl_train exists only in a SkyRL runtime environment.
    import ray  # noqa: PLC0415
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from skyrl_train.entrypoints.main_base import config_dir  # noqa: PLC0415
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient  # noqa: PLC0415
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import (  # noqa: PLC0415
        create_ray_wrapped_inference_engines,
    )
    from skyrl_train.utils import initialize_ray  # noqa: PLC0415

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    cfg.generator.sampling_params = OmegaConf.create({"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": -1})

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    initialize_ray(cfg)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=1,
        model_dtype="bfloat16",
        pretrain=args.model_path,
        seed=args.seed,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        engine_init_timeout_seconds=1800,
        expert_parallel_size=8,
        data_parallel_size=8,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        tokenizer=tokenizer,
        backend="vllm",
        vllm_attention_backend="FLASH_ATTN",
    )
    client = InferenceEngineClient(engines, tokenizer, cfg)

    requests = select_prompts(
        read_jsonl(args.reserved_file), read_jsonl(args.x_tasks), count_per_source=args.num_per_source, seed=args.seed
    )
    responses = asyncio.run(generate(client, tokenizer, requests, master_seed=args.seed))
    rows = sequence_rows(requests, responses, tokenizer)
    write_jsonl(args.out, rows)
    lengths = [len(row["token_ids"]) for row in rows]
    report = {
        "model_path": args.model_path,
        "out": args.out,
        "rows": len(rows),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
    }
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))
    ray.shutdown()


if __name__ == "__main__":
    main()
