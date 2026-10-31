# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample a served expert's completions on task prompts and write the graded M1 rows.

Phase 1a's M1 arm trains on 4 expert completions per prompt (``phase1a.data``). This
driver serves the expert's HF export with vLLM, samples ``--samples`` completions per
prompt from the prompt's token ids under the pre-registered Snowball decode (temperature
0.6, top_p 0.95, repetition penalty 1.05; decision log 2026-10-24) with the task's
response cap as ``max_tokens``, and stops only on the turn close. A completion that did
not end its turn within the cap is dropped, never truncated. Each kept completion is
graded on its answer text (:mod:`terminus_answers`) by the unchanged X or R grader, and
the graded-correct ones become rows in WP1's tokenized format: the prompt ids plus the
sampled ids, with the assistant mask over the sampled ids, so the rows hold exactly the
expert's tokens.

Run as a small CPU job; serving dispatches its own 8-GPU job::

    python -m experiments.post_training.task_data.sample_m1 --checkpoint s3://.../hf \\
        --name e-x --task X --tasks s3://.../x_train.jsonl --skip 6000 --count 3000 \\
        --out s3://.../m1/x_m1.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

import fsspec
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS
from experiments.post_training.mismatch_probe.protocol import request_seed
from experiments.post_training.probe_sets.buckets import decode_token_bytes
from experiments.post_training.probe_sets.continuations import (
    SNOWBALL_DECODE,
    local_tokenizer,
    read_jsonl,
    render_prompt_ids,
    request_body,
    sample_token_ids,
    write_jsonl,
)
from experiments.post_training.stage0.protocol_gate import checkpoint_model
from experiments.post_training.task_data.build import DATA_SOURCES, TEACHER_ROUTES, TOKEN_CAPS
from experiments.post_training.task_data.expert_outputs import GRADERS, bounded_grade
from experiments.post_training.task_data.terminus_answers import answer_text

logger = logging.getLogger(__name__)

MASTER_SEED = 20261024
CONCURRENT_REQUESTS = 128


def m1_row(
    task: dict[str, Any], prompt: list[dict[str, str]], prompt_ids: list[int], response_ids: list[int], task_name: str
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "prompt": prompt,
        "data_source": DATA_SOURCES[task_name],
        "teacher_route": TEACHER_ROUTES[task_name],
        "input_ids": prompt_ids + response_ids,
        "assistant_masks": [0] * len(prompt_ids) + [1] * len(response_ids),
    }


def verdict(tokenizer, family: str, task_name: str, task: dict[str, Any], response_ids: list[int]) -> tuple[str, bool]:
    """(``ended`` / ``over_cap``, graded correct) for one sampled completion."""
    close = tokenizer.convert_tokens_to_ids(FAMILY_TOKEN_MAPS[family].turn_close)
    if not response_ids or response_ids[-1] != close:
        return "over_cap", False
    reply = decode_token_bytes(tokenizer, response_ids[:-1])
    answer = answer_text(reply, FAMILY_TOKEN_MAPS[family].think_close)
    return "ended", bool(bounded_grade(GRADERS[task_name], task, answer.text))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="the expert's HF export")
    parser.add_argument("--name", required=True, help="slash-free serving name")
    parser.add_argument("--family", default="grug", choices=["grug"])
    parser.add_argument("--task", required=True, choices=sorted(DATA_SOURCES))
    parser.add_argument("--tasks", required=True, help="WP7 task split (task_id, prompt, answer, ...)")
    parser.add_argument("--skip", type=int, required=True, help="leading split rows to skip (the expert's own)")
    parser.add_argument("--count", type=int, required=True, help="prompts to sample")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--out", required=True, help="M1 rows jsonl; raw samples and a manifest go beside it")
    args = parser.parse_args()

    tokenizer = local_tokenizer(args.checkpoint)
    close = tokenizer.convert_tokens_to_ids(FAMILY_TOKEN_MAPS[args.family].turn_close)
    tasks = read_jsonl(args.tasks)[args.skip : args.skip + args.count]
    if len(tasks) != args.count:
        raise ValueError(f"{args.tasks} has only {len(tasks)} rows after skipping {args.skip}")
    prompts = [[{"role": "user", "content": task["prompt"]}] for task in tasks]
    prompt_ids = [render_prompt_ids(tokenizer, args.family, prompt) for prompt in prompts]
    decode = dataclasses.replace(SNOWBALL_DECODE, max_tokens=TOKEN_CAPS[args.task])
    work = [(index, sample) for index in range(len(tasks)) for sample in range(args.samples)]

    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        checkpoint_model(args.name, args.checkpoint),
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        bodies = [
            request_body(
                endpoint.model,
                prompt_ids[index],
                request_seed(MASTER_SEED, tasks[index]["task_id"], sample),
                decode,
                close,
            )
            for index, sample in work
        ]
        started = time.monotonic()
        with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
            responses = list(pool.map(functools.partial(sample_token_ids, endpoint), bodies))
        seconds = time.monotonic() - started

    stem = args.out.removesuffix(".jsonl")
    raw_sha = write_jsonl(
        f"{stem}.raw.jsonl",
        [
            {"task_id": tasks[index]["task_id"], "sample": sample, "response_ids": response}
            for (index, sample), response in zip(work, responses, strict=True)
        ],
    )
    rows, outcomes = [], {"ended": 0, "over_cap": 0, "graded_correct": 0}
    for (index, _sample), response in zip(work, responses, strict=True):
        status, correct = verdict(tokenizer, args.family, args.task, tasks[index], response)
        outcomes[status] += 1
        if correct:
            outcomes["graded_correct"] += 1
            rows.append(m1_row(tasks[index], prompts[index], prompt_ids[index], response, args.task))
    manifest = {
        "checkpoint": args.checkpoint,
        "tasks": args.tasks,
        "skip": args.skip,
        "prompts": args.count,
        "samples_per_prompt": args.samples,
        "decode": asdict(decode),
        "master_seed": MASTER_SEED,
        "completions": len(work),
        **outcomes,
        "m1_rows": len(rows),
        "m1_sha256": write_jsonl(args.out, rows),
        "raw_sha256": raw_sha,
        "seconds": round(seconds),
    }
    with fsspec.open(f"{stem}.manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("M1 sampling: %s", manifest)


if __name__ == "__main__":
    main()
