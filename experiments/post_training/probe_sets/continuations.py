# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference continuations from an architecture's ``B`` for the single-turn probe sets.

Each probe prompt is rendered through the shared template with ``B``'s tokenizer, sent to
a vLLM server as token ids, and sampled once under the pre-registered decode. The server
returns the sampled token ids, so the stored sequence (prompt ids + sampled ids) is
exactly what ``B`` produced; the bucket tags come from ``buckets.tag_conversation`` on the
decoded reply, which keeps the think markers verbatim. A reply that ran to the token cap
has no closing turn token, and the template's closing token is dropped from its tags.

Run as a small CPU job; serving dispatches its own 8-GPU job::

    python -m experiments.post_training.probe_sets.continuations --checkpoint s3://.../hf/step-N \\
        --name snowball-b --family grug --probe-dir s3://.../probe-sets/20261023 \\
        --sets P_chat P_think P_X P_R --out s3://.../continuations/B
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

import fsspec
import requests
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
from transformers import AutoTokenizer

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template
from experiments.post_training.mismatch_probe.protocol import request_seed
from experiments.post_training.probe_sets.buckets import tag_conversation
from experiments.post_training.stage0.protocol_gate import Decode, snowball_checkpoint_model

logger = logging.getLogger(__name__)

MASTER_SEED = 20261024
CONCURRENT_REQUESTS = 64
REQUEST_TIMEOUT = 7200
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


# The pre-registered single-turn decode for Snowball's probe and expert continuations
# (2026-10-24): temperature 0.6, top_p 0.95, repetition penalty 1.05, 32,768 new tokens.
SNOWBALL_DECODE = Decode(repetition_penalty=1.05)


def render_prompt_ids(tokenizer, family: str, messages: list[dict[str, Any]]) -> list[int]:
    """The generation prompt's token ids, encoded exactly as the tagger encodes."""
    text = tokenizer.apply_chat_template(
        messages, chat_template=shared_template(FAMILY_TOKEN_MAPS[family]), tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, add_special_tokens=False)


def turn_close_id(tokenizer, family: str) -> int:
    return tokenizer.convert_tokens_to_ids(FAMILY_TOKEN_MAPS[family].turn_close)


def continuation_row(
    tokenizer, family: str, set_name: str, probe: dict[str, Any], prompt_ids: list[int], response_ids: list[int]
) -> dict[str, Any]:
    """One scored-sequence row: ``prompt_ids + response_ids`` with the tagger's bucket tags.

    The reply goes to the tagger verbatim, so the tags line up with the sampled tokens
    whenever the decoded text re-encodes to them; ``retokenized`` marks a row where it
    does not, and the row then keeps the re-encoded ids.
    """
    close = turn_close_id(tokenizer, family)
    ended = bool(response_ids) and response_ids[-1] == close
    reply_ids = response_ids[:-1] if ended else response_ids
    reply = tokenizer.decode(reply_ids, skip_special_tokens=False)
    messages = [*probe["messages"], {"role": "assistant", "content": reply}]
    tagged = tag_conversation(tokenizer, family, messages)
    keep = len(tagged.input_ids) if ended else len(tagged.input_ids) - 1  # no closing token B never sampled
    sampled = prompt_ids + response_ids
    tagged_ids = tagged.input_ids[:keep]
    return {
        "sample_id": probe["probe_id"],
        "bucket_group": set_name,
        "source": probe["source"],
        "token_ids": sampled if tagged_ids == sampled else tagged_ids,
        "token_type_tags": [bucket.value for bucket in tagged.buckets[:keep]],
        "position_tags": tagged.assistant_turn[:keep],
        "reasoning_decile": tagged.reasoning_decile[:keep],
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(response_ids),
        "turn_ended": ended,
        "retokenized": tagged_ids != sampled,
    }


def request_body(model: str, prompt_ids: list[int], seed: int, decode: Decode, stop_id: int) -> dict[str, Any]:
    """A completions request on token ids that returns the sampled ids and stops only on the turn close."""
    return {
        "model": model,
        "prompt": prompt_ids,
        "temperature": decode.temperature,
        "top_p": decode.top_p,
        "repetition_penalty": decode.repetition_penalty,
        "max_tokens": decode.max_tokens,
        "seed": seed,
        "skip_special_tokens": False,
        "ignore_eos": True,
        "stop_token_ids": [stop_id],
        "return_token_ids": True,
    }


def _sample(endpoint: OpenAIEndpoint, body: dict[str, Any]) -> list[int]:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    response = requests.post(endpoint.url("completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("token_ids") is None:
        raise ValueError("the server did not return the sampled token ids (return_token_ids)")
    return [int(token) for token in choice["token_ids"]]


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with fsspec.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> str:
    """Write ``rows`` and return the file's sha256."""
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    with fsspec.open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def local_tokenizer(checkpoint: str):
    """``B``'s tokenizer, copied out of the (object-store) checkpoint."""
    directory = tempfile.mkdtemp()
    filesystem, root = fsspec.core.url_to_fs(checkpoint)
    for name in TOKENIZER_FILES:
        filesystem.get(f"{root.rstrip('/')}/{name}", f"{directory}/{name}")
    return AutoTokenizer.from_pretrained(directory)


def set_summary(rows: list[dict[str, Any]], sha256: str, seconds: float) -> dict[str, Any]:
    tokens = sorted(row["completion_tokens"] for row in rows)
    return {
        "rows": len(rows),
        "sha256": sha256,
        "turn_ended": sum(row["turn_ended"] for row in rows),
        "retokenized": sum(row["retokenized"] for row in rows),
        "completion_tokens_p50": tokens[len(tokens) // 2],
        "completion_tokens_p90": tokens[int(len(tokens) * 0.9)],
        "completion_tokens_max": tokens[-1],
        "seconds": round(seconds),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="slash-free serving name")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_TOKEN_MAPS))
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.family != "grug":
        raise ValueError("only Snowball serving is wired; other families need their serve config")

    tokenizer = local_tokenizer(args.checkpoint)
    stop_id = turn_close_id(tokenizer, args.family)
    probes = {name: read_jsonl(f"{args.probe_dir.rstrip('/')}/{name}.jsonl") for name in args.sets}
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        snowball_checkpoint_model(args.name, args.checkpoint),
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    manifest: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "family": args.family,
        "decode": asdict(SNOWBALL_DECODE),
        "master_seed": MASTER_SEED,
        "probe_dir": args.probe_dir,
        "sets": {},
    }
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        for name, rows in probes.items():
            prompt_ids = [render_prompt_ids(tokenizer, args.family, row["messages"]) for row in rows]
            bodies = [
                request_body(
                    endpoint.model, ids, request_seed(MASTER_SEED, row["probe_id"], 0), SNOWBALL_DECODE, stop_id
                )
                for row, ids in zip(rows, prompt_ids, strict=True)
            ]
            started = time.monotonic()
            with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
                responses = list(pool.map(functools.partial(_sample, endpoint), bodies))
            seconds = time.monotonic() - started
            out_rows = [
                continuation_row(tokenizer, args.family, name, row, ids, response)
                for row, ids, response in zip(rows, prompt_ids, responses, strict=True)
            ]
            digest = write_jsonl(f"{args.out.rstrip('/')}/{name}.jsonl", out_rows)
            manifest["sets"][name] = set_summary(out_rows, digest, seconds)
            logger.info("%s: %s", name, manifest["sets"][name])
            with fsspec.open(f"{args.out.rstrip('/')}/manifest.json", "w") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
