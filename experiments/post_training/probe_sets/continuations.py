# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference continuations from an architecture's ``B`` for the single-turn probe sets.

Each probe prompt is rendered through the shared template with ``B``'s tokenizer, sent to
a vLLM server as token ids, and sampled once under the pre-registered decode. The server
returns the sampled token ids, so the stored sequence (prompt ids + sampled ids) is
exactly what ``B`` produced; the bucket tags come from ``buckets.tag_token_ids``, which
aligns the sampled tokens by bytes to the conversation rendered with the reply's think
markers verbatim. A reply that ran to the token cap has no closing turn token.

Run as a small CPU job; serving dispatches its own 8-GPU job::

    python -m experiments.post_training.probe_sets.continuations --checkpoint s3://.../hf/step-N \\
        --name snowball-b --family grug --probe-dir s3://.../probe-sets/20261023 \\
        --sets P_chat P_think P_X P_R --out s3://.../continuations/B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from experiments.post_training.probe_sets.buckets import decode_token_bytes, tag_token_ids
from experiments.post_training.stage0.protocol_gate import Decode, checkpoint_model

logger = logging.getLogger(__name__)

MASTER_SEED = 20261024
CONCURRENT_REQUESTS = 64
REQUEST_TIMEOUT = 7200
# A shared vLLM endpoint occasionally answers a completion with a transient
# status or drops the connection; a hours-long sampling pass retries instead of
# dying. Seeded bodies reproduce the same sample on retry.
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
ENDPOINT_ATTEMPTS = 6
BACKOFF_SECONDS = 4
# The raw file is rewritten every this many new completions: a killed pass (job
# timeout, endpoint loss) loses at most that many rows, and a restart samples
# only the probes still missing.
RAW_WRITE_EVERY = 100
# Every render passes the shared template explicitly, so the export's own template is not needed.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


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

    The sampled ids are tagged by byte alignment against the rendered conversation, so
    the row keeps exactly B's tokens, stray bytes included.
    """
    close = turn_close_id(tokenizer, family)
    ended = bool(response_ids) and response_ids[-1] == close
    reply = decode_token_bytes(tokenizer, response_ids[:-1] if ended else response_ids)
    messages = [*probe["messages"], {"role": "assistant", "content": reply}]
    tagged = tag_token_ids(tokenizer, family, messages, prompt_ids + response_ids)
    return {
        "sample_id": probe["probe_id"],
        "bucket_group": set_name,
        "source": probe["source"],
        "token_ids": tagged.input_ids,
        "token_type_tags": [bucket.value for bucket in tagged.buckets],
        "position_tags": tagged.assistant_turn,
        "reasoning_decile": tagged.reasoning_decile,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(response_ids),
        "turn_ended": ended,
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


def sample_token_ids(endpoint: OpenAIEndpoint, body: dict[str, Any]) -> list[int]:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    for attempt in range(ENDPOINT_ATTEMPTS):
        try:
            response = requests.post(endpoint.url("completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            transient = status is None or status in TRANSIENT_STATUS_CODES
            if attempt == ENDPOINT_ATTEMPTS - 1 or not transient:
                raise
            delay = BACKOFF_SECONDS * 2**attempt
            logger.warning("transient endpoint failure (%s); attempt %d, retrying in %ds", error, attempt + 1, delay)
            time.sleep(delay)
    choice = response.json()["choices"][0]
    if choice.get("token_ids") is None:
        raise ValueError("the server did not return the sampled token ids (return_token_ids)")
    return [int(token) for token in choice["token_ids"]]


def _sampled_responses(
    endpoint: OpenAIEndpoint, rows: list[dict[str, Any]], prompt_ids: list[list[int]], stop_id: int, raw_path: str
) -> list[list[int]]:
    """Each probe's sampled ids, checkpointed to ``raw_path`` as they land.

    A raw file already there is reused: the sampled ids are the expensive part,
    and a restart after a killed pass samples only the probes still missing (the
    seeds reproduce the rest). Rewrites happen every ``RAW_WRITE_EVERY`` new
    completions and once at the end.
    """
    filesystem, path = fsspec.core.url_to_fs(raw_path)
    sampled: dict[str, dict[str, Any]] = {}
    if filesystem.exists(path):
        stored = {row["probe_id"]: row for row in read_jsonl(raw_path)}
        logger.info("reusing %d sampled replies from %s", len(stored), path)
        sampled.update(stored)
    todo = []
    for row, ids in zip(rows, prompt_ids, strict=True):
        if row["probe_id"] in sampled:
            if sampled[row["probe_id"]]["prompt_ids"] != ids:
                raise ValueError(f"{raw_path} holds a different prompt for {row['probe_id']}")
            continue
        todo.append((row, ids))
    if todo:
        bodies = [
            request_body(endpoint.model, ids, request_seed(MASTER_SEED, row["probe_id"], 0), SNOWBALL_DECODE, stop_id)
            for row, ids in todo
        ]
        with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
            futures = {
                pool.submit(sample_token_ids, endpoint, body): (row, ids)
                for (row, ids), body in zip(todo, bodies, strict=True)
            }
            pending_write = 0
            for future in as_completed(futures):
                row, ids = futures[future]
                sampled[row["probe_id"]] = {
                    "probe_id": row["probe_id"],
                    "prompt_ids": ids,
                    "response_ids": future.result(),
                }
                pending_write += 1
                if pending_write >= RAW_WRITE_EVERY:
                    pending_write = 0
                    _write_raw(path, rows, sampled)
        _write_raw(path, rows, sampled)
    return [sampled[row["probe_id"]]["response_ids"] for row in rows]


def _write_raw(path: str, rows: list[dict[str, Any]], sampled: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, [sampled[row["probe_id"]] for row in rows if row["probe_id"] in sampled])


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
    """A checkpoint's tokenizer, copied out of the (object-store) export."""
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
        checkpoint_model(args.name, args.checkpoint),
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
            raw_path = f"{args.out.rstrip('/')}/{name}.raw.jsonl"
            started = time.monotonic()
            responses = _sampled_responses(endpoint, rows, prompt_ids, stop_id, raw_path)
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
