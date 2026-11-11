# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample fixed probes under a checkpoint's native chat template."""

import argparse
import json
import logging
import time
from dataclasses import asdict, replace

import fsspec
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference

from experiments.post_training.probe_sets.continuations import (
    MASTER_SEED,
    SNOWBALL_DECODE,
    _sampled_responses,
    local_tokenizer,
    read_jsonl,
    set_summary,
    write_jsonl,
)
from experiments.post_training.stage0.protocol_gate import END_OF_TURN_ID, MAX_MODEL_LEN, checkpoint_model

logger = logging.getLogger(__name__)


def native_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(rendered, add_special_tokens=False)


def model_stop_ids(checkpoint: str) -> tuple[int, ...]:
    with fsspec.open(f"{checkpoint.rstrip('/')}/generation_config.json") as handle:
        eos_ids = json.load(handle)["eos_token_id"]
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if not isinstance(eos_ids, list) or not eos_ids or any(not isinstance(token_id, int) for token_id in eos_ids):
        raise ValueError(f"{checkpoint}: invalid generation-config eos_token_id")
    if len(set(eos_ids)) != len(eos_ids):
        raise ValueError(f"{checkpoint}: duplicate generation-config eos_token_id")
    return tuple(eos_ids)


def scored_row(
    probe: dict, set_name: str, prompt_ids: list[int], response_ids: list[int], stop_ids: tuple[int, ...]
) -> dict:
    token_ids = prompt_ids + response_ids
    return {
        "sample_id": probe["probe_id"],
        "bucket_group": set_name,
        "source": probe["source"],
        "token_ids": token_ids,
        "token_type_tags": ["context"] * len(prompt_ids) + ["reasoning"] * len(response_ids),
        "position_tags": [0] * len(token_ids),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(response_ids),
        "turn_ended": bool(response_ids) and response_ids[-1] in stop_ids,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-tokens", type=int, default=SNOWBALL_DECODE.max_tokens)
    parser.add_argument("--max-tokens-for-set", action="append", default=[], metavar="SET=TOKENS")
    args = parser.parse_args()

    tokenizer = local_tokenizer(args.checkpoint)
    if tokenizer.convert_tokens_to_ids("<|eot_id|>") != END_OF_TURN_ID:
        raise ValueError("checkpoint uses a different end-of-turn token")
    stop_ids = model_stop_ids(args.checkpoint)
    caps = dict(spec.split("=", 1) for spec in args.max_tokens_for_set)
    if set(caps) - set(args.sets):
        raise ValueError("response caps must name a requested probe set")
    decodes = {name: replace(SNOWBALL_DECODE, max_tokens=int(caps.get(name, args.max_tokens))) for name in args.sets}
    probes = {name: read_jsonl(f"{args.probe_dir.rstrip('/')}/{name}.jsonl") for name in args.sets}
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        checkpoint_model(args.name, args.checkpoint, max_model_len=args.max_model_len),
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    manifest = {
        "checkpoint": args.checkpoint,
        "family": "grug",
        "template": "native",
        "decode_by_set": {name: asdict(decode) for name, decode in decodes.items()},
        "master_seed": MASTER_SEED,
        "context_window": args.max_model_len,
        "stop_token_ids": list(stop_ids),
        "probe_dir": args.probe_dir,
        "sets": {},
    }
    manifest_path = f"{args.out.rstrip('/')}/manifest.json"
    filesystem, path = fsspec.core.url_to_fs(manifest_path)
    if filesystem.exists(path):
        with filesystem.open(path) as handle:
            previous = json.load(handle)
        for key in (
            "checkpoint",
            "family",
            "template",
            "decode_by_set",
            "master_seed",
            "context_window",
            "stop_token_ids",
            "probe_dir",
        ):
            if previous[key] != manifest[key]:
                raise ValueError(f"{manifest_path} records different generation settings")
        manifest["sets"] = previous["sets"]
    with remote_inference(inference) as session:
        for name, rows in probes.items():
            prompts = [native_prompt_ids(tokenizer, row["messages"]) for row in rows]
            started = time.monotonic()
            responses = _sampled_responses(
                session.model.endpoint,
                rows,
                prompts,
                stop_ids,
                f"{args.out.rstrip('/')}/{name}.raw.jsonl",
                decode=decodes[name],
                context_window=args.max_model_len,
            )
            scored = [
                scored_row(row, name, prompt, response, stop_ids)
                for row, prompt, response in zip(rows, prompts, responses, strict=True)
            ]
            digest = write_jsonl(f"{args.out.rstrip('/')}/{name}.jsonl", scored)
            manifest["sets"][name] = set_summary(scored, digest, time.monotonic() - started)
            with fsspec.open(manifest_path, "w") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
            logger.info("%s: %s", name, manifest["sets"][name])


if __name__ == "__main__":
    main()
