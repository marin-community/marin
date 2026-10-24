# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Stage 0 gate's protocol check on a Snowball ``B``: turn ends and thinking closure.

DESIGN, "Stage 0 gate": on 200 prompts per probe family, at least 98% of turns end with the
end-of-turn token and at least 98% of thinking blocks close. This module serves the
checkpoint with vLLM (no reasoning parser, special tokens kept), samples one completion per
prompt and scores the raw text. Generation ignores the tokenizer's EOS list and stops only on
the end-of-turn token, so a turn counts as ended exactly when vLLM reports that token as its
stop reason; a completion that emits ``<|end_of_text|>`` instead runs on and fails. Tool-call parsing is
measured on the P_agent trajectories, where the tool calls are.

Run as a small CPU job; serving dispatches its own 8-GPU job::

    python -m experiments.post_training.stage0.protocol_gate --checkpoint s3://.../hf/step-N \\
        --prompts P_chat=s3://.../P_chat.jsonl --prompts P_think=s3://.../P_think.jsonl \\
        --out s3://.../protocol.json
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import fsspec
import requests
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
from transformers import AutoTokenizer

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.post_training.chat_template import (
    FAMILY_TOKEN_MAPS,
    FAMILY_TOKENIZER_REVISIONS,
    GRUG_TOKENS,
    ChatTokens,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Delimiters:
    """One family's think markers and the id of its end-of-turn token."""

    tokens: ChatTokens
    end_of_turn_id: int


# Grug's <|eot_id|> in the Llama 3 vocabulary.
END_OF_TURN_ID = 128009
GRUG_DELIMITERS = Delimiters(GRUG_TOKENS, END_OF_TURN_ID)


def family_delimiters(family: str) -> Delimiters:
    """A family's delimiters, with the end-of-turn id read from its pinned tokenizer."""
    tokens = FAMILY_TOKEN_MAPS[family]
    repo, revision = FAMILY_TOKENIZER_REVISIONS[family]
    (end_of_turn_id,) = AutoTokenizer.from_pretrained(repo, revision=revision).encode(
        tokens.turn_close, add_special_tokens=False
    )
    return Delimiters(tokens, end_of_turn_id)


PROMPTS_PER_FAMILY = 200
MIN_TURN_END_SHARE = 0.98
MIN_THINK_CLOSE_SHARE = 0.98
MAX_COMPLETION_TOKENS = 32768
MAX_MODEL_LEN = 65536


@dataclass(frozen=True)
class Decode:
    """The sampling settings of one gate run; the chosen one becomes B's probe decode."""

    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = MAX_COMPLETION_TOKENS


CONCURRENT_REQUESTS = 64
REQUEST_TIMEOUT = 3600


@dataclass(frozen=True)
class Completion:
    family: str
    probe_id: str
    finish_reason: str
    stop_reason: int | str | None
    text: str


def completion_protocol(
    text: str, finish_reason: str, stop_reason: int | str | None, delimiters: Delimiters = GRUG_DELIMITERS
) -> dict[str, bool | int]:
    """One completion's protocol facts: did the turn end, and did every think block close."""
    think_open, think_close = delimiters.tokens.think_open, delimiters.tokens.think_close
    opens = text.count(think_open)
    closes = text.count(think_close)
    return {
        "turn_ended": finish_reason == "stop" and stop_reason == delimiters.end_of_turn_id,
        "think_blocks": opens,
        "think_closed": opens > 0 and opens == closes and text.rfind(think_close) > text.rfind(think_open),
    }


def protocol_report(completions: list[Completion], delimiters: Delimiters = GRUG_DELIMITERS) -> dict:
    """Per-family shares and the gate verdict."""
    families: dict[str, dict] = {}
    for family in sorted({completion.family for completion in completions}):
        facts = [
            completion_protocol(c.text, c.finish_reason, c.stop_reason, delimiters)
            for c in completions
            if c.family == family
        ]
        with_think = [fact for fact in facts if fact["think_blocks"]]
        turn_share = sum(bool(fact["turn_ended"]) for fact in facts) / len(facts)
        think_share = sum(bool(fact["think_closed"]) for fact in with_think) / len(with_think) if with_think else None
        families[family] = {
            "completions": len(facts),
            "turn_end_share": turn_share,
            "completions_with_thinking": len(with_think),
            "think_close_share": think_share,
            "passed": turn_share >= MIN_TURN_END_SHARE and (think_share is None or think_share >= MIN_THINK_CLOSE_SHARE),
        }
    return {
        "passed": all(entry["passed"] for entry in families.values()),
        "thresholds": {"turn_end": MIN_TURN_END_SHARE, "think_close": MIN_THINK_CLOSE_SHARE},
        "families": families,
    }


def _read_prompts(path: str, count: int, seed: int) -> list[dict]:
    with fsspec.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    return random.Random(seed).sample(rows, min(count, len(rows)))


def _complete(
    endpoint: OpenAIEndpoint, family: str, row: dict, seed: int, decode: Decode, end_of_turn_id: int
) -> Completion:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    body = {
        "model": endpoint.model,
        "messages": row["messages"],
        "temperature": decode.temperature,
        "top_p": decode.top_p,
        "repetition_penalty": decode.repetition_penalty,
        "max_tokens": decode.max_tokens,
        "seed": seed,
        # Keep the think tokens in the text: they are what the gate counts.
        "skip_special_tokens": False,
        "ignore_eos": True,
        "stop_token_ids": [end_of_turn_id],
    }
    response = requests.post(endpoint.url("chat/completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return Completion(
        family, row["probe_id"], choice["finish_reason"], choice.get("stop_reason"), choice["message"]["content"] or ""
    )


def _complete_item(
    endpoint: OpenAIEndpoint, seed: int, decode: Decode, end_of_turn_id: int, item: tuple[str, dict]
) -> Completion:
    return _complete(endpoint, item[0], item[1], seed, decode, end_of_turn_id)


def checkpoint_model(name: str, checkpoint: str) -> ModelConfig:
    """Serve an exported Stage 0 MoE checkpoint with its embedded template and no reasoning parser.

    Data parallel with expert parallelism over one 8-GPU node serves Snowball and the 30B-class
    Axolotl bases alike.
    """
    return ModelConfig(
        name=name,
        location=checkpoint,
        tokenizer=checkpoint,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g", disk="500g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=7168,
            max_num_seqs=32,
            auto_overrides=False,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="slash-free serving name")
    parser.add_argument("--prompts", action="append", required=True, help="FAMILY=path to a probe JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", choices=sorted(FAMILY_TOKEN_MAPS), default="grug")
    parser.add_argument("--prompts-per-family", type=int, default=PROMPTS_PER_FAMILY)
    parser.add_argument("--temperature", type=float, default=Decode.temperature)
    parser.add_argument("--top-p", type=float, default=Decode.top_p)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        action="append",
        help="repeatable: each value is scored in the same serving session, into --out with a -rpX suffix",
    )
    args = parser.parse_args()
    decodes = [
        Decode(temperature=args.temperature, top_p=args.top_p, repetition_penalty=penalty)
        for penalty in args.repetition_penalty or [Decode.repetition_penalty]
    ]

    delimiters = GRUG_DELIMITERS if args.family == "grug" else family_delimiters(args.family)
    work = [
        (family, row)
        for family, path in (spec.split("=", 1) for spec in args.prompts)
        for row in _read_prompts(path, args.prompts_per_family, args.seed)
    ]
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        checkpoint_model(args.name, args.checkpoint),
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        for decode in decodes:
            with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
                completions = list(
                    pool.map(
                        functools.partial(_complete_item, endpoint, args.seed, decode, delimiters.end_of_turn_id), work
                    )
                )
            out = (
                args.out
                if len(decodes) == 1
                else args.out.removesuffix(".json") + f"-rp{decode.repetition_penalty:g}.json"
            )
            _write_report(args.checkpoint, decode, completions, out, delimiters)


def _write_report(
    checkpoint: str, decode: Decode, completions: list[Completion], out: str, delimiters: Delimiters
) -> None:
    report = {"checkpoint": checkpoint, "decode": decode.__dict__}
    report |= protocol_report(completions, delimiters)
    with fsspec.open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    with fsspec.open(out.removesuffix(".json") + "-completions.jsonl", "w") as handle:
        for completion in completions:
            handle.write(json.dumps(completion.__dict__) + "\n")
    logger.info("protocol gate %s at %s: %s", "PASSED" if report["passed"] else "FAILED", decode, report["families"])


if __name__ == "__main__":
    main()
