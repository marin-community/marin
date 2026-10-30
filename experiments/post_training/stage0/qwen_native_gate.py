# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protocol check for pinned Qwen bases or native-template exports, with both stop tokens.

Run one family per CPU coordinator. Each dispatches one 8-H100 serving worker.
Completed replies are saved individually so a retry can reuse them.
"""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from enum import StrEnum

import requests
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
from rigging.filesystem.buckets import filesystem_for
from transformers import AutoTokenizer
from verifyit.modes.extract import extract_boxed

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS, QWEN_TOKENS
from experiments.post_training.curriculum_sft.generation import answers_match
from experiments.post_training.stage0.axolotl_stage0 import QWEN_NATIVE_TEMPLATES
from experiments.post_training.stage0.protocol_gate import (
    CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    Decode,
    Delimiters,
    checkpoint_model,
    completion_protocol,
    ends_in_repetition_loop,
)
from experiments.post_training.stage0.serve_check import checked_templates

logger = logging.getLogger(__name__)
FAMILIES = ("qwen3_moe", "qwen3_5_moe")
THINK_PREFILL = "<think>\n"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
QWEN3_CONTEXT = 131072
QWEN35_CONTEXT = 262144
QWEN3_YARN = json.dumps(
    {
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        },
        "max_position_embeddings": QWEN3_CONTEXT,
    }
)


class Thinking(StrEnum):
    ENABLED = "thinking"
    DISABLED = "no-thinking"


def native_request(family: str, messages: list[dict], thinking: Thinking, decode: Decode, stop_ids: list[int]) -> dict:
    """The chat API request, preserving the pinned template's prefill behavior."""
    body = {
        "messages": messages,
        "temperature": decode.temperature,
        "top_p": decode.top_p,
        "repetition_penalty": decode.repetition_penalty,
        "max_tokens": decode.max_tokens,
        "seed": 0,
        "skip_special_tokens": False,
        # Explicitly stopping on both tokens makes stop_reason unambiguous even for EOS.
        "ignore_eos": True,
        "stop_token_ids": stop_ids,
        "return_token_ids": True,
    }
    if family == "qwen3_moe" and thinking == Thinking.ENABLED:
        body.update(
            messages=[*messages, {"role": "assistant", "content": THINK_PREFILL}],
            continue_final_message=True,
            add_generation_prompt=False,
        )
    elif family == "qwen3_5_moe":
        body["chat_template_kwargs"] = {"enable_thinking": thinking == Thinking.ENABLED}
    return body


def native_facts(
    text: str, finish_reason: str, stop_reason: int | str | None, prefill: str, stop_ids: list[int]
) -> dict:
    facts = completion_protocol(text, finish_reason, stop_reason, Delimiters(QWEN_TOKENS, stop_ids[0]), prefill=prefill)
    answer = text.rsplit("</think>", 1)[-1]
    if facts["think_blocks"] and not facts["think_closed"]:
        answer = ""
    for token in STOP_TOKENS:
        answer = answer.replace(token, "")
    facts["think_closed_with_answer"] = bool(facts["think_closed"] and answer.strip())
    return {
        **facts,
        "end_token": (
            dict(zip(stop_ids, STOP_TOKENS, strict=True)).get(stop_reason, "none") if finish_reason == "stop" else "none"
        ),
        "answer_text": answer,
    }


def read_json(path: str):
    fs, resolved = filesystem_for(path)
    with fs.open(resolved, "rt") as handle:
        return json.load(handle)


def write_json(path: str, value: dict) -> None:
    fs, resolved = filesystem_for(path)
    fs.makedirs(resolved.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(resolved, "wt") as handle:
        json.dump(value, handle, ensure_ascii=False)


def sample_reply(endpoint: OpenAIEndpoint, body: dict) -> dict:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    response = requests.post(
        endpoint.url("chat/completions"),
        json={**body, "model": endpoint.model},
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice["finish_reason"],
        "stop_reason": choice.get("stop_reason"),
        "last_token_id": (choice.get("token_ids") or [None])[-1],
        "usage": result["usage"],
    }


def gate_summary(records: list[dict]) -> dict:
    facts = [record["facts"] for record in records]
    eligible = [fact for fact in facts if not fact["length_limited"]]
    thinking = [fact for fact in eligible if fact["think_blocks"]]
    graded = [record["correct"] for record in records if record["correct"] is not None]
    return {
        "count": len(records),
        "think_closed_with_answer_share": (
            sum(f["think_closed_with_answer"] for f in thinking) / len(thinking) if thinking else None
        ),
        "think_close_denominator": len(thinking),
        # A B that never opens a block would pass closure vacuously.
        "replies_opening_think": sum(f["think_blocks"] > 0 for f in facts),
        "replies_containing_think_close": sum("</think>" in r["text"] for r in records),
        "turn_end_share": sum(f["turn_ended"] for f in eligible) / len(eligible) if eligible else None,
        "turn_end_denominator": len(eligible),
        "length_limited": len(facts) - len(eligible),
        "length_limited_repetition_loops": sum(
            ends_in_repetition_loop(r["text"]) for r, f in zip(records, facts, strict=True) if f["length_limited"]
        ),
        "end_tokens": {token: sum(f["end_token"] == token for f in facts) for token in (*STOP_TOKENS, "none")},
        "length_capped": sum(r["finish_reason"] == "length" for r in records),
        "accuracy": sum(graded) / len(graded) if graded else None,
        "graded": len(graded),
        "samples": [{"probe_id": r["probe_id"], "text": r["text"]} for r in records[:3]],
    }


def math_correct(reference: str, text: str) -> bool:
    """Grade answer content independently of reasoning-tag compliance."""
    answer = text.rsplit("</think>", 1)[-1]
    for token in STOP_TOKENS:
        answer = answer.replace(token, "")
    candidate = extract_boxed(answer) or answer
    return bool(candidate.strip() and answers_match(reference, candidate))


def sample_set(
    endpoint: OpenAIEndpoint,
    family: str,
    repo: str,
    revision: str,
    name: str,
    rows: list[dict],
    thinking: Thinking,
    decode: Decode,
    stop_ids: list[int],
    out: str,
) -> list[dict]:
    key = f"{name}-{thinking}-rp{decode.repetition_penalty:g}"
    records = {}
    with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
        pending = {}
        for row in rows:
            body = native_request(family, row["messages"], thinking, decode, stop_ids)
            fingerprint = hashlib.sha256(
                json.dumps({"base": repo, "revision": revision, "body": body, "row": row}, sort_keys=True).encode()
            ).hexdigest()
            path = f"{out}/{key}/{fingerprint}.json"
            fs, resolved = filesystem_for(path)
            if fs.exists(resolved):
                cached = read_json(path)
                prefill = THINK_PREFILL if thinking == Thinking.ENABLED else ""
                stop_reason = cached["stop_reason"]
                if stop_reason is None and cached["finish_reason"] == "stop":
                    stop_reason = cached["last_token_id"]
                cached["facts"] = native_facts(cached["text"], cached["finish_reason"], stop_reason, prefill, stop_ids)
                cached["correct"] = math_correct(str(row["answer"]), cached["text"]) if name == "P_think" else None
                records[row["probe_id"]] = cached
            else:
                pending[pool.submit(sample_reply, endpoint, body)] = (row, path)
        for future in as_completed(pending):
            row, path = pending[future]
            result = future.result()
            prefill = THINK_PREFILL if thinking == Thinking.ENABLED else ""
            stop_reason = result["stop_reason"]
            if stop_reason is None and result["finish_reason"] == "stop":
                stop_reason = result["last_token_id"]
            facts = native_facts(result["text"], result["finish_reason"], stop_reason, prefill, stop_ids)
            correct = math_correct(str(row["answer"]), result["text"]) if name == "P_think" else None
            record = {**result, "probe_id": row["probe_id"], "facts": facts, "correct": correct}
            write_json(path, record)
            records[row["probe_id"]] = record
            logger.info("%s: %d/%d complete", key, len(records), len(rows))
    return [records[row["probe_id"]] for row in rows]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", help="native-template B export; omit for the pinned base check")
    parser.add_argument("--chat-prompts", required=True)
    parser.add_argument("--think-prompts", required=True)
    parser.add_argument("--out", required=True, help="new output directory, reused only for this exact check")
    parser.add_argument("--dry-run", action="store_true", help="validate prompts and serving configuration without GPUs")
    parser.add_argument(
        "--thinking",
        type=Thinking,
        choices=list(Thinking),
        action="append",
        required=True,
        help="repeatable; for Qwen3, no-thinking is the plain generation prompt with no prefill",
    )
    parser.add_argument("--repetition-penalty", type=float, action="append", required=True, help="repeatable")
    args = parser.parse_args()
    if not args.name.startswith(("opd-0b-qn-", "opd-0b-q3n-")):
        parser.error("job names must start with opd-0b-qn- or opd-0b-q3n-")
    repo, revision = FAMILY_TOKENIZER_REVISIONS[args.family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    stop_ids = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
    prompt_sets = {}
    for name, path in (("P_chat", args.chat_prompts), ("P_think", args.think_prompts)):
        fs, resolved = filesystem_for(path)
        with fs.open(resolved, "rt") as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != 200 or len({row["probe_id"] for row in rows}) != 200:
            raise ValueError(f"{name}: expected 200 distinct gate prompts, got {len(rows)}")
        prompt_sets[name] = rows
    context = QWEN3_CONTEXT if args.family == "qwen3_moe" else QWEN35_CONTEXT
    max_prompt_tokens = 0
    for thinking in Thinking:
        for rows in prompt_sets.values():
            for row in rows:
                body = native_request(args.family, row["messages"], thinking, Decode(), stop_ids)
                ids = tokenizer.apply_chat_template(
                    body["messages"],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=body.get("add_generation_prompt", True),
                    continue_final_message=body.get("continue_final_message", False),
                    **body.get("chat_template_kwargs", {}),
                )
                max_prompt_tokens = max(max_prompt_tokens, len(ids))
    if max_prompt_tokens + Decode.max_tokens > context:
        raise ValueError(
            f"Prompt plus full output cap exceeds context: {max_prompt_tokens} + {Decode.max_tokens} > {context}"
        )
    model = replace(checkpoint_model(args.name, repo, context), revision=revision, tokenizer_revision=revision)
    if args.family == "qwen3_moe":
        model = replace(
            model,
            serve=replace(model.serve, vllm_extra_args=(*model.serve.vllm_extra_args, "--hf-overrides", QWEN3_YARN)),
        )
    else:
        # Text-only Qwen3.5 serving flags from serving_config, without its reasoning parser:
        # the gate needs the raw reasoning tags in message.content.
        model = replace(
            model,
            serve=replace(
                model.serve,
                vllm_extra_args=(
                    *model.serve.vllm_extra_args,
                    "--gdn-prefill-backend",
                    "triton",
                    "--limit-mm-per-prompt",
                    '{"image":0,"video":0}',
                ),
            ),
        )
    if args.checkpoint:
        checked_templates(args.checkpoint, QWEN_NATIVE_TEMPLATES[args.family])
        # Stage exports keep fp32 masters; serve them in bf16 like the bf16 B copy.
        model = replace(
            model,
            location=args.checkpoint,
            tokenizer=args.checkpoint,
            revision=None,
            tokenizer_revision=None,
            serve=replace(model.serve, vllm_extra_args=(*model.serve.vllm_extra_args, "--dtype", "bfloat16")),
        )
    inference = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a"),
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        request_timeout_seconds=REQUEST_TIMEOUT,
    )
    report = {
        "base": repo,
        "checkpoint": args.checkpoint,
        "revision": revision,
        "template_sha256": (
            hashlib.sha256(
                (QWEN_NATIVE_TEMPLATES[args.family] if args.checkpoint else tokenizer.chat_template).encode()
            ).hexdigest()
        ),
        "repetition_loop_rule": "terminal substring of 32-512 characters repeated at least three times",
        "context": context,
        "vllm_extra_args": model.serve.vllm_extra_args,
        "max_prompt_tokens": max_prompt_tokens,
        "max_completion_tokens": Decode.max_tokens,
        "stop_ids": dict(zip(STOP_TOKENS, stop_ids, strict=True)),
        "request_timeout": REQUEST_TIMEOUT,
        "temperature": Decode.temperature,
        "top_p": Decode.top_p,
        "repetition_penalties": args.repetition_penalty,
        "thinking": [str(mode) for mode in args.thinking],
        "sets": {},
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    with remote_inference(inference) as session:
        for penalty in args.repetition_penalty:
            for thinking in args.thinking:
                for name, rows in prompt_sets.items():
                    key = f"{name}-{thinking}-rp{penalty:g}"
                    records = sample_set(
                        session.model.endpoint,
                        args.family,
                        args.checkpoint or repo,
                        revision,
                        name,
                        rows,
                        thinking,
                        Decode(repetition_penalty=penalty),
                        stop_ids,
                        args.out,
                    )
                    report["sets"][key] = gate_summary(records)
                    write_json(f"{args.out}/report.json", report)
                    logger.info("%s: %s", key, {k: v for k, v in report["sets"][key].items() if k != "samples"})


if __name__ == "__main__":
    main()
