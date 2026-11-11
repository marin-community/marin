# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixed terminal turn probes from exact Terminus-2 rollout tokens."""

import argparse
import hashlib
import json
from dataclasses import dataclass

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.native_math_continuations import native_prompt_ids
from experiments.post_training.probe_sets.buckets import _token_bytes, decode_token_bytes
from experiments.post_training.probe_sets.continuations import local_tokenizer, write_jsonl
from experiments.post_training.task_data.terminus_answers import json_string_spans


@dataclass(frozen=True)
class Turn:
    task_name: str
    task_checksum: str
    index: int
    messages: list[dict[str, str]]
    prompt_ids: list[int]
    response_ids: list[int]

    @property
    def sample_id(self) -> str:
        return f"{self.task_name}#turn-{self.index:03d}"


def prompt_messages(tokenizer, prompt_ids: list[int]) -> list[dict[str, str]]:
    header_start = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    header_end = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    turn_end = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    message_end = tokenizer.convert_tokens_to_ids("<|eom_id|>")
    starts = [index for index, token in enumerate(prompt_ids) if token == header_start]
    if not starts:
        raise ValueError("prompt has no header start")
    messages = []
    normalized_segments = [
        tokenizer.decode(prompt_ids[: starts[0]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ]
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(prompt_ids)
        segment = prompt_ids[start:stop]
        if header_end not in segment:
            raise ValueError("prompt header has no end token")
        header = segment.index(header_end)
        role = tokenizer.decode(segment[1:header], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        close = next(
            (i for i, token in enumerate(segment[header + 1 :], header + 1) if token in (turn_end, message_end)), None
        )
        if close is None:
            if index != len(starts) - 1 or role != "assistant":
                raise ValueError("incomplete non-final message in prompt")
            normalized_segments.append(
                tokenizer.decode(segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
            continue
        content_ids = segment[header + 1 : close]
        content_start = header + 1
        if content_ids and content_ids[0] == tokenizer.encode("\n", add_special_tokens=False)[0]:
            content_ids = content_ids[1:]
            content_start += 1
        content = tokenizer.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if role == "assistant":
            normalized_segments.append(
                tokenizer.decode(segment[:content_start], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                + content.strip()
                + tokenizer.decode(segment[close:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
        else:
            normalized_segments.append(
                tokenizer.decode(segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
        if role == "system" and content == "Reasoning: /think":
            continue
        messages.append({"role": role, "content": content})
    rendered = native_prompt_ids(tokenizer, messages)
    if tokenizer.decode(rendered, skip_special_tokens=False, clean_up_tokenization_spaces=False) != "".join(
        normalized_segments
    ):
        raise ValueError("recorded T1 prompt text does not replay under T1's pinned native template")
    return messages


def response_tags(tokenizer, response_ids: list[int]) -> list[str]:
    content = decode_token_bytes(tokenizer, response_ids)
    tags = ["other"] * len(content)
    think_close = "<|end_think|>"
    json_start = content.find(think_close)
    json_start = len(content) if json_start < 0 else json_start + len(think_close)
    tags[:json_start] = ["analysis_plan"] * json_start

    in_string = False
    escaped = False
    for index in range(json_start, len(content)):
        char = content[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
            tags[index] = "json_structural"
        elif not in_string and char in "{}[],:":
            tags[index] = "json_structural"

    for key, bucket in (("analysis", "analysis_plan"), ("plan", "analysis_plan"), ("keystrokes", "command_strings")):
        for start, stop in json_string_spans(content[json_start:], key):
            tags[json_start + start : json_start + stop] = [bucket] * (stop - start)

    for marker in ("<|start_think|>", "<|end_think|>", "<|eot_id|>"):
        start = 0
        while (index := content.find(marker, start)) >= 0:
            tags[index : index + len(marker)] = ["delimiter"] * len(marker)
            start = index + len(marker)

    byte_chars = [index for index, char in enumerate(content) for _ in char.encode("utf-8", "surrogateescape")]
    result = []
    offset = 0
    for piece in _token_bytes(tokenizer, response_ids):
        result.append(tags[byte_chars[offset]])
        offset += len(piece)
    if offset != len(byte_chars):
        raise ValueError("response bytes and tags do not align")
    return result


def result_paths(root: str) -> list[str]:
    filesystem, path = filesystem_for(root)
    return sorted(
        item
        for item in filesystem.find(path)
        if "/task_" in item and item.endswith("/result.json") and "/attempts/" not in item
    )


def recorded_turns(root: str, tokenizer) -> tuple[list[Turn], dict[str, float]]:
    filesystem, _ = filesystem_for(root)
    turns = []
    scores = {}
    for path in result_paths(root):
        with filesystem.open(path) as handle:
            result = json.load(handle)
        task = result["task_name"]
        if task in scores:
            raise ValueError(f"duplicate task result: {task}")
        verifier = result["verifier_result"]
        scores[task] = float(verifier["rewards"]["reward"]) if verifier is not None else 0.0
        agent = result["agent_result"]
        details = agent["rollout_details"] if agent is not None else None
        if not details or not details[0].get("prompt_token_ids"):
            if result["exception_info"] is not None:
                continue
            raise ValueError(f"{task}: missing exact rollout details")
        prompts = details[0]["prompt_token_ids"]
        responses = details[0]["completion_token_ids"]
        if len(prompts) != len(responses):
            raise ValueError(f"{task}: prompt and response turn counts differ")
        for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            if not response or any(token in (128005, 128011, 128000, 128006, 128007) for token in response):
                raise ValueError(f"{task} turn {index}: empty response or reserved/role token")
            turns.append(
                Turn(task, result["task_checksum"], index, prompt_messages(tokenizer, prompt), prompt, response)
            )
    return turns, scores


def selected_turns(turns: list[Turn], limit: int) -> list[Turn]:
    by_task: dict[str, list[Turn]] = {}
    for turn in turns:
        by_task.setdefault(turn.task_name, []).append(turn)
    first = [min(rows, key=lambda row: hashlib.sha256(row.sample_id.encode()).digest()) for rows in by_task.values()]
    if limit < len(first):
        first = sorted(first, key=lambda row: hashlib.sha256(row.sample_id.encode()).digest())[:limit]
    first_ids = {row.sample_id for row in first}
    rest = sorted(
        (row for row in turns if row.sample_id not in first_ids),
        key=lambda row: hashlib.sha256(row.sample_id.encode()).digest(),
    )
    return sorted((first + rest[: max(limit - len(first), 0)])[:limit], key=lambda row: row.sample_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--model", action="append", required=True, help="ROLE=checkpoint URI")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    models = dict(spec.split("=", 1) for spec in args.model)
    if set(models) != {"B", "T1", "M1"} or args.limit < 1:
        parser.error("provide B, T1 and M1 checkpoint URIs and a positive limit")
    configure_coreweave_s3()
    tokenizers = {role: local_tokenizer(uri) for role, uri in models.items()}
    turns, scores = recorded_turns(args.results_root, tokenizers["T1"])
    selected = selected_turns(turns, args.limit)
    if not selected:
        raise ValueError("no recorded turns")
    manifest = {
        "results_root": args.results_root,
        "models": models,
        "tasks": len(scores),
        "task_reward_mean": sum(scores.values()) / len(scores),
        "available_turns": len(turns),
        "requested_turns": args.limit,
        "selected_turns": len(selected),
        "sets": {},
    }
    prompts = [
        {
            "probe_id": turn.sample_id,
            "source": turn.task_name,
            "task_checksum": turn.task_checksum,
            "turn_index": turn.index,
            "messages": turn.messages,
        }
        for turn in selected
    ]
    manifest["prompts"] = {
        "rows": len(prompts),
        "sha256": write_jsonl(f"{args.out.rstrip('/')}/P_J.jsonl", prompts),
    }
    for role, tokenizer in tokenizers.items():
        rows = []
        for turn in selected:
            prompt = turn.prompt_ids if role == "T1" else native_prompt_ids(tokenizer, turn.messages)
            response = turn.response_ids
            rows.append(
                {
                    "sample_id": turn.sample_id,
                    "source": turn.task_name,
                    "task_checksum": turn.task_checksum,
                    "bucket_group": "P_J",
                    "token_ids": prompt + response,
                    "token_type_tags": ["context"] * len(prompt) + response_tags(tokenizers["T1"], response),
                    "position_tags": [turn.index] * (len(prompt) + len(response)),
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(response),
                }
            )
        manifest["sets"][role] = {
            "rows": len(rows),
            "sha256": write_jsonl(f"{args.out.rstrip('/')}/P_J-{role}.jsonl", rows),
        }
    filesystem, path = filesystem_for(f"{args.out.rstrip('/')}/manifest.json")
    with filesystem.open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
