# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade native math-teacher attempts and render complete ones for B's SFT template."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS
from experiments.post_training.phase1b.native_math_continuations import native_prompt_ids
from experiments.post_training.probe_sets.buckets import decode_token_bytes
from experiments.post_training.probe_sets.continuations import local_tokenizer, render_prompt_ids
from experiments.post_training.task_data.expert_outputs import GRADERS, bounded_grade
from experiments.post_training.task_data.terminus_answers import answer_text

MAX_PROMPT_TOKENS = 4_096
MAX_RESPONSE_TOKENS = 16_384
RESERVED_IDS = frozenset((128005, 128011, 128000, 128006, 128007))


def read_jsonl(uri: str) -> list[dict]:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(uri: str, rows: list[dict]) -> None:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)


def grade_attempts(
    tasks: list[dict],
    requests: list[dict],
    responses: list[dict],
    b_tokenizer,
    teacher_tokenizer,
    teacher_route: str,
    expected_tasks: int,
    expected_attempts: int,
) -> tuple[list[dict], list[dict]]:
    """Return per-attempt verdicts and turn-ended B-template fixed-completion rows."""
    if len(tasks) != expected_tasks or len(requests) != len(responses) or len(requests) != expected_attempts:
        raise ValueError(f"expected {expected_tasks} tasks and {expected_attempts} aligned native teacher responses")
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("duplicate math task IDs")
    response_by_id = {row["probe_id"]: row for row in responses}
    if len(response_by_id) != len(responses) or set(response_by_id) != {row["probe_id"] for row in requests}:
        raise ValueError("missing or duplicate math attempts")
    close = teacher_tokenizer.convert_tokens_to_ids(FAMILY_TOKEN_MAPS["grug"].turn_close)
    if close != b_tokenizer.convert_tokens_to_ids(FAMILY_TOKEN_MAPS["grug"].turn_close):
        raise ValueError("B and teacher disagree on the turn close")

    verdicts = []
    candidates = []
    b_prompts = {
        task_id: render_prompt_ids(b_tokenizer, "grug", [{"role": "user", "content": task["prompt"]}])
        for task_id, task in task_by_id.items()
    }
    if any(len(prompt) > MAX_PROMPT_TOKENS for prompt in b_prompts.values()):
        raise ValueError("a B-template math prompt exceeds the Phase 1a prompt cap")
    for request in requests:
        sample_id = request["probe_id"]
        task_id = request["source"]
        if task_id not in task_by_id or not sample_id.startswith(f"{task_id}#sample-"):
            raise ValueError(f"invalid math attempt ID: {sample_id}")
        raw = response_by_id[sample_id]
        if raw["prompt_ids"] != native_prompt_ids(teacher_tokenizer, request["messages"]):
            raise ValueError(f"native teacher prompt changed: {sample_id}")
        response_ids = raw["response_ids"]
        ended = bool(response_ids) and response_ids[-1] == close
        final_token_id = response_ids[-1] if response_ids else None
        drop_reason = None if ended else "other_eos" if final_token_id == 128001 else "cap_or_unterminated"
        if len(response_ids) > MAX_RESPONSE_TOKENS or any(token in RESERVED_IDS for token in response_ids):
            raise ValueError(f"invalid response token stream: {sample_id}")
        verdict = None
        if ended:
            reply = decode_token_bytes(teacher_tokenizer, response_ids[:-1])
            answer = answer_text(reply, FAMILY_TOKEN_MAPS["grug"].think_close)
            verdict = bounded_grade(GRADERS["R"], task_by_id[task_id], answer.text)
            prompt_ids = b_prompts[task_id]
            candidates.append(
                {
                    "task_id": task_id,
                    "sample_id": sample_id,
                    "data_source": "task_R",
                    "teacher_route": teacher_route,
                    "input_ids": prompt_ids + response_ids,
                    "assistant_masks": [0] * len(prompt_ids) + [1] * len(response_ids),
                }
            )
        verdicts.append(
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "response_tokens": len(response_ids),
                "turn_ended": ended,
                "final_token_id": final_token_id,
                "drop_reason": drop_reason,
                "graded_correct": verdict,
            }
        )
    return verdicts, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--b-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-route", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-attempts", type=int, required=True)
    args = parser.parse_args()
    if args.expected_tasks < 1 or args.expected_attempts < 1:
        parser.error("expected tasks and attempts must be positive")
    for uri in (args.tasks, args.requests, args.responses, args.b_model, args.teacher_model, args.out):
        if not uri.startswith("s3://marin-rno2a/"):
            parser.error("all Phase 1b grading paths must stay in marin-rno2a")
    configure_coreweave_s3()
    tasks = read_jsonl(args.tasks)
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    b_tokenizer = local_tokenizer(args.b_model)
    teacher_tokenizer = local_tokenizer(args.teacher_model)
    verdicts, candidates = grade_attempts(
        tasks,
        requests,
        responses,
        b_tokenizer,
        teacher_tokenizer,
        args.teacher_route,
        args.expected_tasks,
        args.expected_attempts,
    )
    write_jsonl(f"{args.out.rstrip('/')}/verdicts.jsonl", verdicts)
    write_jsonl(f"{args.out.rstrip('/')}/ended-candidates.jsonl", candidates)
    counts = Counter((row["turn_ended"], row["graded_correct"]) for row in verdicts)
    manifest = {
        "tasks": args.tasks,
        "requests": args.requests,
        "responses": args.responses,
        "b_model": args.b_model,
        "teacher_model": args.teacher_model,
        "teacher_route": args.teacher_route,
        "total_attempts": len(verdicts),
        "ended_candidates": len(candidates),
        "ended_correct": counts[(True, True)],
        "ended_incorrect": counts[(True, False)],
        "ended_ungraded": counts[(True, None)],
        "non_turn_close": sum(not row["turn_ended"] for row in verdicts),
        "other_eos": sum(row["drop_reason"] == "other_eos" for row in verdicts),
        "cap_or_unterminated": sum(row["drop_reason"] == "cap_or_unterminated" for row in verdicts),
    }
    filesystem, path = filesystem_for(f"{args.out.rstrip('/')}/manifest.json")
    with filesystem.open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
