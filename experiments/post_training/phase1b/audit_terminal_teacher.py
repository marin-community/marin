# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit T1's complete Harbor teacher trials before constructing SFT windows."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.phase1b.extract_terminal_probe import prompt_messages
from experiments.post_training.probe_sets.continuations import local_tokenizer, render_prompt_ids

PROMPT_CAP = 4_096
RESPONSE_CAP = 16_384
TURN_CLOSE = 128009
INVALID_RESPONSE_IDS = frozenset((128005, 128011, 128000, 128006, 128007))


def result_paths(root: str) -> list[str]:
    filesystem, path = filesystem_for(root)
    return sorted(
        item
        for item in filesystem.find(path)
        if item.endswith("/result.json") and "/attempts/" not in item and "/harbor_jobs/" in item and "/task_" in item
    )


def trial_summary(result: dict, t1_tokenizer, b_tokenizer) -> dict:
    """Record complete-trial outcome and every usable B-template response window."""
    agent = result.get("agent_result")
    details = agent.get("rollout_details") if agent else None
    detail = details[0] if details else None
    prompts = detail.get("prompt_token_ids", []) if detail else []
    responses = detail.get("completion_token_ids", []) if detail else []
    if len(prompts) != len(responses):
        raise ValueError(f"{result['trial_name']}: prompt/response turn count differs")
    eligible = []
    all_turns = []
    invalid_response_turns = 0
    cap_hit_turns = 0
    over_prompt_cap = 0
    over_response_cap = 0
    unreplayable_prompt_turns = 0
    for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        turn = {"turn": index, "native_prompt_tokens": len(prompt), "response_tokens": len(response)}
        all_turns.append(turn)
        if not response or any(token in INVALID_RESPONSE_IDS for token in response):
            turn["ineligible_reason"] = "invalid_response"
            invalid_response_turns += 1
            continue
        if response[-1] != TURN_CLOSE:
            turn["ineligible_reason"] = "cap_hit"
            cap_hit_turns += 1
            continue
        if len(response) > RESPONSE_CAP:
            turn["ineligible_reason"] = "over_response_cap"
            over_response_cap += 1
            continue
        try:
            messages = prompt_messages(t1_tokenizer, prompt)
            b_prompt = render_prompt_ids(b_tokenizer, "grug", messages)
        except ValueError as error:
            turn["ineligible_reason"] = "unreplayable_prompt"
            turn["prompt_error"] = str(error)
            unreplayable_prompt_turns += 1
            continue
        turn["b_prompt_tokens"] = len(b_prompt)
        if len(b_prompt) > PROMPT_CAP:
            turn["ineligible_reason"] = "over_prompt_cap"
            over_prompt_cap += 1
            continue
        turn["eligible"] = True
        eligible.append({"turn": index, "prompt_tokens": len(b_prompt), "response_tokens": len(response)})
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if verifier else None
    reward = rewards.get("reward") if rewards else None
    exception = result.get("exception_info") or {}
    return {
        "task_name": result["task_name"],
        "task_checksum": result["task_checksum"],
        "trial_name": result["trial_name"],
        "trial_uri": result["trial_uri"],
        "reward": reward,
        "exception_type": exception.get("exception_type"),
        "no_model_turn": not prompts,
        "turns": len(prompts),
        "all_turns": all_turns,
        "eligible_turns": eligible,
        "invalid_response_turns": invalid_response_turns,
        "cap_hit_turns": cap_hit_turns,
        "over_prompt_cap": over_prompt_cap,
        "over_response_cap": over_response_cap,
        "unreplayable_prompt_turns": unreplayable_prompt_turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--t1-model", required=True)
    parser.add_argument("--b-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-tasks", type=int, default=1_000)
    parser.add_argument("--attempts-per-task", type=int, default=4)
    args = parser.parse_args()
    if args.expected_tasks < 1 or args.attempts_per_task < 1:
        parser.error("expected tasks and attempts per task must be positive")
    for uri in (args.results_root, args.t1_model, args.b_model, args.out):
        if not uri.startswith("s3://marin-rno2a/"):
            parser.error("all Phase 1b audit paths must stay in marin-rno2a")
    configure_coreweave_s3()
    t1_tokenizer = local_tokenizer(args.t1_model)
    b_tokenizer = local_tokenizer(args.b_model)
    paths = result_paths(args.results_root)
    expected_trials = args.expected_tasks * args.attempts_per_task
    if len(paths) != expected_trials:
        raise ValueError(f"expected {expected_trials} canonical trial results, found {len(paths)}")
    filesystem, _ = filesystem_for(args.results_root)
    rows = []
    for path in paths:
        with filesystem.open(path) as handle:
            result = json.load(handle)
        row = trial_summary(result, t1_tokenizer, b_tokenizer)
        row["result_uri"] = path if path.startswith("s3://") else f"s3://{path}"
        rows.append(row)
    if len({row["trial_name"] for row in rows}) != len(rows):
        raise ValueError("duplicate trial names")
    by_task = Counter(row["task_name"] for row in rows)
    if len(by_task) != args.expected_tasks or set(by_task.values()) != {args.attempts_per_task}:
        raise ValueError(f"expected {args.attempts_per_task} T1 trials for each of {args.expected_tasks} tasks")
    retained = [row for row in rows if not row["no_model_turn"]]
    verified = [row for row in retained if row["reward"] is not None]
    summary = {
        "results_root": args.results_root,
        "t1_model": args.t1_model,
        "b_model": args.b_model,
        "tasks": len(by_task),
        "trials": len(rows),
        "solved": sum(row["reward"] == 1 for row in rows),
        "verified": sum(row["reward"] is not None for row in rows),
        "unverified": sum(row["reward"] is None for row in rows),
        "agent_timeouts": sum(row["exception_type"] == "AgentTimeoutError" for row in rows),
        "exception_types": dict(Counter(row["exception_type"] for row in rows if row["exception_type"])),
        "infrastructure_no_turn": sum(row["no_model_turn"] for row in rows),
        "retained_trajectories": sum(not row["no_model_turn"] for row in rows),
        "retained_solve_rate": sum(row["reward"] == 1 for row in retained) / len(retained) if retained else None,
        "verified_solve_rate": sum(row["reward"] == 1 for row in verified) / len(verified) if verified else None,
        "retained_timeout_fraction": (
            sum(row["exception_type"] == "AgentTimeoutError" for row in retained) / len(retained) if retained else None
        ),
        "trials_with_eligible_turn": sum(bool(row["eligible_turns"]) for row in rows),
        "eligible_turns": sum(len(row["eligible_turns"]) for row in rows),
        "invalid_response_turns": sum(row["invalid_response_turns"] for row in rows),
        "cap_hit_turns": sum(row["cap_hit_turns"] for row in rows),
        "over_prompt_cap": sum(row["over_prompt_cap"] for row in rows),
        "over_response_cap": sum(row["over_response_cap"] for row in rows),
        "unreplayable_prompt_turns": sum(row["unreplayable_prompt_turns"] for row in rows),
    }
    filesystem, output_path = filesystem_for(f"{args.out.rstrip('/')}/trials.jsonl")
    with filesystem.open(output_path, "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    filesystem, output_path = filesystem_for(f"{args.out.rstrip('/')}/manifest.json")
    with filesystem.open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
