# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select one turn from each complete T1 episode and materialize B-template SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template
from experiments.post_training.phase1b.audit_terminal_teacher import INVALID_RESPONSE_IDS, PROMPT_CAP, RESPONSE_CAP
from experiments.post_training.phase1b.extract_terminal_probe import prompt_messages
from experiments.post_training.probe_sets.continuations import (
    local_tokenizer,
    read_jsonl,
    render_prompt_ids,
    write_jsonl,
)

ROWS = 4_000
TASKS = 1_000
ROWS_PER_TASK = 4
ATTEMPT_CAP = 8
TURN_CLOSE = 128009
TURN_SALT = "phase1b-j-latest-biased-seed0-20261109"


@dataclass(frozen=True)
class PlannedTurn:
    task_name: str
    trial_name: str
    result_uri: str
    turn_index: int
    native_prompt_tokens: int
    prompt_tokens: int
    response_tokens: int
    reward: float | None
    exception_type: str | None
    domain: str
    complexity: str

    @property
    def sample_id(self) -> str:
        digest = hashlib.sha256(self.result_uri.encode()).hexdigest()[:12]
        return f"{self.trial_name}#{digest}#turn-{self.turn_index:03d}"


def selected_turn(eligible: list[dict], result_uri: str) -> dict | None:
    """Draw one of the last four eligible turns with a strong, seeded latest-turn preference."""
    by_index = {row["turn"]: row for row in eligible}
    if len(by_index) != len(eligible):
        raise ValueError(f"{result_uri}: duplicate eligible turn index")
    if not by_index:
        return None
    choices = sorted(by_index)[-4:]
    weights = [4**rank for rank in range(len(choices))]
    draw = int.from_bytes(hashlib.sha256(f"{TURN_SALT}:{result_uri}".encode()).digest()) % sum(weights)
    for turn, weight in zip(choices, weights, strict=True):
        if draw < weight:
            return by_index[turn]
        draw -= weight
    raise AssertionError("weighted turn draw exceeded the available choices")


def quantiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {"p50": ordered[len(ordered) // 2], "p90": ordered[int(len(ordered) * 0.9)], "max": ordered[-1]}


def outcome_label(reward: float | None) -> str:
    if reward == 1:
        return "solved"
    if reward == 0:
        return "failed"
    return "unverified"


def plan_turns(trials: list[dict], metadata: list[dict]) -> tuple[list[PlannedTurn], dict, list[dict]]:
    """Choose four usable episodes per task without filtering on verifier outcome."""
    task_by_name = {row["task_name"]: row for row in metadata}
    if len(task_by_name) != len(metadata) or len({row["task_id"] for row in metadata}) != len(metadata):
        raise ValueError("duplicate candidate terminal task")
    if len({row["result_uri"] for row in trials}) != len(trials):
        raise ValueError("duplicate terminal result URI")
    if {row["task_name"] for row in trials} - set(task_by_name):
        raise ValueError("audited trial outside the screened terminal candidate pool")
    trials_by_task: dict[str, list[dict]] = defaultdict(list)
    for trial in trials:
        trials_by_task[trial["task_name"]].append(trial)
    attempts = Counter(row["task_name"] for row in trials)
    if any(count > ATTEMPT_CAP for count in attempts.values()):
        raise ValueError("terminal task exceeded the eight-attempt cap")
    successes = Counter(row["task_name"] for row in trials if row["reward"] == 1)
    usable_episodes = Counter(row["task_name"] for row in trials if not row["no_model_turn"] and row["eligible_turns"])
    selected = []
    for task in metadata:
        name = task["task_name"]
        episodes = sorted(trials_by_task[name], key=lambda row: (row["result_uri"], row["trial_name"]))
        usable = [episode for episode in episodes if not episode["no_model_turn"] and episode["eligible_turns"]]
        if len(usable) < ROWS_PER_TASK:
            continue
        for episode in usable[:ROWS_PER_TASK]:
            turn = selected_turn(episode["eligible_turns"], episode["result_uri"])
            if turn is None:
                raise AssertionError("usable episode has no eligible turn")
            native_lengths = {item["turn"]: item["native_prompt_tokens"] for item in episode["all_turns"]}
            selected.append(
                PlannedTurn(
                    task_name=name,
                    trial_name=episode["trial_name"],
                    result_uri=episode["result_uri"],
                    turn_index=turn["turn"],
                    native_prompt_tokens=native_lengths[turn["turn"]],
                    prompt_tokens=turn["prompt_tokens"],
                    response_tokens=turn["response_tokens"],
                    reward=episode["reward"],
                    exception_type=episode["exception_type"],
                    domain=task["domain"],
                    complexity=task["complexity"],
                )
            )
        if len(selected) == ROWS:
            break
    selected_counts = Counter(row.task_name for row in selected)
    items = [
        {
            **task,
            "attempts": attempts[task["task_name"]],
            "verified_solves": successes[task["task_name"]],
            "usable_episodes": usable_episodes[task["task_name"]],
            "no_model_turn": sum(row["no_model_turn"] for row in trials_by_task[task["task_name"]]),
            "no_eligible_turn": sum(
                not row["no_model_turn"] and not row["eligible_turns"] for row in trials_by_task[task["task_name"]]
            ),
            "selected_rows": selected_counts[task["task_name"]],
        }
        for task in metadata
    ]
    all_turns = [turn for trial in trials for turn in trial.get("all_turns", [])]
    eligible_turns = [turn for trial in trials for turn in trial["eligible_turns"]]
    summary = {
        "candidate_tasks": len(metadata),
        "attempted_tasks": len(attempts),
        "trials": len(trials),
        "verified_solved_episodes": sum(successes.values()),
        "usable_episodes": sum(usable_episodes.values()),
        "selected_rows": len(selected),
        "selected_tasks": len(selected_counts),
        "selected_episodes": len(selected),
        "selected_outcome_counts": dict(Counter(outcome_label(row.reward) for row in selected)),
        "all_outcome_counts": dict(Counter(outcome_label(row["reward"]) for row in trials)),
        "attempted_tasks_short_of_four_usable": sum(
            0 < attempts[task["task_name"]] and usable_episodes[task["task_name"]] < ROWS_PER_TASK for task in metadata
        ),
        "infrastructure_no_turn": sum(row["no_model_turn"] for row in trials),
        "episodes_no_eligible_turn": sum(not row["no_model_turn"] and not row["eligible_turns"] for row in trials),
        "agent_timeouts": sum(row["exception_type"] == "AgentTimeoutError" for row in trials),
        "cap_hit_turns": sum(row["cap_hit_turns"] for row in trials),
        "unreplayable_prompt_turns": sum(row["unreplayable_prompt_turns"] for row in trials),
        "candidate_domain_counts": dict(Counter(task["domain"] for task in metadata)),
        "selected_domain_task_counts": dict(Counter(task_by_name[name]["domain"] for name in selected_counts)),
        "selected_domain_row_counts": dict(Counter(row.domain for row in selected)),
        "candidate_complexity_counts": dict(Counter(task["complexity"] for task in metadata)),
        "selected_complexity_task_counts": dict(Counter(task_by_name[name]["complexity"] for name in selected_counts)),
        "selected_complexity_row_counts": dict(Counter(row.complexity for row in selected)),
        "all_turn_index_counts": dict(sorted(Counter(turn["turn"] for turn in all_turns).items())),
        "eligible_turn_index_counts": dict(sorted(Counter(turn["turn"] for turn in eligible_turns).items())),
        "selected_turn_index_counts": dict(sorted(Counter(row.turn_index for row in selected).items())),
        "all_native_prompt_tokens": quantiles([turn["native_prompt_tokens"] for turn in all_turns]),
        "all_rendered_b_prompt_tokens": quantiles(
            [turn["b_prompt_tokens"] for turn in all_turns if "b_prompt_tokens" in turn]
        ),
        "selected_native_prompt_tokens": quantiles([row.native_prompt_tokens for row in selected]),
        "selected_b_prompt_tokens": quantiles([row.prompt_tokens for row in selected]),
        "selected_response_tokens": quantiles([row.response_tokens for row in selected]),
        "selection": "one seeded latest-biased eligible turn per episode; four episodes per task; no outcome filter",
        "turn_salt": TURN_SALT,
        "max_prompt_tokens": PROMPT_CAP,
        "max_response_tokens": RESPONSE_CAP,
    }
    return selected, summary, items


def materialize_rows(planned: list[PlannedTurn], t1_tokenizer, b_tokenizer) -> list[dict]:
    """Render each selected turn's full recorded prefix without left cropping."""
    by_uri: dict[str, list[PlannedTurn]] = defaultdict(list)
    for turn in planned:
        by_uri[turn.result_uri].append(turn)
    rows = []
    for uri, choices in by_uri.items():
        filesystem, path = filesystem_for(uri)
        with filesystem.open(path) as handle:
            result = json.load(handle)
        verifier = result.get("verifier_result") or {}
        reward = (verifier.get("rewards") or {}).get("reward")
        if result["trial_name"] != choices[0].trial_name or reward != choices[0].reward:
            raise ValueError(f"selected trial or outcome changed: {uri}")
        detail = result["agent_result"]["rollout_details"][0]
        for choice in choices:
            prompt = detail["prompt_token_ids"][choice.turn_index]
            response = detail["completion_token_ids"][choice.turn_index]
            if not response or response[-1] != TURN_CLOSE or set(response) & INVALID_RESPONSE_IDS:
                raise ValueError(f"{choice.sample_id}: ineligible response in sealed result")
            messages = prompt_messages(t1_tokenizer, prompt)
            b_prompt = render_prompt_ids(b_tokenizer, "grug", messages)
            if len(b_prompt) != choice.prompt_tokens or len(response) != choice.response_tokens:
                raise ValueError(f"{choice.sample_id}: audited token lengths changed")
            if len(b_prompt) > PROMPT_CAP or len(response) > RESPONSE_CAP:
                raise ValueError(f"{choice.sample_id}: exceeds unchanged Phase 1a SFT caps")
            response_text = t1_tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not response_text.endswith("<|eot_id|>"):
                raise ValueError(f"{choice.sample_id}: recorded response has no turn close")
            replay = b_tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": response_text.removesuffix("<|eot_id|>")}],
                chat_template=shared_template(FAMILY_TOKEN_MAPS["grug"]),
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            replay_response = b_tokenizer.decode(
                replay["input_ids"][len(b_prompt) :],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if replay["input_ids"][: len(b_prompt)] != b_prompt or replay_response.rstrip("\n") != response_text:
                raise ValueError(f"{choice.sample_id}: B-template response text does not replay")
            rows.append(
                {
                    "sample_id": choice.sample_id,
                    "task_id": choice.task_name,
                    "data_source": "task_J",
                    "teacher_route": "T1",
                    "turn_index": choice.turn_index,
                    "prompt_tokens": len(b_prompt),
                    "response_tokens": len(response),
                    "outcome_label": outcome_label(choice.reward),
                    "reward": choice.reward,
                    "exception_type": choice.exception_type,
                    "input_ids": b_prompt + response,
                    "assistant_masks": [0] * len(b_prompt) + [1] * len(response),
                }
            )
    by_sample = {row["sample_id"]: row for row in rows}
    if len(by_sample) != len(rows):
        raise ValueError("duplicate selected terminal sample IDs")
    return [by_sample[turn.sample_id] for turn in planned]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", action="append", required=True)
    parser.add_argument("--task-metadata", required=True)
    parser.add_argument("--t1-model")
    parser.add_argument("--b-model")
    parser.add_argument("--out")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not args.plan_only and not all((args.t1_model, args.b_model, args.out)):
        parser.error("materialization needs --t1-model, --b-model and --out")
    uris = [*args.audit_root, args.task_metadata, *filter(None, (args.t1_model, args.b_model, args.out))]
    if any(not uri.startswith("s3://marin-rno2a/") for uri in uris):
        parser.error("all Phase 1b terminal data paths must stay in marin-rno2a")
    configure_coreweave_s3()
    metadata = read_jsonl(args.task_metadata)
    trials = [trial for root in args.audit_root for trial in read_jsonl(f"{root.rstrip('/')}/trials.jsonl")]
    selected, summary, items = plan_turns(trials, metadata)
    summary.update({"audit_roots": args.audit_root, "task_metadata": args.task_metadata})
    if args.plan_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if len(selected) != ROWS or summary["selected_tasks"] != TASKS:
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(f"need {TASKS} tasks and {ROWS} eligible terminal rows")
    t1_tokenizer = local_tokenizer(args.t1_model)
    b_tokenizer = local_tokenizer(args.b_model)
    rows = materialize_rows(selected, t1_tokenizer, b_tokenizer)
    selected_names = {row.task_name for row in selected}
    selected_tasks = [task for task in metadata if task["task_name"] in selected_names]
    write_jsonl(args.out, rows)
    write_jsonl(f"{args.out}.items.jsonl", items)
    write_jsonl(f"{args.out}.tasks.jsonl", selected_tasks)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    summary.update(
        {
            "out": args.out,
            "rows_sha256": hashlib.sha256(payload).hexdigest(),
            "per_item_audit_uri": f"{args.out}.items.jsonl",
            "selected_tasks_uri": f"{args.out}.tasks.jsonl",
            "t1_model": args.t1_model,
            "b_model": args.b_model,
        }
    )
    filesystem, path = filesystem_for(f"{args.out}.manifest.json")
    with filesystem.open(path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
