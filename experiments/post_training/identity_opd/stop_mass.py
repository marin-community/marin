# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher stop mass on the dry run's tool family: does the teacher want a tool call where the student stopped?

Input: the dry run's retained step-0 rollouts. For every tool-family rollout that ended on ``<|eot_id|>`` without
a tool call, the teacher prompt is rebuilt exactly as in training (the student's prompt tokens with the identity
system message inserted by the run's rule), followed by the student's response up to its stop. Two continuations
are scored there with ``prompt_logprobs`` on an OpenAI-compatible endpoint serving v0.1:

- end of turn: ``<|eot_id|>``;
- tool-call opening: the tokens of ``<tool_call>\\n{"``, the start the chat template renders for every call.

Each continuation's log-prob is the sum over its tokens. Per sample, and as medians, the report gives both
log-probs and probabilities, and the tool family's no-tool-call fraction. The plan's flag: a student tool-call
rate below 5% with a median teacher end-of-turn probability above 0.2.
"""

import argparse
import json
import math
from collections.abc import Callable

import numpy as np
import requests
from marinskyrl.distillation import TeacherPromptInsertion, conditioned_teacher_prompt
from transformers import AutoTokenizer

from experiments.post_training.identity_opd.dry_run_report import END_OF_TURN, TOOL_CALL, _write_new, trajectory_records
from experiments.post_training.identity_opd.teacher_insertion import insertion_rule

TOOL_OPEN = '<tool_call>\n{"'
LOW_CALL_RATE = 0.05
HIGH_STOP_PROBABILITY = 0.2

Scorer = Callable[[list[int]], list[float | None]]


def endpoint_scorer(base_url: str, model: str) -> Scorer:
    """Log-prob of each prompt token given the ones before it, from ``/v1/completions`` with ``prompt_logprobs``."""

    def score(token_ids: list[int]) -> list[float | None]:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json={"model": model, "prompt": token_ids, "max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
            timeout=600,
        )
        response.raise_for_status()
        entries = response.json()["choices"][0]["prompt_logprobs"]
        if len(entries) != len(token_ids):
            raise ValueError(f"prompt_logprobs covers {len(entries)} of {len(token_ids)} tokens")
        return [
            None if entry is None else entry[str(token)]["logprob"]
            for token, entry in zip(token_ids, entries, strict=True)
        ]

    return score


def stopped_without_call(record: dict) -> bool:
    response = record["response"]
    ids = response["token_ids"]
    return bool(ids) and ids[-1] == END_OF_TURN and TOOL_CALL not in (response["text"] or "")


def continuation_logprob(scorer: Scorer, context: list[int], continuation: list[int]) -> float:
    values = scorer(context + continuation)[len(context) :]
    if any(value is None for value in values):
        raise ValueError("the endpoint returned no log-prob for a continuation token")
    return float(sum(values))


def stop_mass(records: list[dict], rule: TeacherPromptInsertion, tool_open: list[int], scorer: Scorer) -> dict:
    tool = [r for r in records if r["trajectory"]["environment_extras"]["teacher_route"].startswith("tool_")]
    if not tool:
        raise ValueError("no tool-family rollouts at this step")
    samples = []
    for record in tool:
        if not stopped_without_call(record):
            continue
        prompt = conditioned_teacher_prompt(list(record["prompt"]["token_ids"]), (rule,))
        context = prompt + list(record["response"]["token_ids"][:-1])
        end = continuation_logprob(scorer, context, [END_OF_TURN])
        call = continuation_logprob(scorer, context, tool_open)
        samples.append(
            {
                "instance_id": record["trajectory"].get("instance_id"),
                "route": record["trajectory"]["environment_extras"]["teacher_route"],
                "response_tokens": len(record["response"]["token_ids"]),
                "end_of_turn_logprob": end,
                "tool_open_logprob": call,
                "end_of_turn_probability": math.exp(end),
                "tool_open_probability": math.exp(call),
            }
        )
    no_call = len(samples) / len(tool)
    medians = {
        key: float(np.median([sample[key] for sample in samples])) if samples else None
        for key in ("end_of_turn_probability", "tool_open_probability", "end_of_turn_logprob", "tool_open_logprob")
    }
    flagged = (
        1 - no_call < LOW_CALL_RATE
        and medians["end_of_turn_probability"] is not None
        and medians["end_of_turn_probability"] > HIGH_STOP_PROBABILITY
    )
    return {
        "tool_samples": len(tool),
        "stopped_without_call": len(samples),
        "no_tool_call_fraction": no_call,
        "tool_open_tokens": tool_open,
        "medians": medians,
        "flag": flagged,
        "flag_rule": f"tool-call rate < {LOW_CALL_RATE} and median end-of-turn probability > {HIGH_STOP_PROBABILITY}",
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", required=True, help="the dry run's retained-trajectory root")
    parser.add_argument("--step", type=int, default=1, help="global step of the rollouts (1 = step 0)")
    parser.add_argument("--tokenizer", required=True, help="v0.1 adc0078a tokenizer directory")
    parser.add_argument("--endpoint", required=True, help="base URL of the v0.1 serve")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--out", required=True, help="new report file, local or S3; never overwritten")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tool_open = tokenizer.encode(TOOL_OPEN, add_special_tokens=False)
    result = stop_mass(
        trajectory_records(args.trajectories, args.step),
        insertion_rule(tokenizer),
        tool_open,
        endpoint_scorer(args.endpoint, args.model),
    )
    digest = _write_new(args.out, result)
    summary = {key: result[key] for key in ("tool_samples", "stopped_without_call", "medians", "flag")}
    print(json.dumps({"out": args.out, "sha256": digest, **summary}))


if __name__ == "__main__":
    main()
