# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an architecture's probe continuations on P_X, P_R and P_think.

Each continuation's sampled reply is decoded, and the answer text is read out of it with
:func:`~experiments.post_training.task_data.terminus_answers.answer_text`: flattened from a
Terminus-2 turn when the reply is one, the plain reply otherwise. The X and R graders then
run unchanged on that text (decision log 2026-10-24). Per set the report counts the
reply formats, the replies whose grader extractor finds a candidate answer (a ``\\boxed``
for math and X output tasks, a valid ``f(...)`` call for X input tasks), and the
graded-correct ones.

    python -m experiments.post_training.probe_sets.grade_continuations \\
        --continuations s3://.../continuations/snowball-B/xr --set P_X=s3://.../x_probe.jsonl \\
        --set P_R=s3://.../r_probe.jsonl --out s3://.../grades-xr.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import fsspec
from verifyit.modes.extract import extract_boxed

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS
from experiments.post_training.probe_sets.buckets import decode_token_bytes
from experiments.post_training.probe_sets.continuations import local_tokenizer, read_jsonl
from experiments.post_training.task_data.code_x import DIRECTION_INPUT, extract_input_call
from experiments.post_training.task_data.expert_outputs import GRADERS, bounded_grade
from experiments.post_training.task_data.terminus_answers import AnswerFormat, answer_text

# Which grader reads each probe set's answers.
SET_TASK = {"P_X": "X", "P_R": "R", "P_think": "R"}


def candidate_found(task_name: str, task: dict[str, Any], text: str) -> bool:
    """Whether the grader's own extractor finds an answer to grade in ``text``."""
    if task_name == "X" and task["direction"] == DIRECTION_INPUT:
        return extract_input_call(text, answer=str(task["answer"])) is not None
    return extract_boxed(text) is not None


def grade_reply(task_name: str, task: dict[str, Any], reply: str, think_close: str) -> dict[str, Any]:
    answer = answer_text(reply, think_close)
    return {
        "format": answer.format.value,
        "candidate_found": candidate_found(task_name, task, answer.text),
        "graded_correct": bool(bounded_grade(GRADERS[task_name], task, answer.text)),
    }


def set_report(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    formats = Counter(verdict["format"] for verdict in verdicts)
    found = Counter(verdict["format"] for verdict in verdicts if verdict["candidate_found"])
    correct = Counter(verdict["format"] for verdict in verdicts if verdict["graded_correct"])
    return {
        "rows": len(verdicts),
        "agent_format_share": formats[AnswerFormat.TERMINUS] / len(verdicts),
        "by_format": {
            str(kind): {"rows": formats[kind], "candidate_found": found[kind], "graded_correct": correct[kind]}
            for kind in AnswerFormat
        },
        "candidate_found_share": sum(found.values()) / len(verdicts),
        "graded_correct_share": sum(correct.values()) / len(verdicts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--continuations", required=True, help="a continuation directory with manifest.json")
    parser.add_argument("--set", dest="sets", action="append", required=True, help="SET=tasks jsonl (task_id rows)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    directory = args.continuations.rstrip("/")
    with fsspec.open(f"{directory}/manifest.json") as handle:
        manifest = json.load(handle)
    tokenizer = local_tokenizer(manifest["checkpoint"])
    tokens = FAMILY_TOKEN_MAPS[manifest["family"]]
    close = tokenizer.convert_tokens_to_ids(tokens.turn_close)
    report: dict[str, Any] = {"continuations": directory, "sets": {}}
    for spec in args.sets:
        name, tasks_path = spec.split("=", 1)
        id_key = "task_id" if name != "P_think" else "probe_id"
        tasks = {row[id_key]: row for row in read_jsonl(tasks_path)}
        verdicts = []
        for raw in read_jsonl(f"{directory}/{name}.raw.jsonl"):
            ids = raw["response_ids"]
            reply = decode_token_bytes(tokenizer, ids[:-1] if ids and ids[-1] == close else ids)
            verdicts.append(
                {
                    "probe_id": raw["probe_id"],
                    **grade_reply(SET_TASK[name], tasks[raw["probe_id"]], reply, tokens.think_close),
                }
            )
        report["sets"][name] = set_report(verdicts)
        with fsspec.open(f"{args.out.removesuffix('.json')}-{name}-verdicts.jsonl", "w") as handle:
            handle.writelines(json.dumps(verdict) + "\n" for verdict in verdicts)
        print(name, json.dumps(report["sets"][name]))
    with fsspec.open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
