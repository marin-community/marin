# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the test-exposed CruxEval teacher pool without changing raw generations."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import click
from transformers import AutoTokenizer

from experiments.post_training.cruxeval_transfer.constants import (
    BATCH_SIZE,
    DATA_SHA256,
    STUDENT_MODEL,
    STUDENT_REVISION,
)
from experiments.post_training.cruxeval_transfer.rendering import END_TURN_ID, token_counts, training_row


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_rows(requests: list[dict], responses: list[dict], grades: list[dict]) -> list[dict]:
    """Select one correct, completed, reasoning-bearing response for each request."""
    request_ids = {row["custom_id"] for row in requests}
    responses_by_id = {row["custom_id"]: row for row in responses}
    grades_by_id = {row["custom_id"]: row for row in grades}
    if (
        len(request_ids) != len(requests)
        or len(responses_by_id) != len(responses)
        or len(grades_by_id) != len(grades)
        or request_ids != set(responses_by_id)
        or request_ids != set(grades_by_id)
    ):
        raise ValueError("Requests, responses and grades must match one-to-one")
    rows = []
    for request in requests:
        identity = request["custom_id"]
        grade = grades_by_id[identity]
        if grade["task"] != request["task"] or grade["source_id"] != request["source_id"]:
            raise ValueError(f"Grade identity mismatch: {identity}")
        if not grade["correct"]:
            continue
        choice = responses_by_id[identity]["response"]["body"]["choices"][0]
        message = choice["message"]
        reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
        if choice["finish_reason"] != "stop" or not reasoning:
            continue
        row = training_row(
            {
                "id": identity,
                "problem_id": request["source_id"],
                "messages": (
                    request["b_request"]["messages"]
                    + [
                        {
                            "role": "assistant",
                            "reasoning_content": reasoning,
                            "content": message["content"],
                        }
                    ]
                ),
            }
        )
        rows.append(row | {"task": request["task"]})
    return rows


@click.command()
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--grades", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--max-sequence-tokens", type=click.IntRange(1, 32768), required=True)
def main(source: Path, grades: Path, output: Path, max_sequence_tokens: int) -> None:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["source_sha256"] != DATA_SHA256:
        raise ValueError("Unexpected CruxEval source")
    if output.exists():
        raise ValueError("Use a new output directory; existing artifacts are immutable")
    requests = read_rows(source / "requests.jsonl")
    if Counter(row["task"] for row in requests) != {"input": 800, "output": 800}:
        raise ValueError("Expected all 1600 CruxEval requests")
    rows = selected_rows(requests, read_rows(source / "raw-responses.jsonl"), read_rows(grades))
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION)
    audits = []
    probes = []
    prebuilt = []
    for row in rows:
        total, supervised = token_counts(row, tokenizer)
        if total > max_sequence_tokens:
            raise ValueError(f"Selected trace exceeds context; no silent truncation: {row['id']} ({total})")
        audits.append({"id": row["id"], "task": row["task"], "tokens": total, "supervised_tokens": supervised})
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **row["chat_template_kwargs"],
        )
        padding = max_sequence_tokens - total
        prebuilt.append(
            {
                "input_ids": encoded["input_ids"] + [END_TURN_ID] * padding,
                "loss_weights": encoded["assistant_masks"][1:] + [0] * (padding + 1),
            }
        )
        if sum(probe["task"] == row["task"] for probe in probes) < 2:
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **row["chat_template_kwargs"],
            )
            rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, **row["chat_template_kwargs"])
            probes.append(
                row | {"rendered": rendered, "input_ids": encoded["input_ids"], "loss_mask": encoded["assistant_masks"]}
            )
    output.mkdir(parents=True)
    padding_rows = -len(rows) % BATCH_SIZE
    empty = {"input_ids": [END_TURN_ID] * max_sequence_tokens, "loss_weights": [0] * max_sequence_tokens}
    with gzip.open(output / "train-prebuilt.jsonl.gz", "wt") as stream:
        for row in prebuilt + [empty] * padding_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    for name, values in (("train.jsonl", rows), ("token-audit.jsonl", audits), ("row-probes.jsonl", probes)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    (output / "chat_template.jinja").write_text(tokenizer.get_chat_template())
    report = {
        "test_exposed": True,
        "source_sha256": DATA_SHA256,
        "student": STUDENT_MODEL,
        "student_revision": STUDENT_REVISION,
        "selection": "correct execution grade, finish_reason stop, nonempty reasoning; one per problem/direction",
        "training_rows": len(rows),
        "selected_by_task": dict(Counter(row["task"] for row in rows)),
        "batch_size": BATCH_SIZE,
        "padding_rows": padding_rows,
        "steps_per_epoch": (len(rows) + padding_rows) // BATCH_SIZE,
        "evaluation_denominators": {"input": 800, "output": 800},
        "max_sequence_tokens": max_sequence_tokens,
        "longest_row": max(row["tokens"] for row in audits),
        "input_tokens": sum(row["tokens"] for row in audits),
        "supervised_tokens": sum(row["supervised_tokens"] for row in audits),
        "source_hashes": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("requests.jsonl", "raw-responses.jsonl", "manifest.json")
        },
        "grades_sha256": hashlib.sha256(grades.read_bytes()).hexdigest(),
        "output_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
