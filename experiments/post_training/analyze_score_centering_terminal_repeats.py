# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the two terminal evaluations of the GSM8K and Math500 pool subset.

SkyRL evaluates at the last step and again during finalization, overwriting the
same response dump. The first result remains in the step-40 train mirror; the
second is in the final eval mirror. GSM8K rewards are 0/1 and Math500 rewards
are -1/1, so the completed counts can be recovered from their signed metrics.
The final reconstructed counts must match the saved response analysis CSV.
When an Iris pod log is unavailable, pass retained W&B evaluation history as
JSONL with the run label, W&B run ID, history step, optimizer step, and metrics.
The legacy `*_all` output columns sum only GSM8K and Math500. When the pool
contains other validation suites, they must be compared with the `core-math`
response row rather than the heterogeneous `all` row.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

GSM8K_QUESTIONS = 256
MATH500_QUESTIONS = 500

FIELDS = (
    "run",
    "scheduled_gsm8k",
    "scheduled_math500",
    "scheduled_all",
    "final_gsm8k",
    "final_math500",
    "final_all",
    "final_minus_scheduled",
)


def _integer(value: float, label: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-4:
        raise ValueError(f"{label} is not an integer count: {value}")
    return rounded


def _completed_counts(metrics: dict) -> tuple[int, int]:
    gsm = _integer(metrics["eval/val-gsm8k/completed_stop_score_contribution"] * GSM8K_QUESTIONS, "GSM8K correct")
    math_completed = _integer(
        metrics["eval/val-math500/completed_stop_fraction"] * MATH500_QUESTIONS, "Math500 complete"
    )
    math_signed = _integer(
        metrics["eval/val-math500/completed_stop_score_contribution"] * MATH500_QUESTIONS, "Math500 score"
    )
    if (math_completed + math_signed) % 2:
        raise ValueError("Math500 completed count and signed score have different parity")
    math = (math_completed + math_signed) // 2
    if not 0 <= gsm <= GSM8K_QUESTIONS or not 0 <= math <= math_completed <= MATH500_QUESTIONS:
        raise ValueError("derived completed-correct count is out of range")
    return gsm, math


def _mirrors(path: Path, step: int) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        if "WANDB_MIRROR kind=" not in line or f" step={step} metrics=" not in line:
            continue
        marker = line.split("WANDB_MIRROR kind=", 1)[1]
        kind, remainder = marker.split(" step=", 1)
        logged_step, payload = remainder.split(" metrics=", 1)
        if int(logged_step) == step and kind in {"train", "eval"}:
            selected[kind] = json.loads(payload)
    if set(selected) != {"train", "eval"}:
        raise ValueError(f"{path}: expected scheduled train and final eval mirrors for step {step}")
    return selected


def _wandb_history_mirrors(path: Path, run: str, step: int) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["run"] == run]
    run_ids = {row["wandb_run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError(f"{run}: expected one W&B run identity in {path}")
    scheduled = [row for row in rows if row.get("trainer/global_step") == step]
    if len(scheduled) != 1:
        raise ValueError(f"{run}: expected one scheduled step-{step} W&B evaluation")
    final = [row for row in rows if row["_step"] == scheduled[0]["_step"] + 1 and row.get("trainer/global_step") is None]
    if len(final) != 1:
        raise ValueError(f"{run}: expected one final W&B evaluation after scheduled step {step}")
    return {"train": scheduled[0], "eval": final[0]}


def summarize(run: str, log: Path, evaluations: list[dict[str, str]], step: int, *, wandb_history: bool = False) -> dict:
    rows = {row["dataset"]: row for row in evaluations if row["run"] == run and int(row["step"]) == step}
    if not {"all", "val-gsm8k", "val-math500"} <= set(rows):
        raise ValueError(f"{run}: missing final response analysis for step {step}")
    if [int(rows[dataset]["questions"]) for dataset in ("val-gsm8k", "val-math500")] != [
        GSM8K_QUESTIONS,
        MATH500_QUESTIONS,
    ]:
        raise ValueError(f"{run}: frozen evaluation suite sizes changed")
    mirrors = _wandb_history_mirrors(log, run, step) if wandb_history else _mirrors(log, step)
    scheduled_gsm, scheduled_math = _completed_counts(mirrors["train"])
    final_gsm, final_math = _completed_counts(mirrors["eval"])
    final_all = final_gsm + final_math
    combined = rows["core-math"] if "core-math" in rows else rows["all"]
    if int(combined["questions"]) != GSM8K_QUESTIONS + MATH500_QUESTIONS:
        raise ValueError(f"{run}: combined math evaluation size changed")
    if [final_gsm, final_math, final_all] != [
        int(rows[dataset]["completed_correct"]) for dataset in ("val-gsm8k", "val-math500")
    ] + [int(combined["completed_correct"])]:
        raise ValueError(f"{run}: final mirror disagrees with saved responses")
    return {
        "run": run,
        "scheduled_gsm8k": scheduled_gsm,
        "scheduled_math500": scheduled_math,
        "scheduled_all": scheduled_gsm + scheduled_math,
        "final_gsm8k": final_gsm,
        "final_math500": final_math,
        "final_all": final_all,
        "final_minus_scheduled": final_all - scheduled_gsm - scheduled_math,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", action="append", required=True, type=Path)
    parser.add_argument("--iris-log", action="append", default=[], metavar="RUN=PATH")
    parser.add_argument("--wandb-history", action="append", default=[], metavar="RUN=JSONL")
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluations = []
    seen_evaluations: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in args.evaluations:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row["run"], row["step"], row["dataset"])
                if key in seen_evaluations and seen_evaluations[key] != row:
                    raise ValueError(f"conflicting evaluation row for {key}")
                seen_evaluations[key] = row
                evaluations.append(row)
    result = []
    labels: set[str] = set()
    for source, items in (("--iris-log", args.iris_log), ("--wandb-history", args.wandb_history)):
        for item in items:
            run, separator, path = item.partition("=")
            if not separator or not run or not path or run in labels:
                parser.error(f"invalid {source} {item!r}; expected RUN=PATH")
            labels.add(run)
            result.append(summarize(run, Path(path), evaluations, args.step, wandb_history=source == "--wandb-history"))
    if not result:
        parser.error("pass at least one --iris-log or --wandb-history")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} paired evaluation records to {args.output}")


if __name__ == "__main__":
    main()
