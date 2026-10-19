# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-trial generation metrics for a set of runs restricted to a task list, for the reasoning-effort pilot.

For each run prefix (a traces shard or an effort-pilot run holding ``rebuilt.jsonl`` and ``requests.jsonl``) it reads
the final attempt of every trial on the listed tasks and prints one JSON summary:

- pass rate, and clean-pass rate with the row builder's rule (passed, no trace problems, no harness exception, not
  summarized, last assistant turn is mini-swe-agent's submit or Terminus-2's ``"task_complete": true``);
- share of trials cut by the window: SWE-smith trials whose mini-swe-agent exit status is ``ContextLimit``, TMax
  trials Terminus-2 summarized (more than one conversation);
- GLM reasoning tokens per teacher turn (p50/p90/p99) from ``requests.jsonl`` usage;
- GLM completion tokens (reasoning plus reply, the supervised part of a row before v0 re-tokenization) per passing
  and per clean-passing trial;
- context tokens per trial: the largest prompt plus completion of any of its turns;
- wall time per trial (Harbor trial seconds).

    uv run python experiments/post_training/sft_shootout/effort_report.py --label max --family swesmith \
      --task-list s3://.../generation/effort-pilot-20261019/tasks-swesmith.json --runs s3://.../traces/swesmith/shard-*
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

SWE_SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TASK_COMPLETE = re.compile(r'"task_complete"\s*:\s*true')


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 1) if values else None,
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p99": percentile(values, 0.99),
    }


def share(values: list[int], test) -> float | None:
    return round(sum(map(test, values)) / len(values), 4) if values else None


def clean_completion(row: dict) -> bool:
    assistant = [m for m in row["messages"] if m["role"] == "assistant"]
    if not assistant:
        return False
    last = (assistant[-1].get("content") or "") + json.dumps(assistant[-1].get("tool_calls") or [])
    return SWE_SUBMIT in last if row["family"] == "swesmith" else bool(TASK_COMPLETE.search(last))


def swe_exit_status(run: str, trial_name: str) -> str | None:
    attempts = StoragePath(f"{run}/{trial_name}/attempts")
    final = sorted(str(p).rstrip("/") for p in attempts.ls()) if attempts.exists() else [f"{run}/{trial_name}"]
    path = StoragePath(f"{final[-1]}/agent/mini-swe-agent.trajectory.json")
    return json.loads(path.read_text()).get("info", {}).get("exit_status") if path.exists() else None


def read_run(run: str, tasks: set[str]) -> tuple[dict[str, dict], list[int], dict[str, int], int]:
    """({trial_id: trial}, reasoning tokens per teacher turn, {trial_id: context tokens}) for the listed tasks.

    rebuilt.jsonl holds every Harbor attempt of a trial (retries after a harness error); only the final attempt counts.
    """
    rows = [json.loads(line) for line in StoragePath(f"{run}/rebuilt.jsonl").read_text().splitlines()]
    final: dict[str, int] = {}
    for row in rows:
        final[row["trial_name"]] = max(final.get(row["trial_name"], -1), row.get("retry", 0))
    retried_attempts = len(
        {row["trial_id"] for row in rows if row["task_id"] in tasks and row.get("retry", 0) < final[row["trial_name"]]}
    )
    trials: dict[str, dict] = {}
    for row in rows:
        if row["task_id"] not in tasks or row.get("retry", 0) != final[row["trial_name"]]:
            continue
        trial = trials.setdefault(
            row["trial_id"],
            {
                "run": run,
                "family": row["family"],
                "task_id": row["task_id"],
                "trial_name": row["trial_name"],
                "passed": row["passed"],
                "exception": row.get("exception"),
                "n_conversations": row.get("n_conversations", 1),
                "seconds": row.get("trial_seconds"),
                "completion": 0,
                "turns": 0,
                "clean": False,
                "problems": False,
            },
        )
        trial["completion"] += sum(author["completion"] for author in row["tokens"].values())
        trial["turns"] += sum(author["turns"] for author in row["tokens"].values())
        trial["problems"] |= bool(row["problems"])
        if row["conversation"] == row.get("n_conversations", 1) - 1:
            trial["clean"] = clean_completion(row)
    reasoning, context = [], defaultdict(int)
    for line in StoragePath(f"{run}/requests.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("trial") not in trials or record.get("status") != 200 or not record.get("usage"):
            continue
        usage = record["usage"]
        reasoning.append((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        context[record["trial"]] = max(context[record["trial"]], usage["prompt_tokens"] + usage["completion_tokens"])
    return trials, reasoning, context, retried_attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument("--task-list", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--trials-out", help="Also write one JSON line per trial here.")
    args = parser.parse_args()

    tasks = set(json.loads(StoragePath(args.task_list).read_text())["tasks"])
    # "<dir>/*" stands for every run directory under <dir> that holds a rebuilt.jsonl.
    runs = []
    for run in args.runs:
        if run.endswith("/*"):
            runs += sorted(str(p).rstrip("/") for p in StoragePath(run[:-2]).ls() if (p / "rebuilt.jsonl").exists())
        else:
            runs.append(run.rstrip("/"))
    trials, reasoning, context, retried = {}, [], {}, 0
    with ThreadPoolExecutor(8) as pool:
        for run_trials, run_reasoning, run_context, run_retried in pool.map(lambda r: read_run(r, tasks), runs):
            retried += run_retried
            trials.update(run_trials)
            reasoning += run_reasoning
            context.update(run_context)
    if args.family == "swesmith":
        with ThreadPoolExecutor(32) as pool:
            statuses = pool.map(lambda t: swe_exit_status(t["run"], t["trial_name"]), trials.values())
            for trial, status in zip(trials.values(), statuses, strict=True):
                trial["cut"] = status == "ContextLimit"
    else:
        for trial in trials.values():
            trial["cut"] = trial["n_conversations"] > 1
    for trial in trials.values():
        trial["clean_pass"] = (
            trial["passed"]
            and trial["clean"]
            and not trial["problems"]
            and not trial["exception"]
            and trial["n_conversations"] == 1
        )
    values = list(trials.values())
    if args.trials_out:
        keys = ("task_id", "passed", "clean_pass", "cut", "completion", "turns", "seconds")
        lines = [
            json.dumps({"trial_id": k, **{key: t[key] for key in keys}, "context": context.get(k)})
            for k, t in trials.items()
        ]
        StoragePath(args.trials_out).write_text("\n".join(lines) + "\n")
    passing = [t for t in values if t["passed"]]
    clean = [t for t in values if t["clean_pass"]]
    print(
        json.dumps(
            {
                "label": args.label,
                "family": args.family,
                "runs": len(runs),
                "tasks": len({t["task_id"] for t in values}),
                "trials": len(values),
                "pass_rate": round(len(passing) / len(values), 4),
                "clean_pass_rate": round(len(clean) / len(values), 4),
                "clean_share_of_passes": round(len(clean) / len(passing), 4) if passing else None,
                "tasks_passed": len({t["task_id"] for t in passing}),
                "tasks_clean_passed": len({t["task_id"] for t in clean}),
                "cut_by_window": round(sum(t["cut"] for t in values) / len(values), 4),
                "exceptions": sum(bool(t["exception"]) for t in values),
                "retried_attempts": retried,
                "reasoning_per_turn": summary(reasoning),
                "zero_reasoning_turn_share": share(reasoning, lambda r: r == 0),
                "turns_over_2k_reasoning": share(reasoning, lambda r: r > 2000),
                "turns_over_8k_reasoning": share(reasoning, lambda r: r > 8000),
                "steps_per_passing_trial": summary([t["turns"] for t in passing]),
                "completion_per_passing_trial": summary([t["completion"] for t in passing]),
                "completion_per_clean_pass": summary([t["completion"] for t in clean]),
                "context_per_trial": summary([context[k] for k in trials if k in context]),
                "seconds_per_trial": summary([t["seconds"] for t in values if t["seconds"] is not None]),
            }
        )
    )


if __name__ == "__main__":
    main()
