# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pass-rate pilot numbers for one run: Harbor outcomes plus token use from ``rebuilt.jsonl``.

Per trial (a Harbor trial's final attempt): pass, no reward (no verifier result: an infrastructure
or agent error before grading), exception type, wall time. Per task: pass@1 (mean over its trials)
and pass@k (any of its k trials passed). Per trial attempt's model calls (from ``rebuilt.jsonl``):
sequence length (the last call's prompt + completion tokens, per conversation), assistant tokens
(completion, reasoning included), and trials with more than one conversation (Terminus-2 context
summarization). With ``--manifest`` (a pool ``tasks.jsonl``), also pass@1 / pass@k per repo. Writes
``pilot_report.json`` next to the run and prints it.

    uv run python experiments/post_training/sft_shootout/pilot_report.py <run prefix> [--manifest <tasks.jsonl>]
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout.rebuild_traces import harbor_results

WINDOW = 32768


def spread(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
    }


def relay_stats(rows: list[dict], final: dict[str, dict]) -> dict:
    """Trigger frequencies, author shares and episode classes for a relay run.

    A relay episode (training data) passed and has at least one GLM turn; a v0-solo pass has none.
    """
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_trial[row["trial_id"]].append(row)
    passed = {t["trial_id"]: t["passed"] for t in final.values()}
    triggers = Counter()
    first_trigger = Counter()
    tokens = defaultdict(lambda: Counter())
    classes = Counter()
    for trial_id, trial_rows in by_trial.items():
        trial_triggers = [trigger for row in trial_rows for trigger in row["triggers"]]
        triggers.update(set(trial_triggers))
        first_trigger[trial_triggers[0] if trial_triggers else "none"] += 1
        glm_turns = 0
        for row in trial_rows:
            for author, counts in row["tokens"].items():
                tokens[author].update(counts)
            glm_turns += row["tokens"].get("glm", {}).get("turns", 0)
        outcome = "pass" if passed.get(trial_id) else "fail"
        classes[f"{outcome}_{'with_glm' if glm_turns else 'v0_solo'}"] += 1
    total = {key: sum(t[key] for t in tokens.values()) for key in ("turns", "completion", "prompt")}
    return {
        "trials": len(by_trial),
        "trials_with_trigger": {k: v for k, v in sorted(triggers.items())},
        "first_trigger": dict(first_trigger),
        "episodes": dict(classes),
        "v0_share": {key: round(tokens["v0"][key] / total[key], 4) if total[key] else None for key in total},
        "tokens_by_author": {author: dict(counts) for author, counts in tokens.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix")
    parser.add_argument("--manifest", help="Pool tasks.jsonl, for a per-repo breakdown.")
    args = parser.parse_args()
    prefix = args.prefix.rstrip("/")
    attempts = harbor_results(StoragePath(prefix))
    final: dict[str, dict] = {}
    for trial_id, attempt in attempts.items():
        current = final.get(attempt["trial_name"])
        if current is None or attempt["retry"] > current["retry"]:
            final[attempt["trial_name"]] = {**attempt, "trial_id": trial_id}
    trials = list(final.values())

    by_task: dict[str, list[bool]] = defaultdict(list)
    for trial in trials:
        by_task[trial["task_id"]].append(trial["passed"])
    graded = [t for t in trials if t["reward"] is not None]

    rows = [json.loads(line) for line in StoragePath(f"{prefix}/rebuilt.jsonl").read_text().splitlines()]
    final_ids = {t["trial_id"] for t in trials}
    rows = [row for row in rows if row["trial_id"] in final_ids]
    lengths, assistant, reasoning, calls = [], [], [], []
    conversations = Counter(row["trial_id"] for row in rows)
    for row in rows:
        usages = [m["usage"] for m in row["messages"] if m.get("role") == "assistant" and m.get("usage")]
        if usages:
            lengths.append(usages[-1]["prompt_tokens"] + usages[-1]["completion_tokens"])
    for trial_id in conversations:
        trial_rows = [row for row in rows if row["trial_id"] == trial_id]
        totals = [author for row in trial_rows for author in row["tokens"].values()]
        assistant.append(sum(t["completion"] for t in totals))
        reasoning.append(sum(t["reasoning"] for t in totals))
        calls.append(sum(t["turns"] for t in totals))

    report = {
        "run": prefix,
        "tasks": len(by_task),
        "trials": len(trials),
        "harbor_retries": sum(1 for a in attempts.values() if a["retry"] > 0),
        "pass@1": round(statistics.fmean(statistics.fmean(v) for v in by_task.values()), 4),
        "pass@1_graded_only": round(statistics.fmean(t["passed"] for t in graded), 4) if graded else None,
        "pass@k": round(statistics.fmean(any(v) for v in by_task.values()), 4),
        "k": dict(Counter(len(v) for v in by_task.values())),
        "no_reward_rate": round(1 - len(graded) / len(trials), 4),
        "exceptions": dict(Counter(t["exception"] for t in trials if t["exception"])),
        "trial_seconds": spread([t["trial_seconds"] for t in trials if t["trial_seconds"] is not None]),
        "agent_seconds": spread([t["agent_seconds"] for t in trials if t["agent_seconds"] is not None]),
        "sequence_tokens": spread(lengths),
        f"conversations_over_{WINDOW}": round(sum(n > WINDOW for n in lengths) / len(lengths), 4) if lengths else None,
        "assistant_tokens_per_trial": spread(assistant),
        "reasoning_tokens_per_trial": spread(reasoning),
        "model_calls_per_trial": spread(calls),
        "trials_with_model_calls": len(conversations),
        "trials_summarized": sum(1 for n in conversations.values() if n > 1),
        "dropped_replies": sum(len(row.get("dropped_replies", [])) for row in rows),
        "trials_with_dropped_replies": len({row["trial_id"] for row in rows if row.get("dropped_replies")}),
    }
    if any("v0" in row["tokens"] for row in rows):
        report["relay"] = relay_stats(rows, final)
    if args.manifest:
        repos = {
            row["task_id"]: row["repo"]
            for row in map(json.loads, StoragePath(args.manifest).read_text().splitlines())
        }
        per_repo: dict[str, list[list[bool]]] = defaultdict(list)
        for task, passes in by_task.items():
            per_repo[repos.get(task) or "unknown"].append(passes)
        report["by_repo"] = {
            repo: {
                "tasks": len(tasks),
                "pass@1": round(statistics.fmean(statistics.fmean(v) for v in tasks), 4),
                "pass@k": round(statistics.fmean(any(v) for v in tasks), 4),
            }
            for repo, tasks in sorted(per_repo.items())
        }
    StoragePath(f"{prefix}/pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
