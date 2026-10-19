# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task list for the next relay round of a shard.

A relay episode is a passing trial (its final Harbor attempt) with at least one GLM turn, as in
``pilot_report.py``. A follow-up round runs every non-held-out task of the shard that has fewer than
``--need`` episodes and fewer than ``--max-runs`` runs so far (one run = one trial, any round; a trial that
raised before any GLM turn, e.g. on proxy errors, is not a run). Reads each
earlier round's Harbor results and ``rebuilt.jsonl``, writes ``--out`` (a ``harbor_glm.py --task-list``)
with the remaining tasks and the per-task episode and run counts, and prints the number of remaining
tasks as the last line.

    python experiments/post_training/sft_shootout/relay_followup.py --shard <shard.json> \
      --rounds <round-1 prefix> [...] --need 4 --max-runs 5 --out <next round tasks.json>
"""

import argparse
import json
from collections import Counter, defaultdict

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout.rebuild_traces import harbor_results


def round_counts(prefix: str) -> tuple[Counter, Counter]:
    """(relay episodes, runs) per task in one round. A trial that raised and has no GLM turn is not a run."""
    final: dict[str, dict] = {}
    for trial_id, attempt in harbor_results(StoragePath(prefix)).items():
        current = final.get(attempt["trial_name"])
        if current is None or attempt["retry"] > current["retry"]:
            final[attempt["trial_name"]] = {**attempt, "trial_id": trial_id}
    glm_turns: dict[str, int] = defaultdict(int)
    for line in StoragePath(f"{prefix}/rebuilt.jsonl").read_text().splitlines():
        row = json.loads(line)
        glm_turns[row["trial_id"]] += row["tokens"].get("glm", {}).get("turns", 0)
    episodes = Counter(t["task_id"] for t in final.values() if t["passed"] and glm_turns[t["trial_id"]])
    runs = Counter(t["task_id"] for t in final.values() if not t["exception"] or glm_turns[t["trial_id"]])
    return episodes, runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", required=True, help="The shard's task list (shard_tasks.py output).")
    parser.add_argument("--rounds", nargs="+", required=True, help="Output prefixes of the earlier relay rounds.")
    parser.add_argument("--need", type=int, default=4, help="Relay episodes per task to stop at.")
    parser.add_argument("--max-runs", type=int, default=5, help="Relay runs per task, all rounds together.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    shard = json.loads(StoragePath(args.shard).read_text())
    episodes: Counter = Counter()
    runs: Counter = Counter()
    for prefix in args.rounds:
        round_episodes, round_runs = round_counts(prefix.rstrip("/"))
        episodes.update(round_episodes)
        runs.update(round_runs)
    heldout = set(shard["heldout"])
    remaining = [
        task
        for task in shard["tasks"]
        if task not in heldout and episodes[task] < args.need and runs[task] < args.max_runs
    ]
    record = {
        **{key: shard[key] for key in ("family", "tasks_jsonl_sha256", "package_revision", "shard", "shards")},
        "tasks": remaining,
        "heldout": [],
        "rounds": args.rounds,
        "need": args.need,
        "max_runs": args.max_runs,
        "episodes": {task: episodes[task] for task in shard["tasks"]},
        "runs": {task: runs[task] for task in shard["tasks"]},
    }
    StoragePath(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shard": shard["shard"],
                "tasks": len(shard["tasks"]),
                "episodes": sum(episodes.values()),
                "runs": sum(runs.values()),
            }
        )
    )
    print(len(remaining))


if __name__ == "__main__":
    main()
