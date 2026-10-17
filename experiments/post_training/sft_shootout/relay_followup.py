# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task list for the next relay round of a shard: tasks that still lack ``--need`` relay episodes.

A relay episode is a passing trial (its final Harbor attempt) with at least one GLM turn, as in
``pilot_report.py``. Reads each earlier round's Harbor results and ``rebuilt.jsonl``, writes ``--out`` (a
``harbor_glm.py --task-list``) with the remaining tasks and the per-task episode counts, and prints the
number of remaining tasks as the last line.

    python experiments/post_training/sft_shootout/relay_followup.py --shard <shard.json> \
      --rounds <round-1 prefix> [...] --need 2 --out <round-2 tasks.json>
"""

import argparse
import json
from collections import Counter, defaultdict

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout.rebuild_traces import harbor_results


def episodes(prefix: str) -> Counter:
    """Relay episodes per task in one round."""
    final: dict[str, dict] = {}
    for trial_id, attempt in harbor_results(StoragePath(prefix)).items():
        current = final.get(attempt["trial_name"])
        if current is None or attempt["retry"] > current["retry"]:
            final[attempt["trial_name"]] = {**attempt, "trial_id": trial_id}
    glm_turns: dict[str, int] = defaultdict(int)
    for line in StoragePath(f"{prefix}/rebuilt.jsonl").read_text().splitlines():
        row = json.loads(line)
        glm_turns[row["trial_id"]] += row["tokens"].get("glm", {}).get("turns", 0)
    return Counter(t["task_id"] for t in final.values() if t["passed"] and glm_turns[t["trial_id"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", required=True, help="The shard's task list (shard_tasks.py output).")
    parser.add_argument("--rounds", nargs="+", required=True, help="Output prefixes of the earlier relay rounds.")
    parser.add_argument("--need", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    shard = json.loads(StoragePath(args.shard).read_text())
    counts: Counter = Counter()
    for prefix in args.rounds:
        counts.update(episodes(prefix.rstrip("/")))
    remaining = [task for task in shard["tasks"] if counts[task] < args.need]
    record = {
        **{key: shard[key] for key in ("family", "tasks_jsonl_sha256", "package_revision", "shard", "shards")},
        "tasks": remaining,
        "heldout": [task for task in shard["heldout"] if task in remaining],
        "rounds": args.rounds,
        "need": args.need,
        "episodes": {task: counts[task] for task in shard["tasks"]},
    }
    StoragePath(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"shard": shard["shard"], "tasks": len(shard["tasks"]), "with_enough": len(shard["tasks"]) - len(remaining)}))
    print(len(remaining))


if __name__ == "__main__":
    main()
