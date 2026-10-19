# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw the #207 pass-rate pilot sample from a family's pool of Harbor task folders.

Eligible tasks are the family's non-held-out rows of the #206 pool manifest (``pool/tasks.jsonl``).
TMax is sampled uniformly; SWE-smith is stratified by the manifest's ``repo`` (proportional allocation,
largest remainder). The draw is deterministic for a given seed and manifest. With ``--copy-to`` the
sampled folders are copied from ``--pool`` there, ready for ``harbor_glm.py --tasks``.

    python -m experiments.post_training.sft_shootout.pilot_sample --family tmax \
      --manifest pool/tasks.jsonl --pool <dir of task folders> --n 200 \
      --output pilot-tmax.json --copy-to experiments/post_training/sft_shootout/pilot_tasks/tmax
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 20261016


def stratified(rng: random.Random, groups: dict[str, list[str]], n: int) -> list[str]:
    total = sum(len(members) for members in groups.values())
    quotas = {key: n * len(members) / total for key, members in groups.items()}
    counts = {key: int(quota) for key, quota in quotas.items()}
    by_remainder = sorted(groups, key=lambda key: (counts[key] - quotas[key], key))
    for key in by_remainder[: n - sum(counts.values())]:
        counts[key] += 1
    return sorted(task for key in sorted(groups) for task in rng.sample(sorted(groups[key]), counts[key]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="The #206 pool tasks.jsonl.")
    parser.add_argument("--pool", type=Path, help="Directory of the pool's task folders, for --copy-to.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    rows = [row for row in rows if row["family"] == args.family]
    eligible = sorted(row["task_id"] for row in rows if not row["heldout"])
    rng = random.Random(args.seed)
    if args.family == "tmax":
        sample = sorted(rng.sample(eligible, args.n))
        strata = None
    else:
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if not row["heldout"]:
                groups[row["repo"]].append(row["task_id"])
        sample = stratified(rng, groups, args.n)
        strata = {key: sum(1 for task in sample if task in set(members)) for key, members in sorted(groups.items())}
    record = {
        "family": args.family,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "package_revision": sorted({row["package_revision"] for row in rows}),
        "pool_tasks": len(rows),
        "heldout": sum(1 for row in rows if row["heldout"]),
        "eligible": len(eligible),
        "seed": args.seed,
        "n": args.n,
        "strata": strata,
        "tasks": sample,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("tasks", "strata")}))
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for task in sample:
            shutil.copytree(args.pool / task, args.copy_to / task)


if __name__ == "__main__":
    main()
