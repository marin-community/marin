# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw the #207 pass-rate pilot sample from a family's pool of Harbor task folders.

TMax is sampled uniformly; SWE-smith is stratified by repository (proportional allocation, largest
remainder). Held-out task ids are excluded first. The draw is deterministic for a given seed and pool.
With ``--copy-to`` the sampled folders are copied there, ready for ``harbor_glm.py --tasks``.

    python -m experiments.post_training.sft_shootout.pilot_sample --family tmax \
      --pool <dir of task folders> --exclude <held-out ids file> --n 200 \
      --output pilot-tmax.json --copy-to experiments/post_training/sft_shootout/pilot_tasks/tmax
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 20261016
SWESMITH_IMAGE = re.compile(r"^FROM jyangballin/swesmith\.x86_64\.(\S+)\.[0-9a-f]{8}\s*$", re.MULTILINE)


def swesmith_repo(task: Path) -> str:
    """The repository of a SWE-smith package, from its base image (e.g. ``oauthlib_1776_oauthlib``)."""
    match = SWESMITH_IMAGE.search((task / "environment" / "Dockerfile").read_text())
    if match is None:
        raise ValueError(f"{task.name}: no SWE-smith base image in its Dockerfile")
    return match.group(1)


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
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, help="File of held-out task ids, one per line or a JSON list.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    tasks = sorted(path.name for path in args.pool.iterdir() if (path / "task.toml").exists())
    excluded: set[str] = set()
    if args.exclude:
        text = args.exclude.read_text()
        excluded = set(json.loads(text)) if text.lstrip().startswith("[") else set(text.split())
    eligible = [task for task in tasks if task not in excluded]
    rng = random.Random(args.seed)
    if args.family == "tmax":
        sample = sorted(rng.sample(eligible, args.n))
        strata = None
    else:
        groups: dict[str, list[str]] = defaultdict(list)
        for task in eligible:
            groups[swesmith_repo(args.pool / task)].append(task)
        sample = stratified(rng, groups, args.n)
        strata = {key: sum(1 for task in sample if task in set(members)) for key, members in sorted(groups.items())}
    record = {
        "family": args.family,
        "pool": str(args.pool),
        "pool_tasks": len(tasks),
        "excluded_file": str(args.exclude) if args.exclude else None,
        "excluded_in_pool": len(set(tasks) & excluded),
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
