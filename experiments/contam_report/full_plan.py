# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choose the files the full scan reads (Gate 5): every normalized file not in the pilot.

Usage: full_plan.py SOURCES_JSON PILOT_PLAN_JSON OUT_PLAN_JSON
"""

import json
import sys


def main() -> None:
    with open(sys.argv[1]) as f:
        sources = json.load(f)
    with open(sys.argv[2]) as f:
        pilot = {p for paths in json.load(f).values() for p in paths}
    plan = {}
    for name, info in sorted(sources.items()):
        paths = [p for p, _ in info["files"] if p not in pilot]
        if paths:
            plan[name] = paths
    sizes = {p: s for info in sources.values() for p, s in info["files"]}
    chosen = [p for paths in plan.values() for p in paths]
    print(f"{len(chosen)} files from {len(plan)} sources, {sum(sizes[p] for p in chosen) / 1e12:.2f} TB")
    with open(sys.argv[3], "w") as f:
        json.dump(plan, f, indent=1)


if __name__ == "__main__":
    main()
