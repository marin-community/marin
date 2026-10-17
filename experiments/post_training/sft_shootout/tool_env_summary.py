# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Union of mini-swe-agent's tool env over a ``harbor_glm.py --agent tool-env-probe`` run.

Reads each trial's ``agent/tool_env.txt`` and writes ``tool_env.json`` next to the run:
``distributions`` (name, versions seen), ``top_level`` (import names), ``python`` versions,
``other_site_packages`` (site-packages directories outside the tool env), and per-task coverage.

    uv run python experiments/post_training/sft_shootout/tool_env_summary.py <run prefix>
"""

import json
import sys
from collections import defaultdict

from rigging.filesystem.storage_path import StoragePath


def sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    current = None
    for line in text.splitlines():
        if line.startswith("== "):
            current = line[3:].strip()
        elif current is not None and line.strip():
            out[current].append(line.strip())
    return out


def main() -> None:
    prefix = sys.argv[1].rstrip("/")
    versions: dict[str, set[str]] = defaultdict(set)
    top_level: set[str] = set()
    pythons: set[str] = set()
    others: set[str] = set()
    tasks = {}
    for directory, _, names in StoragePath(prefix).walk():
        if "tool_env.txt" not in names:
            continue
        task = str(directory).split("/attempts/")[0].rsplit("/", 1)[-1].split("__")[0]
        parts = sections(StoragePath(f"{directory}/tool_env.txt").read_text())
        distributions = json.loads(" ".join(parts["distributions"]) or "[]")
        for dist in distributions:
            versions[dist["name"].lower()].add(dist["version"])
        for line in parts["top_level"]:
            names_part = line.split(":", 1)[1].replace("(no top_level.txt)", "")
            top_level.update(n for n in names_part.split() if n[:1].isalpha() and n.replace(".", "_").isidentifier())
        pythons.update(parts["python"])
        others.update(parts["other site-packages dirs"])
        tasks[task] = len(distributions)
    summary = {
        "run": prefix,
        "tasks": dict(sorted(tasks.items())),
        "python": sorted(pythons),
        "distributions": [{"name": name, "versions": sorted(v)} for name, v in sorted(versions.items())],
        "top_level": sorted(top_level),
        "other_site_packages": sorted(others),
    }
    StoragePath(f"{prefix}/tool_env.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
