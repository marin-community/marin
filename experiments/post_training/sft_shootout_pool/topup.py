# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pools v3+: drop SWE-smith repos with readable clean copies in the agent's sandbox, then top up.

v3 (--version 3 --tool-env <pilot tool-env probe> --exclude mahmoud/boltons=image_level_clean_copy=...): the repos below.
v4 (--version 4 --exclude <repo>=image_level_clean_copy=<where> ...): repos whose clean source the base image holds
outside /testbed, found by image_check.py (PROBE_METHOD): chardet and tomli (vendored by pip/setuptools), inflect
(vendored by setuptools), dataset (an installed egg). The steps below are the same for every version; seeds carry the
version number.


mini-swe-agent is installed at agent setup into /root/.local/share/uv/tools/mini-swe-agent/, inside the agent's
sandbox, and its site-packages hold clean copies of some pool repos (found in the generation pilot's run swe-net2: GLM
read tenacity/retry.py there as "the reference"). Every pool repo whose distribution or import name appears in that
environment is removed (leak_risk=agent_tool_clean_copy), and so is mahmoud/boltons, whose clean copy sits in conda's
package cache in every conda image (leak_risk=image_level_clean_copy). The SWE-smith family is then topped back up to
SWESMITH_TASKS from DCAgent packages in the remaining repos: the shortfall is allocated across those repos in
proportion to their unused packages (largest remainder) and drawn at random within each repo, seed
'<SEED>:v<version>-topup'. No model's pass/fail is used. Added tasks get fixed packages (#10212) and are
decontaminated like the build. Held-out tasks still in the pool stay held out; the set is topped up to round(5%) from
added tasks that are not in allkimi's kimi_trials.txt. The remaining tasks of v<version - 1> keep their packages, so
samples drawn from v<version - 1> stay valid.

Usage:
    python -m experiments.post_training.sft_shootout_pool.topup --pool <prefix> --version <N> [--tool-env <json>]
        [--exclude <repo>=<reason>=<where> ...] [--exclude-task <task>=<reason>=<evidence> ...]
        [--staged <tasks.parquet>] [--dry-run]
"""

import argparse
import hashlib
import io
import json
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_pool import decon
from experiments.post_training.sft_shootout_pool.build_swesmith import fix_package, upstream_packages
from experiments.post_training.sft_shootout_pool.image_check import PROBE_METHOD
from experiments.post_training.sft_shootout_pool.pool import allkimi_trials, trial_tasks
from experiments.post_training.sft_shootout_pool.pool_io import (
    MANIFEST,
    SWESMITH_PACKAGES,
    jsonl_bytes,
    kept_manifest,
    parquet_bytes,
    read_jsonl,
    read_pool,
    swesmith_revision,
    write_pool,
)
from experiments.post_training.sft_shootout_pool.repo_names import IMPORT_NAMES
from experiments.post_training.sft_shootout_pool.sample import (
    HELDOUT_FRACTION,
    SEED,
    SWEBENCH_VERIFIED_REPOS,
    SWESMITH_TASKS,
    proportional,
)
from experiments.post_training.sft_shootout_pool.swesmith_packages import read_package

REASON = "agent_tool_clean_copy"
REPO_LEVEL_REASONS = {"vendored_clean_copy", "agent_tool_clean_copy", "image_level_clean_copy"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def tool_env_repos(repos: set[str], tool_env: dict) -> dict[str, str]:
    """{repo: matched name} for repos whose name matches a tool-env distribution or top-level import name."""
    distributions = [entry["name"] if isinstance(entry, dict) else entry for entry in tool_env["distributions"]]
    names = {normalize(name) for name in distributions} | {normalize(name) for name in tool_env["top_level"]}
    matched = {}
    for repo in repos:
        name = repo.split("/", 1)[1]
        for candidate in {name, name.removeprefix("python-"), name.removesuffix(".six")}:
            if normalize(candidate) in names:
                matched[repo] = candidate
                break
    return matched


def parse_entries(entries: list[str], detail: str) -> dict[str, dict[str, str]]:
    """{key: {"reason": reason, detail: value}} from repeated 'key=reason=value' arguments."""
    parsed = {}
    for entry in entries:
        key, reason, value = entry.split("=", 2)
        parsed[key] = {"reason": reason, detail: value}
    return parsed


class AddedPackages(NamedTuple):
    # The new swesmith-packages.parquet: kept packages, then added ones, each sorted by task.
    parquet: bytes
    # Restore records of the kept tasks, then the added ones.
    records: list[dict]
    # Decon marks of the added tasks.
    marks: dict[str, dict]


def build_added(
    files: dict[str, bytes],
    in_pool: set[str],
    added: list[str],
    upstream: dict[str, bytes],
    staged_uri: str | None,
    version: int,
) -> AddedPackages:
    """Build fixed packages for the added tasks (or take them from `staged_uri`) and decontaminate them."""
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        packages = {row["path"]: row for row in pq.read_table(io.BytesIO(files[SWESMITH_PACKAGES])).to_pylist()}
        packages = {task: row for task, row in packages.items() if task in in_pool}
        records = [record for record in read_jsonl(files["swesmith-packages.jsonl"]) if record["task_id"] in in_pool]
        mirrors = work / "mirrors"
        mirrors.mkdir()
        new_rows = []
        staged = {}
        if staged_uri:
            staged = {
                row["path"]: row["task_binary"]
                for row in pq.read_table(io.BytesIO(StoragePath(staged_uri).read_bytes())).to_pylist()
            }
            if set(staged) != set(added):
                raise ValueError(f"Staged packages {sorted(staged)} differ from the draw {added}")
        for task in added:
            fixed = fix_package(upstream[task], mirrors, staged.get(task))
            if task in staged and read_package(fixed.binary)["tests/config.json"] != fixed.upstream["tests/config.json"]:
                raise ValueError(f"Staged package for {task} is not built from its upstream package")
            new_rows.append({"path": task, "task_binary": fixed.binary})
            records.append({"task_id": task} | fixed.record | {"added_in": version})
        parquet = parquet_bytes(
            [packages[task] for task in sorted(packages)] + sorted(new_rows, key=lambda row: row["path"])
        )
        added_parquet = work / "added.parquet"
        added_parquet.write_bytes(parquet_bytes(new_rows))
        evals = decon.eval_documents(work)
        marks = decon.run(decon.swesmith_documents(added_parquet, set(added)), evals, work / "decon")
    return AddedPackages(parquet, records, marks)


def top_up_tmax_heldout(
    kept: list[dict], tmax_removed: list[dict], task_exclusions: dict, manifest: dict, version: int
) -> None:
    """Record removed TMax tasks in the manifest and top TMax held-out back up to round(5%), in place.

    TMax-SFT-16.5K has no spare environments, so removed TMax tasks are not replaced; held-out is topped back up to
    round(5%) of the remaining TMax tasks from non-held-out ones if needed.
    """
    tmax_rows = [row for row in kept if row["family"] == "tmax"]
    tmax_target = round(HELDOUT_FRACTION * len(tmax_rows))
    tmax_held = sum(row["heldout"] for row in tmax_rows)
    tmax_pool = sorted(row["task_id"] for row in tmax_rows if not row["heldout"])
    tmax_rng = random.Random(f"{SEED}:v{version}-heldout-tmax")
    tmax_new = set(tmax_rng.sample(tmax_pool, max(0, tmax_target - tmax_held)))
    for row in tmax_rows:
        if row["task_id"] in tmax_new:
            row["heldout"] = True
            row["heldout_added_in"] = version
    tmax = manifest["families"]["tmax"]
    tmax["pool_tasks"] = len(tmax_rows)
    tmax["heldout_tasks"] = sum(row["heldout"] for row in tmax_rows)
    tmax.setdefault("excluded_after_build", []).append(
        {
            "version": version,
            "tasks": {row["task_id"]: task_exclusions[row["task_id"]] for row in tmax_removed},
            "removed": len(tmax_removed),
            "removed_heldout": sum(row["heldout"] for row in tmax_removed),
            "replaced": 0,
            "heldout_added": sorted(tmax_new),
        }
    )


def merge_decon(manifest: dict, kept: list[dict], added: list[str], marks: dict, contaminated: list[str]) -> None:
    """Add the added tasks' decon results to the manifest and drop near-threshold entries for removed tasks."""
    manifest["decon"]["contaminated_tasks"] = sorted(set(manifest["decon"]["contaminated_tasks"]) | set(contaminated))
    manifest["decon"]["near_threshold_tasks"] = {
        task: value
        for task, value in manifest["decon"]["near_threshold_tasks"].items()
        if task in {row["task_id"] for row in kept}
    } | {
        task: round(marks[task]["max_overlap"], 3)
        for task in added
        if not marks[task]["contaminated"] and marks[task]["max_overlap"] >= 0.2
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--tool-env", help="JSON {distributions: [...], top_level: [...]} from the generation pilot (v3)"
    )
    parser.add_argument(
        "--exclude", action="append", default=[], help="repo=reason=where the clean copy lives (repeatable)"
    )
    parser.add_argument(
        "--exclude-task", action="append", default=[], help="task=reason=evidence, either family (repeatable)"
    )
    parser.add_argument(
        "--staged",
        help="tasks.parquet of already-built packages for the added tasks (integrity-checked); used byte for byte",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the SWE-smith removals and replacement draw; write nothing"
    )
    args = parser.parse_args()
    version = args.version
    pool = StoragePath(args.pool)
    if (pool / kept_manifest(version - 1)).exists():
        raise ValueError(f"Pool v{version} was already built here")
    previous_bytes, manifest, files = read_pool(pool, f"v{version - 1}")
    tool_env_bytes = StoragePath(args.tool_env).read_bytes() if args.tool_env else None
    tool_env = json.loads(tool_env_bytes) if tool_env_bytes else {"distributions": [], "top_level": []}
    explicit = parse_entries(args.exclude, "where")
    task_exclusions = parse_entries(args.exclude_task, "evidence")

    rows = read_jsonl(files["tasks.jsonl"])
    excluded = read_jsonl(files["excluded.jsonl"])
    upstream = upstream_packages()
    configs, repo_of = upstream.configs, upstream.repo_of

    leaking = tool_env_repos(set(repo_of.values()), tool_env)
    reasons = {repo: REASON for repo in leaking} | {repo: item["reason"] for repo, item in explicit.items()}
    removed = [
        row
        for row in rows
        if (row["family"] == "swesmith" and row["repo"] in reasons) or row["task_id"] in task_exclusions
    ]
    if missing := sorted(set(task_exclusions) - {row["task_id"] for row in rows}):
        raise ValueError(f"Excluded tasks are not in the pool: {missing}")
    kept = [row for row in rows if row not in removed]
    previously_excluded = {row["task_id"] for row in excluded}
    excluded_repos = (
        SWEBENCH_VERIFIED_REPOS
        | set(reasons)
        | {row["repo"] for row in excluded if row.get("leak_risk") in REPO_LEVEL_REASONS}
    )
    in_pool = {row["task_id"] for row in kept}
    removed_ids = {row["task_id"] for row in removed}
    unused: dict[str, list[str]] = {}
    for task, repo in sorted(repo_of.items()):
        if (
            task not in in_pool
            and task not in previously_excluded
            and task not in removed_ids
            and repo not in excluded_repos
        ):
            unused.setdefault(repo, []).append(task)
    shortfall = SWESMITH_TASKS - sum(row["family"] == "swesmith" for row in kept)
    allocation = proportional({repo: len(tasks) for repo, tasks in unused.items()}, shortfall)
    rng = random.Random(f"{SEED}:v{version}-topup")
    added = sorted(task for repo in sorted(unused) for task in rng.sample(unused[repo], allocation[repo]))
    if args.dry_run:
        # The draw depends only on SWE-smith state and the seed, so the final run reproduces this list.
        print(
            "TOPUP_DRY_RUN "
            + json.dumps(
                {
                    "version": version,
                    "removed": sorted(row["task_id"] for row in removed),
                    "removed_heldout": sorted(row["task_id"] for row in removed if row["heldout"]),
                    "shortfall": shortfall,
                    "allocation": {r: c for r, c in allocation.items() if c},
                    "added": added,
                    "added_repos": {task: repo_of[task] for task in added},
                }
            ),
            flush=True,
        )
        return

    built = build_added(files, in_pool, added, upstream.binaries, args.staged, version)
    files[SWESMITH_PACKAGES] = built.parquet
    records, marks = built.records, built.marks

    allkimi_tasks = trial_tasks(allkimi_trials())
    total = sum(row["family"] == "swesmith" for row in kept) + len(added)
    target = round(HELDOUT_FRACTION * total)
    held = sum(row["family"] == "swesmith" and row["heldout"] for row in kept)
    candidates = sorted(task for task in added if task not in allkimi_tasks)
    new_heldout = set(random.Random(f"{SEED}:v{version}-heldout").sample(candidates, max(0, target - held)))

    revision = swesmith_revision(args.pool, files)
    for row in kept:
        if row["family"] == "swesmith":
            row["package_revision"] = revision
            row["import_names"] = IMPORT_NAMES[row["repo"]]
    template = next(row for row in kept if row["family"] == "swesmith")
    kept += [
        {
            "task_id": task,
            "family": "swesmith",
            "heldout": task in new_heldout,
            "repo": repo_of[task],
            "instance_id": configs[task]["instance_id"],
            "leak_risk": None,
            "import_names": IMPORT_NAMES[repo_of[task]],
            "decon": marks[task],
            "package_source": template["package_source"],
            "package_revision": revision,
            "added_in": version,
        }
        for task in added
    ]
    excluded += [
        {
            "task_id": row["task_id"],
            "family": row["family"],
            "repo": row["repo"],
            "heldout": row["heldout"],
            "leak_risk": (
                task_exclusions[row["task_id"]]["reason"] if row["task_id"] in task_exclusions else reasons[row["repo"]]
            ),
        }
        for row in removed
    ]
    files["excluded.jsonl"] = jsonl_bytes(excluded)
    files["swesmith-packages.jsonl"] = jsonl_bytes(records)

    swesmith = manifest["families"]["swesmith"]
    tmax_removed = [row for row in removed if row["family"] == "tmax"]
    if tmax_removed:
        top_up_tmax_heldout(kept, tmax_removed, task_exclusions, manifest, version)
    # Written after the top-up so held-out flags it sets reach tasks.jsonl, not only the manifest counts.
    files["tasks.jsonl"] = jsonl_bytes(kept)
    swesmith_task_exclusions = {
        row["task_id"]: task_exclusions[row["task_id"]]
        for row in removed
        if row["family"] == "swesmith" and row["task_id"] in task_exclusions
    }
    swesmith["pool_tasks"] = sum(row["family"] == "swesmith" for row in kept)
    swesmith["heldout_tasks"] = sum(row["family"] == "swesmith" and row["heldout"] for row in kept)
    swesmith["repos"] = len({row["repo"] for row in kept if row["family"] == "swesmith"})
    swesmith["package_revision"] = revision
    rounds = swesmith.get("excluded_after_build") or []
    swesmith["excluded_after_build"] = [{"version": 2} | rounds] if isinstance(rounds, dict) else rounds
    swesmith["excluded_after_build"].append(
        {
            "version": version,
            "repos": dict(sorted(reasons.items())),
            "tasks": swesmith_task_exclusions,
            "matched_names": leaking,
            "removed": sum(row["family"] == "swesmith" for row in removed),
            "removed_heldout": sum(row["heldout"] for row in removed if row["family"] == "swesmith"),
        }
    )
    swesmith[f"topped_up_in_v{version}"] = {
        "shortfall": shortfall,
        "seed": f"{SEED}:v{version}-topup",
        "allocation": {repo: count for repo, count in allocation.items() if count},
        "added": len(added),
        "added_heldout": len(new_heldout),
        "heldout_seed": f"{SEED}:v{version}-heldout",
        "decon_contaminated": sorted(task for task in added if marks[task]["contaminated"]),
    }
    removal_counts = Counter(row["leak_risk"] for row in excluded[len(excluded) - len(removed) :])
    removal_summary = ", ".join(f"{reason} {count}" for reason, count in sorted(removal_counts.items()))
    history = manifest.get("supersedes") or []
    manifest["version"] = version
    manifest["supersedes"] = [
        *([history] if isinstance(history, dict) else history),
        {
            "manifest": kept_manifest(version - 1),
            "sha256": hashlib.sha256(previous_bytes).hexdigest(),
            "change": f"v{version}: drop {len(removed)} tasks ({removal_summary}), top SWE-smith up to 2,100",
        },
    ]
    if tool_env_bytes:
        manifest["inputs"]["mini_swe_agent_tool_env"] = {"sha256": hashlib.sha256(tool_env_bytes).hexdigest()} | tool_env
    if explicit:
        manifest.setdefault("clean_copy_probe", {})[f"v{version}"] = {
            "method": PROBE_METHOD,
            "evidence": {
                repo: item | {"removed_tasks": sum(r["repo"] == repo for r in removed)}
                for repo, item in explicit.items()
            },
        }
    merge_decon(manifest, kept, added, marks, swesmith[f"topped_up_in_v{version}"]["decon_contaminated"])
    manifest["file_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

    (pool / kept_manifest(version - 1)).write_bytes(previous_bytes)
    write_pool(pool, files, manifest)
    print(
        f"POOL_V{version} "
        + json.dumps(
            {
                "manifest_sha256": hashlib.sha256((pool / MANIFEST).read_bytes()).hexdigest(),
                "leaking_repos": leaking,
                "removed": len(removed),
                "swesmith": {
                    k: v for k, v in swesmith.items() if k not in ("repo_allocation", "swebench_verified_repos")
                },
                "file_sha256": manifest["file_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
