# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool v3: drop SWE-smith repos whose clean code ships in mini-swe-agent's tool environment, then top up (task #206).

mini-swe-agent is installed at agent setup into /root/.local/share/uv/tools/mini-swe-agent/, inside the agent's
sandbox, and its site-packages hold clean copies of some pool repos (found in #207 run swe-net2: GLM read
tenacity/retry.py there as "the reference"). Every pool repo whose distribution or import name appears in that
environment is removed (leak_risk=agent_tool_clean_copy). The SWE-smith family is then topped back up to
SWESMITH_TASKS from DCAgent packages in the remaining repos: the shortfall is allocated across those repos in
proportion to their unused packages (largest remainder) and drawn at random within each repo, seed
'<SEED>:v3-topup'. No model's pass/fail is used. Added tasks get fixed packages (#10212) and are decontaminated like
the build. Held-out tasks still in the pool stay held out; the set is topped up to round(5%) from added tasks that are
not in allkimi's kimi_trials.txt. v2's remaining tasks are unchanged, so samples drawn from v2 stay valid.

Usage:
    python -m experiments.post_training.sft_shootout_pool.topup --pool <prefix> --tool-env <packages.json>
"""

import argparse
import hashlib
import io
import json
import random
import re
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_pool import decon
from experiments.post_training.sft_shootout_pool.build_swesmith import MIRROR_NAME, PACKAGES
from experiments.post_training.sft_shootout_pool.pool import ALLKIMI, parquet_bytes
from experiments.post_training.sft_shootout_pool.sample import (
    HELDOUT_FRACTION,
    SEED,
    SWEBENCH_VERIFIED_REPOS,
    SWESMITH_TASKS,
    proportional,
    upstream_repo,
)
from experiments.post_training.sft_shootout_pool.swesmith_packages import (
    fixed_package,
    instance_id,
    read_package,
    write_package,
)
from experiments.post_training.sft_shootout_pool.swesmith_restore import mirror, restore_payload

VERSION = 3
REASON = "agent_tool_clean_copy"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def tool_env_repos(repos: set[str], tool_env: dict) -> dict[str, str]:
    """{repo: matched name} for repos whose name matches a tool-env distribution or top-level import name."""
    names = {normalize(name) for name in tool_env["distributions"]} | {normalize(n) for n in tool_env["top_level"]}
    matched = {}
    for repo in repos:
        name = repo.split("/", 1)[1]
        for candidate in {name, name.removeprefix("python-"), name.removesuffix(".six")}:
            if normalize(candidate) in names:
                matched[repo] = candidate
                break
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--tool-env", required=True, help="JSON {distributions: [...], top_level: [...]} from #207")
    args = parser.parse_args()
    pool = StoragePath(args.pool)
    if (pool / f"manifest-v{VERSION - 1}.json").exists():
        raise ValueError(f"Pool v{VERSION} was already built here")
    previous_bytes = (pool / "manifest.json").read_bytes()
    manifest = json.loads(previous_bytes)
    files = {name: (pool / name).read_bytes() for name in manifest["file_sha256"]}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError(f"{name} changed since v{VERSION - 1}")
    tool_env_bytes = StoragePath(args.tool_env).read_bytes()
    tool_env = json.loads(tool_env_bytes)

    rows = [json.loads(line) for line in files["tasks.jsonl"].decode().splitlines()]
    excluded = [json.loads(line) for line in files["excluded.jsonl"].decode().splitlines()]
    source = Path(hf_hub_download(PACKAGES[0], "tasks.parquet", repo_type="dataset", revision=PACKAGES[1]))
    upstream = {row["path"]: row["task_binary"] for row in pq.read_table(source).to_pylist()}
    configs = {task: json.loads(read_package(binary)["tests/config.json"]) for task, binary in upstream.items()}
    repo_of = {task: upstream_repo(config["repo"]) for task, config in configs.items()}

    leaking = tool_env_repos(set(repo_of.values()), tool_env)
    removed = [row for row in rows if row["family"] == "swesmith" and row["repo"] in leaking]
    kept = [row for row in rows if row not in removed]
    previously_excluded = {row["task_id"] for row in excluded}
    excluded_repos = (
        SWEBENCH_VERIFIED_REPOS
        | set(leaking)
        | {row["repo"] for row in excluded if row.get("leak_risk") == "vendored_clean_copy"}
    )
    in_pool = {row["task_id"] for row in kept}
    unused: dict[str, list[str]] = {}
    for task, repo in sorted(repo_of.items()):
        if task not in in_pool and task not in previously_excluded and repo not in excluded_repos:
            unused.setdefault(repo, []).append(task)
    shortfall = SWESMITH_TASKS - sum(row["family"] == "swesmith" for row in kept)
    allocation = proportional({repo: len(tasks) for repo, tasks in unused.items()}, shortfall)
    rng = random.Random(f"{SEED}:v3-topup")
    added = sorted(task for repo in sorted(unused) for task in rng.sample(unused[repo], allocation[repo]))

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        packages = {
            row["path"]: row for row in pq.read_table(io.BytesIO(files["swesmith-packages.parquet"])).to_pylist()
        }
        packages = {task: row for task, row in packages.items() if task in in_pool}
        records = [
            json.loads(line)
            for line in files["swesmith-packages.jsonl"].decode().splitlines()
            if json.loads(line)["task_id"] in in_pool
        ]
        mirrors = work / "mirrors"
        mirrors.mkdir()
        new_rows = []
        for task in added:
            package = read_package(upstream[task])
            instance = instance_id(package)
            restored, deleted = restore_payload(mirror(MIRROR_NAME.match(instance)[1], mirrors), instance)
            binary = write_package(fixed_package(package, restored, deleted))
            new_rows.append({"path": task, "task_binary": binary})
            records.append(
                {
                    "task_id": task,
                    "instance_id": instance,
                    "upstream_sha256": hashlib.sha256(upstream[task]).hexdigest(),
                    "fixed_sha256": hashlib.sha256(binary).hexdigest(),
                    "restored_files": sorted(restored),
                    "deleted_files": deleted,
                    "added_in": VERSION,
                }
            )
        files["swesmith-packages.parquet"] = parquet_bytes(
            [packages[task] for task in sorted(packages)] + sorted(new_rows, key=lambda row: row["path"])
        )
        added_parquet = work / "added.parquet"
        added_parquet.write_bytes(parquet_bytes(new_rows))
        evals = decon.eval_documents(work)
        marks = decon.run(decon.swesmith_documents(added_parquet, set(added)), evals, work / "decon")

    allkimi_bytes = Path(hf_hub_download(ALLKIMI[0], ALLKIMI[2], repo_type="dataset", revision=ALLKIMI[1])).read_bytes()
    allkimi_tasks = {line.strip().rsplit("__", 1)[0] for line in allkimi_bytes.decode().splitlines() if line.strip()}
    total = sum(row["family"] == "swesmith" for row in kept) + len(added)
    target = round(HELDOUT_FRACTION * total)
    held = sum(row["family"] == "swesmith" and row["heldout"] for row in kept)
    candidates = sorted(task for task in added if task not in allkimi_tasks)
    new_heldout = set(random.Random(f"{SEED}:v3-heldout").sample(candidates, max(0, target - held)))

    revision = (
        f"{args.pool.rstrip('/')}/swesmith-packages.parquet@sha256:"
        f"{hashlib.sha256(files['swesmith-packages.parquet']).hexdigest()}"
    )
    for row in kept:
        if row["family"] == "swesmith":
            row["package_revision"] = revision
    template = next(row for row in kept if row["family"] == "swesmith")
    kept += [
        {
            "task_id": task,
            "family": "swesmith",
            "heldout": task in new_heldout,
            "repo": repo_of[task],
            "instance_id": configs[task]["instance_id"],
            "leak_risk": None,
            "decon": marks[task],
            "package_source": template["package_source"],
            "package_revision": revision,
            "added_in": VERSION,
        }
        for task in added
    ]
    excluded += [
        {
            "task_id": row["task_id"],
            "family": "swesmith",
            "repo": row["repo"],
            "heldout": row["heldout"],
            "leak_risk": REASON,
        }
        for row in removed
    ]
    files["tasks.jsonl"] = "".join(json.dumps(row) + "\n" for row in kept).encode()
    files["excluded.jsonl"] = "".join(json.dumps(row) + "\n" for row in excluded).encode()
    files["swesmith-packages.jsonl"] = "".join(json.dumps(record) + "\n" for record in records).encode()

    swesmith = manifest["families"]["swesmith"]
    swesmith["pool_tasks"] = sum(row["family"] == "swesmith" for row in kept)
    swesmith["heldout_tasks"] = sum(row["family"] == "swesmith" and row["heldout"] for row in kept)
    swesmith["repos"] = len({row["repo"] for row in kept if row["family"] == "swesmith"})
    swesmith["package_revision"] = revision
    rounds = swesmith.get("excluded_after_build") or []
    swesmith["excluded_after_build"] = [{"version": 2} | rounds] if isinstance(rounds, dict) else rounds
    swesmith["excluded_after_build"].append(
        {
            "version": VERSION,
            "repos": {repo: REASON for repo in sorted(leaking)},
            "matched_names": leaking,
            "removed": len(removed),
            "removed_heldout": sum(row["heldout"] for row in removed),
        }
    )
    swesmith["topped_up_in_v3"] = {
        "shortfall": shortfall,
        "seed": f"{SEED}:v3-topup",
        "allocation": {repo: count for repo, count in allocation.items() if count},
        "added": len(added),
        "added_heldout": len(new_heldout),
        "heldout_seed": f"{SEED}:v3-heldout",
        "decon_contaminated": sorted(task for task in added if marks[task]["contaminated"]),
    }
    history = manifest.get("supersedes") or []
    manifest["version"] = VERSION
    manifest["supersedes"] = [
        *([history] if isinstance(history, dict) else history),
        {
            "manifest": f"manifest-v{VERSION - 1}.json",
            "sha256": hashlib.sha256(previous_bytes).hexdigest(),
            "change": "v3: drop repos in mini-swe-agent's tool env, top SWE-smith up to 2,100",
        },
    ]
    manifest["inputs"]["mini_swe_agent_tool_env"] = {"sha256": hashlib.sha256(tool_env_bytes).hexdigest()} | tool_env
    manifest["decon"]["contaminated_tasks"] = sorted(
        set(manifest["decon"]["contaminated_tasks"]) | set(swesmith["topped_up_in_v3"]["decon_contaminated"])
    )
    manifest["decon"]["near_threshold_tasks"] = {
        task: value
        for task, value in manifest["decon"]["near_threshold_tasks"].items()
        if task in {row["task_id"] for row in kept}
    } | {
        task: round(marks[task]["max_overlap"], 3)
        for task in added
        if not marks[task]["contaminated"] and marks[task]["max_overlap"] >= 0.2
    }
    manifest["file_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

    (pool / f"manifest-v{VERSION - 1}.json").write_bytes(previous_bytes)
    for name, data in files.items():
        (pool / name).write_bytes(data)
        if hashlib.sha256((pool / name).read_bytes()).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError(f"Upload hash mismatch: {name}")
    (pool / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        "POOL_V3 "
        + json.dumps(
            {
                "manifest_sha256": hashlib.sha256((pool / "manifest.json").read_bytes()).hexdigest(),
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
