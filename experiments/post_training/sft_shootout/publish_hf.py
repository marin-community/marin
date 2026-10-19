# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish rebuilt #212 traces to the Hub dataset open-athena/GLM-5.3-tmax-swesmith-k8-traces-2026.10.

python experiments/post_training/sft_shootout/publish_hf.py --list <traces root>
    prints every run (plain shard or relay round) that has rebuilt.jsonl, one prefix per line.
python experiments/post_training/sft_shootout/publish_hf.py <prefix>... [--upload]
    converts each run's rebuilt.jsonl to one parquet file (config "attempts" for plain runs, "relay" for
    relay rounds), drops the excluded tasks, tags held-out tasks from the shard files, and counts capability
    tokens in every row (scrub_run.count_tokens). Any token stops the script before anything is uploaded.
    With --upload (needs HF_TOKEN in the env, never printed) it adds the files and the run provenance to
    the repo in one commit, merges the runs into runs.json and regenerates README.md (hf_README.md plus a
    section built from runs.json and the pool manifest); files already there are replaced by the same path.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

sys.path.insert(0, str(Path(__file__).parent))
from scrub_run import count_tokens

REPO = "open-athena/GLM-5.3-tmax-swesmith-k8-traces-2026.10"
README = Path(__file__).parent / "hf_README.md"
CONSTRAINTS = Path(__file__).parent / "harbor_agents" / "mini-swe-agent-2.1.0-constraints.txt"
POOL_MANIFEST = "s3://marin-us-east-02a/marin/experiments/sft-shootout/pool/manifest.json"
# Pool v5 drops these TMax tasks (broken verifiers 000601, 001780; verifier hangs 000441, 001986); the running
# TMax lanes still run them, so they are dropped here. The 7 SWE-smith removals were never sharded.
EXCLUDED = {"task_000601_e01cc3c2", "task_001780_dcd2e021", "task_000441_9c3e397f", "task_001986_dd88e877"}
RUN = re.compile(r"/traces/(?P<kind>[a-z]+(?:-relay)?)/shard-(?P<shard>[^/]+)(?:/round-(?P<round>\d+))?$")
# Kept from driver_config.json; "harbor" and args.output hold the run's proxy address and local paths.
PROVENANCE_KEYS = ("policy_changes", "teacher", "agent_network", "capability_url", "earlier_reasoning_resent", "relay")
ARGS_KEYS = ("family", "agent", "n_attempts", "n_concurrent", "package_revision", "relay")
JSON_COLUMNS = ("tokens", "messages", "tools", "dropped_replies")
SCHEMA = pa.schema(
    [
        ("family", pa.string()),
        ("agent", pa.string()),
        ("run", pa.string()),
        ("shard", pa.string()),
        ("round", pa.int32()),
        ("task_id", pa.string()),
        ("heldout", pa.bool_()),
        ("trial_id", pa.string()),
        ("trial_name", pa.string()),
        ("sample", pa.int32()),
        ("retry", pa.int32()),
        ("reward", pa.float64()),
        ("passed", pa.bool_()),
        ("exception", pa.string()),
        ("trial_seconds", pa.float64()),
        ("agent_seconds", pa.float64()),
        ("conversation", pa.int32()),
        ("n_conversations", pa.int32()),
        ("triggers", pa.list_(pa.string())),
        ("problems", pa.list_(pa.string())),
        *((name, pa.string()) for name in JSON_COLUMNS),
        ("package_revision", pa.string()),
    ]
)


def list_runs(root: str) -> list[str]:
    runs = []
    for kind in sorted(str(p).rstrip("/") for p in StoragePath(root).ls() if p.isdir()):
        if kind.rsplit("/", 1)[-1] == "shards":
            continue
        for shard in sorted(str(p).rstrip("/") for p in StoragePath(kind).ls() if p.isdir()):
            for prefix in [f"{shard}/round-{r}" for r in range(1, 9)] if kind.endswith("-relay") else [shard]:
                if StoragePath(f"{prefix}/rebuilt.jsonl").exists():
                    runs.append(prefix)
    return runs


def convert(prefix: str) -> tuple[str, bytes, dict]:
    match = RUN.search(prefix)
    if not match:
        raise ValueError(f"not a traces run: {prefix}")
    kind, shard, round_ = match["kind"], match["shard"], match["round"]
    family = kind.removesuffix("-relay")
    traces = prefix[: prefix.index("/traces/") + len("/traces")]
    heldout = set(json.loads(StoragePath(f"{traces}/shards/{family}-shard-{shard}.json").read_text())["heldout"])
    relative = prefix[len(traces) - len("traces") :]
    rows, dropped, tokens = [], 0, 0
    for line in StoragePath(f"{prefix}/rebuilt.jsonl").read_text().splitlines():
        tokens += count_tokens(line)
        row = json.loads(line)
        if row.get("task_id") in EXCLUDED:
            dropped += 1
            continue
        rows.append(
            {
                **{name: row.get(name) for name in SCHEMA.names if name not in JSON_COLUMNS},
                **{name: json.dumps(row.get(name)) for name in JSON_COLUMNS},
                "run": relative,
                "shard": shard,
                "round": int(round_) if round_ else None,
                "heldout": row.get("task_id") in heldout,
            }
        )
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), buffer)
    config = json.loads(StoragePath(f"{prefix}/driver_config.json").read_text())
    provenance = {
        "run": relative,
        "args": {key: config["args"].get(key) for key in ARGS_KEYS},
        "policy": Path(config["policy"]).name,
        "model_info": config["harbor"]["agents"][0]["kwargs"].get("model_info"),
        **{key: config.get(key) for key in PROVENANCE_KEYS},
    }
    tokens += count_tokens(json.dumps(provenance))
    name = f"{'relay' if round_ else 'attempts'}/{family}/shard-{shard}" + (f"-round-{round_}" if round_ else "")
    summary = {
        "file": f"data/{name}.parquet",
        "run": relative,
        "family": family,
        "config": "relay" if round_ else "attempts",
        "package_revision": config["args"].get("package_revision"),
        "rows": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "excluded_rows": dropped,
        "capability_tokens": tokens,
    }
    return name, buffer.getvalue(), {"summary": summary, "provenance": provenance}


def readme(runs: dict[str, dict], manifest_sha256: str) -> str:
    lines = [
        README.read_text().rstrip(),
        "",
        "## Runs in this repo",
        "",
        f"- Pool v5 `manifest.json` sha256: `{manifest_sha256}`.",
        f"- mini-swe-agent 2.1.0 dependency pins: `provenance/{CONSTRAINTS.name}` "
        f"(sha256 `{hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest()}`).",
    ]
    for family in sorted({run["family"] for run in runs.values()}):
        revisions = sorted({run["package_revision"] for run in runs.values() if run["family"] == family})
        lines.append(f"- {family} task package revisions: {', '.join(f'`{r}`' for r in revisions)}.")
    lines += ["", "| Config | Family | Run | Rows | Passed |", "|---|---|---|---|---|"]
    for file, run in sorted(runs.items(), key=lambda item: (item[1]["config"], item[0])):
        lines.append(f"| {run['config']} | {run['family']} | `{run['run']}` | {run['rows']:,} | {run['passed']:,} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="*")
    parser.add_argument("--list", help="traces root to list runs under")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.list:
        for prefix in list_runs(args.list):
            print("RUN", prefix)
        return
    converted = [convert(prefix.rstrip("/")) for prefix in args.prefixes]
    for _, _, info in converted:
        print(json.dumps(info["summary"]))
    tokens = sum(info["summary"]["capability_tokens"] for _, _, info in converted)
    print(
        json.dumps(
            {
                "runs": len(converted),
                "rows": sum(i["summary"]["rows"] for _, _, i in converted),
                "capability_tokens": tokens,
            }
        )
    )
    if tokens:
        sys.exit("capability tokens found; nothing uploaded")
    if not args.upload:
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    runs = {}
    if api.file_exists(REPO, "runs.json", repo_type="dataset"):
        runs = json.loads(Path(api.hf_hub_download(REPO, "runs.json", repo_type="dataset")).read_text())
    runs.update({info["summary"]["file"]: info["summary"] for _, _, info in converted})
    manifest_sha256 = hashlib.sha256(StoragePath(POOL_MANIFEST).read_bytes()).hexdigest()
    operations = [
        CommitOperationAdd("README.md", readme(runs, manifest_sha256).encode()),
        CommitOperationAdd("runs.json", json.dumps(dict(sorted(runs.items())), indent=1).encode()),
        CommitOperationAdd(f"provenance/{CONSTRAINTS.name}", CONSTRAINTS.read_bytes()),
    ]
    for name, data, info in converted:
        operations.append(CommitOperationAdd(f"data/{name}.parquet", data))
        operations.append(
            CommitOperationAdd(f"provenance/{name}.json", json.dumps(info["provenance"], indent=2).encode())
        )
    commit = api.create_commit(
        REPO, operations=operations, repo_type="dataset", commit_message=f"Add {len(converted)} runs"
    )
    print(json.dumps({"commit": commit.oid, "files": len(operations)}))


if __name__ == "__main__":
    main()
