# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and audit disjoint source/recovery outcomes before publishing a score."""

import argparse
import copy
import hashlib
import json
import runpy
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rigging.filesystem.buckets import filesystem_for

AUDIT = runpy.run_path(str(Path(__file__).with_name("audit.py")))
PREPARE = runpy.run_path(str(Path(__file__).with_name("prepare.py")))


def combine(manifest: dict, retained: list[dict], recovered: list[dict]) -> dict:
    """Count each task once, preserving model failures and excluding infrastructure."""
    old = {row["task_name"] for row in manifest["retained_trials"]}
    new = set(manifest["recovery_tasks"])
    if old & new or len(old | new) != 500 or len(old) != len(manifest["retained_trials"]):
        raise ValueError("expected a disjoint 500-task partition")
    if {row["task_name"] for row in retained} != old:
        raise ValueError("retained evidence is missing")
    seen = set()
    rows = []
    for batch, allowed in ((retained, old), (recovered, new)):
        for row in batch:
            name = row["task_name"]
            if name not in allowed or name in seen:
                raise ValueError(f"unexpected or duplicate task: {name}")
            seen.add(name)
            reason = PREPARE["recovery_reason"](row)
            if name in old and reason is not None:
                raise ValueError(f"retained outcome no longer scoreable: {name}")
            rows.append(dict(row, scored=reason is None, exclusion_reason=reason))
    rewards = [row["reward"] for row in rows if row["scored"]]
    return {
        "n_benchmark": 500,
        "n_scored": len(rewards),
        "n_unscored": len(rows) - len(rewards),
        "pending_tasks": sorted((old | new) - seen),
        "mean_reward": statistics.mean(rewards) if rewards else None,
        "stderr": statistics.stdev(rewards) / len(rewards) ** 0.5 if len(rewards) > 1 else None,
        "sealable": len(rewards) >= 450,
        "trials": sorted(rows, key=lambda row: row["task_name"]),
    }


def freeze(manifest_path: Path, recovery_results: str, reviews: Path | None) -> dict:
    """Read exact retained evidence and audit new outcomes without changing sources."""
    return freeze_shards(
        manifest_path,
        [{"config": str(manifest_path.with_name("swebench-verified-recovery.yaml")), "results": recovery_results}],
        reviews,
    )


def validate_shards(base_config: dict, configs: list[dict], recovery_tasks: list[str]) -> None:
    """Require an exhaustive disjoint partition with unchanged benchmark settings."""
    expected = set(recovery_tasks)
    seen = set()
    for config in configs:
        names = config["datasets"][0]["task_names"]
        tasks = set(names)
        if not tasks or len(tasks) != len(names) or seen & tasks or tasks - expected:
            raise ValueError("empty, duplicate, overlapping, or unexpected shard tasks")
        normalized = copy.deepcopy(config)
        normalized["datasets"][0]["task_names"] = base_config["datasets"][0]["task_names"]
        if normalized != base_config:
            raise ValueError("shard changes benchmark settings")
        seen.update(tasks)
    if seen != expected:
        raise ValueError("shards do not cover the complete recovery task set")


def freeze_shards(manifest_path: Path, shards: list[dict], reviews: Path | None) -> dict:
    """Audit independent recovery roots, counting each assigned task at most once."""
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    config_payload = manifest_path.with_name("swebench-verified-recovery.yaml").read_bytes()
    config = yaml.safe_load(config_payload)
    if set(config["datasets"][0]["task_names"]) != set(manifest["recovery_tasks"]):
        raise ValueError("canonical task selection differs from manifest")
    configs = [yaml.safe_load(Path(shard["config"]).read_text()) for shard in shards]
    validate_shards(config, configs, manifest["recovery_tasks"])

    def retained_read(row):
        source_fs, path = filesystem_for(row["source_uri"])
        payload = source_fs.cat(path)
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"retained source changed: {row['task_name']}")
        return dict(row, result=json.loads(payload))

    def recovery_read(item):
        fs, path, allowed = item
        if not fs.exists(path + "/result.json"):
            return None
        result_path = path + "/result.json"
        row = AUDIT["inspect_trial"](("s3://" + result_path, fs, result_path))
        if row["task_name"] not in allowed:
            raise ValueError("trial belongs to another shard")
        payload = fs.cat(result_path)
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("recovery result changed during audit; retry")
        return dict(row, result=json.loads(payload))

    items = []
    provenance = []
    for shard, canonical in zip(shards, configs, strict=True):
        fs, root = filesystem_for(shard["results"].rstrip("/"))
        identity = json.loads(fs.cat(root + "/harbor_resume_identity.json"))
        if identity["runtime"] != manifest["recovery_runtime"]:
            raise ValueError("recovery runtime differs from manifest")
        jobs = fs.ls(root + "/harbor_jobs", detail=False)
        if len(jobs) != 1:
            raise ValueError("expected exactly one recovery job per shard")
        job = jobs[0]
        effective_payload = fs.cat(job + "/config.json")
        effective = json.loads(effective_payload)
        allowed = set(canonical["datasets"][0]["task_names"])
        launched = effective["datasets"][0]["task_names"]
        if set(launched) != allowed or len(launched) != len(allowed):
            raise ValueError("launched task selection differs from canonical shard")
        directories = [p for p in fs.ls(job, detail=False) if "__" in p.rsplit("/", 1)[-1]]
        items.extend((fs, p, allowed) for p in directories)
        provenance.append(
            {
                "results": shard["results"],
                "canonical_config": canonical,
                "effective_config_sha256": hashlib.sha256(effective_payload).hexdigest(),
                "effective_config": effective,
                "recovery_identity": identity,
            }
        )
    with ThreadPoolExecutor(max_workers=12) as pool:
        retained = list(pool.map(retained_read, manifest["retained_trials"]))
        recovered = [r for r in pool.map(recovery_read, items) if r is not None]
    report = {"model": manifest["model"], "trials": recovered}
    PREPARE["enrich_retry_evidence"](report)
    if reviews is not None:
        PREPARE["apply_reviews"](report, json.loads(reviews.read_text()))
    return {
        "kind": "immutable_combined_recovery_audit",
        "observed_at": datetime.now(UTC).isoformat(),
        "model": manifest["model"],
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest": manifest,
        "canonical_config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "shards": provenance,
        **combine(manifest, retained, recovered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recovery-results")
    source.add_argument("--shards", type=Path, help="JSON list of canonical config paths and results URLs")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shards:
        report = freeze_shards(args.manifest, json.loads(args.shards.read_text()), args.reviews)
    else:
        report = freeze(args.manifest, args.recovery_results, args.reviews)
    args.output.write_text(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("model", "n_scored", "n_unscored", "mean_reward", "sealable")}))


if __name__ == "__main__":
    main()
