#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review failed QEMU verifier setup without changing benchmark generations."""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from marin.evaluation.harbor.driver_config import HarborErrorTaxonomy
from marin.evaluation.harbor.runner import _aggregate, _job_benchmark, _read_trial, _read_trials
from marin.external_dependencies import HARBOR
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

BENCHMARK = "tb2-recovery"
BENCHMARK_TRIALS = 267
MINIMUM_COVERAGE = 0.9
CELL = re.compile(r"([0-9.]+) \((s3://[^)]+)\)")


@dataclasses.dataclass(frozen=True)
class Aggregate:
    count: int
    mean: float
    sem: float


def remove_rewards(original: Aggregate, rewards: list[float]) -> Aggregate:
    """Remove invalid verifier outcomes from a stored mean and sample SEM."""
    count = original.count - len(rewards)
    if count < 2:
        raise ValueError("insufficient valid outcomes after verifier review")
    total = original.mean * original.count - sum(rewards)
    squares = (
        original.sem**2 * original.count * (original.count - 1)
        + original.count * original.mean**2
        - sum(reward**2 for reward in rewards)
    )
    variance = (squares - total**2 / count) / (count - 1)
    if variance < -1e-12:
        raise ValueError("stored aggregate has inconsistent variance")
    return Aggregate(count, total / count, math.sqrt(max(variance, 0.0) / count))


def failed_verifier_setup(stdout: str, stderr: str) -> bool:
    """Identify the observed QEMU package bootstrap failure before tests run."""
    output = stdout + "\n" + stderr
    return (
        "404" in output
        and "Not Found" in output
        and "curl: command not found" in output
        and "uvx: command not found" in output
        and re.search(r"\d+ (?:failed|passed)|collected \d+ items", output) is None
    )


def read_json(uri: str) -> tuple[dict, str] | None:
    filesystem, key = filesystem_for(uri)
    if not filesystem.exists(key):
        return None
    with filesystem.open(key, "rb") as source:
        payload = source.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def runtime_taxonomy() -> HarborErrorTaxonomy:
    """Read scoring categories from the declared Harbor runtime."""
    command = [
        "uv",
        "run",
        "--project",
        "config/external/harbor",
        "python",
        "-c",
        "import json; from harbor_config.errors import ErrorCategory, errors_by_category; "
        "print(json.dumps({category.value: sorted(errors_by_category(category)) for category in ErrorCategory}))",
    ]
    categories = json.loads(subprocess.check_output(command, text=True))
    return HarborErrorTaxonomy(
        infrastructure=frozenset(categories["infrastructure"]),
        agent=frozenset(categories["agent"]),
        passthrough=frozenset(categories["passthrough"]),
        undecided=frozenset(categories["unknown"]),
        commit=HARBOR.commit,
    )


def original_aggregate(uri: str, taxonomy: HarborErrorTaxonomy) -> tuple[Aggregate, dict]:
    """Prefer the native final aggregate over stale operator count metadata."""
    native = read_json(uri + "/harbor_result.json")
    if native is not None:
        result, digest = native
        return (
            Aggregate(result["scored_trials"], result["mean_reward"], result["reward_stderr"]),
            {"uri": uri + "/harbor_result.json", "sha256": digest, "payload": result},
        )
    filesystem, key = filesystem_for(uri + "/harbor_jobs")
    jobs = [path for path in filesystem.ls(key, detail=False) if filesystem.exists(path.rstrip("/") + "/result.json")]
    if len(jobs) != 1:
        raise ValueError(f"{uri}: expected one native Harbor job")
    job = StoragePath.parse("s3://" + jobs[0])
    trials = _read_trials(job, taxonomy)
    aggregate = _aggregate(trials, BENCHMARK, _job_benchmark(job), 3)
    return (
        Aggregate(aggregate.scored_trials, aggregate.mean_reward, aggregate.reward_stderr),
        {
            "kind": "native_trial_aggregation",
            "job": str(job),
            "harbor_commit": taxonomy.commit,
            "errors": dict(aggregate.errors),
        },
    )


def review(model: str, uri: str, recovered: dict, taxonomy: HarborErrorTaxonomy) -> dict:
    """Check each QEMU trial and retain evidence for every invalid reward."""
    existing = next((value for value in recovered.values() if value.get("source") == uri), None)
    if existing and ("qemu" in existing.get("recovery_reason", "").lower()):
        return {"model": model, "source": uri, "already_reviewed": True}
    original, provenance = original_aggregate(uri, taxonomy)
    filesystem, key = filesystem_for(uri + "/harbor_jobs")
    excluded = []
    reviewed = []
    for job in filesystem.ls(key, detail=False):
        bucket, prefix, _ = filesystem.split_path(job.rstrip("/") + "/qemu-alpine-ssh__")
        listing = filesystem.call_s3("list_objects_v2", Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000)
        if listing.get("NextContinuationToken"):
            raise ValueError("QEMU trial listing is incomplete")
        for directory in listing.get("CommonPrefixes", []):
            trial_uri = "s3://" + bucket + "/" + directory["Prefix"] + "result.json"
            trial = read_json(trial_uri)
            if trial is None:
                continue
            result, digest = trial
            verifier = result.get("verifier_result") or {}
            scored = _read_trial(StoragePath.parse(trial_uri), taxonomy).scored
            setup_failed = failed_verifier_setup(verifier.get("stdout", ""), verifier.get("stderr", ""))
            evidence = {
                "trial": result["trial_name"],
                "uri": trial_uri,
                "sha256": digest,
                "task_checksum": result["task_checksum"],
                "primary_exception": (result.get("exception_info") or {}).get("exception_type"),
                "originally_scored": scored,
                "verifier_setup_failed": setup_failed,
            }
            reviewed.append(evidence)
            # The native taxonomy already omits infrastructure failures, including
            # failures with a verifier reward. Subtracting them again loses valid trials.
            if setup_failed and scored:
                excluded.append({**evidence, "invalid_reward": verifier["rewards"]["reward"]})
    if len(reviewed) != 3:
        raise ValueError(f"{model}: expected evidence for all three QEMU repetitions")
    corrected = remove_rewards(original, [row["invalid_reward"] for row in excluded])
    return {
        "model": model,
        "benchmark": BENCHMARK,
        "source": uri,
        "benchmark_trials": BENCHMARK_TRIALS,
        "original": dataclasses.asdict(original),
        "corrected": dataclasses.asdict(corrected),
        "sealable": corrected.count / BENCHMARK_TRIALS >= MINIMUM_COVERAGE,
        "reviewed_trials": reviewed,
        "excluded_trials": excluded,
        "original_aggregate": provenance,
        "reason": "QEMU repetitions reviewed; scored package-bootstrap failures excluded because tests never ran.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, type=Path)
    parser.add_argument("--recovered-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publish-prefix")
    args = parser.parse_args()
    configure_coreweave_s3()
    taxonomy = runtime_taxonomy()
    recovered = json.loads(args.recovered_metrics.read_text())
    lines = [line for line in args.tracker.read_text().splitlines() if line.startswith("|")]
    columns = [part.strip() for part in lines[0].strip("|").split("|")]
    index = columns.index(BENCHMARK)
    sources = []
    for line in lines[3:]:
        cells = [part.strip() for part in line.strip("|").split("|")]
        if match := CELL.fullmatch(cells[index]):
            sources.append((cells[0], match.group(2)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(review, model, uri, recovered, taxonomy) for model, uri in sources]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["review_commit"] = commit
            if args.publish_prefix:
                uri = args.publish_prefix.rstrip("/") + "/" + result["model"].replace("/", "--") + ".json"
                filesystem, key = filesystem_for(uri)
                if filesystem.exists(key):
                    raise FileExistsError(uri)
                with filesystem.open(key, "wb") as target:
                    target.write((json.dumps(result, indent=2) + "\n").encode())
                result["review_uri"] = uri
            rows.append(result)
            print(result["model"], result.get("corrected", "already reviewed"), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
