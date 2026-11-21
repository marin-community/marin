# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute deferred code on a Docker-capable Iris CPU worker, or audit parity."""

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.weight_merging.playground import evaluation
from experiments.weight_merging.playground.evaluation import DOCKER_IMAGE, score
from experiments.weight_merging.playground.remote import grades as grade_cache
from experiments.weight_merging.playground.remote.grades import GradeStore

# The cleanup-only patch preserves grading and sandbox limits. Reuse requires an
# explicit, source-pinned manifest and selects this one known old contract.
PRECLEANUP_CACHE_CONTRACT = {
    "docker_image": DOCKER_IMAGE,
    "evaluation_sha256": "dccb00a00219aeeabbd389c9355b01eababa23b7c9460ba8e0cc264e618bf3fa",
    "rescore_sha256": "f5ad7eba3404486b95829620ecdfe6874a52482ae6da624c193af0b5cb5b487e",
}


def check_cleanup_cache(rows: list[dict], items: dict[str, dict], cache: GradeStore) -> list[dict]:
    """Check five saved normal grades and a synthetic infinite-loop timeout."""
    selected = []
    for row in rows:
        if row["dataset"] != "mbpp":
            continue
        item = items[row["item_id"]]
        cached = cache.get(cache.identity(row, item))
        if cached is None or cached.get("error") == "timeout":
            continue
        selected.append((row, cached))
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise ValueError("Cleanup cache audit requires five saved normal grades")
    checks = []
    for row, cached in selected:
        started = time.monotonic()
        actual = score(items[row["item_id"]], row["completion"])
        elapsed = time.monotonic() - started
        if actual != cached:
            raise ValueError(f"Cleanup patch changed cached grade: {row['candidate']} {row['item_id']}")
        checks.append(
            {
                "key": [row[key] for key in ("candidate", "item_id", "mode", "seed")],
                "grade": actual,
                "purpose": "cached_normal",
                "elapsed_seconds": elapsed,
            }
        )
    started = time.monotonic()
    actual = score({"dataset": "mbpp", "setup": [], "tests": []}, "while True:\n    pass\n")
    elapsed = time.monotonic() - started
    if actual != {"correct": False, "error": "timeout"}:
        raise ValueError(f"Synthetic infinite-loop control did not time out and fail: {actual}")
    checks.append(
        {
            "key": ["synthetic-timeout-control", "mbpp/control/infinite-loop", "greedy", 0],
            "grade": actual,
            "purpose": "synthetic_timeout",
            "elapsed_seconds": elapsed,
        }
    )
    print(
        "Cleanup cache sanity: five saved normal grades exact; synthetic infinite loop timed out and failed", flush=True
    )
    return checks


def score_rows(
    rows: list[dict],
    items: dict[str, dict],
    *,
    cache: GradeStore,
    parity: bool = False,
    execution_timings: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    pending = [row for row in rows if row["dataset"] == "mbpp" and (parity or row.get("pending_execution"))]

    def durable_score(row):
        item = items[row["item_id"]]
        identity = cache.identity(row, item)
        key = [row[field] for field in ("candidate", "item_id", "mode", "seed")]
        cached = cache.get_record(identity)
        if cached is not None:
            if execution_timings is not None and "elapsed_seconds" in cached:
                execution_timings.append({"key": key, "elapsed_seconds": cached["elapsed_seconds"], "reused": True})
            return cached["grade"]
        try:
            started = time.monotonic()
            grade = score(item, row["completion"])
            elapsed = time.monotonic() - started
        except Exception as error:
            error.add_note(f"Scoring observation: {row['candidate']} {row['item_id']} {row['mode']} seed={row['seed']}")
            raise
        cache.put(identity, grade, elapsed_seconds=elapsed)
        if execution_timings is not None:
            execution_timings.append({"key": key, "elapsed_seconds": elapsed, "reused": False})
        return grade

    with ThreadPoolExecutor(max_workers=4) as pool:
        grades = []
        for index, grade in enumerate(pool.map(durable_score, pending), 1):
            grades.append(grade)
            if index % 50 == 0 or index == len(pending):
                print(f"Durable code grades: {index}/{len(pending)}", flush=True)
    replacements, differences = {}, []
    for row, grade in zip(pending, grades, strict=True):
        key = row["candidate"], row["item_id"], row["mode"], row["seed"]
        if parity and row["correct"] != grade["correct"]:
            differences.append({"key": key, "before": row["correct"], "after": grade["correct"]})
        replacements[key] = {**row, **grade, "pending_execution": False}
    result = [replacements.get((r["candidate"], r["item_id"], r["mode"], r["seed"]), r) for r in rows]
    return result, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cleanup-cache-manifest", type=Path)
    parser.add_argument("--parity-items", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = json.loads((args.run / "panel.json").read_text())
    items = {item["id"]: item for item in panel["items"]}
    rows = [json.loads(line) for line in (args.run / "outputs.jsonl").open()]
    if args.parity_items:
        selected = {item["id"] for item in panel["items"] if item["dataset"] == "mbpp"}
        selected = set(sorted(selected)[: args.parity_items])
        rows = [
            row for row in rows if row["dataset"] == "mbpp" and row["mode"] == "greedy" and row["item_id"] in selected
        ]
    scoring_contract = {
        "docker_image": DOCKER_IMAGE,
        "evaluation_sha256": hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest(),
        "rescore_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grade_store_sha256": hashlib.sha256(Path(grade_cache.__file__).read_bytes()).hexdigest(),
    }
    compatibility = json.loads(args.cleanup_cache_manifest.read_text()) if args.cleanup_cache_manifest else None
    if compatibility is not None and (
        compatibility["old_contract"] != PRECLEANUP_CACHE_CONTRACT
        or compatibility["new_contract"] != scoring_contract
        or not compatibility["patch_diff"]
        or compatibility["audit_policy"] != "five_cached_normal_plus_synthetic_timeout"
    ):
        raise ValueError("Cleanup cache manifest does not pin the exact old/new scorer sources and patch")
    cache_contract = PRECLEANUP_CACHE_CONTRACT if compatibility is not None else scoring_contract
    cache = GradeStore(args.cache, cache_contract)
    sanity = check_cleanup_cache(rows, items, cache) if compatibility is not None else []
    timings = []
    scored, differences = score_rows(rows, items, cache=cache, parity=bool(args.parity_items), execution_timings=timings)
    args.output.mkdir(parents=True)
    for name in ("panel.json", "model-paths.json", "model-sources.json"):
        shutil.copy2(args.run / name, args.output / name)
    with (args.output / "outputs.jsonl").open("w") as stream:
        for row in scored:
            stream.write(json.dumps(row) + "\n")
    metadata = json.loads((args.run / "run-metadata.json").read_text())
    metadata["scoring"] = {
        "docker_image": DOCKER_IMAGE,
        "cache": args.cache,
        "scoring_contract": scoring_contract,
        "cache_contract": cache_contract,
        "cleanup_cache_compatibility": compatibility,
        "cleanup_cache_sanity": sanity,
        "execution_timing": {
            "limit_seconds": 10,
            "near_limit_threshold_seconds": 8,
            "observations_with_timing": len(timings),
            "cached_without_timing": sum(row["dataset"] == "mbpp" for row in rows) - len(timings),
            "near_limit_items": [value for value in timings if value["elapsed_seconds"] >= 8],
        },
        "parity_items": args.parity_items,
        "rows": len(scored),
        "differences": differences,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["scoring"], indent=2), flush=True)
    if differences:
        raise ValueError(f"Remote scorer differs on {len(differences)} baseline observations")


if __name__ == "__main__":
    main()
