# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute deferred code on a Docker-capable Iris CPU worker, or audit parity."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.weight_merging.playground import evaluation
from experiments.weight_merging.playground.evaluation import DOCKER_IMAGE, score
from experiments.weight_merging.playground.remote.grades import GradeStore

# The cleanup-only patch preserves grading and sandbox limits. Reuse requires an
# explicit, source-pinned manifest and selects this one known old contract.
PRECLEANUP_CACHE_CONTRACT = {
    "docker_image": DOCKER_IMAGE,
    "evaluation_sha256": "dccb00a00219aeeabbd389c9355b01eababa23b7c9460ba8e0cc264e618bf3fa",
    "rescore_sha256": "f5ad7eba3404486b95829620ecdfe6874a52482ae6da624c193af0b5cb5b487e",
}


def check_cleanup_cache(rows: list[dict], items: dict[str, dict], cache: GradeStore, timeout_replay: dict) -> list[dict]:
    """Check five saved normal grades and replay an actual pre-patch timeout."""
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
        actual = score(items[row["item_id"]], row["completion"])
        if actual != cached:
            raise ValueError(f"Cleanup patch changed cached grade: {row['candidate']} {row['item_id']}")
        checks.append(
            {
                "key": [row[key] for key in ("candidate", "item_id", "mode", "seed")],
                "grade": actual,
                "purpose": "cached_normal",
            }
        )
    replay = next(row for row in rows if all(row[key] == value for key, value in timeout_replay.items()))
    actual = score(items[replay["item_id"]], replay["completion"])
    if actual != {"correct": False, "error": "timeout"}:
        raise ValueError(f"Previously crashing observation did not time out in replay: {timeout_replay}: {actual}")
    checks.append(
        {
            "key": [replay[key] for key in ("candidate", "item_id", "mode", "seed")],
            "grade": actual,
            "purpose": "previously_crashing_timeout",
        }
    )
    print("Cleanup cache sanity: five saved normal grades exact; crashed-item replay timed out and failed", flush=True)
    return checks


def score_rows(
    rows: list[dict], items: dict[str, dict], *, cache: GradeStore, parity: bool = False
) -> tuple[list[dict], list[dict]]:
    pending = [row for row in rows if row["dataset"] == "mbpp" and (parity or row.get("pending_execution"))]

    def durable_score(row):
        item = items[row["item_id"]]
        identity = cache.identity(row, item)
        cached = cache.get(identity)
        if cached is not None:
            return cached
        try:
            grade = score(item, row["completion"])
        except Exception as error:
            error.add_note(f"Scoring observation: {row['candidate']} {row['item_id']} {row['mode']} seed={row['seed']}")
            raise
        cache.put(identity, grade)
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
    }
    compatibility = json.loads(args.cleanup_cache_manifest.read_text()) if args.cleanup_cache_manifest else None
    if compatibility is not None and (
        compatibility["old_contract"] != PRECLEANUP_CACHE_CONTRACT
        or compatibility["new_contract"] != scoring_contract
        or not compatibility["patch_diff"]
        or compatibility["audit_policy"] != "five_cached_normal_plus_exact_crashed_timeout"
        or set(compatibility["timeout_replay"]) != {"candidate", "item_id", "mode", "seed"}
    ):
        raise ValueError("Cleanup cache manifest does not pin the exact old/new scorer sources and patch")
    cache_contract = PRECLEANUP_CACHE_CONTRACT if compatibility is not None else scoring_contract
    cache = GradeStore(args.cache, cache_contract)
    sanity = (
        check_cleanup_cache(rows, items, cache, compatibility["timeout_replay"]) if compatibility is not None else []
    )
    scored, differences = score_rows(rows, items, cache=cache, parity=bool(args.parity_items))
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
