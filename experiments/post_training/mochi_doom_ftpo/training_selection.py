# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select training pools only after the fixed-cohort baseline is sealed."""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe.cohort_identity import cohort_hash
from taskcompendium.evaluation.corpus import parquet_rows
from taskcompendium.models import TaskSpec

from experiments.post_training.mochi_doom_ftpo.cohorts import publish_task_files
from experiments.post_training.mochi_doom_ftpo.constants import (
    REGIONAL_ROOT,
    SELECTION_NAME,
    SELECTION_ROOT,
    STRUCT_COMPONENTS,
    TRAIN_BATCH_SIZE,
    VERSION,
)
from experiments.post_training.mochi_doom_ftpo.grades import full_credit_task

logger = logging.getLogger(__name__)
DOMAIN_POOL_LIMIT = 4000


def training_groups(statuses: Mapping[str, str]) -> dict[str, int]:
    """Return family prompt counts totaling 64, approximating math 2 : each other target 1."""
    if any(status not in ("target", "non-target") for status in statuses.values()):
        raise ValueError("training requires a sealed baseline, including all required reserve evaluations")
    weights = {
        "math": 2,
        **{name: 1 for name, status in statuses.items() if name not in ("math16k", "math32k") and status == "target"},
    }
    total = sum(weights.values())
    counts = {name: TRAIN_BATCH_SIZE * weight // total for name, weight in weights.items()}
    order = sorted(weights, key=lambda name: (-(TRAIN_BATCH_SIZE * weights[name] % total), name))
    for name in order[: TRAIN_BATCH_SIZE - sum(counts.values())]:
        counts[name] += 1
    return counts


def selected_training(rows: list[dict[str, Any]], groups: Mapping[str, int]) -> list[dict[str, Any]]:
    """Keep eligible target pools and balance the structured-output components."""
    pools = defaultdict(list)
    for row in rows:
        pools[row["family"]].append(row)
    for pool in pools.values():
        pool.sort(key=lambda row: TaskSpec.model_validate_json(row["task_spec"]).id)
    selected = list(pools["math"])
    for family in groups:
        if family == "math":
            continue
        if family == "struct32k":
            component_count = min(
                DOMAIN_POOL_LIMIT // len(STRUCT_COMPONENTS), *(len(pools[name]) for name in STRUCT_COMPONENTS)
            )
            selected.extend(
                dict(row, family=family) for name in STRUCT_COMPONENTS for row in pools[name][:component_count]
            )
        else:
            selected.extend(pools[family][:DOMAIN_POOL_LIMIT])
    actual = Counter(row["family"] for row in selected)
    if actual.keys() != groups.keys() or any(actual[name] < count for name, count in groups.items()):
        raise ValueError(f"training pools cannot supply the declared per-batch counts: {dict(actual)} vs {dict(groups)}")
    return selected


def full_credit_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "task_spec": full_credit_task(TaskSpec.model_validate_json(row["task_spec"])).model_dump_json()}
        for row in rows
    ]


def effective_panels(
    full: list[dict[str, Any]], manifest: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Select the exact measured cohorts, adding reserves only where baseline required them."""
    by_id = {TaskSpec.model_validate_json(row["task_spec"]).id: row for row in full}
    files = {"math16k": [], "panel32k": []}
    for name, cohort in manifest["cohorts"].items():
        rows = cohort["rows"]
        if baseline["domains"][name]["N"] != len(rows):
            rows = [*rows, *cohort["reserve"]]
        if baseline["domains"][name]["N"] != len(rows):
            raise ValueError(f"Incorrect effective cohort size for {name}")
        destination = "math16k" if name == "math16k" else "panel32k"
        files[destination].extend(by_id[row["task_id"]] for row in rows)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    baseline_bytes = StoragePath(args.baseline).read_bytes()
    baseline = json.loads(baseline_bytes)
    parent = StoragePath(REGIONAL_ROOT) / "cohorts" / VERSION
    manifest = json.loads((parent / "manifest.json").read_text())
    if set(baseline["domains"]) != set(manifest["cohorts"]):
        raise ValueError("baseline must audit every fixed cohort")
    for name, cohort in manifest["cohorts"].items():
        count = baseline["domains"][name]["N"]
        rows = cohort["rows"] if count == len(cohort["rows"]) else [*cohort["rows"], *cohort["reserve"]]
        digest = cohort_hash(row["prompt_hash"] for row in rows)
        if count != len(rows) or digest != baseline["cohort_hashes"][name]:
            raise ValueError(f"baseline identity does not match the frozen {name} cohort")
    groups = training_groups(baseline["target_status"])
    train = full_credit_records(selected_training(list(parquet_rows(parent / "train.parquet")), groups))
    full = full_credit_records(list(parquet_rows(parent / "full_validation.parquet")))
    pilot = full_credit_records(
        [
            row
            for row in parquet_rows(parent / "validation.parquet")
            if ("struct32k" if row["family"] in STRUCT_COMPONENTS else row["family"]) in groups
        ]
    )
    panels = effective_panels(full, manifest, baseline)
    parent_summary = json.loads((parent / "summary.json").read_text())
    recipe = {
        **parent_summary["recipe"],
        "baseline": {"uri": args.baseline, "sha256": hashlib.sha256(baseline_bytes).hexdigest()},
        "target_status": baseline["target_status"],
        "groups_per_family": groups,
        "domain_pool_limit": DOMAIN_POOL_LIMIT,
        "correctness": "native full credit; Verifyit=1, StdoutReward.pass_at_least=1",
        "effective_cohort_hashes": baseline["cohort_hashes"],
        "exclusions": [
            *parent_summary["recipe"]["exclusions"],
            {
                "uri": str(StoragePath(SELECTION_ROOT) / SELECTION_NAME / VERSION / "full_validation.parquet"),
                "version": VERSION,
                "rows": len(full),
                "format": "task_spec",
            },
            *(
                {
                    "uri": str(
                        StoragePath(SELECTION_ROOT)
                        / (SELECTION_NAME + suffix)
                        / args.version
                        / "full_validation.parquet"
                    ),
                    "version": args.version,
                    "rows": len(full),
                    "format": "task_spec",
                }
                for suffix in ("-targets", "-targets-math")
            ),
        ],
    }
    for suffix, rows in (("-targets", train), ("-targets-math", [row for row in train if row["family"] == "math"])):
        name = SELECTION_NAME + suffix
        for root in (
            StoragePath(REGIONAL_ROOT) / "selections" / name / args.version,
            StoragePath(SELECTION_ROOT) / name / args.version,
        ):
            publish_task_files(
                root,
                name,
                args.version,
                {"train": rows, "validation": pilot, "full_validation": full, **panels},
                recipe,
                "sealed baseline targets; per-update groups recorded in recipe",
            )
    logger.info(
        "Published target pools; groups_per_family=%s; train=%s; pilot=%d",
        groups,
        dict(Counter(row["family"] for row in train)),
        len(pilot),
    )


if __name__ == "__main__":
    main()
