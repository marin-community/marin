# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select training pools only after the fixed-cohort baseline is sealed."""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.strata import MembershipState, SelectionDefinition, TaskStrata
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe.cohort_identity import cohort_hash
from taskcompendium.models import TaskSpec

from experiments.post_training.mochi_doom_ftpo.cohorts import read_parquet
from experiments.post_training.mochi_doom_ftpo.constants import REGIONAL_ROOT, SELECTION_NAME, SELECTION_ROOT, VERSION
from experiments.post_training.task_curation import publish_once
from experiments.post_training.task_curation.selection import (
    TASK_MANIFEST_SCHEMA,
    TASK_SCHEMA,
    _publish_selection,
    _selection_row,
)

logger = logging.getLogger(__name__)
BATCH_SIZE = 64
DOMAIN_POOL_LIMIT = 4000
STRUCT_COMPONENTS = ("calendar", "closure")


def training_groups(statuses: Mapping[str, str]) -> dict[str, int]:
    """Round the specified 2:1 domain weights with the largest-remainder rule."""
    if any(status not in ("target", "non-target") for status in statuses.values()):
        raise ValueError("training requires a sealed baseline, including all required reserve evaluations")
    weights = {
        "math": 2,
        **{name: 1 for name, status in statuses.items() if name not in ("math16k", "math32k") and status == "target"},
    }
    total = sum(weights.values())
    counts = {name: BATCH_SIZE * weight // total for name, weight in weights.items()}
    order = sorted(weights, key=lambda name: (-(BATCH_SIZE * weights[name] % total), name))
    for name in order[: BATCH_SIZE - sum(counts.values())]:
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
            component_count = min(DOMAIN_POOL_LIMIT // 2, *(len(pools[name]) for name in STRUCT_COMPONENTS))
            selected.extend(
                dict(row, family=family) for name in STRUCT_COMPONENTS for row in pools[name][:component_count]
            )
        else:
            selected.extend(pools[family][:DOMAIN_POOL_LIMIT])
    actual = Counter(row["family"] for row in selected)
    if actual.keys() != groups.keys() or any(actual[name] < count for name, count in groups.items()):
        raise ValueError(f"training pools cannot supply the declared per-batch counts: {dict(actual)} vs {dict(groups)}")
    return selected


def publish_selection(
    root: StoragePath, name: str, version: str, train: list[dict], pilot: list[dict], full: list[dict], recipe: dict
) -> None:
    """Publish validated task files and their native evaluation identity manifest."""
    files = {"train": train, "validation": pilot, "full_validation": full}
    with TemporaryDirectory(prefix="mochi-doom-targets-") as temporary:
        directory = Path(temporary)
        for file_name, rows in files.items():
            pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), directory / f"{file_name}.parquet")
        manifest = []
        for row in [*train, *full]:
            task = TaskSpec.model_validate_json(row["task_spec"])
            strata = TaskStrata.model_validate({key: row[key] for key in TaskStrata.model_fields})
            manifest.append(_selection_row(task, strata, state=MembershipState.ACTIVE).model_dump(mode="json"))
        pq.write_table(pa.Table.from_pylist(manifest, schema=TASK_MANIFEST_SCHEMA), directory / "task_manifest.parquet")
        summary = {
            "selection_strata": (
                SelectionDefinition(
                    name=name,
                    version=version,
                    task_files=("train.parquet", "full_validation.parquet"),
                    active_files=("train.parquet", "full_validation.parquet"),
                    sampling="sealed baseline targets; per-update groups recorded in recipe",
                ).model_dump(mode="json")
            ),
            **{f"{name}_rows": len(rows) for name, rows in files.items()},
            "training_rows": len(train),
            "training_family_counts": dict(Counter(row["family"] for row in train)),
            "recipe": recipe,
        }
        validated = directory / "validated"
        _publish_selection(directory, str(validated), tuple(f"{name}.parquet" for name in files), summary)
        publish_once(root, {path.name: path.read_bytes() for path in validated.iterdir()})


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
    train = selected_training(list(read_parquet(str(parent / "train.parquet"))), groups)
    full = list(read_parquet(str(parent / "full_validation.parquet")))
    pilot = [
        row
        for row in read_parquet(str(parent / "validation.parquet"))
        if ("struct32k" if row["family"] in STRUCT_COMPONENTS else row["family"]) in groups
    ]
    parent_summary = json.loads((parent / "summary.json").read_text())
    recipe = {
        **parent_summary["recipe"],
        "baseline": {"uri": args.baseline, "sha256": hashlib.sha256(baseline_bytes).hexdigest()},
        "target_status": baseline["target_status"],
        "groups_per_family": groups,
        "domain_pool_limit": DOMAIN_POOL_LIMIT,
        "exclusions": [
            *parent_summary["recipe"]["exclusions"],
            {
                "uri": str(StoragePath(SELECTION_ROOT) / SELECTION_NAME / VERSION / "full_validation.parquet"),
                "version": VERSION,
                "rows": len(full),
                "format": "task_spec",
            },
        ],
    }
    for suffix, rows in (("-targets", train), ("-targets-math", [row for row in train if row["family"] == "math"])):
        name = SELECTION_NAME + suffix
        for root in (
            StoragePath(REGIONAL_ROOT) / "selections" / name / args.version,
            StoragePath(SELECTION_ROOT) / name / args.version,
        ):
            publish_selection(root, name, args.version, rows, pilot, full, recipe)
    logger.info(
        "Published target pools; groups_per_family=%s; train=%s; pilot=%d",
        groups,
        dict(Counter(row["family"] for row in train)),
        len(pilot),
    )


if __name__ == "__main__":
    main()
