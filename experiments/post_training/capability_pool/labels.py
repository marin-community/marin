# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract one capability label per task from an RL Data Atlas alignment release.

The Atlas stores the top subject and the top capability candidates per task, with cosine similarities, inside a
nested ``matches`` struct. This stage keeps only the subject-domain match, the best capability within that subject
and both similarities, so pool building can apply its own thresholds without rescanning the descriptions.
"""

import argparse
import json
from dataclasses import dataclass

from finestore.schema import arrow_schema
from fray.types import ResourceConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.curriculum.models import AlignmentRelease
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

SUBJECT_FACET = "subject_domain"
LABELS_METADATA = "labels.json"
ASSIGNMENT_COLUMNS = ["source_id", "task_id", "content_hash", "intended_use", "description_status", "matches"]


@dataclass(frozen=True)
class TaskLabel:
    """A task's nearest subject and capability; similarities are cosine, so distance is one minus them."""

    content_hash: str
    source_id: str
    task_id: str
    intended_use: str
    subject_id: str | None
    subject_similarity: float | None
    capability_id: str | None
    capability_similarity: float | None


def task_label(row: dict) -> dict:
    subjects = [match for match in (row["matches"] or {}).get("subjects") or [] if match["facet"] == SUBJECT_FACET]
    if len(subjects) > 1:
        raise ValueError(f"Task {row['source_id']}/{row['task_id']} has {len(subjects)} {SUBJECT_FACET} matches")
    label = {
        "content_hash": row["content_hash"],
        "source_id": row["source_id"],
        "task_id": row["task_id"],
        "intended_use": row["intended_use"],
        "subject_id": None,
        "subject_similarity": None,
        "capability_id": None,
        "capability_similarity": None,
    }
    if row["description_status"] != "ready" or not subjects:
        return label
    match = subjects[0]
    best = match["candidates"][0]
    if best["capability_id"] != match["capability_id"]:
        raise ValueError(f"Task {row['task_id']}: top capability candidate disagrees with its match")
    return {
        **label,
        "subject_id": match["subject_id"],
        "subject_similarity": match["membership_similarity"],
        "capability_id": best["capability_id"],
        "capability_similarity": best["similarity"],
    }


def extract_labels(context: ZephyrContext, release: AlignmentRelease, release_uri: str, output_root: str) -> list[str]:
    """Write one ``TaskLabel`` per assignment row, sharded like the assignments."""
    root = StoragePath(output_root)
    result = context.execute(
        Dataset.from_list(list(release.paths))
        .load_parquet(columns=ASSIGNMENT_COLUMNS)
        .map(task_label)
        .write_parquet(str(root / "labels/part-{shard:05d}.parquet"), schema=arrow_schema(TaskLabel)),
        name="capability-labels",
    )
    paths = sorted(result.results)
    metadata = {
        "release_uri": release_uri,
        "release_id": release.release_id,
        "curriculum_key": release.curriculum_key,
        "facet": SUBJECT_FACET,
        "paths": paths,
    }
    (root / LABELS_METADATA).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return paths


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="Atlas publication release.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=64)
    args = parser.parse_args()
    release = AlignmentRelease.model_validate_json(StoragePath(args.release).read_bytes())
    with ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="4g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
        name="capability-labels",
    ) as context:
        paths = extract_labels(context, release, args.release, args.output)
    print(json.dumps({"labels": len(paths), "output": args.output}))


if __name__ == "__main__":
    main()
