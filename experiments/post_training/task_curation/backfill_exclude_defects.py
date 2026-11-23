# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exclude typed grader-defect tasks from a backfilled population.

The negative-control campaign found tasks whose hidden tests accept a deliberately wrong
solution (reward 1.0 on a module that does not implement the expected functions) — the
grader cannot fail closed. Such tasks are excluded from the readmitted population with the
campaign evidence recorded per task, never flagged-and-kept.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from typing import Any

import pyarrow.parquet as pq
from marin.execution.artifact import write_artifact
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import (
    RENDERER_VERSION,
    write_text_population,
)

LOGGER = logging.getLogger(__name__)


def spec_rows(population: str) -> list[tuple[str, str, str]]:
    rows = []
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["id", "task_spec", "runtime"])
        rows.extend(
            zip(
                table.column("id").to_pylist(),
                table.column("task_spec").to_pylist(),
                table.column("runtime").to_pylist(),
                strict=True,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--parent-population", required=True)
    parser.add_argument("--parent-task-specs-sha256", required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--exclude", action="append", required=True, help="task_id=negative_campaign_evidence_uri")
    parser.add_argument("--exclusion-reason", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--converter-version", required=True)
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    excluded = {}
    for assignment in args.exclude:
        task_id, _, evidence_uri = assignment.partition("=")
        excluded[task_id] = evidence_uri

    digest = hashlib.sha256()
    rows = spec_rows(args.parent_population)
    for _, encoded, _ in rows:
        digest.update(encoded.encode())
    if digest.hexdigest() != args.parent_task_specs_sha256:
        raise ValueError(
            f"The parent population's task specs changed: {digest.hexdigest()} != {args.parent_task_specs_sha256}"
        )
    if len(rows) != args.task_count:
        raise ValueError(f"The parent population holds {len(rows)} tasks, not the declared {args.task_count}")

    missing = sorted(set(excluded) - {task_id for task_id, _, _ in rows})
    if missing:
        raise ValueError(f"Excluded task ids absent from the population: {missing}")

    counts: Counter[str] = Counter()
    documents = []
    exclusions = []
    for task_id, encoded, runtime in rows:
        counts["task:total"] += 1
        if task_id in excluded:
            counts["task:excluded_grader_defect"] += 1
            exclusions.append(
                {
                    "task_id": task_id,
                    "parent_task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    "reason": args.exclusion_reason,
                    "evidence_uri": excluded[task_id],
                }
            )
            continue
        counts["task:retained"] += 1
        documents.append((json.loads(encoded), json.loads(runtime) if runtime else {}))

    files = write_text_population(documents, args.output)
    if files.task_count != len(documents):
        raise ValueError(f"The excluded population holds {files.task_count} of {len(documents)} tasks")
    LOGGER.info("Wrote the excluded population: %s (%d tasks)", args.output, files.task_specs_sha256[:16])

    record = SourceTextPopulation(
        path=args.output,
        source_id=args.source_id,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        source_version=args.converter_version,
        converter_revision=args.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )
    write_artifact(record.model_dump(mode="json", exclude={"path"}), args.output)
    (StoragePath(args.output) / "population.json").write_text(record.model_dump_json(indent=2) + "\n")

    receipt: dict[str, Any] = {
        "source_id": args.source_id,
        "parent_population": args.parent_population,
        "parent_task_specs_sha256": args.parent_task_specs_sha256,
        "population": args.output,
        "task_specs_sha256": files.task_specs_sha256,
        "task_count": files.task_count,
        "converter_version": args.converter_version,
        "converter_revision": args.converter_revision,
        "exclusion": {
            "reason": args.exclusion_reason,
            "excluded_tasks": len(exclusions),
            "retained_tasks": counts["task:retained"],
        },
    }
    output = StoragePath(args.output)
    (output / "exclusion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "exclusions.json").write_text(json.dumps(exclusions, indent=1) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
