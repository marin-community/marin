# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the repaired identity-following population (ruling 2026-11-23).

Every ``laion__nemotron-gym-identity-following-v4`` task carries the false NVIDIA persona, so the
repair produces the whole population: unpack each pinned TaskTrove archive, apply the versioned
identity repair, convert the judged task, and publish the result as one private ``@repaired``
revision with its per-task before/after ledger. Conversion is mechanical: no review, deduplication
or admission runs here — the repaired revision re-enters the normal gates afterward.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.conversion import convert_raw_row, row_source, row_task_id
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizedTask, RawRow, SourceRecipe
from taskcompendium.pipeline.sources import conversion_context

from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_REPO, TASKTROVE_REVISION
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import unpack_task_binary
from experiments.post_training.task_curation.datasets.tasktrove.identity_following import (
    identity_converter,
    repair_row_data,
)
from experiments.post_training.task_curation.identity_following_repair import (
    CONFIG,
    EXCLUSION_RULING,
    IMPORTER_REVISION,
    REPAIR_LABEL,
    REPAIR_RULE,
    SOURCE_ID,
)
from experiments.post_training.task_curation.license_publication import publish_licensed_population
from experiments.post_training.task_curation.remediation_publish import TASK_SCHEMA

logger = logging.getLogger(__name__)

RECIPE_NAME = "tasktrove-identity-following"
REPLY_ONLY_RUNTIME = {"task_machine": None, "verifier_machine": None}
LEDGER_FILE = "repair-ledger.jsonl"
MANIFEST_FILE = "manifest.json"
TASKS_FILE = "tasks.parquet"


def repair_recipe() -> SourceRecipe:
    """The mechanical recipe: pinned upstream, versioned importer revision, identity converter."""
    return SourceRecipe(
        name=RECIPE_NAME,
        version=IMPORTER_REVISION,
        source=SourceFiles(
            dataset=TASKTROVE_REPO,
            revision=TASKTROVE_REVISION,
            patterns=(f"{CONFIG}/tasks.parquet",),
            format=SourceFormat.PARQUET,
            decode=unpack_task_binary,
        ),
        convert=identity_converter(),
        rubric=None,
        controls=None,
        intended_use=IntendedUse.TRAIN,
    )


@dataclass(frozen=True)
class BuildCounts:
    """What one build produced and rejected."""

    converted: int
    rejected: int


def build_population(parquet_path: str, output: Path) -> BuildCounts:
    """Convert every pinned archive row into a repaired task, writing the population and ledger.

    The output directory receives ``tasks.parquet``, the per-task ``repair-ledger.jsonl`` and the
    ``manifest.json``. Every rejected row is ledgered; the build fails when any row rejects,
    because the repair must cover the complete declared population.
    """
    output.mkdir(parents=True, exist_ok=False)
    recipe = repair_recipe()
    context = conversion_context(recipe)
    tasks = output / TASKS_FILE
    ledger = output / LEDGER_FILE
    converted = 0
    rejected: list[dict[str, str]] = []
    with (
        pq.ParquetFile(parquet_path) as table,
        pq.ParquetWriter(tasks, TASK_SCHEMA, compression="zstd") as writer,
        ledger.open("w") as evidence,
    ):
        for batch in table.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                unpacked = unpack_task_binary(row, None)
                archive_sha256 = unpacked["archive_sha256"]
                task_id, locator = _row_identity(recipe, row["path"])
                repaired_data, repair_ledger = repair_row_data(unpacked)
                raw = RawRow(task_id, row_source(recipe, locator), repaired_data)
                outcome = convert_raw_row(raw, recipe.convert, context)
                if isinstance(outcome.result, ImportRejection):
                    rejected.append(
                        {
                            "archive_path": row["path"],
                            "task_id": task_id,
                            "reason": outcome.result.reason,
                            "detail": outcome.result.detail,
                        }
                    )
                    continue
                assert isinstance(outcome.result, NormalizedTask)
                encoded = outcome.result.task.model_dump_json()
                writer.write_batch(
                    pa.RecordBatch.from_pylist(
                        [
                            {
                                "id": outcome.result.task.id,
                                "path": row["path"],
                                "task_spec": encoded,
                                "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                                "runtime": json.dumps(REPLY_ONLY_RUNTIME),
                            }
                        ],
                        schema=TASK_SCHEMA,
                    )
                )
                converted += 1
                evidence.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "archive_path": row["path"],
                            "original_archive_sha256": archive_sha256,
                            "repair_rule": REPAIR_RULE,
                            "repair_label": REPAIR_LABEL,
                            "files": repair_ledger,
                            "normalization_changes": [change.model_dump() for change in outcome.result.changes],
                            "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                        }
                    )
                    + "\n"
                )
    if rejected:
        raise ValueError(f"{len(rejected)} rows rejected during the repair build: {rejected[:3]}")
    manifest = {
        "source_id": SOURCE_ID,
        "repair_label": REPAIR_LABEL,
        "repair_rule": REPAIR_RULE,
        "parent_exclusion_ruling": EXCLUSION_RULING,
        "input": {
            "repo": TASKTROVE_REPO,
            "revision": TASKTROVE_REVISION,
            "config": CONFIG,
            "file": f"{CONFIG}/tasks.parquet",
            "file_sha256": file_sha256(StoragePath(parquet_path)),
            "task_count": converted,
        },
        "output": {
            "task_count": converted,
            "rejected_count": 0,
            "tasks_parquet_sha256": file_sha256(StoragePath(str(tasks))),
            "ledger_sha256": file_sha256(StoragePath(str(ledger))),
        },
        "recipe": {"name": RECIPE_NAME, "version": IMPORTER_REVISION},
    }
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    logger.info("Built %d repaired tasks in %s", converted, output)
    return BuildCounts(converted=converted, rejected=0)


def _row_identity(recipe: SourceRecipe, archive_path: str) -> tuple[str, str]:
    locator = f"{CONFIG}/{archive_path}"
    source = row_source(recipe, locator)
    return row_task_id(recipe, source), locator


def publish_population(output: Path, repo: str) -> str:
    """Publish the built population through the licensed publication chokepoint.

    The repaired population derives from the pinned TaskTrove archive's receipt: the
    card declares that receipt's resolved upstream, the upload is read back, and the
    new pin's receipt registers before the revision is returned.
    """
    manifest = json.loads((output / MANIFEST_FILE).read_text())
    body = (
        f"# {SOURCE_ID} {REPAIR_LABEL}\n\n"
        "Private repaired identity-following population (ruling 2026-11-23). Every task carries the\n"
        "versioned identity repair applied to the pinned TaskTrove archive; the per-task before/after\n"
        f"ledger rides beside the tasks. Repair rule: {REPAIR_RULE}\n\n"
        f"Input archive: {TASKTROVE_REPO}@{TASKTROVE_REVISION}, config {CONFIG}\n"
        f"({manifest['input']['file_sha256']}); importer {RECIPE_NAME} at {IMPORTER_REVISION}.\n"
        f"Exclusion ruling on record: {EXCLUSION_RULING}\n"
        "Conversion is mechanical: no review, deduplication or admission runs here — the repaired\n"
        "revision re-enters the normal gates afterward.\n"
    )
    publication = publish_licensed_population(
        SOURCE_ID,
        repo,
        output,
        (TASKS_FILE, LEDGER_FILE, MANIFEST_FILE),
        body=body,
        pretty_name=f"{CONFIG} repaired identity-following tasks",
        use=IntendedUse.TRAIN.value,
    )
    logger.info("Published %s at revision %s", publication.repo, publication.revision)
    return publication.revision


def main() -> None:
    """Run the build CLI: convert the pinned population and optionally publish it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="Local path of the pinned config's tasks.parquet")
    parser.add_argument("--output", required=True, help="Build output directory")
    parser.add_argument("--publish-repo", default=None, help="Publish the build to this HF dataset repo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = build_population(args.parquet, Path(args.output))
    if args.publish_repo:
        revision = publish_population(Path(args.output), args.publish_repo)
        print(json.dumps({"converted": counts.converted, "repo": args.publish_repo, "revision": revision}))


if __name__ == "__main__":
    main()
