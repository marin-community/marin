# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce one sealed branch lexical measurement from main's lexical stage.

The sealed branch measured the converted census of a source — every converter-emitted
TaskSpec, exact contracts deduplicated — rendered with the shared conversation renderer,
and marked it against a frozen eval corpus. This script rebuilds that population from
main's registered converter, adopts the receipt's frozen bloom and drop sets, marks and
attributes through main's stage engine, and compares the result with the sealed
evidence: the branch's frozen text population and its frozen per-task marks.

A converter that no longer emits a sealed row (for example, a reference the current
math grader cannot parse) cannot be remeasured; the script reports such rows as a
population delta and compares the stage output per task over the shared population.

```bash
REGISTRY=s3://marin-us-east-02a/marin/rl-data/atlas/lexical/scan/df7033bcf80e518f3684
TEXT=s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text/9d7386fb9abe0a4c659b/2026.11.12.2

uv run python -m experiments.post_training.task_curation.lexical_reproduction \
    --source-id ASDiv:asdiv \
    --receipt receipts/part-000.json \
    --bloom s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_eca06d42 \
    --global-drop-set $REGISTRY/df/2026.11.12.6/_global \
    --frozen-source-drop-set $REGISTRY/df/2026.11.12.6/9d7386fb9abe0a4c659b \
    --branch-population $TEXT \
    --branch-flags $REGISTRY/report/2026.11.12.6/9d7386fb9abe0a4c659b/task_flags \
    --branch-drop-manifest $REGISTRY/report/2026.11.12.6/9d7386fb9abe0a4c659b \
    --branch-population-record $TEXT \
    --artifact-root s3://marin-us-east-02a/marin/rl-data/lexical-reproduction \
    --output reproduction.json --run
```
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, format_conversation
from taskcompendium.pipeline.conversion import convert_source_row
from taskcompendium.pipeline.lexical import (
    POPULATION_SCHEMA,
    TEXT_RENDERER_VERSION,
    MarkedPopulation,
    TextPopulation,
    population_summary,
    summarize_flags,
)
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask
from taskcompendium.pipeline.sources import conversion_context, source_shards, staged_raw_file_rows
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.lexical_engine import (
    DEFAULT_WORKER_RESOURCES,
    DatakitDecontamination,
)
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.pipeline import download_step, source_recipe
from experiments.post_training.task_curation.sources import all_sources

logger = logging.getLogger(__name__)

RECEIPT_FIELDS = (
    "measured_tasks",
    "contaminated_tasks",
    "contaminated_fraction",
    "max_overlap",
    "mean_max_overlap",
    "p95_max_overlap",
    "matched_evals",
)
FLAG_FIELDS = ("contaminated", "max_overlap", "matched_hashes", "matched_eval_ids", "matched_eval_families")


def census_population(source_id: str, source_input: str) -> tuple[list[TaskSpec], dict[str, int]]:
    """Every converter-emitted TaskSpec of one staged source, before admission gates."""
    source = all_sources()[source_id]
    recipe_config = source.config
    if not hasattr(recipe_config, "convert"):
        raise ValueError(f"Source has no registered TaskSpec converter: {source_id}")
    recipe = source_recipe(recipe_config, {}, None)
    context = conversion_context(recipe)
    tasks: list[TaskSpec] = []
    rejections: Counter[str] = Counter()
    for shard in source_shards(source_input, recipe.source):
        for record in staged_raw_file_rows(source_input, shard, recipe.source, context):
            conversion = convert_source_row(record, recipe)
            if isinstance(conversion.result, NormalizedTask | TaskSpec):
                tasks.append(conversion.result.task)
            elif isinstance(conversion.result, ImportRejection):
                rejections[f"{conversion.result.kind.value}:{conversion.result.reason}"] += 1
    return tasks, dict(rejections)


def contract_identity(task: TaskSpec) -> str:
    """The complete task contract without provenance, matching the sealed census's dedup key."""
    document = task.model_dump(mode="json")
    for field in ("id", "source", "tags", "labels"):
        document.pop(field, None)
    return canonical_sha256(document)


def deduplicate_contracts(tasks: list[TaskSpec]) -> tuple[list[TaskSpec], int]:
    """Keep the lowest task ID per exact contract, as the sealed census did."""
    kept: dict[str, TaskSpec] = {}
    for task in sorted(tasks, key=lambda task: task.id):
        identity = contract_identity(task)
        if identity in kept:
            continue
        kept[identity] = task
    return list(kept.values()), len(tasks) - len(kept)


def write_population(tasks: list[TaskSpec], output_path: str, source_name: str) -> TextPopulation:
    table = pa.Table.from_pylist(
        [{"id": task.id, "text": format_conversation(task.context.events), "partition_id": 0} for task in tasks],
        schema=POPULATION_SCHEMA,
    )
    path = StoragePath(prefix_join(output_path, "population/part-00000.parquet"))
    with path.open("wb", auto_mkdir=True) as stream:
        pq.write_table(table, stream, compression="zstd")
    return TextPopulation(source=source_name, path=str(StoragePath(output_path) / "population"), task_count=len(tasks))


def text_digest(tasks: list[TaskSpec]) -> str:
    """The sealed census's text digest: a SHA-256 chain over ``[task_id, rendered text]`` in task-ID order."""
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda task: task.id):
        digest.update(json.dumps([task.id, format_conversation(task.context.events)], ensure_ascii=False).encode())
    return digest.hexdigest()


def _parquet_files(pattern: str) -> list[str]:
    files = sorted(str(path) for path in StoragePath(pattern).glob())
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern}")
    return files


def read_branch_population(path: str) -> dict[str, str]:
    """The sealed branch's frozen text population, keyed by task ID."""
    population: dict[str, str] = {}
    for file in _parquet_files(prefix_join(path, "outputs/main/*.parquet")):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["id", "text"]):
                for task_id, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist(), strict=True):
                    if task_id in population:
                        raise ValueError(f"Duplicate task ID in the branch population: {task_id}")
                    population[task_id] = text
    if not population:
        raise FileNotFoundError(f"The branch population is empty: {path}")
    return population


def read_flags(path: str, pattern: str = "*.parquet") -> dict[str, dict[str, Any]]:
    """Per-task marks keyed by task ID: contamination, overlap, matches and attribution."""
    flags: dict[str, dict[str, Any]] = {}
    for file in _parquet_files(prefix_join(path, pattern)):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["id", *FLAG_FIELDS]):
                for row in batch.to_pylist():
                    if row["id"] in flags:
                        raise ValueError(f"Duplicate task ID in the marks: {row['id']}")
                    flags[row["id"]] = {field: row[field] for field in FLAG_FIELDS}
    if not flags:
        raise FileNotFoundError(f"The marks are empty: {path}")
    return flags


def read_population_record(path: str) -> dict[str, Any]:
    """The sealed population artifact's record, with its digests and counts."""
    with StoragePath(prefix_join(path, ".artifact.json")).open("rt") as stream:
        record = json.load(stream)
    result = record.get("result")
    if not isinstance(result, dict) or "text_sha256" not in result:
        raise ValueError(f"The artifact record is not a text population: {path}")
    return result


def read_dropped_ids(path: str) -> set[str]:
    """The sealed release's dropped task IDs from its drop manifest."""
    dropped = set()
    for file in _parquet_files(prefix_join(path, "drop_manifest*.parquet")):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["id"]):
                dropped.update(batch.column("id").to_pylist())
    if not dropped:
        raise FileNotFoundError(f"The sealed drop manifest is empty: {path}")
    return dropped


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True, help="Published receipt JSON for this source")
    parser.add_argument("--bloom", required=True)
    parser.add_argument("--global-drop-set", required=True)
    parser.add_argument("--frozen-source-drop-set", required=True)
    parser.add_argument("--branch-population", required=True, help="The sealed population's artifact root")
    parser.add_argument("--branch-flags", required=True, help="The sealed per-task marks' directory")
    parser.add_argument("--branch-drop-manifest", help="The sealed drop manifest (dropped contaminated task IDs)")
    parser.add_argument("--branch-population-record", help="The sealed population artifact root (.artifact.json)")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    configure_coreweave_s3()
    receipt = json.loads(args.receipt.read_text())
    measured_input = receipt["input"]
    source = all_sources()[args.source_id]
    dataset = source.dataset
    assert dataset is not None and dataset.revision is not None
    if (dataset.name, dataset.revision, receipt["source_id"]) != (
        measured_input["dataset_id"],
        measured_input["dataset_revision"],
        args.source_id,
    ):
        raise ValueError(f"The receipt belongs to another pinned input: {args.source_id}")

    if not args.run:
        args.output.write_text(json.dumps({"source_id": args.source_id, "plan": "planned"}, indent=2) + "\n")
        return

    marked, census = _measure(args, source, receipt)
    result = _compare(marked, census, receipt, args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with StoragePath(prefix_join(args.artifact_root, "reproduction.json")).open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Reproduction result written to %s and %s: reproduced=%s",
        args.output,
        prefix_join(args.artifact_root, "reproduction.json"),
        result["reproduced"],
    )


def _measure(args: argparse.Namespace, source: Any, receipt: dict[str, Any]) -> tuple[MarkedPopulation, dict[str, Any]]:
    """Rebuild the census, adopt the frozen corpus, and mark it through the stage engine."""
    with ExitStack() as cleanup:
        client = LocalClient()
        cleanup.callback(client.shutdown)
        cleanup.enter_context(set_current_client(client))
        runtime = CampaignRuntime()
        with artifact_prefix(args.artifact_root):
            with ZephyrContext(
                client=client, max_workers=args.max_workers, resources=DEFAULT_WORKER_RESOURCES
            ) as context:
                with runtime.activate(context):
                    staged = run(download_step(source.config.source, runtime), max_concurrent=1)[0]
                    source_input = staged.path
                tasks, rejections = census_population(args.source_id, source_input)
                unique, duplicates = deduplicate_contracts(tasks)
                logger.info(
                    "%s: %d converted rows, %d exact duplicates, %d census tasks; rejections %s",
                    args.source_id,
                    len(tasks),
                    duplicates,
                    len(unique),
                    rejections,
                )
                population = write_population(unique, str(args.output.parent / "population"), source.config.name)
                engine = DatakitDecontamination(
                    bloom_dir=args.bloom,
                    global_drop_set_dir=args.global_drop_set,
                    eval_corpus_version=receipt["measurement"]["eval_corpus_version"],
                    frozen_source_drop_sets={source.config.name: args.frozen_source_drop_set},
                )
                marked = engine.mark_population(population, str(args.output.parent / "marks"), context=context)
        return marked, {
            "tasks": tasks,
            "unique": unique,
            "converted_rows": len(tasks),
            "exact_duplicates": duplicates,
            "converter_rejections": rejections,
        }


def _compare(
    marked: MarkedPopulation, census: dict[str, Any], receipt: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Compare main's marks with the sealed population, flags, drop manifest and digests."""
    unique: list[TaskSpec] = census["unique"]
    main_flags = read_flags(marked.flags_dir)
    branch_population = read_branch_population(args.branch_population)
    branch_flags = read_flags(args.branch_flags)
    rendered = {task.id: format_conversation(task.context.events) for task in unique}
    shared = sorted(set(rendered) & set(branch_flags))
    mismatches = [
        {"id": task_id, "field": field, "main": main_flags[task_id][field], "branch": branch_flags[task_id][field]}
        for task_id in shared
        for field in FLAG_FIELDS
        if main_flags[task_id][field] != branch_flags[task_id][field]
    ]
    main_summary = summarize_flags([main_flags[task_id] for task_id in shared])
    branch_summary = summarize_flags([branch_flags[task_id] for task_id in shared])

    retained: dict[str, Any] = {}
    if args.branch_drop_manifest:
        branch_dropped = read_dropped_ids(args.branch_drop_manifest)
        main_dropped = {task_id for task_id in shared if main_flags[task_id]["contaminated"]}
        retained = {
            "branch_dropped_tasks": len(branch_dropped),
            "main_dropped_shared_tasks": len(main_dropped),
            "shared_dropped_ids_equal": {task_id for task_id in shared if task_id in branch_dropped} == main_dropped,
            "shared_retained_tasks": len(shared) - len(main_dropped),
        }

    digests: dict[str, Any] = {}
    if args.branch_population_record:
        record = read_population_record(args.branch_population_record)
        shared_tasks = [task for task in unique if task.id in branch_population]
        digests = {
            "branch_text_sha256": record["text_sha256"],
            "main_census_text_sha256": text_digest(unique),
            "main_shared_text_sha256": text_digest(shared_tasks),
            "branch_task_count": record["task_count"],
            "main_census_task_count": len(unique),
            "note": (
                "task_specs_sha256 is not comparable: the sealed digest pins schema-0.30 "
                "serializations, main's census is schema-0.31; the population is identified by the "
                "text digest and per-task text equality instead."
            ),
        }

    full_summary = population_summary(marked.flags_dir)
    fields = {
        name: {
            "receipt": receipt["measurement"][name],
            "main_full_population": getattr(full_summary, name),
            "main_shared_population": getattr(main_summary, name),
            "branch_shared_population": getattr(branch_summary, name),
        }
        for name in RECEIPT_FIELDS
    }
    return {
        "source_id": args.source_id,
        "population": {
            "converted_rows": census["converted_rows"],
            "exact_duplicates": census["exact_duplicates"],
            "converter_rejections": dict(census["converter_rejections"]),
            "main_census_tasks": len(unique),
            "branch_population_tasks": len(branch_population),
            "shared_tasks": len(shared),
            "branch_only_tasks": sorted(set(branch_population) - set(rendered)),
            "main_only_tasks": sorted(set(rendered) - set(branch_flags)),
            "shared_texts_equal": all(rendered[task_id] == branch_population[task_id] for task_id in shared),
        },
        "shared_measurement_equal": main_summary == branch_summary,
        "retained": retained,
        "digests": digests,
        "per_task_flag_mismatch_count": len(mismatches),
        "per_task_flag_mismatches": mismatches[:20],
        "measurement_fields": fields,
        "renderer": TEXT_RENDERER_VERSION,
        "reproduced": (
            not mismatches
            and not (set(rendered) - set(branch_flags))
            and main_summary == branch_summary
            and all(rendered[task_id] == branch_population[task_id] for task_id in shared)
            and (not retained or retained["shared_dropped_ids_equal"])
        ),
    }


if __name__ == "__main__":
    main()
