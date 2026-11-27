# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce one sealed branch lexical measurement through main's lexical stage.

The sealed branch measured the converted census of a source — every converter-emitted
TaskSpec, exact contracts deduplicated — rendered with the shared conversation renderer,
and marked it against a frozen eval corpus. This script runs main's FULL source
pipeline on the same pinned source with the receipt's frozen bloom and drop sets, so
the stage path itself (population render, mark, attribution, training gate, export)
produces the evidence, and compares the result with the sealed release: the branch's
frozen text population, its frozen per-task marks, and its retained population.

The pipeline runs without model review (the recipe's rubric is cleared), so the kept
population is every task the converter emits — the same census the sealed receipt
measured, minus rows main's converter now rejects. A converter that no longer emits
a sealed row (for example, a reference the current math grader cannot parse) cannot
be remeasured; the script reports such rows as a population delta and compares the
stage output per task over the shared population.

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
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.models import TaskSpec, format_conversation
from taskcompendium.pipeline.lexical import (
    TEXT_RENDERER_VERSION,
    summarize_flags,
)
from taskcompendium.pipeline.models import FilterPolicy
from taskcompendium.pipeline.source_processing import (
    SourcePipelineConfig,
    SourcePipelineResult,
    SourceProcessingMode,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution
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


def text_digest_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    """The sealed census's text digest: a SHA-256 chain over ``[task_id, text]`` in task-ID order."""
    digest = hashlib.sha256()
    for task_id, text in sorted(pairs):
        digest.update(json.dumps([task_id, text], ensure_ascii=False).encode())
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


def read_final_population(result: SourcePipelineResult) -> dict[str, str]:
    """The stage's retained population, rendered from the ``final/`` view."""
    population: dict[str, str] = {}
    for file in _parquet_files(prefix_join(result.final_path, "*.parquet")):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["task_id", "task_json"]):
                for task_id, task_json in zip(
                    batch.column("task_id").to_pylist(), batch.column("task_json").to_pylist(), strict=True
                ):
                    task = TaskSpec.model_validate_json(task_json)
                    if task_id in population:
                        raise ValueError(f"Duplicate task ID in final/: {task_id}")
                    population[task_id] = format_conversation(task.context.events)
    if not population:
        raise FileNotFoundError(f"The stage retained no tasks: {result.final_path}")
    return population


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

    result = _run_stage(args, source, receipt)
    comparison = _compare(result, receipt, args)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    with StoragePath(prefix_join(args.artifact_root, "reproduction.json")).open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Reproduction result written to %s and %s: reproduced=%s",
        args.output,
        prefix_join(args.artifact_root, "reproduction.json"),
        comparison["reproduced"],
    )


def _run_stage(args: argparse.Namespace, source: Any, receipt: dict[str, Any]) -> SourcePipelineResult:
    """Run the FULL source pipeline on the staged source with the frozen corpus engine."""
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
                engine = DatakitDecontamination(
                    bloom_dir=args.bloom,
                    global_drop_set_dir=args.global_drop_set,
                    eval_corpus_version=receipt["measurement"]["eval_corpus_version"],
                    frozen_source_drop_sets={source.config.name: args.frozen_source_drop_set},
                )
                recipe = source_recipe(source.config, {}, None)
                # The sealed receipt measured the converter's census, before model review.
                # Clearing the rubric keeps this run's kept population equal to that census.
                recipe = replace(recipe, rubric=None)
                config = SourcePipelineConfig(
                    mode=SourceProcessingMode.FULL,
                    quality_policy=SourceQualityPolicy(),
                    verification_policy=SourceVerificationPolicy(20, 0, 1, 0.95),
                    review=None,
                    execution=AuditExecution(max_workers=args.max_workers, worker_resources=DEFAULT_WORKER_RESOURCES),
                    filter_policy=FilterPolicy(),
                    normalized_shards=2,
                    machines=None,
                    decontamination=engine,
                )
                output = prefix_join(args.artifact_root, "pipeline")
                return run_source_pipeline(
                    recipe,
                    context,
                    source_input,
                    output,
                    config,
                    mode=SourceProcessingMode.FULL,
                    canonical_source=recipe.name,
                )


def _compare(result: SourcePipelineResult, receipt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Compare the stage run's marks, retained population and digests with the sealed release."""
    manifest = json.loads(StoragePath(result.manifest_path).read_text())
    lexical = manifest["lexical"]
    if lexical is None:
        raise ValueError("The stage run recorded no lexical measurement")
    main_flags = read_flags(prefix_join(result.lexical_path, "task_flags"))
    branch_population = read_branch_population(args.branch_population)
    branch_flags = read_flags(args.branch_flags)
    rendered = read_final_population(result)
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
        sealed_retained = {task_id for task_id in branch_population if task_id not in branch_dropped}
        main_retained = set(rendered)
        retained = {
            "branch_retained_tasks": len(sealed_retained),
            "main_retained_tasks": len(main_retained),
            "shared_retained_ids_equal": (
                {task_id for task_id in sealed_retained if task_id in branch_flags} == main_retained
            ),
            "shared_retained_tasks": len(shared) - sum(1 for task_id in shared if task_id in branch_dropped),
        }

    digests: dict[str, Any] = {}
    if args.branch_population_record:
        record = read_population_record(args.branch_population_record)
        shared_tasks = {task_id: text for task_id, text in rendered.items() if task_id in branch_population}
        digests = {
            "branch_text_sha256": record["text_sha256"],
            "main_retained_text_sha256": text_digest_pairs(rendered.items()),
            "main_shared_text_sha256": text_digest_pairs(shared_tasks.items()),
            "branch_task_count": record["task_count"],
            "main_retained_task_count": len(rendered),
            "note": (
                "task_specs_sha256 is not comparable: the sealed digest pins schema-0.30 "
                "serializations, main's pipeline is schema-0.31; the population is identified by the "
                "text digest and per-task text equality instead."
            ),
        }

    fields = {
        name: {
            "receipt": receipt["measurement"][name],
            "stage_full_population": lexical[name],
            "stage_shared_population": getattr(main_summary, name),
            "branch_shared_population": getattr(branch_summary, name),
        }
        for name in RECEIPT_FIELDS
    }
    return {
        "source_id": args.source_id,
        "stage": {
            "status": manifest["status"],
            "raw_population_count": manifest["raw_population_count"],
            "converted_rows": manifest["raw_population_count"],
            "measured_population": (
                "kept rows of the checked audit (the converter census here, because the " "harness clears the rubric)"
            ),
            "admission_counts": manifest["admission_counts"],
        },
        "population": {
            "branch_population_tasks": len(branch_population),
            "stage_task_flags": len(main_flags),
            "main_retained_tasks": len(rendered),
            "shared_tasks": len(shared),
            "branch_only_tasks": sorted(set(branch_population) - set(main_flags)),
            "main_only_tasks": sorted(set(main_flags) - set(branch_flags)),
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
            and not (set(main_flags) - set(branch_flags))
            and main_summary == branch_summary
            and all(rendered[task_id] == branch_population[task_id] for task_id in shared)
            and (not retained or retained["shared_retained_ids_equal"])
        ),
    }


if __name__ == "__main__":
    main()
