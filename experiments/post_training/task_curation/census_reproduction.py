# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce one sealed branch census receipt through main's census stage.

The sealed branch measured the complete converted census of a source — every
converter-emitted TaskSpec — and bound it to the dataset pin, the converter
identity and the population digest. This script runs main's FULL source
pipeline on the same pinned source with the census engine, so the stage path
itself (convert, export, measure) produces the evidence, and compares the
result with the sealed receipt: the measured population, the machine demands
and refusals, and the integrity-contract groups.

```bash
uv run python -m experiments.post_training.task_curation.census_reproduction \\
    --source-id ASDiv:asdiv \\
    --receipt asdiv_census_receipt.json \\
    --branch-population s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text/9d7386fb9abe0a4c659b/2026.11.12.2 \\
    --artifact-root s3://marin-us-east-02a/marin/rl-data/census-reproduction \\
    --output reproduction.json --run
```

The pipeline runs without model review (the recipe's rubric is cleared), so the
measured population is every task the converter emits — the census the sealed
receipt measured, minus rows main's converter now rejects. A converter that no
longer emits a sealed row cannot be remeasured; the script reports such rows as
a population delta and compares the receipts' structural fields, which are
population-size independent.
"""

import argparse
import json
import logging
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.lazy import run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.pipeline.source_processing import (
    SourcePipelineResult,
    SourceProcessingMode,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SOURCE_QUALITY_SAMPLE_SIZE
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.pipeline import _pipeline_config, download_step, source_recipe
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import all_sources

logger = logging.getLogger(__name__)

REPRODUCTION_JSON = "reproduction.json"
MEASUREMENT_FIELDS = (
    "daytona_machine_slots",
    "daytona_snapshot_count",
    "daytona_execution_failures",
    "daytona_roles",
    "daytona_unsafe_reasons",
    "daytona_unsafe_details",
    "daytona_unsafe_counts",
)
CONTRACT_FIELDS = ("distinct_groups", "required_shared_tasks", "shared_missing_contract_tasks")


def _parquet_files(pattern: str) -> list[str]:
    files = sorted(str(path) for path in StoragePath(pattern).glob())
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern}")
    return files


def read_branch_population_ids(path: str) -> set[str]:
    """The sealed branch's frozen population task IDs, from its published text shards."""
    population: set[str] = set()
    for file in _parquet_files(prefix_join(path, "outputs/main/*.parquet")):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["id"]):
                population.update(batch.column("id").to_pylist())
    if not population:
        raise FileNotFoundError(f"The branch population is empty: {path}")
    return population


def read_stage_population_ids(result: SourcePipelineResult) -> set[str]:
    """The stage's measured population: every converted task in the normalize view."""
    population: set[str] = set()
    for file in _parquet_files(prefix_join(result.normalize_path, "*.parquet")):
        with StoragePath(file).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["task_id", "task_json"]):
                ids = batch.column("task_id").to_pylist()
                tasks = batch.column("task_json").to_pylist()
                population.update(task_id for task_id, task_json in zip(ids, tasks, strict=True) if task_json)
    if not population:
        raise FileNotFoundError(f"The stage converted no tasks: {result.normalize_path}")
    return population


def _run_stage(args: argparse.Namespace, source: RlDataSource, run_root: str) -> SourcePipelineResult:
    """Run the FULL source pipeline on the staged source through main's production settings."""
    with ExitStack() as cleanup:
        client = LocalClient()
        cleanup.callback(client.shutdown)
        cleanup.enter_context(set_current_client(client))
        runtime = CampaignRuntime()
        resources = ResourceConfig(cpu=2, ram="8g")
        with artifact_prefix(args.artifact_root):
            with ZephyrContext(client=client, max_workers=args.max_workers, resources=resources) as context:
                with runtime.activate(context):
                    staged = run(download_step(source.config.source, runtime), max_concurrent=1)[0]
                    source_input = staged.path
                recipe = source_recipe(source.config, {}, None)
                # The sealed receipt measured the converter's census, before model review.
                # Clearing the rubric keeps this run's measured population equal to that census.
                recipe = replace(recipe, rubric=None)
                config = _pipeline_config(
                    SourceProcessingMode.FULL,
                    None,
                    None,
                    None,
                    seed=0,
                    quality_sample_size=SOURCE_QUALITY_SAMPLE_SIZE,
                    verification_sample_size=20,
                    max_workers=args.max_workers,
                    worker_resources=resources,
                    normalized_shards=2,
                )
                output = prefix_join(run_root, "pipeline")
                return run_source_pipeline(
                    recipe,
                    context,
                    source_input,
                    output,
                    config,
                    mode=SourceProcessingMode.FULL,
                    canonical_source=recipe.name,
                )


def _compare(
    result: SourcePipelineResult, receipt: dict[str, Any], branch_population: set[str] | None
) -> dict[str, Any]:
    """Compare the stage run's census with the sealed receipt."""
    manifest = json.loads(StoragePath(result.manifest_path).read_text())
    census = manifest["census"]
    if census is None:
        raise ValueError("The stage run recorded no census measurement")
    measurement = census["measurement"]
    contracts = census["integrity_contracts"]
    stage_ids = read_stage_population_ids(result)
    population: dict[str, Any] = {
        "receipt_measured_tasks": receipt["input"]["measured_tasks"],
        "stage_measured_tasks": census["input"]["measured_tasks"],
    }
    if branch_population is not None:
        population.update(
            {
                "branch_population_tasks": len(branch_population),
                "stage_only_tasks": sorted(stage_ids - branch_population),
                "branch_only_tasks": sorted(branch_population - stage_ids),
            }
        )
    fields = {name: {"receipt": receipt["measurement"][name], "stage": measurement[name]} for name in MEASUREMENT_FIELDS}
    contract_comparison = {
        name: {"receipt": receipt["integrity_contracts"][name], "stage": contracts[name]} for name in CONTRACT_FIELDS
    }
    receipt_absent = receipt["integrity_contracts"]["absent_contract_tasks"]
    stage_absent = contracts["absent_contract_tasks"]
    population_delta = population["receipt_measured_tasks"] - population["stage_measured_tasks"]
    return {
        "source_id": receipt["source_id"],
        "stage": {
            "status": manifest["status"],
            "raw_population_count": manifest["raw_population_count"],
            "measured_population": (
                "every converted task of the checked audit (the converter census here, "
                "because the harness clears the rubric)"
            ),
            "admission_counts": manifest["admission_counts"],
            "rejections": census["rejections"],
        },
        "population": population,
        "measurement_fields": fields,
        "contract_fields": contract_comparison,
        "contract_population_delta": {
            "receipt_absent_contract_tasks": receipt_absent,
            "stage_absent_contract_tasks": stage_absent,
            "matches_population_delta": (receipt_absent - stage_absent) == population_delta,
        },
        "pins": {
            "dataset_id": {"receipt": receipt["input"]["dataset_id"], "stage": census["input"]["dataset_id"]},
            "dataset_revision": {
                "receipt": receipt["input"]["dataset_revision"],
                "stage": census["input"]["dataset_revision"],
            },
            "note": (
                "converter_revision and task_specs_sha256 are not comparable: the receipt pins the branch's "
                "git revision and schema-0.30 serializations; main records its recipe revision and digests "
                "schema-0.31 serializations."
            ),
        },
        "grading_environment": census["grading_environment"],
        "reproduced": (
            all(field["receipt"] == field["stage"] for field in fields.values())
            and all(field["receipt"] == field["stage"] for field in contract_comparison.values())
            and (receipt["integrity_contracts"]["example"] is None) == (contracts["example"] is None)
            and (branch_population is None or not population["stage_only_tasks"])
            and population_delta >= 0
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--receipt", required=True, help="Sealed census receipt JSON for this source")
    parser.add_argument(
        "--branch-population",
        help="The sealed population's artifact root, to identify the sealed census's task IDs",
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Key for this run's artifacts below the artifact root; defaults to the UTC start time.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    # Stage artifacts key by source and run, never by a shared version string:
    # two lanes over the same root must not overwrite each other's reports.
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = prefix_join(args.artifact_root, f"runs/{args.source_id.replace(':', '-')}/{run_id}")

    configure_coreweave_s3()
    receipt = json.loads(StoragePath(args.receipt).read_text())
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
    branch_population = read_branch_population_ids(args.branch_population) if args.branch_population else None

    if not args.run:
        args.output.write_text(json.dumps({"source_id": args.source_id, "plan": "planned"}, indent=2) + "\n")
        return

    result = _run_stage(args, source, run_root)
    comparison = _compare(result, receipt, branch_population)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    per_run = prefix_join(run_root, REPRODUCTION_JSON)
    with StoragePath(per_run).open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    with StoragePath(prefix_join(args.artifact_root, REPRODUCTION_JSON)).open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Reproduction result written to %s, %s and %s: reproduced=%s",
        args.output,
        per_run,
        prefix_join(args.artifact_root, REPRODUCTION_JSON),
        comparison["reproduced"],
    )


if __name__ == "__main__":
    main()
