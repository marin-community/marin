# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze retained Atlas task contracts before a full quality review."""

import argparse
import json
import logging
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.source_identity import canonical_source_id, original_source_id
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_environment import assess_grading_environment
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import has_oracle_solution

from experiments.post_training.task_curation.full_qc_artifacts import FullQCInputs
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.published_census import population_documents
from experiments.post_training.task_curation.retention_plan import RetentionPlan

logger = logging.getLogger(__name__)
INDEX_SCHEMA = pa.schema([("task_id", pa.string()), ("task_sha256", pa.string()), ("runtime_sha256", pa.string())])
INDEX_BATCH_SIZE = 1000


@dataclass(frozen=True)
class InputConfig:
    population_path: str
    retention: RetentionPlan
    drop_manifest_sha256: str
    output_path: str


def prepare_inputs(config: InputConfig) -> FullQCInputs:
    """Index the complete remainder, preserving separate execution diagnostics."""
    plan = config.retention
    population = SourceTextPopulation.raw_load(config.population_path)
    if (
        canonical_source_id(population.source_id),
        population.dataset_id,
        population.dataset_revision,
        population.source_version,
        population.converter_revision,
        population.task_specs_sha256,
        population.task_count,
    ) != (
        canonical_source_id(plan.source_id),
        plan.source_dataset_id,
        plan.source_revision,
        plan.converter_version,
        plan.converter_revision,
        plan.task_specs_sha256,
        plan.raw_count,
    ):
        raise ValueError(f"Retention plan differs from frozen population: {plan.source_id}")
    if plan.lexical_use != "train" or plan.raw_count != plan.retained_count + plan.combined_drop_count:
        raise ValueError(f"Incomplete training retention accounting: {plan.source_id}")
    if file_sha256(StoragePath(plan.combined_drop_manifest_uri)) != config.drop_manifest_sha256:
        raise ValueError(f"Drop manifest changed after planning: {plan.source_id}")
    with StoragePath(plan.combined_drop_manifest_uri).open("rb") as stream:
        dropped = pq.read_table(stream, columns=["task_id"]).column(0).to_pylist()
    drop_ids = set(dropped)
    if len(drop_ids) != len(dropped) or len(dropped) != plan.combined_drop_count:
        raise ValueError(f"Drop manifest count differs from plan: {plan.source_id}")

    # Each preempted attempt keeps its partial index; completed artifacts are reused.
    index_path = str(StoragePath(config.output_path) / "attempts" / uuid4().hex / "contracts.parquet")
    seen: set[str] = set()
    count = task_bytes = oracle_tasks = missing_requirements = 0
    environments: Counter[str] = Counter()
    rows = []
    with StoragePath(index_path).open("wb", auto_mkdir=True) as stream:
        with pq.ParquetWriter(stream, INDEX_SCHEMA, compression="zstd") as writer:
            for document, runtime in population_documents(population):
                task = TaskSpec.model_validate(document)
                if task.id in seen:
                    raise ValueError(f"Duplicate task ID in frozen population: {task.id}")
                seen.add(task.id)
                if task.id in drop_ids:
                    continue
                rows.append(
                    {
                        "task_id": task.id,
                        "task_sha256": canonical_sha256(task.model_dump(mode="json")),
                        "runtime_sha256": canonical_sha256(runtime),
                    }
                )
                count += 1
                task_bytes += len(task.model_dump_json().encode())
                oracle_tasks += has_oracle_solution(task)
                missing_requirements += task.execution_requirements is None
                boundary = assess_grading_environment(task)
                environments[f"{boundary.required}:{boundary.status}"] += 1
                if len(rows) == INDEX_BATCH_SIZE:
                    writer.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA))
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA))
    if len(seen) != plan.raw_count or count != plan.retained_count or not drop_ids <= seen:
        raise ValueError(f"Frozen input does not cover the complete retention plan: {plan.source_id}")
    logger.info("%s: indexed %d retained tasks; %d proposed drops", plan.source_id, count, len(drop_ids))
    return FullQCInputs(
        path=config.output_path,
        source_id=canonical_source_id(plan.source_id),
        population=population,
        retention=plan,
        drop_manifest_sha256=config.drop_manifest_sha256,
        index_path=index_path,
        index_sha256=file_sha256(StoragePath(index_path)),
        retained_count=count,
        task_bytes=task_bytes,
        oracle_tasks=oracle_tasks,
        missing_execution_requirements=missing_requirements,
        grading_environments=dict(environments),
    )


def input_config(population: ArtifactStep, plan: RetentionPlan, drop_digest: str, ctx: StepContext) -> InputConfig:
    return InputConfig(ctx.artifact_path(population), plan, drop_digest, ctx.output_path)


def input_steps(plans: list[RetentionPlan], version: str) -> list[ArtifactStep]:
    steps = []
    for plan in plans:
        key = canonical_sha256(original_source_id(plan.source_id))[:20]
        population = ArtifactStep.adopt(
            f"rl-data/atlas/full-qc/input/{key}", version, plan.input_artifact, kind=SourceTextPopulation
        )
        steps.append(
            ArtifactStep(
                name=f"rl-data/atlas/full-qc/contracts/{key}",
                version=version,
                artifact_type=FullQCInputs,
                deps=[population],
                run=prepare_inputs,
                build_config=partial(
                    input_config, population, plan, file_sha256(StoragePath(plan.combined_drop_manifest_uri))
                ),
            )
        )
    return steps


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-plans", required=True)
    parser.add_argument("--source-ids", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    selected = {canonical_source_id(value) for value in json.loads(StoragePath(args.source_ids).read_text())}
    plans = [
        RetentionPlan.model_validate(row)
        for row in json.loads(StoragePath(args.retention_plans).read_text())
        if canonical_source_id(row["source_id"]) in selected
    ]
    if len(plans) != len(selected) or {canonical_source_id(plan.source_id) for plan in plans} != selected:
        raise ValueError("Every requested source requires exactly one retention plan")
    with artifact_prefix(args.artifact_root):
        steps = input_steps(plans, args.artifact_version)
        if not args.run:
            for step in steps:
                logger.info("Planned %s", step.path())
            return
        with ExitStack() as cleanup:
            client = LocalClient()
            cleanup.callback(client.shutdown)
            cleanup.enter_context(set_current_client(client))
            results = run(*steps, max_concurrent=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
