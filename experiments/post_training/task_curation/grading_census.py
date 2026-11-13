# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assess every frozen TaskSpec with the ingestion grading-boundary census."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import ActorCensusInput, GradingEnvironmentEvidence
from pydantic import BaseModel, ConfigDict, model_validator
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_environment import (
    assess_grading_environment,
    count_grading_environment,
    grading_environment_census,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import NoGrader, TaskSpec

from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.prebuilt import validate_native_package

LOGGER = logging.getLogger(__name__)
TASK_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("task_sha256", pa.string()),
        ("required", pa.string()),
        ("declared", pa.string()),
        ("status", pa.string()),
        ("reason", pa.string()),
    ]
)


class GradingCensusInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    population: SourceTextPopulation
    input: ActorCensusInput

    @model_validator(mode="after")
    def _exact_population(self) -> "GradingCensusInput":
        population, binding = self.population, self.input
        if (
            population.dataset_id,
            population.dataset_revision,
            population.source_version,
            population.converter_revision,
            population.task_specs_sha256,
            population.task_count,
        ) != (
            binding.dataset_id,
            binding.dataset_revision,
            binding.converter_version,
            binding.converter_revision,
            binding.task_specs_sha256,
            binding.measured_tasks,
        ):
            raise ValueError(f"Grading population differs from its actor receipt: {population.source_id}")
        return self


def census_source(config: GradingCensusInput, output_root: str) -> GradingCensusReceipt:
    """Retain each assessment and bind the complete scan to its frozen population."""
    population = config.population
    if SourceTextPopulation.raw_load(population.path) != population:
        raise ValueError(f"Grading population manifest changed: {population.source_id}")
    output = StoragePath(output_root) / canonical_sha256(population.source_id)[:20]
    if output.exists():
        raise FileExistsError(f"Grading census requires a new output version: {output}")
    counts: Counter[str] = Counter()
    unassessable: Counter[str] = Counter()
    digest = hashlib.sha256()
    task_count = 0
    task_evidence = output / "tasks.parquet"
    with task_evidence.open("wb", auto_mkdir=True) as stream, pq.ParquetWriter(stream, TASK_SCHEMA) as writer:
        for path in sorted((StoragePath(population.specs_output_dir) / "*.parquet").glob(), key=str):
            with path.open("rb") as source:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=32, columns=["task_spec"]):
                    assessments = []
                    for row in batch.to_pylist():
                        encoded = row["task_spec"]
                        task_hash = hashlib.sha256(encoded.encode()).hexdigest()
                        digest.update(encoded.encode())
                        task_count += 1
                        document = json.loads(encoded)
                        try:
                            task = TaskSpec.model_validate(document)
                            if isinstance(task.grader, NoGrader) and any(
                                resource.path == "task.toml" for resource in task.resources.verifier
                            ):
                                validate_native_package(task)
                            assessment = assess_grading_environment(task).model_dump(mode="json")
                        except ValueError as error:
                            reason = f"{type(error).__name__}: {error}"
                            unassessable[reason] += 1
                            assessment = {"required": None, "declared": None, "status": "unassessable", "reason": reason}
                        else:
                            count_grading_environment(counts, assessment)
                        assessments.append({"task_id": document["id"], "task_sha256": task_hash, **assessment})
                    writer.write_table(pa.Table.from_pylist(assessments, schema=TASK_SCHEMA))
    if task_count != population.task_count:
        raise ValueError(f"Grading census input count differs from its actor receipt: {population.source_id}")
    receipt = GradingCensusReceipt(
        source_id=population.source_id,
        input=config.input,
        population_path=population.path,
        scanned_task_specs_sha256=digest.hexdigest(),
        evidence=GradingEnvironmentEvidence(
            **grading_environment_census(counts), evidence_url=str(output / "receipt.json")
        ),
        task_evidence_uri=str(task_evidence),
        unassessable_reasons=dict(unassessable),
    )
    (output / "receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    LOGGER.info("Grading census %s: %s", receipt.source_id, receipt.evidence.required_counts)
    return receipt


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    inputs = [GradingCensusInput.model_validate(row) for row in json.loads(StoragePath(args.inputs).read_text())]
    if len({row.population.source_id for row in inputs}) != len(inputs):
        raise ValueError("Grading census inputs contain duplicate sources")
    inputs.sort(key=lambda row: -row.population.task_count)
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"), initializer=configure_coreweave_s3
    ) as pool:
        pending = {pool.submit(census_source, row, args.output_root): row.population.source_id for row in inputs}
        for future in as_completed(pending):
            results.append(future.result().model_dump(mode="json"))
    results.sort(key=lambda row: row["source_id"])
    target = StoragePath(args.output_root) / "receipts.json"
    if target.exists():
        raise FileExistsError(f"Grading census requires a new receipt index: {target}")
    target.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
