# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive every frozen source population's contract tags and grader types.

The late-stage ingestion step scans the final converted TaskSpecs of each measured population,
derives each task's contract shape (agentic, turn count, reply-only delivery, workspace grading)
and its actual grader type, and binds a complete summary receipt. Declared shape tags that
contradict the derived shape fail the source instead of reaching the Atlas.
"""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import pyarrow.parquet as pq
from finestore.rl.curation import ActorCensusInput
from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict, model_validator
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.derived_contract import (
    DerivedContractSummary,
    derive_contract,
    summarize_contracts,
    summary_contradictions,
)
from taskcompendium.grader_type import GraderType, judge_details
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, judge_spec

from experiments.post_training.task_curation.datasets.grading_measurements import grading_census_receipts
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation

LOGGER = logging.getLogger(__name__)


class DerivedContractSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_count: int
    grader_type_counts: dict[str, int]
    agentic_count: int
    single_turn_count: int
    reply_only_count: int
    workspace_graded_count: int
    judge_rubric_counts: dict[str, int]
    judge_reference_count: int
    judge_no_reference_count: int

    def summary(self) -> DerivedContractSummary:
        return DerivedContractSummary(**self.model_dump())


class DerivedContractCensusInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    population: SourceTextPopulation
    input: ActorCensusInput

    @model_validator(mode="after")
    def _exact_population(self) -> "DerivedContractCensusInput":
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
            raise ValueError(f"Derived contract population differs from its actor receipt: {population.source_id}")
        return self


class DerivedContractReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    input: ActorCensusInput
    population_path: str
    scanned_task_specs_sha256: str
    summary: DerivedContractSummaryRecord
    evidence_url: str

    @model_validator(mode="after")
    def _complete_population(self) -> "DerivedContractReceipt":
        if self.summary.task_count != self.input.measured_tasks:
            raise ValueError(f"Derived contract census does not cover the complete population: {self.source_id}")
        return self

    def contract_summary(self) -> DerivedContractSummary:
        """The lib summary this receipt records."""
        return DerivedContractSummary(**self.summary.model_dump())

    def contradictions(self, declared_tags: frozenset[str]) -> tuple[str, ...]:
        return summary_contradictions(declared_tags, self.contract_summary())


def census_source(config: DerivedContractCensusInput, output_root: str) -> DerivedContractReceipt:
    """Scan the frozen TaskSpecs and bind the complete derived-shape summary."""
    population = config.population
    if SourceTextPopulation.raw_load(population.path) != population:
        raise ValueError(f"Derived contract population manifest changed: {population.source_id}")
    output = StoragePath(output_root) / canonical_sha256(population.source_id)[:20]
    if output.exists():
        raise FileExistsError(f"Derived contract census requires a new output version: {output}")
    digest = hashlib.sha256()
    contracts = []
    judges = []
    task_count = 0
    for path in sorted((StoragePath(population.specs_output_dir) / "*.parquet").glob(), key=str):
        with path.open("rb") as source:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=32, columns=["task_spec"]):
                for row in batch.to_pylist():
                    encoded = row["task_spec"]
                    task = TaskSpec.model_validate_json(encoded)
                    contract = derive_contract(task)
                    digest.update(encoded.encode())
                    task_count += 1
                    contracts.append(contract)
                    if contract.grader_type is GraderType.LLM_JUDGE:
                        spec = judge_spec(task.grader)
                        if spec is None:
                            raise ValueError(f"LLM-judge task without a judge contract: {task.id}")
                        judges.append(spec)
                        LOGGER.debug("Judge details %s: %s", task.id, judge_details(spec))
    if task_count != population.task_count:
        raise ValueError(f"Derived contract census count differs from its actor receipt: {population.source_id}")
    summary = summarize_contracts(tuple(contracts), tuple(judges))
    receipt = DerivedContractReceipt(
        source_id=population.source_id,
        input=config.input,
        population_path=population.path,
        scanned_task_specs_sha256=digest.hexdigest(),
        summary=DerivedContractSummaryRecord(**summary.__dict__),
        evidence_url=str(output / "receipt.json"),
    )
    (output / "receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    LOGGER.info("Derived contract census %s: %s", receipt.source_id, receipt.summary.grader_type_counts)
    return receipt


def census_inputs_from_grading_receipts(prefix: str | None = None) -> list[DerivedContractCensusInput]:
    """Reuse the frozen population and actor bindings the grading census already measured."""
    inputs = []
    for receipt in sorted(grading_census_receipts().values(), key=lambda receipt: receipt.source_id):
        if prefix is not None and not receipt.population_path.startswith(prefix):
            continue
        population = SourceTextPopulation.raw_load(receipt.population_path)
        inputs.append(DerivedContractCensusInput(population=population, input=receipt.input))
    return inputs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", required=True, help="Inputs JSON path, or 'grading' to reuse the grading census bindings"
    )
    parser.add_argument(
        "--population-prefix", default=None, help="Restrict reused grading bindings to one storage prefix"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--upload-hf", default=None, help="Upload receipts.json to this path in the private registry repository"
    )
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.inputs == "grading":
        inputs = census_inputs_from_grading_receipts(args.population_prefix)
    else:
        inputs = [
            DerivedContractCensusInput.model_validate(row) for row in json.loads(StoragePath(args.inputs).read_text())
        ]
    if len({row.population.source_id for row in inputs}) != len(inputs):
        raise ValueError("Derived contract census inputs contain duplicate sources")
    inputs.sort(key=lambda row: -row.population.task_count)
    results = []
    failures = {}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"), initializer=configure_coreweave_s3
    ) as pool:
        pending = {pool.submit(census_source, row, args.output_root): row.population.source_id for row in inputs}
        for future in as_completed(pending):
            source_id = pending[future]
            try:
                results.append(future.result().model_dump(mode="json"))
            except Exception as error:
                LOGGER.error("Derived contract census failed for %s: %s", source_id, error)
                failures[source_id] = f"{type(error).__name__}: {error}"
    if failures:
        raise ValueError(f"Derived contract census failed for {len(failures)} sources: {sorted(failures)}")
    results.sort(key=lambda row: row["source_id"])
    target = StoragePath(args.output_root) / "receipts.json"
    if target.exists():
        raise FileExistsError(f"Derived contract census requires a new receipt index: {target}")
    payload = json.dumps(results, indent=2) + "\n"
    target.write_text(payload)
    if args.upload_hf:
        HfApi().upload_file(
            path_or_fileobj=payload.encode(),
            path_in_repo=args.upload_hf,
            repo_id="open-athena/marin-rl-data-atlas-registry",
            repo_type="dataset",
            commit_message=f"Staged derived contract census receipts: {args.output_root}",
        )
        LOGGER.info("Staged receipts at %s", args.upload_hf)


if __name__ == "__main__":
    main()
