# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure every actor and grader slot in pinned, normalized TaskSpec populations."""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.grading_environment import GradingEnvironmentMode, assess_grading_environment
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.daytona_census import (
    daytona_unsafe_counts,
    daytona_unsafe_reasons,
    measure_envelopes,
)
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from infra.marina.apps.rl_data_catalog.server.source_identity import original_source_id


@dataclass(frozen=True)
class CensusConfig:
    population_path: str
    output_path: str


def population_rows(population: SourceTextPopulation) -> Iterable[dict[str, str]]:
    """Read original serialized contracts in the census's stable shard order."""
    for path in sorted(StoragePath(prefix_join(population.specs_output_dir, "*.parquet")).glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=["task_spec", "runtime"]):
                yield from batch.to_pylist()


def population_documents(population: SourceTextPopulation) -> Iterable[tuple[dict, dict]]:
    for row in population_rows(population):
        yield json.loads(row["task_spec"]), json.loads(row["runtime"])


@dataclass
class ContractCensus:
    groups: dict[str, dict] = field(default_factory=dict)
    example: dict | None = None
    required_shared: int = 0
    shared_missing: int = 0
    task_count: int = 0

    def observe(self, task: dict) -> None:
        self.task_count += 1
        contract = task.get("integrity_contract")
        shared = (
            assess_grading_environment(TaskSpec.model_validate({**task, "integrity_contract": None})).required
            is GradingEnvironmentMode.SHARED
        )
        self.required_shared += shared
        self.shared_missing += shared and contract is None
        descriptor = {key: (contract or {}).get(key) for key in ("id", "version", "dependency_state", "harness_policy")}
        descriptor["native_family"] = ((contract or {}).get("native") or {}).get("family")
        if contract is not None:
            descriptor["harness_policy"] = {
                name: contract[name]
                for name in (
                    "harness",
                    "harness_sha256",
                    "protected_root",
                    "agent_uid",
                    "grader_uid",
                    "protected_test_patterns",
                    "pinned_plugins",
                    "plugin_discovery",
                )
                if name in contract
            }
        descriptor["present"] = contract is not None
        key = canonical_sha256(descriptor)
        group = self.groups.setdefault(key, {**descriptor, "count": 0})
        group["count"] += 1
        if self.example is None and contract is not None and len(json.dumps(contract).encode()) <= 16384:
            self.example = {"task_id": task["id"], "contract": contract}

    def rows(self) -> list[dict]:
        return sorted(self.groups.values(), key=lambda group: (-group["count"], canonical_sha256(group)))


def write_contract_summary(census: ContractCensus, population: SourceTextPopulation, output_path: str) -> dict:
    rows = census.rows()
    uri = prefix_join(output_path, "integrity-contracts.json")
    with StoragePath(uri).open("w", auto_mkdir=True) as stream:
        json.dump({"groups": rows, "example": census.example, "input_artifact": population.path}, stream)
    return {
        "schema_version": 2,
        "groups": rows[:64],
        "distinct_groups": len(rows),
        "absent_contract_tasks": sum(group["count"] for group in rows if not group["present"]),
        "required_shared_tasks": census.required_shared,
        "shared_missing_contract_tasks": census.shared_missing,
        "example": census.example,
        "evidence_url": uri,
        "dataset_id": population.dataset_id,
        "dataset_revision": population.dataset_revision,
        "task_specs_sha256": population.task_specs_sha256,
    }


def measure_population(config: CensusConfig) -> ActorCensus:
    population = SourceTextPopulation.raw_load(config.population_path)
    contracts = ContractCensus()

    def documents():
        for task, runtime in population_documents(population):
            contracts.observe(task)
            yield task, runtime

    result = measure_envelopes(documents())
    if result.converted != population.task_count:
        raise ValueError(f"{population.source_id}: {result.converted} measured for {population.task_count} TaskSpecs")
    details = daytona_unsafe_reasons(result)
    slots = {
        role: sum(backend["machine_slots"] for backend in backends.values()) for role, backends in result.roles.items()
    }
    return ActorCensus(
        path=config.output_path,
        source_id=population.source_id,
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        converter_revision=population.converter_revision,
        input_artifact=population.path,
        task_specs_sha256=population.task_specs_sha256,
        task_count=result.converted,
        daytona_safety="not_applicable" if not result.machine_slots else "unsafe" if details else "safe",
        daytona_snapshot_count=len(result.snapshot_sizes),
        daytona_machine_slots=result.machine_slots,
        daytona_execution_failures=result.execution_failure_tasks,
        daytona_count_basis=(
            f"Complete {result.converted}-TaskSpec census from {population.dataset_id}@"
            f"{population.dataset_revision}; native Harbor task.toml or published runtime allocations; "
            f"actor only on Daytona, grader roles separate; no provider calls; {population.path}"
        ),
        daytona_unsafe_reasons=sorted(details),
        daytona_unsafe_details=details,
        daytona_unsafe_counts=daytona_unsafe_counts(result),
        daytona_roles=result.roles,
        actor_environment_counts=dict(result.actor_sources),
        empty_declared_environment_tasks={role: result.converted - count for role, count in slots.items()},
        snapshot_sizes=dict(result.snapshot_sizes),
        execution_failures=dict(result.execution_failures),
        mapping_failures=dict(result.mapping_failures),
        integrity_contracts=write_contract_summary(contracts, population, config.output_path),
    )


def census_config(population: ArtifactStep, ctx: StepContext) -> CensusConfig:
    return CensusConfig(ctx.artifact_path(population), ctx.output_path)


def census_steps(populations: list[SourceTextPopulation], version: str) -> list[ArtifactStep]:
    steps = []
    for population in populations:
        key = canonical_sha256(original_source_id(population.source_id))[:20]
        source = ArtifactStep.adopt(
            f"rl-data/atlas/actor-census/input/{key}", version, population.path, kind=SourceTextPopulation
        )
        steps.append(
            ArtifactStep(
                name=f"rl-data/atlas/actor-census/{key}",
                version=version,
                artifact_type=ActorCensus,
                deps=[source],
                run=measure_population,
                build_config=partial(census_config, source),
            )
        )
    return steps


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    populations = [SourceTextPopulation.model_validate(row) for row in json.loads(args.populations.read_text())]
    with artifact_prefix(args.artifact_root):
        steps = census_steps(populations, args.artifact_version)
        if not args.run:
            args.output.write_text(
                json.dumps(
                    [
                        {"source_id": p.source_id, "path": step.path()}
                        for p, step in zip(populations, steps, strict=True)
                    ],
                    indent=2,
                )
                + "\n"
            )
            return
        with local_client():
            results = run(*steps, max_concurrent=2)
        args.output.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
