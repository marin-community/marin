# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Refresh contract metadata over persisted TaskSpecs without repeating actor measurements."""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.lexical_artifacts import IntegrityCensus, SourceTextPopulation
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.published_census import (
    CensusConfig,
    ContractCensus,
    census_config,
    population_documents,
    write_contract_summary,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import original_source_id


def measure_integrity(config: CensusConfig) -> IntegrityCensus:
    population = SourceTextPopulation.raw_load(config.population_path)
    counts = ContractCensus()
    for task, _ in population_documents(population):
        counts.observe(task)
    if counts.task_count != population.task_count:
        raise ValueError(f"Incomplete integrity census: {population.source_id}")
    return IntegrityCensus(
        path=config.output_path,
        source_id=population.source_id,
        input_artifact=population.path,
        converter_version=population.source_version,
        converter_revision=population.converter_revision,
        task_count=counts.task_count,
        summary=write_contract_summary(counts, population, config.output_path),
    )


def integrity_steps(populations: list[SourceTextPopulation], version: str) -> list[ArtifactStep]:
    steps = []
    for population in populations:
        key = canonical_sha256(original_source_id(population.source_id))[:20]
        source = ArtifactStep.adopt(
            f"rl-data/atlas/integrity-census/input/{key}", version, population.path, kind=SourceTextPopulation
        )
        steps.append(
            ArtifactStep(
                name=f"rl-data/atlas/integrity-census/{key}",
                version=version,
                artifact_type=IntegrityCensus,
                deps=[source],
                run=measure_integrity,
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
    args = parser.parse_args()
    configure_coreweave_s3()
    populations = [SourceTextPopulation.model_validate(row) for row in json.loads(args.populations.read_text())]
    if any(not population.path for population in populations):
        raise ValueError("Integrity inputs require concrete population artifact paths")
    with artifact_prefix(args.artifact_root):
        with local_client():
            results = run(*integrity_steps(populations, args.artifact_version), max_concurrent=2)
        args.output.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
