# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register completed lexical measurements without publishing dataset versions."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from finestore.rl.curation import ActorCensusInput, LexicalContaminationEvidence
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt
from experiments.post_training.task_curation.lexical_policy import LexicalInputUse
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards, write_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_map


def measurement_receipts(
    measurements: Mapping[str, LexicalContaminationEvidence],
    populations: Sequence[SourceTextPopulation],
    uses: Sequence[LexicalInputUse],
    actor_receipts: Mapping[str, ActorCensusReceipt],
) -> dict[str, LexicalMeasurementReceipt]:
    """Join full measurements to exact normalization, Use and actor-census inputs."""
    measured = canonical_source_map(measurements)
    normalized = canonical_source_map({row.source_id: row for row in populations})
    declared = canonical_source_map({row.source_id: row for row in uses})
    if len(normalized) != len(populations) or len(declared) != len(uses):
        raise ValueError("Duplicate population or Use declaration")
    if set(measured) != set(normalized) or set(measured) != set(declared):
        raise ValueError("Measurements, populations and Use must cover the same sources")
    receipts = {}
    for source_id, evidence in measured.items():
        population = normalized[source_id]
        intended = declared[source_id]
        population_input = ActorCensusInput(
            dataset_id=population.dataset_id,
            dataset_revision=population.dataset_revision,
            converter_version=population.source_version,
            converter_revision=population.converter_revision,
            measured_tasks=population.task_count,
            task_specs_sha256=population.task_specs_sha256,
            evidence_url=str(StoragePath(population.path) / ".artifact.json"),
        )
        actor = actor_receipts.get(source_id)
        if actor is None or population_input.model_dump(exclude={"evidence_url"}) != actor.input.model_dump(
            exclude={"evidence_url"}
        ):
            raise ValueError(f"Lexical input differs from the actor census: {source_id}")
        if (intended.dataset_id, intended.dataset_revision, intended.converter_revision) != (
            population_input.dataset_id,
            population_input.dataset_revision,
            population_input.converter_revision,
        ) or evidence.text_renderer_version != population.text_renderer_version:
            raise ValueError(f"Use or text renderer belongs to another measured input: {source_id}")
        receipts[source_id] = LexicalMeasurementReceipt(
            source_id=source_id, input=population_input, use=intended.use, measurement=evidence
        )
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--source-uses", type=Path, required=True)
    parser.add_argument("--actor-receipts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcomes = json.loads(args.outcomes.read_text())
    receipts = measurement_receipts(
        {key: LexicalContaminationEvidence.model_validate(row) for key, row in outcomes["measurements"].items()},
        [SourceTextPopulation.model_validate(row) for row in json.loads(args.populations.read_text())],
        [LexicalInputUse.model_validate(row) for row in json.loads(args.source_uses.read_text())],
        read_receipt_shards(args.actor_receipts_dir, ActorCensusReceipt, "census", load_json=read_registry_json),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, {key: receipt.model_dump(mode="json") for key, receipt in receipts.items()})


if __name__ == "__main__":
    main()
