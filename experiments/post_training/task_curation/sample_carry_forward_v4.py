# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the SAMPLE review carry-forward plan for both repaired v4 populations.

Loads each source's prior oracle-admitted receipt (the qualified pin's panel), the earlier
population it reviewed, and the sealed v4 population, then plans the carry-forward: kept
members (unchanged public content), replaced slots, and members needing a fresh review.
"""

import json

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.datasets.oracle_admitted_releases import (
    oracle_admitted_release_receipts,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.review_carry_forward import (
    plan_review_carry_forward,
    population_identities,
)

configure_coreweave_s3()
CURRENT = {
    "RST:laion/rts-1k-rootless-apptainer": (
        "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.17.1/rst-repaired-admission-v4/rst-population",
        "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.17.1/rst-repaired-admission-v4/rst-census-result.json",
    ),
    "Lego-X:Terminal-Lego-15k": (
        "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/terminal-lego/2026.11.20.1/root-delivery-repaired-admission-v4/lego-population",
        "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/terminal-lego/2026.11.20.1/root-delivery-repaired-admission-v4/lego-census-result.json",
    ),
}


def earlier_population_path(actor: ActorCensusReceipt) -> str:
    """The earlier population path: the census record at the actor receipt's evidence_url."""
    record = json.loads(StoragePath(actor.input.evidence_url).read_text())
    return record["result"]["input_artifact"]


def main() -> None:
    receipts = oracle_admitted_release_receipts()
    for source_id, (current_path, result_uri) in CURRENT.items():
        receipt = receipts[source_id]
        panel = receipt.sample.review.panel
        earlier_population = SourceTextPopulation.raw_load(earlier_population_path(receipt.actor))
        current_population = SourceTextPopulation.raw_load(current_path)
        earlier_identities = population_identities(earlier_population)
        current_identities = population_identities(current_population)
        plan = plan_review_carry_forward(
            source_id,
            receipt.sample,
            earlier_population,
            current_population,
            earlier_identities,
            current_identities,
            reviewer=None,
        )
        document = plan.model_dump(mode="json")
        summary = {
            "source_id": source_id,
            "earlier_panel": {
                "dataset_id": panel.dataset_id,
                "dataset_revision": panel.dataset_revision,
                "retained_tasks": panel.retained_tasks,
                "sample_size": panel.sample_size,
                "sample_seed": panel.sample_seed,
            },
            "current_population": {
                "dataset_id": current_population.dataset_id,
                "dataset_revision": current_population.dataset_revision,
                "task_count": current_population.task_count,
            },
            "kept_members": len(document.get("kept_members", document.get("kept", []))),
            "replaced_slots": document.get("replaced_slots", document.get("replaced", 0)),
            "fresh_reviews": len(document.get("fresh_review_members", document.get("fresh", []))),
        }
        out = StoragePath(result_uri.rsplit("/", 1)[0] + f"/{source_id.split(':')[0].lower()}-sample-carry-plan.json")
        out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print("CARRY-PLAN " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
