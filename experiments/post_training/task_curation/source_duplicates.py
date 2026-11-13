# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint converted sources and run a bounded, advisory duplicate scan."""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.source_fingerprints import DuplicateLabel, SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation.backend_capabilities import converted_specs
from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey
from experiments.post_training.task_curation.duplicate_review import review_pairs
from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
    pair_similarity,
)
from experiments.post_training.task_curation.source_neighbor_survey import cache_population, read_population
from experiments.post_training.task_curation.task_neighbors import SAMPLE_SIZE, PopulationIndex, content_rows
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, canonical_source_map

logger = logging.getLogger(__name__)


def fingerprint_manifest(manifest: dict) -> SourceIdentitySurvey:
    fingerprints: dict[str, SourceFingerprint | None] = canonical_source_map(dict.fromkeys(manifest["source_ids"]))
    for entry in manifest["inputs"]:
        source_id = canonical_source_id(entry["source_id"])
        if source_id not in fingerprints or fingerprints[source_id] is not None:
            raise ValueError(f"Unknown or duplicate fingerprint input: {source_id}")
        paths = [Path(path) for path in entry["paths"]]
        inputs = {}
        for path in paths:
            with path.open("rb") as stream:
                inputs[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
        fingerprint = fingerprint_source(converted_specs(paths, entry["column"]), inputs)
        if fingerprint.task_count != entry["expected_tasks"]:
            raise ValueError(f"Fingerprint does not cover the converted population: {source_id}")
        fingerprints[source_id] = fingerprint
        logger.info("Fingerprinted %s: %d tasks", source_id, fingerprint.task_count)
    return SourceIdentitySurvey(
        catalog_root=manifest["catalog_root"],
        snapshot_id=manifest["snapshot_id"],
        generated_at=datetime.now(UTC).isoformat(),
        fingerprints=fingerprints,
        reviews=[],
        groups=[],
        judge=None,
    )


def scan(
    survey: SourceIdentitySurvey,
    changed: set[str],
    top_k: int,
    populations: dict[str, PopulationIndex],
    acknowledgements: dict[str, str],
    labels: list[dict],
    receipts: Path,
    sample_size: int = SAMPLE_SIZE,
) -> SourceIdentitySurvey:
    fingerprints = {key: value for key, value in survey.fingerprints.items() if value is not None}
    for source_id, population in populations.items():
        fingerprint = fingerprints.get(source_id)
        if fingerprint is None or len(population.rows) != fingerprint.task_count:
            raise ValueError(f"Full matching population differs from fingerprint: {source_id}")
    pairs = candidate_pairs(fingerprints, changed, top_k)
    labelled = [
        pair_similarity(row["left"], fingerprints[row["left"]], row["right"], fingerprints[row["right"]])
        for row in labels
    ]
    selected = {(pair.left, pair.right): pair for pair in [*labelled, *pairs]}
    ordered = list(selected.values())
    expected = {(row["left"], row["right"]): DuplicateLabel(row["expected"]) for row in labels}
    reviews = []
    responses = StoragePath(str(receipts.with_suffix(".neighbors")))
    with receipts.open("w") as output:
        for review in review_pairs(ordered, populations, acknowledgements, responses, sample_size):
            reviews.append(review)
            output.write(review.model_dump_json() + "\n")
            output.flush()
            pair = review.similarity
            logger.info("Pair %s / %s: %s", pair.left, pair.right, review.verdict.label)
            label = expected.get((pair.left, pair.right))
            if label is not None and label != review.verdict.label:
                raise ValueError(f"Labelled duplicate control disagrees: {pair.left} / {pair.right}")

    return survey.model_copy(
        update={
            "generated_at": datetime.now(UTC).isoformat(),
            "reviews": reviews,
            "groups": duplicate_groups(reviews),
            "judge": None,
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("--manifest", type=Path, required=True)
    fingerprint.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("scan")
    review.add_argument("--fingerprints", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--changed-source", action="append", default=[])
    review.add_argument("--changed-sources", type=Path)
    review.add_argument("--top-k", type=int, required=True)
    review.add_argument(
        "--manifest", type=Path, help="Local complete-population inputs; otherwise use pinned S3 fingerprint inputs"
    )
    review.add_argument("--cache-dir", type=Path, default=Path("/tmp/source-overlap-populations"))
    review.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    review.add_argument("--labels", type=Path)
    review.add_argument("--acknowledgements", type=Path)
    review.add_argument("--require-acknowledged", action="store_true")
    args = parser.parse_args()
    if args.command == "fingerprint":
        result = fingerprint_manifest(json.loads(args.manifest.read_text()))
    else:
        if args.top_k <= 0:
            raise ValueError("Candidate count must be positive")
        source = canonical_identity_survey(json.loads(args.fingerprints.read_text()))
        populations = {}
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
            for entry in manifest["inputs"]:
                populations[canonical_source_id(entry["source_id"])] = PopulationIndex(
                    list(content_rows(converted_specs([Path(path) for path in entry["paths"]], entry["column"])))
                )
        else:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            for source_id, fingerprint in source.fingerprints.items():
                if fingerprint is not None:
                    populations[source_id] = read_population(cache_population(source_id, fingerprint, args.cache_dir))
        acknowledgements = json.loads(args.acknowledgements.read_text()) if args.acknowledgements else {}
        labels = json.loads(args.labels.read_text()) if args.labels else []
        changed = set(args.changed_source)
        if args.changed_sources:
            changed.update(json.loads(args.changed_sources.read_text()))
        result = scan(
            source,
            changed,
            args.top_k,
            populations,
            acknowledgements,
            labels,
            args.output.with_suffix(".receipts.jsonl"),
            args.sample_size,
        )
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    if args.command == "scan" and args.require_acknowledged:
        unresolved = [
            row for row in result.reviews if row.verdict.label != DuplicateLabel.DISTINCT and not row.acknowledgement
        ]
        if unresolved:
            raise ValueError(f"Resolve or acknowledge {len(unresolved)} source-overlap flags before admission")


if __name__ == "__main__":
    main()
