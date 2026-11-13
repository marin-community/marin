# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint converted sources and run a bounded, advisory duplicate scan."""

import argparse
import hashlib
import json
import logging
import os
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.reviewer_spec import JudgeEndpoint
from taskcompendium.source_fingerprints import DuplicateJudgeRun, DuplicateLabel, SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation.backend_capabilities import converted_specs
from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey
from experiments.post_training.task_curation.duplicate_review import (
    DUPLICATE_MODEL,
    DUPLICATE_MODEL_CONFIG,
    DUPLICATE_MODEL_REVISION,
    DUPLICATE_PROMPT_VERSION,
    DUPLICATE_SYSTEM_PROMPT,
    DUPLICATE_TEMPERATURE,
    LocalDuplicateEndpoint,
    UnresolvedDuplicatePair,
    review_pairs,
)
from experiments.post_training.task_curation.judges import HostedJudgeEndpoint
from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
    pair_similarity,
)
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
        for index, path in enumerate(paths):
            with path.open("rb") as stream:
                inputs[f"{index}:{path.name}"] = hashlib.file_digest(stream, "sha256").hexdigest()
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
    endpoint: JudgeEndpoint,
    acknowledgements: dict[str, str],
    labels: list[dict],
    receipts: Path,
) -> SourceIdentitySurvey:
    fingerprints = {key: value for key, value in survey.fingerprints.items() if value is not None}
    pairs = candidate_pairs(fingerprints, changed, top_k)
    labelled = [
        pair_similarity(row["left"], fingerprints[row["left"]], row["right"], fingerprints[row["right"]])
        for row in labels
    ]
    selected = {(pair.left, pair.right): pair for pair in [*labelled, *pairs]}
    # Labelled controls run first, inside the same endpoint lease as the scan.
    ordered = list({(pair.left, pair.right): pair for pair in [*labelled, *selected.values()]}.values())
    expected = {(row["left"], row["right"]): DuplicateLabel(row["expected"]) for row in labels}
    reviews = []
    unresolved = []
    unresolved_path = receipts.with_suffix(".unresolved.jsonl")
    responses = StoragePath(str(receipts.with_suffix(".responses")))
    with closing(review_pairs(ordered, fingerprints, endpoint, survey.reviews, acknowledgements, responses)) as results:
        with receipts.open("w") as output, unresolved_path.open("w") as failures:
            for review in results:
                if isinstance(review, UnresolvedDuplicatePair):
                    failures.write(review.model_dump_json() + "\n")
                    failures.flush()
                    unresolved.append(review)
                    continue
                reviews.append(review)
                output.write(review.model_dump_json() + "\n")
                output.flush()
                pair = review.similarity
                logger.info("Pair %s / %s: %s", pair.left, pair.right, review.verdict.label)
                label = expected.get((pair.left, pair.right))
                if label is not None and label != review.verdict.label:
                    raise ValueError(f"Labelled duplicate control disagrees: {pair.left} / {pair.right}")
    if unresolved:
        raise ValueError(f"{len(unresolved)} source pairs remain unresolved; evidence: {unresolved_path}")
    local = endpoint if isinstance(endpoint, LocalDuplicateEndpoint) else None
    judge = DuplicateJudgeRun(
        model=DUPLICATE_MODEL,
        revision=DUPLICATE_MODEL_REVISION,
        prompt_version=DUPLICATE_PROMPT_VERSION,
        prompt_sha256=canonical_sha256(DUPLICATE_SYSTEM_PROMPT),
        temperature=DUPLICATE_TEMPERATURE,
        top_k=top_k,
        labelled_controls=len(labels),
        labelled_controls_passed=len(labels),
        route="local" if local is not None else "hosted",
        serving_jobs=local.job_ids if local is not None else [],
        lease_seconds_upper_bound=local.lease_seconds if local is not None else None,
    )
    return survey.model_copy(
        update={
            "generated_at": datetime.now(UTC).isoformat(),
            "reviews": reviews,
            "groups": duplicate_groups(reviews),
            "judge": judge,
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
    review.add_argument("--route", choices=("hosted", "local"), required=True)
    review.add_argument("--base-url")
    review.add_argument("--api-key-env")
    review.add_argument("--model-config", type=Path, default=DUPLICATE_MODEL_CONFIG)
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
        if args.route == "local":
            endpoint: JudgeEndpoint = LocalDuplicateEndpoint(args.model_config)
        else:
            if args.base_url is None or args.api_key_env is None:
                raise ValueError("Hosted duplicate review requires --base-url and --api-key-env")
            endpoint = HostedJudgeEndpoint(DUPLICATE_MODEL, args.base_url, os.environ[args.api_key_env])
        acknowledgements = json.loads(args.acknowledgements.read_text()) if args.acknowledgements else {}
        labels = json.loads(args.labels.read_text()) if args.labels else []
        changed = set(args.changed_source)
        if args.changed_sources:
            changed.update(json.loads(args.changed_sources.read_text()))
        result = scan(
            source, changed, args.top_k, endpoint, acknowledgements, labels, args.output.with_suffix(".receipts.jsonl")
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
