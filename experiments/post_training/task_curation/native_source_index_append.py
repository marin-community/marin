# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append the native conversions' FingerprintReference entries to the atlas registry.

Consumes the native source survey job's SURVEY RESULT (the reviewed identity
survey artifact plus its per-source digests), verifies every value it publishes
against the live declarations and the artifact bytes, then appends the two
per-admission registry documents — the fingerprint index and the population
locators — through the append-only archive flow, and re-pins the repository's
served pointers.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.registry import (
    RegistryReference,
    archive_registry_documents,
    read_registry_json,
)
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from experiments.post_training.task_curation.sources import all_sources
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, original_source_id

logger = logging.getLogger(__name__)

REGISTRY_PATH_PREFIX = "experiments/post_training/task_curation"


def declared_input_sha256(source_id: str) -> str:
    declaration = all_sources()[source_id].declaration.model_dump(mode="json")
    return canonical_sha256(fingerprint_input(declaration).model_dump(mode="json"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-result", type=Path, required=True, help="The survey job's SURVEY RESULT JSON.")
    parser.add_argument("--repo", default="open-athena/marin-rl-data-atlas-registry")
    parser.add_argument("--parent-revision", required=True, help="The registry head revision to append onto.")
    parser.add_argument(
        "--admission", required=True, help="The per-admission document name, e.g. native_taskspec_repin_20261126_a_v1."
    )
    parser.add_argument(
        "--references-root", type=Path, default=Path("."), help="The repository root the pointer files live under."
    )
    parser.add_argument(
        "--commit-message", required=True, help="The registry append's commit message (append-only; never rewrites)."
    )
    args = parser.parse_args()

    configure_coreweave_s3()
    result = json.loads(args.survey_result.read_text())
    survey_uri = result["survey_uri"]
    survey_sha256 = result["survey_sha256"]
    payload = StoragePath(survey_uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != survey_sha256:
        raise ValueError(f"Reviewed survey artifact bytes differ: {survey_uri}")
    artifact = json.loads(payload)
    fingerprints = artifact["fingerprints"]

    served_index = read_registry_json(Path(REGISTRY_PATH_PREFIX) / "source_fingerprint_index.json")
    served_locators = read_registry_json(Path(REGISTRY_PATH_PREFIX) / "source_population_locators.json")

    index = dict(served_index)
    locators = {
        "files": dict(served_locators.get("files", {})),
        "prepared": dict(served_locators.get("prepared", {})),
    }
    for source_id, counts in result["task_counts"].items():
        canonical = canonical_source_id(source_id)
        original = original_source_id(canonical)
        declared = declared_input_sha256(canonical)
        fingerprint = fingerprints.get(original)
        if fingerprint is None:
            raise ValueError(f"Reviewed survey artifact lacks a fingerprint for {source_id}")
        if fingerprint["task_count"] != counts:
            raise ValueError(f"Reviewed survey artifact task count differs for {source_id}")
        index[canonical] = {"input_sha256": declared, "sha256": survey_sha256, "uri": survey_uri}
        prepared = result["prepared"][source_id]
        population_digest = result["input_sha256"][source_id]
        if prepared["inputs_sha256"] != population_digest or prepared["task_count"] != counts:
            raise ValueError(f"Prepared population record disagrees with the measurement for {source_id}")
        locators["files"].update(result["files"])
        locators["prepared"][canonical] = prepared
        logger.info("Referenced %s: %d tasks, input %s…", source_id, counts, declared[:12])

    with TemporaryDirectory(prefix="native-source-index-append-") as temporary:
        staged_root = Path(temporary)
        documents: dict[str, bytes] = {}
        for name, document in (
            (f"source_fingerprint_index_{args.admission}.json", index),
            (f"source_population_locators_{args.admission}.json", locators),
        ):
            destination = staged_root / REGISTRY_PATH_PREFIX / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
            destination.write_bytes(payload)
            documents[f"{REGISTRY_PATH_PREFIX}/{name}"] = payload
        revision = archive_registry_documents(
            staged_root,
            args.references_root,
            repo=args.repo,
            parent_revision=args.parent_revision,
            commit_message=args.commit_message,
        )
    references = {
        name: RegistryReference(repo=args.repo, revision=revision, path=name, sha256=hashlib.sha256(payload).hexdigest())
        for name, payload in documents.items()
    }
    (args.references_root / REGISTRY_PATH_PREFIX / "source_fingerprint_index.json").write_text(
        references[f"{REGISTRY_PATH_PREFIX}/source_fingerprint_index_{args.admission}.json"].model_dump_json(indent=2)
        + "\n"
    )
    (args.references_root / REGISTRY_PATH_PREFIX / "source_population_locators.json").write_text(
        references[f"{REGISTRY_PATH_PREFIX}/source_population_locators_{args.admission}.json"].model_dump_json(indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {
                "registry_revision": revision,
                "index_pointer_sha256": (
                    references[f"{REGISTRY_PATH_PREFIX}/source_fingerprint_index_{args.admission}.json"].sha256
                ),
                "locator_pointer_sha256": (
                    references[f"{REGISTRY_PATH_PREFIX}/source_population_locators_{args.admission}.json"].sha256
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
