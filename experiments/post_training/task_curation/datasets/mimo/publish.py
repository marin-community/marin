# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the repaired MiMo release to its private registry repository.

The repaired variant is a new converted population: the pinned agent images, the
declared grader and the delivery contract replace the unresolved build contexts and
the NoGrader placeholder, and the eight 16-GiB tasks are excluded with typed
evidence. The publication uploads the repaired ``tasks.parquet`` beside its dataset
card and conversion receipt to a fresh private ``open-athena`` repository, then
verifies the stored bytes and records the identity proof the qualification binds.
"""

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import hf_hub_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.calibforge_publish import publish

logger = logging.getLogger(__name__)

REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"
DEFAULT_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-pinned-images-20261124-a-v1"
SOURCE_ID = "FWeindel:validated-tasks/mimo_v1"
INPUT_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-local-images-20261122-b-v1"
INPUT_REVISION = "9efbd74114d232e5d3607af847c75418a1a798d8"
CONVERTER_VERSION = "2026.11.23.pinned-images.1"


@dataclass(frozen=True)
class PublicationInputs:
    prefix: StoragePath
    repo: str


def card_text(summary: dict, package: str) -> str:
    """The publication card body; the chokepoint writes the licensed front matter."""
    return "\n".join(
        (
            "Source pin: "
            f"{INPUT_REPO}@{INPUT_REVISION} repaired {CONVERTER_VERSION}"
            " (converter landing in atlas-env-audit)",
            "",
            "Private TRAIN candidate, not admitted. The backend repair of the pinned MiMo population:",
            f"every task's agent environment is its digest-pinned prepared image ({package},",
            "built in-region from the task's own Dockerfile recipe), the grader is declared from the",
            "source's own verifier package (setup.sh, test.sh, reward fail-closed), the shared grading",
            "boundary carries its root-delivery contract, and the eight 16-GiB LOCAL-routed tasks are",
            "excluded with typed evidence (backend_contract_unsupported; Daytona sandboxes cap at",
            "8 GiB). 2,501 tasks are served; the original release rows stay immutable. Prompts,",
            "native files and role budgets are unchanged. No original oracle exists; executability",
            "is unmeasured under the approved no-oracle policy. The conversion receipt",
            "(conversion-receipt.json) binds the admitted revision to its per-task conversions.",
            "",
        )
    )


def download_verified(prefix: StoragePath, name: str, expected_sha: str, destination: Path) -> Path:
    """Download one repaired artifact and verify it against the summary's recorded hash."""
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with StoragePath(str(prefix / name)).open("rb") as reader, target.open("wb") as writer:
        writer.write(reader.read())
    digest = file_sha256(StoragePath(str(target)))
    if digest != expected_sha:
        raise ValueError(f"{name} differs from its recorded hash: {digest} != {expected_sha}")
    return target


def publish_release(inputs: PublicationInputs, revision: str) -> dict:
    """Publish the repaired population and verify the stored release end to end."""
    configure_coreweave_s3()
    summary = json.loads(StoragePath(str(inputs.prefix / "repair-summary.json")).read_text())
    if summary["version"] != CONVERTER_VERSION:
        raise ValueError(f"the repaired release belongs to another converter version: {summary['version']}")
    if not revision:
        raise ValueError("the publication identity binds the converter's landing revision")
    with tempfile.TemporaryDirectory(prefix="mimo-publish-") as directory:
        staging = Path(directory)
        tasks = download_verified(inputs.prefix, "tasks.parquet", summary["output_sha256"], staging)
        receipt = download_verified(
            inputs.prefix, "conversion-receipt.json", summary["conversion_receipt_sha256"], staging
        )
        card = staging / "README.md"
        card.write_text(card_text(summary, "ghcr.io/marin-community/rl-data-mimo-prepared"))
        publication = publish(
            tasks,
            card,
            inputs.repo,
            data_restriction=DataRestriction.OPEN,
            evidence_files=(receipt,),
            source_id=SOURCE_ID,
            use="train",
            pretty_name="MiMo code v1 backend-repair candidate",
        )
        commit = publication.revision
        published = Path(hf_hub_download(inputs.repo, "tasks.parquet", repo_type="dataset", revision=commit))
        published_sha = file_sha256(StoragePath(str(published)))
        if published_sha != summary["output_sha256"]:
            raise ValueError("the published release readback differs from the repaired population")
        proof = {
            "job": "mimo-publish-v1",
            "published_at": datetime.now(UTC).isoformat(),
            "release": {
                "repo": inputs.repo,
                "revision": commit,
                "url": f"https://huggingface.co/datasets/{inputs.repo}/tree/{commit}",
            },
            "input_pin": {"repo": INPUT_REPO, "revision": INPUT_REVISION},
            "converter": {
                "version": CONVERTER_VERSION,
                "revision": revision,
                "conversion_receipt_sha256": summary["conversion_receipt_sha256"],
            },
            "tasks": summary["tasks"],
            "excluded_tasks": summary["excluded_tasks"],
            "tasks_parquet_sha256": published_sha,
        }
        proof_path = staging / "identity-proof.json"
        proof_path.write_text(json.dumps(proof, indent=1) + "\n")
        destination = inputs.prefix / "identity-proof.json"
        with proof_path.open("rb") as reader, destination.open("wb", auto_mkdir=True) as writer:
            writer.write(reader.read())
        logger.info("published %s at %s (%d tasks)", inputs.repo, commit, summary["tasks"])
        return proof


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", default=REGIONAL_PREFIX + "/rl-data/atlas/ingest-b/fweindel-mimo-v1/2026.11.23.1/repaired-v1"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--converter-revision", required=True)
    arguments = parser.parse_args()
    inputs = PublicationInputs(prefix=StoragePath(arguments.prefix), repo=arguments.repo)
    proof = publish_release(inputs, arguments.converter_revision)
    print(json.dumps(proof, indent=1))


if __name__ == "__main__":
    main()
