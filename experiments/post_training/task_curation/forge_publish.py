# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the corrected forge cohort revision from its staged release root.

The rebuild stages the release artifacts on object storage; this publishes them
as the private cohort repository's next revision. The staged tasks parquet is
verified against its manifest's hash before any upload, and the shared publish
helper verifies the read-back after it. Run on an in-region worker with
CoreWeave and Hugging Face credentials exported::

    uv run python -m experiments.post_training.task_curation.forge_publish \\
        --release-root s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/forge-main/2026.11.21.1/release \\
        --card experiments/post_training/task_curation/datasets/forge_main_card.md \\
        --repo open-athena/forge-main-taskcompendium-20261117 \\
        --data-restriction open
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.calibforge_publish import publish
from experiments.post_training.task_curation.forge_release import LOWERED_FILENAME, MANIFEST_FILENAME
from experiments.post_training.task_curation.license_publication import PublishedLicensedPopulation


def publish_staged_release(
    release_root: str,
    card: Path,
    repo: str,
    restriction: DataRestriction,
    *,
    source_id: str,
    use: str,
    pretty_name: str,
) -> PublishedLicensedPopulation:
    """Publish the staged release as ``repo``'s next revision and return the publication."""
    root = StoragePath(release_root)
    manifest = json.loads((root / MANIFEST_FILENAME).read_bytes())
    with tempfile.TemporaryDirectory(prefix="forge-publish-") as directory:
        staged = Path(directory) / LOWERED_FILENAME
        staged.write_bytes((root / LOWERED_FILENAME).read_bytes())
        if hashlib.sha256(staged.read_bytes()).hexdigest() != manifest["lowered_sha256"]:
            raise ValueError(f"Staged tasks parquet differs from its manifest: {release_root}")
        # The shared helper copies the tasks file under the loader's name.
        return publish(
            staged,
            card,
            repo,
            data_restriction=restriction,
            source_id=source_id,
            use=use,
            pretty_name=pretty_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", required=True, help="object-storage root holding the staged release")
    parser.add_argument("--card", type=Path, required=True, help="dataset card body markdown")
    parser.add_argument("--repo", required=True, help="dataset repo, e.g. open-athena/<name> (private)")
    parser.add_argument(
        "--data-restriction", type=DataRestriction, required=True, help="the source's typed restriction tier"
    )
    parser.add_argument("--source-id", required=True, help="the cohort source id the publication belongs to")
    parser.add_argument("--use", required=True, choices=["train", "eval"], help="the publication use")
    parser.add_argument("--pretty-name", required=True, help="the card's pretty_name")
    args = parser.parse_args()
    configure_coreweave_s3()
    publication = publish_staged_release(
        args.release_root,
        args.card,
        args.repo,
        args.data_restriction,
        source_id=args.source_id,
        use=args.use,
        pretty_name=args.pretty_name,
    )
    print(f"{args.repo}@{publication.revision}")


if __name__ == "__main__":
    main()
