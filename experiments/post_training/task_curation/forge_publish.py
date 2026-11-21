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


def publish_staged_release(release_root: str, card: Path, repo: str, restriction: DataRestriction) -> str:
    """Publish the staged release as ``repo``'s next revision and return that revision."""
    root = StoragePath(release_root)
    manifest = json.loads((root / MANIFEST_FILENAME).read_bytes())
    with tempfile.TemporaryDirectory(prefix="forge-publish-") as directory:
        staged = Path(directory) / LOWERED_FILENAME
        staged.write_bytes((root / LOWERED_FILENAME).read_bytes())
        if hashlib.sha256(staged.read_bytes()).hexdigest() != manifest["lowered_sha256"]:
            raise ValueError(f"Staged tasks parquet differs from its manifest: {release_root}")
        # The shared helper copies the tasks file under the loader's name.
        return publish(staged, card, repo, data_restriction=restriction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", required=True, help="object-storage root holding the staged release")
    parser.add_argument("--card", type=Path, required=True, help="dataset card markdown")
    parser.add_argument("--repo", required=True, help="dataset repo, e.g. open-athena/<name> (private)")
    parser.add_argument(
        "--data-restriction", type=DataRestriction, required=True, help="the source's typed restriction tier"
    )
    args = parser.parse_args()
    configure_coreweave_s3()
    revision = publish_staged_release(args.release_root, args.card, args.repo, args.data_restriction)
    print(f"{args.repo}@{revision}")


if __name__ == "__main__":
    main()
