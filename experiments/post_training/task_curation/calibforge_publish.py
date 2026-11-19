# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a converted cohort as a private TaskCompendium dataset.

Curation that converts data must publish the converted artifact, so the Atlas can
point at something a reader can actually load instead of at the deprecated native
release. This uploads the lowered task spec rows beside the dataset card, verifies the
result through an authenticated read-back, and prints the revision to record.

Campaign datasets stay private until release, so the repo is created private and the
artifact is verified by reading it back through the cached login and comparing hashes.
Run it with that login (``HfApi()``/``hf_hub_download`` need no explicit token)::

    uv run python -m experiments.post_training.task_curation.calibforge_publish \
        --tasks /path/to/lowered-tasks.parquet \
        --card experiments/post_training/task_curation/datasets/calibforge_rebuilt_card.md \
        --repo open-athena/calibforge-v1-rebuilt-taskcompendium-20261111
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

TASKS_FILENAME = "tasks.parquet"
CARD_FILENAME = "README.md"


def publish(tasks: Path, card: Path, repo: str) -> str:
    """Upload ``tasks`` and ``card`` to a private ``repo`` and return the commit revision."""
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="calibforge-publish-") as directory:
        staging = Path(directory)
        shutil.copyfile(tasks, staging / TASKS_FILENAME)
        shutil.copyfile(card, staging / CARD_FILENAME)
        api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish the rebuilt CalibForge TaskCompendium cohort",
        )
    revision = commit.oid
    # Read the artifact back through the login that uploaded it and compare hashes, so
    # a truncated or rewritten file fails here rather than at a reader.
    for source, name in ((tasks, TASKS_FILENAME), (card, CARD_FILENAME)):
        downloaded = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision)
        local, remote = StoragePath(str(source)), StoragePath(str(downloaded))
        if file_sha256(remote) != file_sha256(local):
            raise ValueError(f"{name} at {repo}@{revision} does not match the uploaded file")
    return revision


def main() -> None:
    """Publish the cohort and print the revision to pin in the Atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="lowered task spec parquet")
    parser.add_argument("--card", type=Path, required=True, help="dataset card markdown")
    parser.add_argument("--repo", required=True, help="dataset repo, e.g. open-athena/<name> (created private)")
    args = parser.parse_args()
    revision = publish(args.tasks, args.card, args.repo)
    print(f"{args.repo}@{revision}")


if __name__ == "__main__":
    main()
