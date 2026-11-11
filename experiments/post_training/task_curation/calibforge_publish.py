# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a converted cohort as a public TaskCompendium dataset.

Curation that converts data must publish the converted artifact, so the Atlas can
point at something a reader can actually load instead of at the deprecated native
release. This uploads the lowered task spec rows beside the dataset card, verifies the
result by fetching it anonymously, and prints the revision to record.

Run it with the cached Hugging Face login (``HfApi()`` needs no explicit token)::

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

TASKS_FILENAME = "tasks.parquet"
CARD_FILENAME = "README.md"


def publish(tasks: Path, card: Path, repo: str) -> str:
    """Upload ``tasks`` and ``card`` to a public ``repo`` and return the commit revision."""
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="calibforge-publish-") as directory:
        staging = Path(directory)
        shutil.copyfile(tasks, staging / TASKS_FILENAME)
        shutil.copyfile(card, staging / CARD_FILENAME)
        api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish the rebuilt CalibForge TaskCompendium cohort",
        )
    revision = commit.oid
    # Verify the artifact as a reader sees it: an anonymous download must succeed.
    for name in (TASKS_FILENAME, CARD_FILENAME):
        downloaded = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision, token=False)
        if Path(downloaded).stat().st_size == 0:
            raise ValueError(f"{name} downloaded empty from {repo}@{revision}")
    return revision


def main() -> None:
    """Publish the cohort and print the revision to pin in the Atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="lowered task spec parquet")
    parser.add_argument("--card", type=Path, required=True, help="dataset card markdown")
    parser.add_argument("--repo", required=True, help="public dataset repo, e.g. open-athena/<name>")
    args = parser.parse_args()
    revision = publish(args.tasks, args.card, args.repo)
    print(f"{args.repo}@{revision}")


if __name__ == "__main__":
    main()
