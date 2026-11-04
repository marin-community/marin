# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload the mined thoughtcrime rows to the Hugging Face dataset repository the catalog pins.

The commit also publishes ``dataset_card.md`` as the repository README; update its counts
and baseline when the rows change. Prints the commit of the upload; pin it as ``REVISION`` in
``experiments/post_training/task_curation/datasets/thoughtcrime.py``.

    uv run python -m experiments.post_training.thoughtcrime.upload_prefixes \\
        --rows-root s3://marin-us-east-02a/marin/rl-data/inputs/thoughtcrime/<version>
"""

import os
from pathlib import Path

import click
from huggingface_hub import CommitOperationAdd, HfApi
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.thoughtcrime import REPOSITORY, ROWS_FILE
from experiments.post_training.thoughtcrime.mine_prefixes import REPORT_FILENAME

DATASET_CARD = Path(__file__).with_name("dataset_card.md")


@click.command(help=__doc__)
@click.option("--rows-root", required=True, help="Published rl-data/inputs/thoughtcrime/<version> root.")
@click.option("--repo", default=REPOSITORY, show_default=True)
def main(rows_root: str, repo: str) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise click.UsageError("Uploads use HF_TOKEN; export it first")
    configure_coreweave_s3()
    api = HfApi()
    root = StoragePath(rows_root)
    # The rows republish benchmark prompts and tool outputs, so the repository stays private.
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=[
            *(
                CommitOperationAdd(path_in_repo=filename, path_or_fileobj=(root / filename).read_bytes())
                for filename in (ROWS_FILE, REPORT_FILENAME)
            ),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=DATASET_CARD.read_bytes()),
        ],
        commit_message=f"Mined double-think prefix rows from {rows_root}",
    )
    click.echo(f"{repo}@{commit.oid}")


if __name__ == "__main__":
    main()
