# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload mined double-think prefix rows to the Hugging Face dataset repository the catalog pins.

Prints the commit of the upload; pin it as ``REVISION`` in
``experiments/post_training/task_curation/datasets/double_think.py``.

    uv run python -m experiments.post_training.double_think.upload_prefixes \\
        --rows-root s3://marin-us-east-02a/marin/rl-data/inputs/double-think-prefixes/<version>
"""

import os

import click
from huggingface_hub import CommitOperationAdd, HfApi
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.double_think.mine_prefixes import REPORT_FILENAME
from experiments.post_training.task_curation.datasets.double_think import REPOSITORY, ROWS_FILE


@click.command(help=__doc__)
@click.option("--rows-root", required=True, help="Published rl-data/inputs/double-think-prefixes/<version> root.")
@click.option("--repo", default=REPOSITORY, show_default=True)
def main(rows_root: str, repo: str) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise click.UsageError("Uploads use HF_TOKEN; export it first")
    api = HfApi()
    root = StoragePath(rows_root)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    commit = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=(root / filename).read_bytes())
            for filename in (ROWS_FILE, REPORT_FILENAME)
        ],
        commit_message=f"Mined double-think prefix rows from {rows_root}",
    )
    click.echo(f"{repo}@{commit.oid}")


if __name__ == "__main__":
    main()
