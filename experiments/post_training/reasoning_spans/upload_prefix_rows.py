# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload mined reasoning-span prefix rows to the Hugging Face dataset repositories the catalog pins.

The hazard and control splits go to separate repositories because the task-curation pipeline names
each task's source dataset after its repository, and the quick RL loop reports evaluation metrics per
source dataset. Prints the commit of each upload; pin those revisions in
``experiments/post_training/task_curation/datasets/reasoning_spans.py``.

    uv run python -m experiments.post_training.reasoning_spans.upload_prefix_rows \\
        --rows-root s3://marin-us-east-02a/marin/rl-data/inputs/reasoning-span-prefixes/2026.11.03
"""

import os

import click
from huggingface_hub import HfApi
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.reasoning_spans.mine_pi_prefixes import (
    CONTROL_FILENAME,
    HAZARD_FILENAME,
    REPORT_FILENAME,
)

HAZARD_REPO = "open-athena/Marin-Reasoning-Span-Hazard-Prefixes"
CONTROL_REPO = "open-athena/Marin-Reasoning-Span-Control-Prefixes"
ROWS_FILENAME = "rows.parquet"


def upload_split(api: HfApi, rows_root: StoragePath, filename: str, repo: str, message: str) -> str:
    """Upload one split's rows and the mining report to ``repo``; return the commit."""
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=(rows_root / filename).read_bytes(),
        path_in_repo=ROWS_FILENAME,
        repo_id=repo,
        repo_type="dataset",
        commit_message=message,
    )
    commit = api.upload_file(
        path_or_fileobj=(rows_root / REPORT_FILENAME).read_bytes(),
        path_in_repo=REPORT_FILENAME,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"{message} (mining report)",
    )
    return commit.oid


@click.command(help=__doc__)
@click.option("--rows-root", required=True, help="Published rl-data/inputs/reasoning-span-prefixes/<version> root.")
@click.option("--hazard-repo", default=HAZARD_REPO, show_default=True)
@click.option("--control-repo", default=CONTROL_REPO, show_default=True)
def main(rows_root: str, hazard_repo: str, control_repo: str) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise click.UsageError("Uploads use HF_TOKEN; export it first")
    api = HfApi()
    root = StoragePath(rows_root)
    message = f"Mined prefix rows from {rows_root}"
    for filename, repo in ((HAZARD_FILENAME, hazard_repo), (CONTROL_FILENAME, control_repo)):
        commit = upload_split(api, root, filename, repo, message)
        click.echo(f"{repo}@{commit}")


if __name__ == "__main__":
    main()
