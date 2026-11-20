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
import hashlib
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.publication_gate import require_public_data

TASKS_FILENAME = "tasks.parquet"
CARD_FILENAME = "README.md"


def _require_loader_hash_column(tasks: Path) -> None:
    """The runtime cohort loader requires each row's task content hash.

    A publication whose rows lack the ``task_sha256`` column, or whose recorded hash
    differs from the serialized ``task_spec``, would publish a cohort the loader
    refuses to import. The check runs before any upload (ruling 66).
    """
    table = pq.read_table(tasks)
    if "task_sha256" not in table.column_names:
        raise ValueError(f"Cohort publication lacks the row-level task_sha256 column the loader requires: {tasks}")
    specs = table.column("task_spec").to_pylist()
    recorded = table.column("task_sha256").to_pylist()
    for serialized, digest in zip(specs, recorded, strict=True):
        if hashlib.sha256(serialized.encode()).hexdigest() != digest:
            raise ValueError(f"Published task hash differs from its task_spec content: {tasks}")


def _source_pin_line(card_text: str) -> str | None:
    """The ``Source pin:`` line a publication card declares, for content-pin comparison."""
    for line in card_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source pin:"):
            return stripped
    return None


def publish(
    tasks: Path, card: Path, repo: str, *, data_restriction: DataRestriction, evidence_files: tuple[Path, ...] = ()
) -> str:
    """Upload ``tasks`` and ``card`` to a private ``repo`` and return the commit revision.

    The caller passes the source population's tier: a PRIVATE_ALWAYS payload is
    refused here at the shared helper, never published anywhere.

    A repo left behind by a partial prior run is resumed into, never clobbered:
    the existing card must declare the same ``Source pin`` as the new card, or
    the publication refuses and a new version name is required.
    """
    require_public_data(data_restriction, destination=repo)
    api = HfApi()
    exist_ok = False
    try:
        existing_card_path = hf_hub_download(repo_id=repo, filename=CARD_FILENAME, repo_type="dataset")
        existing_pin = _source_pin_line(Path(existing_card_path).read_text())
        new_pin = _source_pin_line(card.read_text())
        if existing_pin is not None and existing_pin != new_pin:
            raise ValueError(
                f"Publication destination holds another content pin: {repo}; "
                f"existing {existing_pin!r} != new {new_pin!r} — use a new version name"
            )
        exist_ok = True
    except Exception as error:
        if "Source pin" in str(error):
            raise
    with tempfile.TemporaryDirectory(prefix="calibforge-publish-") as directory:
        staging = Path(directory)
        shutil.copyfile(tasks, staging / TASKS_FILENAME)
        shutil.copyfile(card, staging / CARD_FILENAME)
        for source in evidence_files:
            shutil.copyfile(source, staging / source.name)
        _require_loader_hash_column(staging / TASKS_FILENAME)
        api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=exist_ok)
        if not api.repo_info(repo, repo_type="dataset").private:
            raise ValueError(f"Publication destination must be private: {repo}")
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish the private TaskCompendium cohort",
        )
    revision = commit.oid
    # Read the artifact back through the login that uploaded it and compare hashes, so
    # a truncated or rewritten file fails here rather than at a reader.
    files = ((tasks, TASKS_FILENAME), (card, CARD_FILENAME), *((path, path.name) for path in evidence_files))
    for source, name in files:
        downloaded = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision)
        if file_sha256(StoragePath(downloaded)) != file_sha256(StoragePath(str(source))):
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
