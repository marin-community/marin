# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill recipe evidence from a sealed archive without executing any producer."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.archive import ArchiveInput, digest, write_once
from experiments.post_training.long_loop.recipe_bundle import (
    RECIPE_CATEGORIES,
    require_clean_revision,
    runtime_source_manifest,
    write_recipe_bundle,
)


def backfill_recipe(
    *,
    archive_root: str,
    archive_manifest_sha256: str,
    output: str,
    repository: Path,
    source_revisions: tuple[str, ...],
    backfill_revision: str,
    limitations: tuple[str, ...],
) -> dict:
    """Bundle pinned historical source trees and exact archived recipe/config bytes."""
    raw = (StoragePath(archive_root) / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != archive_manifest_sha256:
        raise ValueError("Historical archive manifest SHA256 mismatch")
    archived = json.loads(raw)
    inputs = []
    for item in archived["objects"]:
        if item["category"] not in RECIPE_CATEGORIES:
            continue
        if digest(item["destination"]) != (item["size"], item["sha256"]):
            raise ValueError(f"Historical recipe input changed: {item['destination']}")
        name = str(StoragePath(item["destination"]).relative_to(StoragePath(archive_root) / "objects"))
        inputs.append(ArchiveInput(item["destination"], name, item["category"]))
    provenance = {
        "schema": 1,
        "archive_root": archive_root,
        "archive_manifest_sha256": archive_manifest_sha256,
        "backfill_revision": backfill_revision,
        "source_revisions": source_revisions,
        "limitations": limitations,
        "producer_replay": False,
        "gpu_allocation": False,
        "historical_archive_modified": False,
    }
    provenance_uri = str(StoragePath(output) / "backfill-provenance.json")
    write_once(provenance_uri, json.dumps(provenance, sort_keys=True, indent=2).encode())
    inputs.append(ArchiveInput(provenance_uri, "backfill-provenance.json", "source"))
    with TemporaryDirectory(prefix="long-loop-recipe-") as temporary:
        workspace = Path(temporary)
        for revision in source_revisions:
            actual = subprocess.check_output(
                ["git", "rev-parse", revision + "^{commit}"], cwd=repository, text=True
            ).strip()
            if actual != revision:
                raise ValueError("Historical source revisions must be full commit SHAs")
            content = subprocess.check_output(["git", "archive", "--format=tar", revision], cwd=repository)
            root = workspace / "revisions" / revision
            root.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
                archive.extractall(root, filter="data")
        snapshot = runtime_source_manifest(workspace, backfill_revision)
        _, receipt = write_recipe_bundle(output, inputs, snapshot, workspace)
    return {**receipt, **provenance}


@click.command(help=__doc__)
@click.option("--archive-root", required=True)
@click.option("--archive-manifest-sha256", required=True)
@click.option("--output", required=True)
@click.option("--source-revision", multiple=True, required=True)
@click.option("--limitation", multiple=True)
def main(
    archive_root: str,
    archive_manifest_sha256: str,
    output: str,
    source_revision: tuple[str, ...],
    limitation: tuple[str, ...],
):
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    require_clean_revision(repository, revision)
    configure_coreweave_s3()
    result = backfill_recipe(
        archive_root=archive_root,
        archive_manifest_sha256=archive_manifest_sha256,
        output=output,
        repository=repository,
        source_revisions=source_revision,
        backfill_revision=revision,
        limitations=limitation,
    )
    click.echo(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
