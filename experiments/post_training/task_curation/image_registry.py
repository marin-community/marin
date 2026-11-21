# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The built-image registry: one map row per served source, task and image.

The registry reads the frozen TaskSpecs of the served sources — the catalog's
``Available`` declarations whose filtration admits the population — and records
every image their environments reference, beside the source metadata that
states how open that image may be. Nothing is entered by hand: the rows come
from the pinned ``tasks.parquet`` payload of each served source's declaration.

The map lives on a private Hugging Face dataset
(``open-athena/atlas-image-registry``), versioned by revision; this repository
keeps the generator and, after a run, the revision pointer — never the data.
The payloads are large, so run the generator in-region::

    uv run python -m experiments.post_training.task_curation.image_registry \
        [--output /path/to/map.parquet]

The visibility rule the map serves is checked by
``experiments/post_training/task_curation/check_image_visibility.py``.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import CurationCatalog
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from infra.marina.apps.rl_data_catalog.load import DEFAULT_CATALOG_ROOT
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.publication_scope import publication_scope_decision
from infra.marina.apps.rl_data_catalog.server.restrictions import declaration_restriction

REGISTRY_REPO = "open-athena/atlas-image-registry"
"""The private dataset repo the map is published to, versioned by revision."""

MAP_FILENAME = "map.parquet"
CARD_FILENAME = "README.md"
PAYLOAD_FILENAME = "tasks.parquet"
POINTER_FILENAME = "image_registry_pointer.json"
POINTER_PATH = Path(__file__).resolve().parent / POINTER_FILENAME

MAP_COLUMNS = (
    "source_id",
    "dataset_id",
    "dataset_revision",
    "data_restriction",
    "publication_scope",
    "task_id",
    "task_row",
    "image_reference",
    "image_digest",
)
"""The map schema: source identity and allowance metadata, task identity, image fields.

Source identity comes from the served declaration; ``data_restriction`` and
``publication_scope`` record the existing metadata tiers the visibility check
derives each source's allowance from. The image fields come from the frozen
TaskSpec environment: the digest-pinned reference and its sha256 digest.
"""

MAP_SCHEMA = pa.schema([(column, pa.string()) for column in MAP_COLUMNS])

REGISTRY_CARD = f"""\
# Atlas built-image registry

One row per (served source, task, image): every image the frozen TaskSpec
environments of the served sources reference, with the source metadata that
states how open the image may be. Generated entirely from the served catalog
and the sources' pinned ``tasks.parquet`` payloads; never edited by hand.

Columns: {", ".join(MAP_COLUMNS)}.

- ``data_restriction`` is the source's typed restriction tier
  (``open`` | ``private_always``), resolved by the Atlas fail-closed rule.
- ``publication_scope`` records the exact-release public-display grant for the
  source's pinned dataset revision, or an empty string when none exists.

Regenerate with:

    uv run python -m experiments.post_training.task_curation.image_registry

Readers pin a revision; the pointer file in the Marin repository records the
revision each generation produced. The check script
(``check_image_visibility.py``) compares every GHCR package in the map against
the allowance this metadata encodes.
"""


def pinned_digest(reference: str) -> str:
    """The sha256 digest a digest-pinned reference carries, or an empty string."""
    _, separator, digest = reference.rpartition("@")
    return digest if separator and digest.startswith("sha256:") else ""


def served_source_columns(root: str) -> Iterator[tuple[dict[str, str], tuple[str, str] | None]]:
    """The map columns of every served source, with its payload pin when it has one.

    Serving mirrors the Atlas load rule: the declaration is ``Available`` and its
    filtration does not exclude the population. The allowance metadata is read
    from the declaration with the same fail-closed resolution the Atlas serves.
    """
    catalog = CurationCatalog(root)
    snapshot = catalog.current()
    for record in catalog.sources(snapshot.snapshot_id):
        declaration = record.declaration.model_dump(mode="json")
        if record.declaration.status != "Available" or filtration(declaration)["state"] == "excluded":
            continue
        dataset_id: str = declaration.get("dataset_id") or ""
        dataset_revision: str = declaration.get("dataset_revision") or ""
        decision = publication_scope_decision(declaration)
        columns = {
            "source_id": record.id,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "data_restriction": declaration_restriction(declaration).value,
            "publication_scope": decision.scope.value if decision is not None else "",
        }
        yield columns, (dataset_id, dataset_revision) if dataset_id and dataset_revision else None


def image_row(source: dict[str, str], task: dict[str, Any]) -> dict[str, str] | None:
    """One map row for the image a frozen task environment references, or None."""
    environment = task.get("environment_requirements") or {}
    reference = environment.get("docker_image")
    if not reference:
        return None
    return {
        **source,
        "task_id": task["id"],
        "task_row": (task.get("source") or {}).get("row", ""),
        "image_reference": reference,
        "image_digest": pinned_digest(reference),
    }


def payload_task_rows(payload: Path) -> Iterator[dict[str, Any]]:
    """The frozen TaskSpecs of one downloaded payload, in either published encoding.

    Canonical rows carry ``task_spec`` beside its ``task_sha256``; lowered rows
    carry the rollout envelope whose ``task`` holds the spec.
    """
    parquet = pq.ParquetFile(payload)
    columns = set(parquet.schema_arrow.names)
    if "task_spec" in columns:
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                serialized = row["task_spec"]
                if row.get("task_sha256") is not None:
                    if hashlib.sha256(serialized.encode()).hexdigest() != row["task_sha256"]:
                        raise ValueError(f"Published task differs from its recorded hash: {payload}")
                yield json.loads(serialized)
    elif "lowered_task_spec" in columns:
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                envelope = json.loads(row["lowered_task_spec"])
                task = envelope.get("task")
                if not task:
                    raise ValueError(f"Lowered envelope carries no task: {payload}")
                yield task
    else:
        raise ValueError(f"Payload carries neither task_spec nor lowered_task_spec: {payload}")


def source_map_rows(source: dict[str, str], payload: Path) -> tuple[list[dict[str, str]], int]:
    """The map rows of one served source's payload, and its tasks without an image.

    One task yields at most one row: a task environment references one image, and
    a task whose environment carries no image (lock-only or build context) has no
    registry image to record.
    """
    rows: list[dict[str, str]] = []
    without_image = 0
    seen: set[str] = set()
    for task in payload_task_rows(payload):
        row = image_row(source, task)
        if row is None:
            without_image += 1
            continue
        if row["task_id"] in seen:
            raise ValueError(f"Payload repeats task {row['task_id']} for source {source['source_id']}: {payload}")
        seen.add(row["task_id"])
        rows.append(row)
    return rows, without_image


def download_payload(dataset_id: str, dataset_revision: str) -> Path | None:
    """Download one served source's pinned payload, or None when it has no payload file.

    Some served sources pin an upstream dataset rather than a converted cohort;
    such a pin carries no ``tasks.parquet`` at any revision. Those sources have
    no frozen TaskSpec payload to read, and the generator counts them instead of
    failing the whole map.
    """
    try:
        return Path(
            hf_hub_download(
                repo_id=dataset_id, repo_type="dataset", filename=PAYLOAD_FILENAME, revision=dataset_revision
            )
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def build_map_rows(root: str) -> tuple[list[dict[str, str]], str]:
    """The complete map over the served catalog, with a one-line summary."""
    rows: list[dict[str, str]] = []
    sources = 0
    unpinned = 0
    unavailable = 0
    without_image = 0
    for columns, pin in served_source_columns(root):
        sources += 1
        payload = download_payload(*pin) if pin is not None else None
        if payload is None:
            if pin is None:
                unpinned += 1
            else:
                unavailable += 1
            continue
        source_rows, skipped = source_map_rows(columns, payload)
        rows.extend(source_rows)
        without_image += skipped
    summary = (
        f"{len(rows)} image rows over {sources} served sources "
        f"({unpinned} without a payload pin, {unavailable} whose pin carries no payload, "
        f"{without_image} tasks without a registry image)"
    )
    return rows, summary


def publish_map(map_path: Path, card: Path, repo: str) -> str:
    """Upload the map and card to the private dataset repo and return the revision.

    The repo is created private and verified private before upload; each file is
    read back through the uploading login and compared by hash, so a truncated
    upload fails here instead of at a reader.
    """
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError(f"Publication destination must be private: {repo}")
    with tempfile.TemporaryDirectory(prefix="image-registry-publish-") as directory:
        staging = Path(directory)
        shutil.copyfile(map_path, staging / MAP_FILENAME)
        shutil.copyfile(card, staging / CARD_FILENAME)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish the built-image registry map",
        )
    revision = commit.oid
    for source, name in ((map_path, MAP_FILENAME), (card, CARD_FILENAME)):
        downloaded = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision)
        if file_sha256(StoragePath(downloaded)) != file_sha256(StoragePath(str(source))):
            raise ValueError(f"{name} at {repo}@{revision} does not match the uploaded file")
    return revision


def write_pointer(path: Path, repo: str, revision: str, row_count: int, summary: str) -> None:
    """Record the revision a generation produced, for committing to this repository."""
    pointer = {
        "repo": repo,
        "revision": revision,
        "rows": row_count,
        "summary": summary,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(pointer, indent=2) + "\n")


def main() -> int:
    """Build the map over the served catalog and publish it to the private repo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", default=DEFAULT_CATALOG_ROOT, help="the FineStore catalog view to read")
    parser.add_argument("--repo", default=REGISTRY_REPO, help="private dataset repo the map is published to")
    parser.add_argument("--pointer", type=Path, default=POINTER_PATH, help="where the revision pointer is written")
    parser.add_argument("--output", type=Path, default=None, help="write the map parquet here and skip publishing")
    args = parser.parse_args()

    configure_coreweave_s3()
    rows, summary = build_map_rows(args.catalog_root)
    table = pa.Table.from_pylist(rows, schema=MAP_SCHEMA)
    if args.output is not None:
        pq.write_table(table, args.output)
        print(f"{args.output}: {summary}")
        return 0

    with tempfile.TemporaryDirectory(prefix="image-registry-") as directory:
        staging = Path(directory)
        map_path = staging / MAP_FILENAME
        pq.write_table(table, map_path)
        card_path = staging / CARD_FILENAME
        card_path.write_text(REGISTRY_CARD)
        revision = publish_map(map_path, card_path, args.repo)
    write_pointer(args.pointer, args.repo, revision, len(rows), summary)
    print(f"{args.repo}@{revision}: {summary}")
    print(f"Pointer written to {args.pointer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
