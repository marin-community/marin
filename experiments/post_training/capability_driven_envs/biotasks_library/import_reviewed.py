# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize BioTasks' individually reviewed notebook inventory as a private source campaign."""

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

import fsspec

BIOTASKS_REVISION = "a342f41d9d0674cbe280a33a2f537285547552d3"
INVENTORY_ROOT = "experiments/5-classification-pilot/full-inventory"
RAW_ROOT = f"https://raw.githubusercontent.com/Open-Athena/biotasks/{BIOTASKS_REVISION}/{INVENTORY_ROOT}"
ANNOTATIONS_URL = f"{RAW_ROOT}/annotations.jsonl"
ACQUISITION_URL = f"{RAW_ROOT}/acquisition.jsonl"
EXTERNAL_PINS_URL = f"{RAW_ROOT}/external-github-pins.json"
DEFAULT_OUTPUT_ROOT = "s3://marin-us-east-02a/marin4sc/taskforge/task-library"
DEFAULT_CAMPAIGN_ID = "reviewed57-a342f41d-v1"
RELEASE_STATUS = "internal_unreviewed_do_not_publish"
USER_AGENT = "Marin-Taskforge-BioTasks/1"


@dataclass(frozen=True)
class ReviewedSource:
    """One content-confirmed, individually reviewed BioTasks document."""

    document_key: str
    canonical_document_key: str
    source_ids: tuple[str, ...]
    repo: str | None
    revision: str | None
    path: str
    format: str
    title: str
    source_role: str
    review: str
    source_sha256: str
    download_url: str
    fields: tuple[str, ...]
    modalities: tuple[str, ...]
    operations: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedSource:
    """A reviewed source after exact-byte verification and object storage."""

    document_key: str
    canonical_document_key: str
    source_ids: tuple[str, ...]
    repo: str | None
    revision: str | None
    path: str
    format: str
    title: str
    source_role: str
    review: str
    source_sha256: str
    source_bytes: int
    object_uri: str
    fields: tuple[str, ...]
    modalities: tuple[str, ...]
    operations: tuple[str, ...]


def fetch(url: str) -> bytes:
    """Fetch one bounded inventory or source object without executing it."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def jsonl(data: bytes) -> list[dict[str, Any]]:
    """Parse nonempty JSONL records."""
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _raw_github_url(repo: str, revision: str, path: str) -> str:
    encoded_path = urllib.parse.quote(path.lstrip("/"), safe="/")
    return f"https://raw.githubusercontent.com/{repo}/{revision}/{encoded_path}"


def _pinned_external_urls(data: bytes) -> dict[str, str]:
    records = json.loads(data)
    result = {}
    for record in records:
        pinned = str(record["pinned_url"])
        marker = "github.com/"
        if marker not in pinned or "/blob/" not in pinned:
            raise ValueError(f"unsupported external pin URL: {pinned}")
        repository_and_path = pinned.split(marker, 1)[1]
        repository, remainder = repository_and_path.split("/blob/", 1)
        revision, path = remainder.split("/", 1)
        result[str(record["document_key"])] = _raw_github_url(repository, revision, path)
    return result


def _facet_terms(annotation: dict[str, Any], facet: str) -> tuple[str, ...]:
    return tuple(sorted({str(label["term"]) for label in annotation["labels"] if label["facet"] == facet}))


def reviewed_sources(
    annotations_data: bytes,
    acquisition_data: bytes,
    external_pins_data: bytes,
) -> list[ReviewedSource]:
    """Select the 57 counted individually reviewed documents and resolve immutable downloads."""
    acquisitions = {record["document_key"]: record for record in jsonl(acquisition_data)}
    external = _pinned_external_urls(external_pins_data)
    selected = []
    for annotation in jsonl(annotations_data):
        if not annotation.get("counted") or annotation["review"] == "not_individually_adjudicated":
            continue
        if annotation["acquisition_status"] != "confirmed":
            raise ValueError(f"reviewed source is not confirmed: {annotation['document_key']}")
        document_key = str(annotation["document_key"])
        repo = annotation.get("repo")
        revision = annotation.get("revision")
        path = str(annotation["path"])
        if repo and revision:
            download_url = _raw_github_url(str(repo), str(revision), path)
        elif document_key in external:
            download_url = external[document_key]
        else:
            download_url = str(acquisitions[document_key]["acquisition_url"])
        selected.append(
            ReviewedSource(
                document_key=document_key,
                canonical_document_key=str(annotation["canonical_document_key"]),
                source_ids=tuple(str(value) for value in annotation["source_ids"]),
                repo=None if repo is None else str(repo),
                revision=None if revision is None else str(revision),
                path=path,
                format=str(annotation["format"]),
                title=str(annotation["title"]),
                source_role=str(annotation["source_role"]),
                review=str(annotation["review"]),
                source_sha256=str(annotation["source_sha256"]),
                download_url=download_url,
                fields=_facet_terms(annotation, "scientific_field"),
                modalities=_facet_terms(annotation, "modality"),
                operations=_facet_terms(annotation, "operation"),
            )
        )
    selected.sort(key=lambda source: source.document_key)
    if len(selected) != 57:
        raise ValueError(f"expected 57 counted reviewed sources, found {len(selected)}")
    if len({source.canonical_document_key for source in selected}) != len(selected):
        raise ValueError("reviewed source selection contains duplicate canonical document keys")
    return selected


def _filesystem(output_root: str):
    if output_root.startswith("s3://"):
        key = os.environ.get("CW_KEY_ID")
        secret = os.environ.get("CW_KEY_SECRET")
        if not key or not secret:
            raise RuntimeError("CW_KEY_ID and CW_KEY_SECRET are required for S3 output")
        fs = fsspec.filesystem(
            "s3",
            key=key,
            secret=secret,
            client_kwargs={"endpoint_url": "https://cwobject.com"},
            config_kwargs={"s3": {"addressing_style": "virtual"}},
        )
        return fs, output_root.removeprefix("s3://").rstrip("/")
    fs, root = fsspec.core.url_to_fs(output_root)
    return fs, root.rstrip("/")


def _write_if_absent(fs: Any, path: str, data: bytes) -> None:
    if fs.exists(path):
        with fs.open(path, "rb") as stream:
            existing = stream.read()
        if existing != data:
            raise ValueError(f"existing content-addressed object differs: {path}")
        return
    parent = str(PurePosixPath(path).parent)
    fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as stream:
        stream.write(data)


def _download(source: ReviewedSource) -> tuple[ReviewedSource, bytes]:
    data = fetch(source.download_url)
    observed = hashlib.sha256(data).hexdigest()
    if observed != source.source_sha256:
        raise ValueError(
            f"source digest mismatch for {source.document_key}: expected {source.source_sha256}, observed {observed}"
        )
    return source, data


def _jsonl_bytes(records: Iterable[dict[str, object]]) -> bytes:
    return ("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records)).encode()


def materialize(
    sources: list[ReviewedSource],
    *,
    output_root: str,
    campaign_id: str,
    max_workers: int,
) -> dict[str, object]:
    """Verify sources, store content-addressed bytes, and publish the campaign manifest last."""
    fs, root = _filesystem(output_root)
    campaign_root = f"{root}/campaigns/biotasks/{campaign_id}"
    manifest_path = f"{campaign_root}/manifest.json"
    if fs.exists(manifest_path):
        raise FileExistsError(f"campaign already exists: {output_root}/campaigns/biotasks/{campaign_id}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        downloaded = list(executor.map(_download, sources))

    catalog = []
    for source, data in downloaded:
        object_path = f"{root}/objects/sha256/{source.source_sha256[:2]}/{source.source_sha256}"
        _write_if_absent(fs, object_path, data)
        catalog.append(
            MaterializedSource(
                document_key=source.document_key,
                canonical_document_key=source.canonical_document_key,
                source_ids=source.source_ids,
                repo=source.repo,
                revision=source.revision,
                path=source.path,
                format=source.format,
                title=source.title,
                source_role=source.source_role,
                review=source.review,
                source_sha256=source.source_sha256,
                source_bytes=len(data),
                object_uri=f"{output_root.rstrip('/')}/objects/sha256/{source.source_sha256[:2]}/{source.source_sha256}",
                fields=source.fields,
                modalities=source.modalities,
                operations=source.operations,
            )
        )
    catalog.sort(key=lambda source: source.document_key)
    catalog_data = _jsonl_bytes(asdict(source) for source in catalog)
    catalog_path = f"{campaign_root}/source-catalog.jsonl"
    fs.makedirs(campaign_root, exist_ok=True)
    with fs.open(catalog_path, "wb") as stream:
        stream.write(catalog_data)

    manifest: dict[str, object] = {
        "schema_version": "1",
        "campaign_id": campaign_id,
        "release_status": RELEASE_STATUS,
        "biotasks_revision": BIOTASKS_REVISION,
        "source_issues": [
            "https://github.com/Open-Athena/biotasks/issues/5",
            "https://github.com/Open-Athena/biotasks/issues/16",
        ],
        "source_count": len(catalog),
        "unique_object_count": len({source.source_sha256 for source in catalog}),
        "source_bytes": sum(source.source_bytes for source in catalog),
        "catalog_sha256": hashlib.sha256(catalog_data).hexdigest(),
        "catalog_uri": f"{output_root.rstrip('/')}/campaigns/biotasks/{campaign_id}/source-catalog.jsonl",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection": "counted, content-confirmed documents with individual source review",
        "execution": "source bytes acquired and hash-verified; no notebook code executed",
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with fs.open(manifest_path, "wb") as stream:
        stream.write(manifest_data)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    sources = reviewed_sources(fetch(ANNOTATIONS_URL), fetch(ACQUISITION_URL), fetch(EXTERNAL_PINS_URL))
    manifest = materialize(
        sources,
        output_root=args.output_root,
        campaign_id=args.campaign_id,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
