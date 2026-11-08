# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize pinned BioTasks repositories with bounded scientific-document evidence."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from experiments.post_training.capability_driven_envs.biotasks_library.import_reviewed import (
    BIOTASKS_REVISION,
    DEFAULT_OUTPUT_ROOT,
    RELEASE_STATUS,
    USER_AGENT,
    _filesystem,
    _write_if_absent,
    jsonl,
)

INVENTORY_ROOT = "experiments/5-classification-pilot/full-inventory"
AUDIT_ROOT = "experiments/16-analysis-discovery/repo-notebook-audit"
RAW_ROOT = f"https://raw.githubusercontent.com/Open-Athena/biotasks/{BIOTASKS_REVISION}"
REPOSITORY_ACQUISITION_URL = f"{RAW_ROOT}/{INVENTORY_ROOT}/repository-acquisition.jsonl"
OBSERVATIONS_URL = f"{RAW_ROOT}/{AUDIT_ROOT}/observations.jsonl"
DEFAULT_CAMPAIGN_ID = "repositories/v1"
MAX_README_BYTES = 2_000_000
MAX_DOCUMENT_BYTES = 1_000_000
MAX_DOCUMENTS_PER_REPOSITORY = 3


@dataclass(frozen=True)
class EvidenceDocument:
    path: str
    format: str
    git_blob_sha1: str
    source_sha256: str
    source_bytes: int
    object_uri: str


@dataclass(frozen=True)
class RepositoryPacket:
    repository: str
    revision: str
    source_id: str
    primary_domain: str
    manual_topic: str
    source_type: str
    notebook_detection_count: int
    notebook_count_complete: bool
    readme_sha256: str | None
    readme_bytes: int | None
    readme_object_uri: str | None
    documents: tuple[EvidenceDocument, ...]
    acquisition_errors: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryCatalogRow:
    repository: str
    revision: str
    source_group_id: str
    source_key: str
    source_sha256: str
    packet_object_uri: str
    primary_domain: str
    manual_topic: str
    source_type: str
    notebook_detection_count: int
    selected_document_count: int
    acquisition_status: str
    review: str
    release_status: str


def fetch_bounded(url: str, maximum: int) -> bytes:
    """Fetch at most ``maximum`` bytes from one immutable source URL."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"source exceeds {maximum} byte acquisition bound")
    return data


def _raw_url(repository: str, revision: str, path: str) -> str:
    encoded = urllib.parse.quote(path.lstrip("/"), safe="/")
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{encoded}"


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _selected_documents(observation: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in observation["notebooks"]
        if int(row.get("size") or MAX_DOCUMENT_BYTES + 1) <= MAX_DOCUMENT_BYTES and row.get("sha")
    ]
    eligible.sort(key=lambda row: (int(row["size"]), str(row["path"])))
    return eligible[:MAX_DOCUMENTS_PER_REPOSITORY]


def _acquire_repository(
    observation: dict[str, Any],
    acquisition: dict[str, Any] | None,
) -> tuple[RepositoryPacket, dict[str, bytes]]:
    repository = str(observation["repo"])
    revision = str(observation["revision"])
    objects: dict[str, bytes] = {}
    errors: list[str] = []
    readme_sha256 = None
    readme_bytes = None
    readme_object_uri = None
    if acquisition and acquisition["status"] == "acquired":
        try:
            data = fetch_bounded(str(acquisition["url"]), MAX_README_BYTES)
            observed = hashlib.sha256(data).hexdigest()
            if observed != acquisition["sha256"]:
                raise ValueError(f"README digest mismatch: expected {acquisition['sha256']}, observed {observed}")
            objects[observed] = data
            readme_sha256 = observed
            readme_bytes = len(data)
            readme_object_uri = f"objects/sha256/{observed[:2]}/{observed}"
        except (OSError, ValueError) as error:
            errors.append(f"README: {type(error).__name__}: {error}")
    else:
        errors.append("README: unavailable in pinned BioTasks acquisition")

    documents = []
    for record in _selected_documents(observation):
        path = str(record["path"])
        try:
            data = fetch_bounded(_raw_url(repository, revision, path), MAX_DOCUMENT_BYTES)
            observed_blob = _git_blob_sha1(data)
            if observed_blob != record["sha"]:
                raise ValueError(f"Git blob mismatch: expected {record['sha']}, observed {observed_blob}")
            source_sha256 = hashlib.sha256(data).hexdigest()
            objects[source_sha256] = data
            documents.append(
                EvidenceDocument(
                    path=path,
                    format=str(record["format"]),
                    git_blob_sha1=observed_blob,
                    source_sha256=source_sha256,
                    source_bytes=len(data),
                    object_uri=f"objects/sha256/{source_sha256[:2]}/{source_sha256}",
                )
            )
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {type(error).__name__}: {error}")
    packet = RepositoryPacket(
        repository=repository,
        revision=revision,
        source_id=str(observation["source_id"]),
        primary_domain=str(observation["primary_domain"]),
        manual_topic=str(observation["manual_topic"]),
        source_type=str(observation["source_type"]),
        notebook_detection_count=len(observation["notebooks"]),
        notebook_count_complete=bool(observation["notebook_count_complete"]),
        readme_sha256=readme_sha256,
        readme_bytes=readme_bytes,
        readme_object_uri=readme_object_uri,
        documents=tuple(documents),
        acquisition_errors=tuple(errors),
    )
    return packet, objects


def _jsonl_bytes(records: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        for record in records
    )


def materialize(
    observations_data: bytes,
    acquisitions_data: bytes,
    *,
    output_root: str,
    campaign_id: str,
    max_workers: int,
) -> dict[str, object]:
    """Acquire bounded evidence for every repository with detected scientific documents."""
    observations = [row for row in jsonl(observations_data) if row["status"] == "present"]
    acquisitions = {str(row["repository"]).lower(): row for row in jsonl(acquisitions_data)}
    observations.sort(key=lambda row: str(row["repo"]).lower())
    if len(observations) != 405:
        raise ValueError(f"expected 405 repositories with detected documents, found {len(observations)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acquired = list(
            executor.map(lambda row: _acquire_repository(row, acquisitions.get(str(row["repo"]).lower())), observations)
        )

    fs, root = _filesystem(output_root)
    campaign_root = f"{root}/campaigns/biotasks/{campaign_id}"
    manifest_path = f"{campaign_root}/manifest.json"
    if fs.exists(manifest_path):
        raise FileExistsError(f"campaign already exists: {output_root}/campaigns/biotasks/{campaign_id}")

    catalog = []
    unique_objects: dict[str, bytes] = {}
    for packet, objects in acquired:
        unique_objects.update(objects)
        packet_dict = asdict(packet)
        for document in packet_dict["documents"]:
            document["object_uri"] = f"{output_root.rstrip('/')}/{document['object_uri']}"
        if packet_dict["readme_object_uri"]:
            packet_dict["readme_object_uri"] = f"{output_root.rstrip('/')}/{packet_dict['readme_object_uri']}"
        packet_data = (json.dumps(packet_dict, sort_keys=True, separators=(",", ":")) + "\n").encode()
        packet_sha256 = hashlib.sha256(packet_data).hexdigest()
        unique_objects[packet_sha256] = packet_data
        catalog.append(
            RepositoryCatalogRow(
                repository=packet.repository,
                revision=packet.revision,
                source_group_id=f"github:{packet.repository.lower()}",
                source_key=f"github-repository:{packet.repository.lower()}@{packet.revision}",
                source_sha256=packet_sha256,
                packet_object_uri=(f"{output_root.rstrip('/')}/objects/sha256/{packet_sha256[:2]}/{packet_sha256}"),
                primary_domain=packet.primary_domain,
                manual_topic=packet.manual_topic,
                source_type=packet.source_type,
                notebook_detection_count=packet.notebook_detection_count,
                selected_document_count=len(packet.documents),
                acquisition_status="usable" if packet.readme_sha256 or packet.documents else "unavailable",
                review="repository_notebook_detection_not_content_review",
                release_status=RELEASE_STATUS,
            )
        )

    for digest, data in unique_objects.items():
        _write_if_absent(fs, f"{root}/objects/sha256/{digest[:2]}/{digest}", data)
    catalog.sort(key=lambda row: row.repository.lower())
    catalog_data = _jsonl_bytes(asdict(row) for row in catalog)
    fs.makedirs(campaign_root, exist_ok=True)
    with fs.open(f"{campaign_root}/source-catalog.jsonl", "wb") as stream:
        stream.write(catalog_data)

    manifest: dict[str, object] = {
        "schema_version": "1",
        "campaign_id": campaign_id,
        "release_status": RELEASE_STATUS,
        "biotasks_revision": BIOTASKS_REVISION,
        "source_issue": "https://github.com/Open-Athena/biotasks/issues/5",
        "selection": "mapped repositories with notebook or literate-document detection at pinned revisions",
        "review_limitation": "format detection and README scope are not individual scientific-content review",
        "repository_count": len(catalog),
        "usable_repository_count": sum(row.acquisition_status == "usable" for row in catalog),
        "selected_document_count": sum(row.selected_document_count for row in catalog),
        "unique_object_count": len(unique_objects),
        "source_bytes": sum(len(data) for data in unique_objects.values()),
        "catalog_sha256": hashlib.sha256(catalog_data).hexdigest(),
        "catalog_uri": f"{output_root.rstrip('/')}/campaigns/biotasks/{campaign_id}/source-catalog.jsonl",
        "bounds": {
            "max_readme_bytes": MAX_README_BYTES,
            "max_document_bytes": MAX_DOCUMENT_BYTES,
            "max_documents_per_repository": MAX_DOCUMENTS_PER_REPOSITORY,
        },
        "execution": "exact source bytes acquired and hash-verified; no repository code executed",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    with fs.open(manifest_path, "wb") as stream:
        stream.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    manifest = materialize(
        fetch_bounded(OBSERVATIONS_URL, 2_000_000),
        fetch_bounded(REPOSITORY_ACQUISITION_URL, 1_000_000),
        output_root=args.output_root,
        campaign_id=args.campaign_id,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
