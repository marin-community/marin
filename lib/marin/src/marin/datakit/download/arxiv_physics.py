# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""License- and category-filtered arXiv physics abstracts with a novelty gate."""

import gzip
import json
import logging
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

import requests
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath

from marin.datakit.download.http_session import build_retrying_session
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
    write_ingestion_metadata_json,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

ARXIV_PHYSICS_NAME = "arxiv/physics-compatible-novel-abstracts"
ARXIV_METADATA_PAGE = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
ARXIV_METADATA_API = "https://www.kaggle.com/api/v1/datasets/download"
ARXIV_METADATA_FILE = "Cornell-University/arxiv/arxiv-metadata-oai-snapshot.json"
ARXIV_METADATA_VERSION = 304
ARXIV_METADATA_PATH = f"{ARXIV_METADATA_API}/{ARXIV_METADATA_FILE}"
ARXIV_METADATA_QUERY = f"datasetVersionNumber={ARXIV_METADATA_VERSION}"
ARXIV_METADATA_URL = "?".join((ARXIV_METADATA_PATH, ARXIV_METADATA_QUERY))
ARXIV_LICENSE_HELP_URL = "https://info.arxiv.org/help/license/index.html"
ARXIV_SNAPSHOT_DATE = "2026-09-19"
ARXIV_SNAPSHOT_BYTES = 5_545_506_967
ARXIV_SNAPSHOT_ETAG = '"7a485208318791dd856edda11af28e32"'
COMMON_PILE_SNAPSHOT_CUTOFF = date(2024, 9, 1)

ARXIV_COMPATIBLE_LICENSES: Mapping[str, str] = MappingProxyType(
    {
        "http://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
        "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
        "http://creativecommons.org/licenses/by-sa/4.0/": "CC BY-SA 4.0",
        "http://creativecommons.org/licenses/by-nc-sa/3.0/": "CC BY-NC-SA 3.0",
        "http://creativecommons.org/licenses/by-nc-sa/4.0/": "CC BY-NC-SA 4.0",
        "http://creativecommons.org/licenses/publicdomain/": "Public Domain",
    }
)
ARXIV_NONCOMMERCIAL_LICENSES = frozenset(
    {
        "http://creativecommons.org/licenses/by-nc-sa/3.0/",
        "http://creativecommons.org/licenses/by-nc-sa/4.0/",
    }
)
PHYSICS_CATEGORY_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "physics",
    "quant-ph",
)

ARXIV_PHYSICS_MANIFEST = IngestionSourceManifest(
    dataset_key="Cornell-University/arxiv",
    slice_key=ARXIV_PHYSICS_NAME,
    source_label=ARXIV_PHYSICS_NAME,
    source_urls=(ARXIV_METADATA_PAGE, ARXIV_METADATA_URL, ARXIV_LICENSE_HELP_URL),
    source_license="Per-record CC0/Public Domain, CC BY, CC BY-SA, or CC BY-NC-SA",
    source_format="pinned arXiv OAI metadata snapshot JSONL",
    surface_form="one title-and-abstract record per eligible arXiv paper version",
    policy=IngestionPolicy(
        usage_policy=UsagePolicy.TRAINING_ALLOWED,
        use_policy=(
            "Noncommercial ShareAlike training only. Preserve arXiv ID, selected version, categories, source URL, "
            "and exact license URI. Exclude the default arXiv distribution license, ND licenses, missing licenses, "
            "and non-physics records."
        ),
        contamination_risk="high: open-license arXiv papers overlap Common Pile, peS2o, and broad PDF corpora",
        provenance_notes=(
            "Retain records absent from Common Pile by license (BY-NC-SA), or updated after its source snapshot. "
            "Exact arXiv ID/version and text deduplication against Snowball replay is still required."
        ),
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
    ),
    staging=StagingMetadata(
        transform_name="stage_arxiv_physics_abstracts",
        serializer_name="arxiv_metadata_record_to_jsonl",
        metadata={
            "snapshot_date": ARXIV_SNAPSHOT_DATE,
            "snapshot_version": ARXIV_METADATA_VERSION,
            "snapshot_bytes": ARXIV_SNAPSHOT_BYTES,
            "snapshot_etag": ARXIV_SNAPSHOT_ETAG,
            "common_pile_snapshot_cutoff": COMMON_PILE_SNAPSHOT_CUTOFF.isoformat(),
            "dedup_identity": "arxiv_id_version",
        },
    ),
    rough_tokens_b=0.1,
    source_metadata={
        "snapshot_date": ARXIV_SNAPSHOT_DATE,
        "snapshot_version": ARXIV_METADATA_VERSION,
        "snapshot_etag": ARXIV_SNAPSHOT_ETAG,
    },
)


@dataclass(frozen=True)
class ArxivPhysicsStageConfig:
    """Output and HTTP controls for the pinned metadata snapshot."""

    output_path: str
    http_timeout: int = 900


def _is_physics_category(category: str) -> bool:
    return any(category == prefix or category.startswith(prefix + ".") for prefix in PHYSICS_CATEGORY_PREFIXES)


def arxiv_record_to_document(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert one metadata row when its subject, license, and novelty gates pass."""
    license_uri = record.get("license")
    if not isinstance(license_uri, str) or license_uri not in ARXIV_COMPATIBLE_LICENSES:
        return None

    categories_value = record.get("categories")
    if not isinstance(categories_value, str):
        return None
    categories = categories_value.split()
    if not any(_is_physics_category(category) for category in categories):
        return None

    update_date_value = record.get("update_date")
    if not isinstance(update_date_value, str):
        return None
    updated = date.fromisoformat(update_date_value)
    novelty_basis = "noncommercial_license"
    if license_uri not in ARXIV_NONCOMMERCIAL_LICENSES:
        if updated < COMMON_PILE_SNAPSHOT_CUTOFF:
            return None
        novelty_basis = "post_common_pile_snapshot"

    arxiv_id = str(record["id"])
    versions = record.get("versions")
    if not isinstance(versions, list) or not versions or not isinstance(versions[-1], dict):
        return None
    version = versions[-1].get("version")
    if not isinstance(version, str):
        return None

    title = str(record.get("title") or "").strip()
    abstract = str(record.get("abstract") or "").strip()
    if not abstract:
        return None
    text = f"# {title}\n\n{abstract}" if title else abstract
    return {
        "id": f"arxiv:{arxiv_id}{version}",
        "text": text,
        "source": ARXIV_PHYSICS_NAME,
        "title": title,
        "license": ARXIV_COMPATIBLE_LICENSES[license_uri],
        "provenance": {
            "arxiv_id": arxiv_id,
            "version": version,
            "source_url": f"https://arxiv.org/abs/{arxiv_id}{version}",
            "license_uri": license_uri,
            "categories": categories,
            "update_date": update_date_value,
            "versions": versions,
            "doi": record.get("doi"),
            "novelty_basis": novelty_basis,
            "metadata_snapshot_date": ARXIV_SNAPSHOT_DATE,
        },
    }


def _validate_snapshot_headers(response: requests.Response) -> None:
    content_length = response.headers.get("content-length")
    etag = response.headers.get("etag")
    if content_length != str(ARXIV_SNAPSHOT_BYTES) or etag != ARXIV_SNAPSHOT_ETAG:
        raise ValueError(
            "arXiv metadata snapshot changed: "
            f"expected bytes={ARXIV_SNAPSHOT_BYTES}, etag={ARXIV_SNAPSHOT_ETAG}; "
            f"found bytes={content_length}, etag={etag}"
        )


def stage_arxiv_physics_abstracts(config: ArxivPhysicsStageConfig) -> dict[str, int | str]:
    """Stream the pinned metadata snapshot into provenance-rich physics JSONL."""
    StoragePath(config.output_path).mkdirs(exist_ok=True)
    output_file = posixpath.join(config.output_path, "data.jsonl.gz")
    session = build_retrying_session()
    input_records = 0
    record_count = 0
    bytes_written = 0
    try:
        with session.get(ARXIV_METADATA_URL, timeout=config.http_timeout, stream=True) as response:
            response.raise_for_status()
            _validate_snapshot_headers(response)
            with atomic_rename(output_file) as temp_path, open_url(temp_path, "wb") as raw_output:
                with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
                    for line in response.iter_lines(chunk_size=8 * 1024 * 1024):
                        if not line:
                            continue
                        input_records += 1
                        document = arxiv_record_to_document(json.loads(line))
                        if document is None:
                            continue
                        encoded = (json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        compressed.write(encoded)
                        bytes_written += len(encoded)
                        record_count += 1
    finally:
        session.close()

    metadata_path = write_ingestion_metadata_json(
        manifest=ARXIV_PHYSICS_MANIFEST,
        materialized_output=MaterializedOutputMetadata(
            input_path=ARXIV_METADATA_URL,
            output_path=config.output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata={
                "input_records": input_records,
                "snapshot_date": ARXIV_SNAPSHOT_DATE,
                "snapshot_version": ARXIV_METADATA_VERSION,
                "snapshot_bytes": ARXIV_SNAPSHOT_BYTES,
                "snapshot_etag": ARXIV_SNAPSHOT_ETAG,
            },
        ),
    )
    logger.info("Retained %d of %d arXiv records", record_count, input_records)
    return {
        "record_count": record_count,
        "bytes_written": bytes_written,
        "output_file": output_file,
        "metadata_file": metadata_path,
    }


def arxiv_physics_normalize_steps() -> tuple[StepSpec, ...]:
    """Return pinned metadata-filter and normalization steps."""
    staged = StepSpec(
        name=f"raw/{ARXIV_PHYSICS_NAME}",
        fn=lambda output_path: stage_arxiv_physics_abstracts(ArxivPhysicsStageConfig(output_path=output_path)),
        hash_attrs={"manifest_content_fingerprint": ARXIV_PHYSICS_MANIFEST.fingerprint()},
    )
    return staged, normalize_step(
        name=f"normalized/{ARXIV_PHYSICS_NAME}",
        download=staged,
        file_extensions=(".jsonl.gz",),
    )
