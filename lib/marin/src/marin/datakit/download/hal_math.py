# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed ingestion of rights-compatible HAL mathematics abstracts."""

import gzip
import json
import logging
import posixpath
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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

HAL_SEARCH_URL = "https://api.hal.science/search/"
HAL_REUSE_TERMS_URL = "https://doc.hal.science/aspects-juridiques/conditions-de-reutilisation/"
HAL_DATASET_REVISION = "2dab727888c4f29eb61d1a3abb9764f5b03ea0c2"
HAL_DATASET_REPO_URL = "https://huggingface.co/datasets/aritol/hal-open-archive-mathematics-resources"
HAL_DATASET_URL = f"{HAL_DATASET_REPO_URL}/tree/{HAL_DATASET_REVISION}"
HAL_MATH_ABSTRACTS_NAME = "hal/mathematics-licensed-abstracts"
HAL_API_PAGE_SIZE = 10_000
HAL_API_FIELDS = (
    "docid",
    "halId_s",
    "uri_s",
    "title_s",
    "abstract_s",
    "fileLicenses_s",
    "arxivId_s",
    "doiId_s",
    "domain_s",
    "submittedDate_s",
)

HAL_COMPATIBLE_LICENSES: Mapping[str, str] = MappingProxyType(
    {
        "https://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/": "CC BY-SA 4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/": "CC BY-NC 4.0",
        "https://creativecommons.org/licenses/by-nc-sa/4.0/": "CC BY-NC-SA 4.0",
        "https://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
        "https://creativecommons.org/public-domain/pdm/": "Public Domain Mark",
        "https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf": (
            "Etalab Open Licence 2.0"
        ),
    }
)

HAL_MATH_ABSTRACTS_MANIFEST = IngestionSourceManifest(
    dataset_key="aritol/hal-open-archive-mathematics-resources",
    slice_key=HAL_MATH_ABSTRACTS_NAME,
    source_label=HAL_MATH_ABSTRACTS_NAME,
    source_urls=(HAL_DATASET_URL, HAL_SEARCH_URL, HAL_REUSE_TERMS_URL),
    source_license="Per-record allowlist: CC0/PDM, CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA, or Etalab 2.0",
    source_format="HAL search API JSON",
    surface_form="one title-and-abstract record per licensed HAL mathematics document",
    policy=IngestionPolicy(
        usage_policy=UsagePolicy.TRAINING_ALLOWED,
        use_policy=(
            "Noncommercial ShareAlike training only. Preserve document identifiers, source URLs, and exact "
            "per-file license URIs. Do not admit HAL Authorization, Copyright, ND, missing-license, or "
            "mixed-compatible/incompatible file records."
        ),
        contamination_risk="high: HAL papers can be cross-posted to arXiv and indexed in broad PDF corpora",
        provenance_notes=(
            "The Hugging Face repository MIT badge covers packaging, not paper rights. Numeric archive filenames "
            "are HAL docids; this source instead queries HAL's authoritative per-file license metadata directly."
        ),
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
    ),
    staging=StagingMetadata(
        transform_name="stage_hal_math_abstracts",
        serializer_name="hal_api_record_to_jsonl",
        metadata={
            "dataset_revision": HAL_DATASET_REVISION,
            "domain_query": "0.math",
            "license_field": "fileLicenses_s",
            "dedup_identity": "docid",
        },
    ),
    rough_tokens_b=0.005,
    source_metadata={"dataset_revision": HAL_DATASET_REVISION},
)


@dataclass(frozen=True)
class HalMathStageConfig:
    """Output and API controls for one HAL metadata snapshot."""

    output_path: str
    page_size: int = HAL_API_PAGE_SIZE
    http_timeout: int = 120


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def hal_record_to_document(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert one HAL API record, rejecting incomplete or rights-ambiguous rows."""
    licenses = set(_string_list(record.get("fileLicenses_s")))
    if not licenses or not licenses.issubset(HAL_COMPATIBLE_LICENSES):
        return None

    abstracts = _string_list(record.get("abstract_s"))
    if not abstracts:
        return None
    titles = _string_list(record.get("title_s"))
    text_parts = [*(f"# {title}" for title in titles), *abstracts]
    docid = str(record["docid"])
    hal_id = str(record["halId_s"])
    source_url = str(record.get("uri_s") or f"https://hal.science/{hal_id}")
    return {
        "id": f"hal:{docid}",
        "text": "\n\n".join(text_parts),
        "source": HAL_MATH_ABSTRACTS_NAME,
        "title": titles[0] if titles else "",
        "license": sorted(HAL_COMPATIBLE_LICENSES[license_uri] for license_uri in licenses),
        "provenance": {
            "docid": docid,
            "hal_id": hal_id,
            "source_url": source_url,
            "license_uris": sorted(licenses),
            "arxiv_ids": _string_list(record.get("arxivId_s")),
            "doi_ids": _string_list(record.get("doiId_s")),
            "domains": _string_list(record.get("domain_s")),
            "submitted_date": record.get("submittedDate_s"),
        },
    }


def _hal_license_query(license_uri: str) -> str:
    license_clause = f'fileLicenses_s:"{license_uri}"'
    return f"(domain_s:0.math) AND (submitType_s:file) AND ({license_clause})"


def _iter_license_records(
    session: requests.Session,
    license_uri: str,
    *,
    page_size: int,
    http_timeout: int,
) -> Iterator[dict[str, Any]]:
    cursor = "*"
    while True:
        response = session.get(
            HAL_SEARCH_URL,
            params={
                "q": _hal_license_query(license_uri),
                "fl": ",".join(HAL_API_FIELDS),
                "rows": page_size,
                "sort": "docid asc",
                "cursorMark": cursor,
                "wt": "json",
            },
            timeout=http_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload["response"]["docs"]
        yield from records
        next_cursor = payload["nextCursorMark"]
        if not records or next_cursor == cursor:
            return
        cursor = next_cursor


def stage_hal_math_abstracts(config: HalMathStageConfig) -> dict[str, int | str]:
    """Snapshot compatible HAL mathematics abstracts with rights metadata."""
    if config.page_size <= 0 or config.page_size > HAL_API_PAGE_SIZE:
        raise ValueError(f"page_size must be in [1, {HAL_API_PAGE_SIZE}]")

    StoragePath(config.output_path).mkdirs(exist_ok=True)
    output_file = posixpath.join(config.output_path, "data.jsonl.gz")
    session = build_retrying_session()
    seen_docids: set[str] = set()
    record_count = 0
    bytes_written = 0
    try:
        with atomic_rename(output_file) as temp_path, open_url(temp_path, "wb") as raw_output:
            with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
                for license_uri in HAL_COMPATIBLE_LICENSES:
                    for record in _iter_license_records(
                        session,
                        license_uri,
                        page_size=config.page_size,
                        http_timeout=config.http_timeout,
                    ):
                        document = hal_record_to_document(record)
                        if document is None:
                            continue
                        docid = document["provenance"]["docid"]
                        if docid in seen_docids:
                            continue
                        seen_docids.add(docid)
                        encoded = (json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        compressed.write(encoded)
                        bytes_written += len(encoded)
                        record_count += 1
    finally:
        session.close()

    metadata_path = write_ingestion_metadata_json(
        manifest=HAL_MATH_ABSTRACTS_MANIFEST,
        materialized_output=MaterializedOutputMetadata(
            input_path=HAL_SEARCH_URL,
            output_path=config.output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata={
                "dataset_revision": HAL_DATASET_REVISION,
                "license_uris": sorted(HAL_COMPATIBLE_LICENSES),
            },
        ),
    )
    logger.info("Staged %d rights-compatible HAL mathematics abstracts", record_count)
    return {
        "record_count": record_count,
        "bytes_written": bytes_written,
        "output_file": output_file,
        "metadata_file": metadata_path,
    }


def hal_math_abstracts_normalize_steps() -> tuple[StepSpec, ...]:
    """Return HAL API snapshot and normalization steps."""
    staged = StepSpec(
        name=f"raw/{HAL_MATH_ABSTRACTS_NAME}",
        fn=lambda output_path: stage_hal_math_abstracts(HalMathStageConfig(output_path=output_path)),
        hash_attrs={"manifest_content_fingerprint": HAL_MATH_ABSTRACTS_MANIFEST.fingerprint()},
    )
    return staged, normalize_step(
        name=f"normalized/{HAL_MATH_ABSTRACTS_NAME}",
        download=staged,
        file_extensions=(".jsonl.gz",),
    )
