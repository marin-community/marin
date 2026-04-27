# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small package-metadata eval ingestion for issue #5061.

This first pass materializes a bounded slice from the public npm registry API. The
emitted records keep the package metadata surfaces that matter for PPL-gap work:

- package names, including scoped packages
- version strings and dist-tags
- dependency edges and semver constraints
- tarball URLs, integrity hashes, and shasums
- package/repository/homepage URLs

We intentionally exclude README blobs and maintainer identity fields so the slice
stays focused on dependency metadata rather than free-form package docs.
"""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url, url_to_fs
from urllib3.util import Retry
from zephyr.writers import atomic_rename

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
    write_ingestion_metadata_json,
)
from marin.execution.executor import THIS_OUTPUT_PATH, ExecutorStep, VersionedValue, versioned
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

EPIC_5005 = 5005
PACKAGE_METADATA_ISSUE = 5061
NPM_REGISTRY_SLICE_KEY = "package_metadata/npm_registry_metadata"
NPM_REGISTRY_DOCS_URL = "https://docs.npmjs.com/policies/crawlers/"
DEFAULT_NPM_REGISTRY_BASE_URL = "https://registry.npmjs.org/"
DEFAULT_HTTP_TIMEOUT_SECONDS = 120
DEFAULT_MAX_VERSIONS_PER_PACKAGE = 4
DEFAULT_OUTPUT_FILENAME = "data.jsonl.gz"
DEFAULT_PACKAGE_NAMES: tuple[str, ...] = (
    "express",
    "vite",
    "react-dom",
    "@babel/core",
    "@types/node",
)
_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "_id",
    "name",
    "dist-tags",
    "homepage",
    "repository",
    "bugs",
    "license",
    "keywords",
    "description",
)
_VERSION_FIELDS: tuple[str, ...] = (
    "name",
    "version",
    "license",
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "peerDependenciesMeta",
    "optionalDependencies",
    "bundleDependencies",
    "bundledDependencies",
    "engines",
    "os",
    "cpu",
    "bin",
    "dist",
    "deprecated",
    "funding",
    "homepage",
    "repository",
    "bugs",
    "keywords",
)


@dataclass(frozen=True)
class NpmRegistryMetadataSource:
    """Describes the bounded npm registry metadata slice for issue #5061."""

    slice_key: str = NPM_REGISTRY_SLICE_KEY
    registry_base_url: str = DEFAULT_NPM_REGISTRY_BASE_URL
    package_names: tuple[str, ...] = DEFAULT_PACKAGE_NAMES
    max_versions_per_package: int = DEFAULT_MAX_VERSIONS_PER_PACKAGE
    source_label: str = "npm_registry"

    def validate(self) -> None:
        if not self.registry_base_url.endswith("/"):
            raise ValueError("registry_base_url must end with '/'")
        if not self.package_names:
            raise ValueError("package_names must not be empty")
        if self.max_versions_per_package <= 0:
            raise ValueError("max_versions_per_package must be positive")

    def manifest(self) -> IngestionSourceManifest:
        """Return the shared ingestion manifest for this source."""

        self.validate()
        return IngestionSourceManifest(
            dataset_key="npm/public_registry_metadata",
            slice_key=self.slice_key,
            source_label=self.source_label,
            source_urls=(self.registry_base_url, NPM_REGISTRY_DOCS_URL),
            source_license=(
                "Mixed per-package metadata from the public npm registry; emitted slice excludes README "
                "content and is eval-only until package-license policy is reviewed."
            ),
            source_format="npm registry package JSON",
            surface_form="registry_json",
            policy=IngestionPolicy(
                usage_policy=UsagePolicy.EVAL_ONLY,
                use_policy="Eval-only package/dependency metadata slice.",
                requires_sanitization=False,
                identity_treatment=IdentityTreatment.PRESERVE,
                secret_redaction=SecretRedaction.NONE,
                contamination_risk="low: operational package metadata, but package licensing is heterogeneous",
                provenance_notes=(
                    "Materialization keeps structural registry metadata only and drops README/maintainer fields."
                ),
            ),
            staging=StagingMetadata(
                transform_name="download_npm_registry_metadata",
                serializer_name="canonical_registry_json",
                output_filename=DEFAULT_OUTPUT_FILENAME,
                record_provenance_fields=("package_name", "version", "source_url"),
                metadata={"excluded_fields": ("readme", "readmeFilename", "maintainers", "users", "contributors")},
            ),
            epic_issue=EPIC_5005,
            issue_numbers=(PACKAGE_METADATA_ISSUE,),
            sample_caps=SampleCapConfig(max_examples=len(self.package_names) * self.max_versions_per_package),
            source_metadata={
                "package_names": list(self.package_names),
                "max_versions_per_package": self.max_versions_per_package,
            },
        )


@dataclass
class DownloadNpmRegistryMetadataConfig:
    """Executor config for :func:`download_npm_registry_metadata`."""

    source: NpmRegistryMetadataSource
    output_path: str | VersionedValue[str] = THIS_OUTPUT_PATH
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS
    cache_key: dict[str, Any] | VersionedValue[dict[str, Any]] = field(default_factory=dict, repr=False)


def _build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _package_url(source: NpmRegistryMetadataSource, package_name: str) -> str:
    return f"{source.registry_base_url}{quote(package_name, safe='')}"


def _selected_versions(payload: dict[str, Any], *, max_versions: int) -> list[str]:
    versions = payload.get("versions", {})
    version_times = payload.get("time", {})
    latest = payload.get("dist-tags", {}).get("latest")

    ordered = sorted(
        versions,
        key=lambda version: (version_times.get(version, ""), version),
        reverse=True,
    )
    selected: list[str] = []
    if latest in versions:
        selected.append(latest)
    for version in ordered:
        if version not in selected:
            selected.append(version)
        if len(selected) >= max_versions:
            break
    return selected


def _pick_fields(payload: dict[str, Any], field_names: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: payload[field_name] for field_name in field_names if field_name in payload}


def _record_payload(
    *,
    raw_payload: dict[str, Any],
    package_name: str,
    version: str,
    source_url: str,
) -> dict[str, Any]:
    version_payload = raw_payload["versions"][version]
    selected = _pick_fields(raw_payload, _TOP_LEVEL_FIELDS)
    selected["source_url"] = source_url
    selected["publish_time"] = raw_payload.get("time", {}).get(version)
    selected["version_metadata"] = _pick_fields(version_payload, _VERSION_FIELDS)
    return selected


def _record_id(slice_key: str, package_name: str, version: str) -> str:
    return f"{slice_key}#{quote(package_name, safe='')}@{version}"


def download_npm_registry_metadata(config: DownloadNpmRegistryMetadataConfig) -> dict[str, Any]:
    """Download a bounded slice of npm registry metadata and write gzipped JSONL output."""

    source = config.source
    source.validate()
    manifest = source.manifest()
    output_path = str(config.output_path)
    fsspec_mkdirs(output_path, exist_ok=True)

    output_file = posixpath.join(output_path, config.output_filename)
    session = _build_session()
    records_written = 0
    package_summaries: list[dict[str, Any]] = []
    try:
        with atomic_rename(output_file) as temp_path:
            with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as handle:
                for package_name in source.package_names:
                    source_url = _package_url(source, package_name)
                    response = session.get(source_url, timeout=config.http_timeout_seconds)
                    response.raise_for_status()
                    raw_payload = response.json()
                    selected_versions = _selected_versions(raw_payload, max_versions=source.max_versions_per_package)
                    package_summaries.append(
                        {
                            "package_name": package_name,
                            "source_url": source_url,
                            "selected_versions": selected_versions,
                        }
                    )
                    for version in selected_versions:
                        payload = _record_payload(
                            raw_payload=raw_payload,
                            package_name=package_name,
                            version=version,
                            source_url=source_url,
                        )
                        record = {
                            "id": _record_id(source.slice_key, package_name, version),
                            "package_name": package_name,
                            "version": version,
                            "source": source.source_label,
                            "source_url": source_url,
                            "text": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                        handle.write("\n")
                        records_written += 1
                    logger.info(
                        "Materialized %d npm registry versions for %s",
                        len(selected_versions),
                        package_name,
                    )
    finally:
        session.close()

    fs, _ = url_to_fs(output_file)
    output_size = int(fs.info(output_file)["size"])
    metadata_path = write_ingestion_metadata_json(
        manifest=manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=source.registry_base_url,
            output_path=output_path,
            output_file=output_file,
            record_count=records_written,
            bytes_written=output_size,
            metadata={"packages": package_summaries},
        ),
    )
    return {
        "metadata_path": metadata_path,
        "output_file": output_file,
        "record_count": records_written,
        "packages": package_summaries,
    }


def npm_registry_metadata_step(
    source: NpmRegistryMetadataSource,
    *,
    name: str | None = None,
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> ExecutorStep[DownloadNpmRegistryMetadataConfig]:
    """Create the executor step that materializes the bounded npm metadata slice."""

    source.validate()
    manifest = source.manifest()
    step_name = name or f"raw/{source.slice_key}"
    return ExecutorStep(
        name=step_name,
        fn=download_npm_registry_metadata,
        config=DownloadNpmRegistryMetadataConfig(
            source=source,
            http_timeout_seconds=http_timeout_seconds,
            cache_key=versioned(
                {
                    "slice_key": source.slice_key,
                    "registry_base_url": source.registry_base_url,
                    "package_names": list(source.package_names),
                    "max_versions_per_package": source.max_versions_per_package,
                    "manifest_fingerprint": manifest.fingerprint(),
                }
            ),
        ),
    )
