# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned, attribution-preserving ingestion for selected MIT OpenCourseWare science courses."""

import gzip
import json
import logging
import posixpath
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from bs4 import BeautifulSoup
from markdownify import markdownify
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

LICENSE_URL = "https://ocw.mit.edu/pages/privacy-and-terms-of-use/"


@dataclass(frozen=True)
class MitOcwCourse:
    """Immutable HTTP identity and domain metadata for one OCW course package."""

    name: str
    slug: str
    archive_name: str
    etag: str
    domain: str
    rough_tokens_b: float

    @property
    def course_url(self) -> str:
        return f"https://ocw.mit.edu/courses/{self.slug}/"

    @property
    def archive_url(self) -> str:
        return self.course_url + self.archive_name

    @property
    def manifest(self) -> IngestionSourceManifest:
        return IngestionSourceManifest(
            dataset_key=f"mit-ocw/{self.slug}",
            slice_key=self.name,
            source_label=self.name,
            source_urls=(self.course_url, self.archive_url, LICENSE_URL),
            source_license="CC BY-NC-SA 4.0 with MIT OCW AI-training conditions",
            source_format="official_ocw_course_zip_html",
            surface_form="one_markdown_record_per_html_content page",
            policy=IngestionPolicy(
                usage_policy=UsagePolicy.TRAINING_ALLOWED,
                use_policy=(
                    "Noncommercial ShareAlike model training approved; preserve course, instructor, and MIT OCW "
                    "attribution in training records and distributed model documentation."
                ),
                contamination_risk="medium: OCW pages and course files may occur in web and PDF corpora",
                provenance_notes=(
                    "MIT OCW explicitly permits AI training for noncommercial models with attribution and "
                    "ShareAlike model distribution. Third-party readings omitted from OCW packages remain excluded."
                ),
                identity_treatment=IdentityTreatment.PRESERVE,
                secret_redaction=SecretRedaction.NONE,
            ),
            staging=StagingMetadata(
                transform_name="stage_mit_ocw_html",
                serializer_name="ocw_html_to_markdown",
                metadata={
                    "slug": self.slug,
                    "archive_etag": self.etag,
                    "member_glob": "**/index.html",
                    "domain": self.domain,
                },
            ),
            rough_tokens_b=self.rough_tokens_b,
            source_metadata={"archive_etag": self.etag, "domain": self.domain},
        )


MIT_OCW_SCIENCE_COURSES: Mapping[str, MitOcwCourse] = MappingProxyType(
    {
        "mit-ocw/physics/classical-mechanics": MitOcwCourse(
            name="mit-ocw/physics/classical-mechanics",
            slug="8-01sc-classical-mechanics-fall-2016",
            archive_name="8.01sc-fall-2016.zip",
            etag='"dea309d37ae74ec51b10f591b42c8790-25"',
            domain="physics",
            rough_tokens_b=0.01,
        ),
        "mit-ocw/biology/fundamentals": MitOcwCourse(
            name="mit-ocw/biology/fundamentals",
            slug="7-01sc-fundamentals-of-biology-fall-2011",
            archive_name="7.01sc-fall-2011.zip",
            etag='"23cd9079791db7de008207531fa53548-4"',
            domain="biology",
            rough_tokens_b=0.01,
        ),
        "mit-ocw/chemistry/solid-state": MitOcwCourse(
            name="mit-ocw/chemistry/solid-state",
            slug="3-091-introduction-to-solid-state-chemistry-fall-2018",
            archive_name="3.091-fall-2018.zip",
            etag='"c0dac84279ee2f6785f63d4b0adc5299-6"',
            domain="chemistry",
            rough_tokens_b=0.01,
        ),
    }
)


def ocw_html_to_markdown(payload: bytes) -> tuple[str, str]:
    """Extract the content-bearing main element from one OCW HTML page."""
    soup = BeautifulSoup(payload, "html.parser")
    for element in soup.select("script, style, nav, footer, form, noscript"):
        element.decompose()
    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body
    if main is None:
        return "", ""
    title_element = main.find("h1") or soup.find("title")
    title = title_element.get_text(" ", strip=True) if title_element else ""
    text = markdownify(str(main), heading_style="ATX").strip()
    return title, text


def download_course_archive(course: MitOcwCourse, output_path: str) -> dict[str, int | str]:
    """Stream an official OCW archive and reject mutable-source drift."""
    StoragePath(output_path).mkdirs(exist_ok=True)
    output_file = posixpath.join(output_path, "course.zip")
    session = build_retrying_session()
    bytes_written = 0
    try:
        with session.get(course.archive_url, timeout=600, stream=True) as response:
            response.raise_for_status()
            observed_etag = response.headers.get("etag")
            if observed_etag != course.etag:
                raise ValueError(f"{course.name} archive ETag changed: expected {course.etag}, found {observed_etag}")
            with atomic_rename(output_file) as temp_path, open_url(temp_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    output.write(chunk)
                    bytes_written += len(chunk)
    finally:
        session.close()
    metadata_path = write_ingestion_metadata_json(
        manifest=course.manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=course.archive_url,
            output_path=output_path,
            output_file=output_file,
            record_count=0,
            bytes_written=bytes_written,
            metadata={"archive_etag": course.etag},
        ),
    )
    return {"bytes_written": bytes_written, "output_file": output_file, "metadata_file": metadata_path}


def stage_course_html(course: MitOcwCourse, input_path: str, output_path: str) -> dict[str, int | str]:
    """Convert content pages from one pinned OCW course package into JSONL."""
    StoragePath(output_path).mkdirs(exist_ok=True)
    output_file = posixpath.join(output_path, "data.jsonl.gz")
    record_count = 0
    bytes_written = 0
    with open_url(posixpath.join(input_path, "course.zip"), "rb") as archive_file:
        with zipfile.ZipFile(archive_file) as archive:
            members = sorted(
                member
                for member in archive.namelist()
                if member.endswith("/index.html") and not member.startswith("__MACOSX/")
            )
            with atomic_rename(output_file) as temp_path, open_url(temp_path, "wb") as raw_output:
                with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
                    for member in members:
                        title, text = ocw_html_to_markdown(archive.read(member))
                        if not text:
                            continue
                        record = {
                            "id": f"{course.slug}:{member}",
                            "text": text,
                            "source": course.name,
                            "title": title,
                            "license": course.manifest.source_license,
                            "provenance": {
                                "course_url": course.course_url,
                                "archive_url": course.archive_url,
                                "archive_etag": course.etag,
                                "member": member,
                                "domain": course.domain,
                            },
                        }
                        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        compressed.write(encoded)
                        bytes_written += len(encoded)
                        record_count += 1
    metadata_path = write_ingestion_metadata_json(
        manifest=course.manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=posixpath.join(input_path, "course.zip"),
            output_path=output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata={"archive_etag": course.etag},
        ),
    )
    logger.info("Staged %d HTML pages from %s", record_count, course.name)
    return {
        "record_count": record_count,
        "bytes_written": bytes_written,
        "output_file": output_file,
        "metadata_file": metadata_path,
    }


def mit_ocw_course_normalize_steps(course: MitOcwCourse) -> tuple[StepSpec, ...]:
    """Return archive download, HTML transform, and normalization steps for one course."""
    download = StepSpec(
        name=f"raw/{course.name}",
        fn=lambda output_path: download_course_archive(course, output_path),
        hash_attrs={"manifest_content_fingerprint": course.manifest.fingerprint()},
    )
    staged = StepSpec(
        name=f"processed/{course.name}",
        deps=[download],
        fn=lambda output_path: stage_course_html(course, download.output_path, output_path),
        hash_attrs={"manifest_content_fingerprint": course.manifest.fingerprint()},
    )
    return (
        download,
        staged,
        normalize_step(
            name=f"normalized/{course.name}",
            download=staged,
            file_extensions=(".jsonl.gz",),
        ),
    )


def mit_ocw_science_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return one pipeline per approved physics, biology, and chemistry course."""
    return {name: mit_ocw_course_normalize_steps(course) for name, course in MIT_OCW_SCIENCE_COURSES.items()}
