# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""License-preserving ingestion for pinned OpenStax CNXML books."""

import gzip
import io
import json
import logging
import posixpath
import re
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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

CNXML_MEMBER_RE = re.compile(r"^[^/]+/modules/(?P<module_id>[^/]+)/index\.cnxml$")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class OpenStaxBook:
    """Pinned source identity and rights metadata for one OpenStax repository."""

    name: str
    repository: str
    revision: str
    license: str
    rough_tokens_b: float

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"

    @property
    def manifest(self) -> IngestionSourceManifest:
        return IngestionSourceManifest(
            dataset_key=self.repository,
            slice_key=self.name,
            source_label=self.name,
            source_urls=(
                f"https://github.com/{self.repository}/tree/{self.revision}",
                self.archive_url,
            ),
            source_license=self.license,
            source_format="openstax_cnxml_git_archive",
            surface_form="one_markdown_record_per_cnxml_module",
            policy=IngestionPolicy(
                usage_policy=UsagePolicy.TRAINING_ALLOWED,
                use_policy=(
                    "Noncommercial ShareAlike training approved; preserve source and license attribution in "
                    "every record and metadata sidecar."
                ),
                contamination_risk="medium: OpenStax chapters may occur in web and OER corpora",
                provenance_notes=f"The source repository LICENSE records {self.license} at the pinned commit.",
                identity_treatment=IdentityTreatment.PRESERVE,
                secret_redaction=SecretRedaction.NONE,
            ),
            staging=StagingMetadata(
                transform_name="stage_openstax_cnxml",
                serializer_name="cnxml_to_markdown",
                metadata={
                    "repository": self.repository,
                    "revision": self.revision,
                    "member_glob": "modules/*/index.cnxml",
                    "dedup_identity": "repository_revision_module_id",
                },
            ),
            rough_tokens_b=self.rough_tokens_b,
            source_metadata={"repository_revision": self.revision},
        )


OPENSTAX_BOOKS: Mapping[str, OpenStaxBook] = MappingProxyType(
    {
        "openstax/physics": OpenStaxBook(
            name="openstax/physics",
            repository="openstax/osbooks-physics",
            revision="dfdfd7a5356ecdd42e504de3df50d9153e33ea49",
            license="CC BY 4.0",
            rough_tokens_b=0.002,
        ),
        "openstax/chemistry": OpenStaxBook(
            name="openstax/chemistry",
            repository="openstax/osbooks-chemistry-bundle",
            revision="3be4b60ff501f29a445f0cacf003e5f5cc16244d",
            license="CC BY-NC-SA 4.0",
            rough_tokens_b=0.004,
        ),
        "openstax/biology": OpenStaxBook(
            name="openstax/biology",
            repository="openstax/osbooks-biology-bundle",
            revision="63f8b6f8d129dd1582989bb755011e9a6d523471",
            license="CC BY-NC-SA 4.0",
            rough_tokens_b=0.006,
        ),
        "openstax/college-physics": OpenStaxBook(
            name="openstax/college-physics",
            repository="openstax/osbooks-college-physics-bundle",
            revision="fd1b25dfd5d8c6580c6e2b2b34a19e29cc69ada9",
            license="CC BY-NC-SA 4.0",
            rough_tokens_b=0.004,
        ),
        "openstax/university-physics": OpenStaxBook(
            name="openstax/university-physics",
            repository="openstax/osbooks-university-physics-bundle",
            revision="2300d5b8ca2571dacc6ec25ebef56ba8d49000a2",
            license="CC BY-NC-SA 4.0",
            rough_tokens_b=0.004,
        ),
        "openstax/organic-chemistry": OpenStaxBook(
            name="openstax/organic-chemistry",
            repository="openstax/osbooks-organic-chemistry",
            revision="8917713cdfb7f74018a8fd43cdcfe3173419bb82",
            license="CC BY-NC-SA 4.0",
            rough_tokens_b=0.004,
        ),
    }
)

OPENSTAX_PHYSICS_MANIFEST = OPENSTAX_BOOKS["openstax/physics"].manifest
OPENSTAX_PHYSICS_ARCHIVE_URL = OPENSTAX_BOOKS["openstax/physics"].archive_url
OPENSTAX_PHYSICS_ROUGH_TOKEN_COUNT_B = OPENSTAX_BOOKS["openstax/physics"].rough_tokens_b


@dataclass(frozen=True)
class OpenStaxStageConfig:
    """Inputs for staging one pinned OpenStax repository."""

    manifest: IngestionSourceManifest
    archive_url: str
    output_path: str
    revision: str
    http_timeout_seconds: int = 600


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalized_text(element: ET.Element) -> str:
    return WHITESPACE_RE.sub(" ", " ".join(element.itertext())).strip()


def cnxml_to_markdown(payload: bytes) -> tuple[str, str]:
    """Render a CNXML module as compact Markdown and return ``(title, text)``."""
    root = ET.fromstring(payload)
    content = next((child for child in root if _local_name(child.tag) == "content"), None)
    if content is None:
        raise ValueError("CNXML document has no content element")

    document_title = next(
        (_normalized_text(child) for child in root if _local_name(child.tag) == "title"),
        "",
    )
    blocks: list[str] = []
    if document_title:
        blocks.append(f"# {document_title}")

    block_names = {"title", "para", "caption", "code"}
    for element in content.iter():
        name = _local_name(element.tag)
        if name not in block_names:
            continue
        text = _normalized_text(element)
        if not text:
            continue
        if name == "title":
            blocks.append(f"## {text}")
        elif name == "code":
            blocks.append(f"```\n{text}\n```")
        else:
            blocks.append(text)

    return document_title, "\n\n".join(blocks)


def _cnxml_members(response_raw: io.BufferedIOBase) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(fileobj=response_raw, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = CNXML_MEMBER_RE.match(member.name)
            if match is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            yield match.group("module_id"), handle.read()


def stage_openstax_cnxml(config: OpenStaxStageConfig) -> dict[str, int | str]:
    """Stream a pinned OpenStax archive into provenance-rich JSONL records."""
    if not config.manifest.policy.training_allowed:
        raise ValueError(f"{config.manifest.slice_key} is not approved for training")

    StoragePath(config.output_path).mkdirs(exist_ok=True)
    output_file = posixpath.join(config.output_path, "data.jsonl.gz")
    session = build_retrying_session()
    record_count = 0
    bytes_written = 0
    try:
        with session.get(config.archive_url, timeout=config.http_timeout_seconds, stream=True) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with atomic_rename(output_file) as temp_path, open_url(temp_path, "wb") as raw_output:
                with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
                    for module_id, payload in _cnxml_members(response.raw):
                        title, text = cnxml_to_markdown(payload)
                        if not text:
                            continue
                        record = {
                            "id": f"{config.manifest.dataset_key}@{config.revision}:{module_id}",
                            "text": text,
                            "source": config.manifest.source_label,
                            "title": title,
                            "license": config.manifest.source_license,
                            "provenance": {
                                "repository": config.manifest.dataset_key,
                                "revision": config.revision,
                                "module_id": module_id,
                                "source_url": config.manifest.source_urls[0],
                            },
                        }
                        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        compressed.write(encoded)
                        bytes_written += len(encoded)
                        record_count += 1
    finally:
        session.close()

    metadata_path = write_ingestion_metadata_json(
        manifest=config.manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=config.archive_url,
            output_path=config.output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata={"repository_revision": config.revision},
        ),
    )
    logger.info("Staged %d OpenStax CNXML modules", record_count)
    return {
        "record_count": record_count,
        "bytes_written": bytes_written,
        "output_file": output_file,
        "metadata_file": metadata_path,
    }


def openstax_normalize_steps(book: OpenStaxBook) -> tuple[StepSpec, ...]:
    """Return staging and normalization steps for one pinned OpenStax book repository."""
    staged = StepSpec(
        name=f"raw/{book.name}",
        fn=lambda output_path: stage_openstax_cnxml(
            OpenStaxStageConfig(
                manifest=book.manifest,
                archive_url=book.archive_url,
                output_path=output_path,
                revision=book.revision,
            )
        ),
        hash_attrs={"manifest_content_fingerprint": book.manifest.fingerprint()},
    )
    return staged, normalize_step(
        name=f"normalized/{book.name}",
        download=staged,
        file_extensions=(".jsonl.gz",),
    )


def openstax_science_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return one provenance-preserving pipeline for each approved science repository."""
    return {name: openstax_normalize_steps(book) for name, book in OPENSTAX_BOOKS.items()}


def openstax_physics_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the legacy OpenStax Physics source chain."""
    return openstax_normalize_steps(OPENSTAX_BOOKS["openstax/physics"])
