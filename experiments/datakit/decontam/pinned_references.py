# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract benchmark references without importing benchmark scoring runtimes.

The catalog pins the dataset commit and SHA-256 of every input file. TaskSpec
sources index only the supplied conversation, excluding graders, dependencies,
and solutions. Field sources select problem and answer fields explicitly.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.datakit.decon import NGramConfig, _record_feature_status
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.reference_io import _write_json, _write_parquet

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).with_name("reference_catalog.json")
COVERAGE_VERSION = "atlas-eval-coverage-v2"
COVERAGE_RELATIVE = f"datakit/decontam/evals/{COVERAGE_VERSION}"
COVERAGE_MANIFEST = "_manifest.json"


class TextPolicy(StrEnum):
    FIELDS = "fields"
    TASK_SPEC = "task_spec"
    QAS = "qas"


@dataclass(frozen=True)
class ReferenceFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ReferenceSource:
    name: str
    repository: str
    revision: str
    files: tuple[ReferenceFile, ...]
    policy: TextPolicy
    fields: tuple[str, ...]
    configured_tasks: tuple[str, ...]
    benchmark: str | None = None


def reference_catalog(path: Path = CATALOG_PATH) -> tuple[ReferenceSource, ...]:
    with path.open() as source:
        catalog = json.load(source)
    return tuple(
        ReferenceSource(
            name=entry["name"],
            repository=entry["repository"],
            revision=entry["revision"],
            files=tuple(ReferenceFile(**file) for file in entry["files"]),
            policy=TextPolicy(entry["policy"]),
            fields=tuple(entry["fields"]),
            configured_tasks=tuple(entry["configured_tasks"]),
            benchmark=entry.get("benchmark"),
        )
        for entry in catalog["sources"]
    )


def reference_sources_sha256(sources: tuple[ReferenceSource, ...]) -> str:
    """Fingerprint source bytes, selections and extraction policies together."""
    content = json.dumps([asdict(source) for source in sources], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _field_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _field_text(item)))
    if isinstance(value, dict):
        return "\n".join(text for item in value.values() if (text := _field_text(item)))
    return ""


def reference_records(rows: Iterator[dict[str, Any]], source: ReferenceSource) -> Iterator[dict[str, str]]:
    """Extract the configured problem text, rejecting silent empty records."""
    for index, row in enumerate(rows):
        if source.benchmark is not None and row["benchmark"] != source.benchmark:
            continue
        identifier = str(row.get("task_id", row.get("instance_id", row.get("id", index))))
        if source.policy == TextPolicy.TASK_SPEC:
            spec = json.loads(row["task_spec"])
            events = spec["context"]["events"]
            text = "\n\n".join(
                _field_text(event["content"])
                for event in events
                if event["type"] == "message" and event["role"] in {"system", "user"}
            )
        elif source.policy == TextPolicy.QAS:
            text = _field_text(row["qas"]["question"])
        else:
            missing = set(source.fields) - row.keys()
            if missing:
                raise ValueError(f"{source.name}: missing fields {sorted(missing)}")
            text = "\n\n".join(part for field in source.fields if (part := _field_text(row[field])))
        if not text.strip():
            raise ValueError(f"{source.name}: empty problem text for {identifier}")
        yield {"id": f"{source.name}:{identifier}", "text": text}


def _reference_bytes(source: ReferenceSource, file: ReferenceFile) -> bytes:
    if source.repository.startswith("https://raw.githubusercontent.com/"):
        url = f"{source.repository}/{source.revision}/{file.path}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
    elif source.repository.startswith("s3://"):
        fs, key = filesystem_for(prefix_join(source.repository, file.path))
        with fs.open(key, "rb") as stream:
            content = stream.read()
    else:
        path = hf_hub_download(source.repository, file.path, repo_type="dataset", revision=source.revision)
        content = Path(path).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != file.sha256:
        raise ValueError(f"{source.name}/{file.path}: SHA-256 {actual}, expected {file.sha256}")
    return content


def _reference_rows(source: ReferenceSource, file: ReferenceFile) -> Iterator[dict[str, Any]]:
    content = _reference_bytes(source, file)
    if file.path.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".csv") or "test" not in Path(member).parts:
                    continue
                with archive.open(member) as stream:
                    yield from csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
        return
    if file.path.endswith((".jsonl", ".txt")):
        for line in content.decode("utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    parquet = pq.ParquetFile(io.BytesIO(content))
    for batch in parquet.iter_batches(batch_size=256):
        yield from batch.to_pylist()


def prepare_references(output_root: str, sources: tuple[ReferenceSource, ...]) -> dict[str, Any]:
    """Write a new reference release; publish its manifest after all families pass."""
    manifest_path = prefix_join(output_root, COVERAGE_MANIFEST)
    if StoragePath(manifest_path).exists():
        raise ValueError(f"Reference release already exists: {manifest_path}; choose a new version")
    entries: list[dict[str, Any]] = []
    for source in sources:
        raw_rows = chain.from_iterable(_reference_rows(source, file) for file in source.files)
        records = list(reference_records(raw_rows, source))
        extracted_records = len(records)
        ngram = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)
        unmatchable_ids = [record["id"] for record in records if not _record_feature_status(record["text"], ngram)[1]]
        omitted = set(unmatchable_ids)
        records = [record for record in records if record["id"] not in omitted]
        if not records:
            raise ValueError(f"{source.name}: configured reference family yielded zero matchable records")
        artifact = f"{source.name}/eval.parquet"
        output_path = prefix_join(output_root, artifact)
        count = _write_parquet(output_path, iter(records))
        with StoragePath(output_path).open("rb") as stream:
            artifact_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append(
            {
                "name": source.name,
                "artifact": artifact,
                "expected_records": count,
                "extracted_records": extracted_records,
                "unmatchable_ids": unmatchable_ids,
                "omission_reason": "No self-match under the canonical 13-gram/minimum-two-feature policy",
                "sha256": artifact_sha256,
                "repository": source.repository,
                "revision": source.revision,
                "files": [{"path": file.path, "sha256": file.sha256} for file in source.files],
                "configured_tasks": list(source.configured_tasks),
            }
        )
        logger.info("%s: %d records, SHA-256 %s", source.name, count, artifact_sha256)
    manifest = {
        "schema_version": 1,
        "corpus_version": COVERAGE_VERSION,
        "source_catalog_sha256": reference_sources_sha256(sources),
        "required": True,
        "status": "complete",
        "benchmarks": entries,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=prefix_join(marin_prefix(), COVERAGE_RELATIVE))
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    args = parser.parse_args()
    configure_logging(logging.INFO)
    prepare_references(args.output_root, reference_catalog(args.catalog))


if __name__ == "__main__":
    main()
