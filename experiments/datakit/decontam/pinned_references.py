# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned Micro210, SWE-bench test and long-loop holdout references."""

import argparse
import hashlib
import io
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.datakit.decon_keys import WINDOW, text_stream
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.prepare_eval_corpus import _write_json, _write_parquet

COVERAGE_VERSION = "atlas-benchmark-references-v1"
COVERAGE_RELATIVE = f"datakit/decontam/evals/{COVERAGE_VERSION}"
COVERAGE_MANIFEST = "_manifest.json"


class TextPolicy(StrEnum):
    PROBLEM = "problem"
    TASK_SPEC = "task_spec"


@dataclass(frozen=True)
class ReferenceFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ReferenceSource:
    name: str
    repository: str
    revision: str
    file: ReferenceFile
    expected_records: int
    policy: TextPolicy
    benchmark: str | None = None


_MICRO_BENCHMARKS = (
    "cruxeval",
    "bixbench-pi",
    "humanevalplus",
    "tb2-recovery",
    "ot-tblite-recovery",
    "gsm8k-0shot",
    "ds-1000-local",
    "ifbench",
    "bfclparity-pi",
    "truthfulqa",
    "olympiadbench",
    "triviaqa",
    "tau3-pi",
    "aime24",
    "gpqa-diamond",
    "math500",
    "mrcr",
    "NUPA",
    "mmlu-pro",
    "swebench-verified",
    "mbppplus",
)


def reference_catalog() -> tuple[ReferenceSource, ...]:
    """Return source pins for the selected tasks and test splits."""
    micro = tuple(
        ReferenceSource(
            name=f"micro210-{benchmark}",
            repository="open-athena/Marin-Micro-Eval-210",
            revision="4564ed081c316bc3231ae5c3a53a46b39248c819",
            file=ReferenceFile(
                "data/validation.parquet", "cde0bc0bda40c24f7d653e282660fac9f8637fa57b809cec9a13e629a632e794"
            ),
            expected_records=10,
            policy=TextPolicy.TASK_SPEC,
            benchmark=benchmark,
        )
        for benchmark in _MICRO_BENCHMARKS
    )
    return (
        *micro,
        ReferenceSource(
            name="swe-bench",
            repository="princeton-nlp/SWE-bench",
            revision="e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6",
            file=ReferenceFile(
                "data/test-00000-of-00001.parquet", "db4f70ef735b3162c74801ddcdf8d7bae8d704193788c6d844f898c20b571cbb"
            ),
            expected_records=2294,
            policy=TextPolicy.PROBLEM,
        ),
        ReferenceSource(
            name="swe-bench-verified",
            repository="princeton-nlp/SWE-bench_Verified",
            revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
            file=ReferenceFile(
                "data/test-00000-of-00001.parquet", "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
            ),
            expected_records=500,
            policy=TextPolicy.PROBLEM,
        ),
        ReferenceSource(
            name="swe-bench-lite",
            repository="princeton-nlp/SWE-bench_Lite",
            revision="6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2",
            file=ReferenceFile(
                "data/test-00000-of-00001.parquet", "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4"
            ),
            expected_records=300,
            policy=TextPolicy.PROBLEM,
        ),
        ReferenceSource(
            name="long-loop-holdouts",
            repository="s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/swe-verified-20261112-v4/release",
            revision="sha256:8601f26d9b16467efdfb022166367a1c413adfa4118f5110a76b12258e254838",
            file=ReferenceFile("validation.parquet", "8601f26d9b16467efdfb022166367a1c413adfa4118f5110a76b12258e254838"),
            expected_records=120,
            policy=TextPolicy.TASK_SPEC,
        ),
    )


def reference_sources_sha256(sources: tuple[ReferenceSource, ...]) -> str:
    """Fingerprint source pins, selections and extraction policies together."""
    content = json.dumps([asdict(source) for source in sources], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def _field_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _field_text(item)))
    if isinstance(value, dict):
        return "\n".join(text for item in value.values() if (text := _field_text(item)))
    return ""


def _record_identifier(row: dict[str, Any], index: int, spec: dict[str, Any] | None) -> str:
    """Trace a record to its task: id columns first, then the TaskSpec payload."""
    for column in ("task_id", "instance_id", "id"):
        if row.get(column):
            return str(row[column])
    if spec is not None and spec.get("id"):
        return str(spec["id"])
    return str(index)


def reference_records(rows: Iterator[dict[str, Any]], source: ReferenceSource) -> Iterator[dict[str, str]]:
    """Extract public problem text without graders, solutions or image bytes."""
    for index, row in enumerate(rows):
        if source.benchmark is not None and row["benchmark"] != source.benchmark:
            continue
        spec = json.loads(row["task_spec"]) if source.policy == TextPolicy.TASK_SPEC else None
        if spec is not None:
            events = spec["context"]["events"]
            text = "\n\n".join(
                _field_text(event["content"])
                for event in events
                if event["type"] == "message" and event["role"] in {"system", "user"}
            )
        else:
            text = row["problem_statement"].strip()
        if not text:
            raise ValueError(f"{source.name}: empty problem text for {_record_identifier(row, index, spec)}")
        yield {"id": f"{source.name}:{_record_identifier(row, index, spec)}", "text": text}


def _reference_rows(source: ReferenceSource) -> Iterator[dict[str, Any]]:
    if source.repository.startswith("s3://"):
        fs, key = filesystem_for(prefix_join(source.repository, source.file.path))
        with fs.open(key, "rb") as stream:
            content = stream.read()
    else:
        path = hf_hub_download(source.repository, source.file.path, repo_type="dataset", revision=source.revision)
        content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != source.file.sha256:
        raise ValueError(f"{source.name}: source SHA-256 differs from its pin")
    for batch in pq.ParquetFile(io.BytesIO(content)).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def prepare_references(output_root: str, sources: tuple[ReferenceSource, ...]) -> dict[str, Any]:
    """Publish a new manifest after every selected benchmark passes validation."""
    manifest_path = prefix_join(output_root, COVERAGE_MANIFEST)
    if StoragePath(manifest_path).exists():
        raise ValueError(f"Reference release already exists: {manifest_path}; choose a new version")
    entries = []
    for source in sources:
        records = list(reference_records(_reference_rows(source), source))
        if len(records) != source.expected_records:
            raise ValueError(f"{source.name}: expected {source.expected_records} records, extracted {len(records)}")
        if any(len(text_stream(record["text"])) < WINDOW for record in records):
            raise ValueError(f"{source.name}: a benchmark record is too short to have a decontamination key")
        artifact = prefix_join(source.name, "eval.parquet")
        output_path = prefix_join(output_root, artifact)
        _write_parquet(output_path, iter(records))
        with StoragePath(output_path).open("rb") as stream:
            artifact_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append(
            {
                "name": source.name,
                "artifact": artifact,
                "expected_records": len(records),
                "sha256": artifact_sha256,
                "source": asdict(source),
            }
        )
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
    args = parser.parse_args()
    configure_logging(logging.INFO)
    prepare_references(args.output_root, reference_catalog())


if __name__ == "__main__":
    main()
