# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SHA-bound population locations and verified local content caches."""

import fcntl
import hashlib
import json
import logging
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.task_neighbors import (
    PopulationIndex,
    TaskContent,
    content_rows,
    strip_boilerplate,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import original_source_id

logger = logging.getLogger(__name__)


CONTENT_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("instruction_hash", pa.string()),
        ("text", pa.large_string()),
        ("reference_hash", pa.string()),
        ("tests_hash", pa.string()),
        ("state_hash", pa.string()),
    ]
)
HASH_COLUMNS = [name for name in CONTENT_SCHEMA.names if name != "text"]


class PreparedPopulation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    uri: str
    sha256: str
    inputs_sha256: str
    task_count: int
    text_version: Literal["complete-context-v1"] = "complete-context-v1"


class PopulationLocations(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    files: dict[str, str] = Field(default_factory=dict)
    prepared: dict[str, PreparedPopulation] = Field(default_factory=dict)


def read_locator_manifest(path: Path) -> PopulationLocations:
    return PopulationLocations.model_validate(read_registry_json(path))


def population_preflight(survey: SourceIdentitySurvey, locations: PopulationLocations, source_bucket: str) -> dict:
    """Reject missing or out-of-region population locators before scheduling reads."""
    prepared_count = 0
    located_files = 0
    for source_id, fingerprint in survey.fingerprints.items():
        if fingerprint is None:
            continue
        prepared = locations.prepared.get(source_id)
        if prepared is not None:
            if prepared.inputs_sha256 != canonical_sha256(fingerprint.inputs):
                raise ValueError(f"Prepared inputs differ: {source_id}")
            if prepared.task_count != fingerprint.task_count:
                raise ValueError(f"Prepared count differs: {source_id}")
            uris = [prepared.uri]
            prepared_count += 1
        else:
            uris = [locations.files.get(digest, key) for key, digest in fingerprint.inputs.items()]
            located_files += sum(digest in locations.files for digest in fingerprint.inputs.values())
        for uri in uris:
            if not uri.startswith(f"s3://{source_bucket}/"):
                raise ValueError(f"Missing co-located population locator for {source_id}: {uri}")
    return {"prepared_sources": prepared_count, "located_files": located_files, "source_bucket": source_bucket}


def cache_population(
    source_id: str,
    fingerprint: SourceFingerprint,
    cache_dir: Path,
    locations: PopulationLocations,
    local_paths: dict[str, str],
    *,
    source_bucket: str,
) -> Path:
    """Verify input bytes before extracting complete task content into a local cache."""
    prepared = locations.prepared.get(source_id)
    cache_key = canonical_sha256(
        {
            "source": source_id,
            "inputs": fingerprint.inputs,
            "task_count": fingerprint.task_count,
            "prepared_sha256": prepared.sha256 if prepared else None,
            "text_version": "complete-context-v1",
        }
    )
    path = cache_dir / f"{cache_key}.parquet"
    if path.exists():
        return path
    configure_coreweave_s3()
    if prepared is not None:
        if (
            prepared.inputs_sha256 != canonical_sha256(fingerprint.inputs)
            or prepared.task_count != fingerprint.task_count
        ):
            raise ValueError(f"Prepared population differs from pinned inputs: {source_id}")
        if not prepared.uri.startswith(f"s3://{source_bucket}/"):
            raise ValueError(f"Prepared population is outside the job region: {prepared.uri}")
        payload = StoragePath(prepared.uri).read_bytes()
        if hashlib.sha256(payload).hexdigest() != prepared.sha256:
            raise ValueError(f"Prepared population digest differs: {source_id}")
        pending = path.with_suffix(".pending")
        pending.write_bytes(payload)
        if pq.ParquetFile(pending).metadata.num_rows != fingerprint.task_count:
            raise ValueError(f"Prepared population count differs: {source_id}")
        pending.rename(path)
        return path
    count = 0
    with pq.ParquetWriter(path.with_suffix(".pending"), CONTENT_SCHEMA, compression="zstd") as writer:
        for key, digest in sorted(fingerprint.inputs.items()):
            uri = locations.files.get(digest, key)
            if not uri.startswith(f"s3://{source_bucket}/"):
                raise ValueError(f"Rerun requires co-located input: {uri}")
            fs, remote_path = filesystem_for(local_paths.get(uri, uri))
            with tempfile.TemporaryFile() as local:
                with fs.open(remote_path, "rb") as remote:
                    sha = hashlib.sha256()
                    while chunk := remote.read(8 * 1024 * 1024):
                        sha.update(chunk)
                        local.write(chunk)
                if sha.hexdigest() != digest:
                    raise ValueError(f"Input digest differs: {uri}")
                local.seek(0)
                parquet = pq.ParquetFile(local)
                column = "lowered_task_spec" if "lowered_task_spec" in parquet.schema_arrow.names else "task_spec"
                for batch in parquet.iter_batches(batch_size=256, columns=[column]):
                    rows = [
                        asdict(row)
                        for row in content_rows(json.loads(encoded) for encoded in batch.column(column).to_pylist())
                    ]
                    writer.write_table(pa.Table.from_pylist(rows, schema=CONTENT_SCHEMA))
                    count += len(rows)
    if count != fingerprint.task_count:
        raise ValueError(f"Population count differs for {source_id}: {count} != {fingerprint.task_count}")
    path.with_suffix(".pending").rename(path)
    logger.info("Cached %s: %d tasks", original_source_id(source_id), count)
    return path


def read_population(path: Path, inputs: dict[str, str]) -> PopulationIndex:
    return PopulationIndex([TaskContent(**row) for row in pq.read_table(path).to_pylist()], inputs)


def read_hash_population(path: Path, inputs: dict[str, str]) -> PopulationIndex:
    """Load the exact census and state gates without reading conversation text."""
    rows = [TaskContent(text="", **row) for row in pq.read_table(path, columns=HASH_COLUMNS).to_pylist()]
    return PopulationIndex(rows, inputs)


def read_normalized_population(path: Path, inputs: dict[str, str]) -> PopulationIndex:
    """Share one atomic normalization cache across local measurement workers."""
    normalized_path = path.with_suffix(".normalized-v1.parquet")
    with path.with_suffix(".normalization.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not normalized_path.exists():
            rows = [TaskContent(**row) for row in pq.read_table(path).to_pylist()]
            normalized = strip_boilerplate(rows)
            pending = normalized_path.with_suffix(".pending")
            pq.write_table(
                pa.Table.from_pylist([asdict(row) for row in normalized], schema=CONTENT_SCHEMA),
                pending,
                compression="zstd",
            )
            pending.rename(normalized_path)
            logger.info("Normalized population %s: %d tasks", path.name, len(rows))
    normalized = [TaskContent(**row) for row in pq.read_table(normalized_path).to_pylist()]
    return PopulationIndex([replace(row, text="") for row in normalized], inputs, normalized)
