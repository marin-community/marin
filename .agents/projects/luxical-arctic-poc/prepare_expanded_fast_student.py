# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack one expanded Arctic rung for bounded fast-student training."""

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from extend_arctic_teacher import expanded_root, teacher_root
from fast_student import packed_document_ids
from ladder_config import MANIFEST_ROOT, read_json, write_json
from rigging.filesystem import atomic_rename

BASE_PREPARED_MANIFEST_URL = f"{MANIFEST_ROOT}/fast-student/prepared-3m/manifest.json"
EXPANDED_MANIFEST_METADATA_KEY = b"luxical_expanded_manifest_sha256"
TEACHER_AUDIT_METADATA_KEY = b"luxical_expanded_teacher_audit_sha256"
TOKENIZER_MAP_METADATA_KEY = b"luxical_tokenizer_map_sha256"
TOKENIZE_BATCH_SIZE = 4_096
PREPARED_ROW_GROUP_ROWS = 8_192
RESULT_FILE = Path("/tmp/luxical-expanded-fast-student-prepare")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def load_numpy(url: str) -> np.ndarray:
    """Load one NumPy array."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path, "rb") as file:
        return np.load(file)


def selected_training_table(url: str, rung: str, columns: list[str]) -> pa.Table:
    """Read the selected training rows for one expanded rung."""
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(path, filesystem=filesystem, columns=[*columns, "split", f"in_{rung}"])
    selected = pc.and_(pc.equal(table["split"], "train"), table[f"in_{rung}"])
    return table.filter(selected).select(columns)


def aligned_training_tables(source: pa.Table, teacher: pa.Table) -> None:
    """Fail unless the source and teacher training rows have exact alignment."""
    if len(source) != len(teacher):
        raise ValueError(f"Source and teacher rows differ: {len(source)}, {len(teacher)}")
    for column in ("raw_sha256", "train_rank"):
        if not pc.all(pc.equal(source[column], teacher[column])).as_py():
            raise ValueError(f"Source and teacher rows differ in {column}")


def expected_metadata(manifest_sha256: str, audit_sha256: str, tokenizer_map_sha256: str) -> dict[bytes, bytes]:
    """Return metadata that binds one prepared source to all inputs."""
    return {
        EXPANDED_MANIFEST_METADATA_KEY: manifest_sha256.encode(),
        TEACHER_AUDIT_METADATA_KEY: audit_sha256.encode(),
        TOKENIZER_MAP_METADATA_KEY: tokenizer_map_sha256.encode(),
    }


def reusable_output(
    url: str,
    rows: int,
    metadata: dict[bytes, bytes],
    saved_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Return one complete prepared-source result when it can be reused."""
    filesystem, path = fsspec.core.url_to_fs(url)
    if not filesystem.exists(path):
        return None
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        actual_rows = parquet_file.metadata.num_rows
        actual_metadata = parquet_file.schema_arrow.metadata or {}
    if actual_rows != rows:
        raise ValueError(f"Existing prepared output has {actual_rows} rows; expected {rows}: {url}")
    if any(actual_metadata.get(key) != value for key, value in metadata.items()):
        raise ValueError(f"Existing prepared output has different input metadata: {url}")
    if saved_result.get("output_url") != url or saved_result.get("rows") != rows:
        return None
    digest = saved_result.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    return {"output_url": url, "rows": rows, "sha256": digest, "reused": True}


def packed_ids(texts: list[str], raw_to_compact: np.ndarray) -> np.ndarray:
    """Pack one source without retaining intermediate tokenizer windows."""
    chunks = [
        packed_document_ids(texts[start : start + TOKENIZE_BATCH_SIZE], raw_to_compact)
        for start in range(0, len(texts), TOKENIZE_BATCH_SIZE)
    ]
    return np.concatenate(chunks)


def prepare_source(
    rung: str,
    source: str,
    source_url: str,
    teacher_url: str,
    expected_rows: int,
    raw_to_compact: np.ndarray,
    metadata: dict[bytes, bytes],
) -> dict[str, Any]:
    """Pack and write one aligned expanded training source."""
    output_url = f"{expanded_root(rung)}/prepared/sources/{Path(source_url).name}"
    source_report_url = f"{expanded_root(rung)}/prepared/source-reports/{Path(source_url).stem}.json"
    report_filesystem, report_path = fsspec.core.url_to_fs(source_report_url)
    if report_filesystem.exists(report_path):
        reusable = reusable_output(output_url, expected_rows, metadata, read_json(source_report_url))
        if reusable is not None:
            logger.info("Reusing complete prepared source %s", source)
            return reusable
    source_table = selected_training_table(source_url, rung, ["raw_sha256", "train_rank", "text"])
    teacher_table = selected_training_table(teacher_url, rung, ["raw_sha256", "train_rank", "embedding"])
    aligned_training_tables(source_table, teacher_table)
    if len(source_table) != expected_rows:
        raise ValueError(f"Source {source} has {len(source_table)} rows; expected {expected_rows}")
    ids_values = packed_ids(source_table["text"].to_pylist(), raw_to_compact)
    ids = pa.FixedSizeListArray.from_arrays(pa.array(ids_values.reshape(-1)), ids_values.shape[1])
    output = pa.table(
        {
            "raw_sha256": source_table["raw_sha256"],
            "train_rank": source_table["train_rank"],
            "ids": ids,
            "embedding": teacher_table["embedding"],
        }
    ).replace_schema_metadata(metadata)
    filesystem, path = fsspec.core.url_to_fs(output_url)
    with tempfile.TemporaryDirectory(prefix="luxical-expanded-prepare-") as temporary_directory:
        local_path = Path(temporary_directory) / "source.parquet"
        pq.write_table(output, local_path, compression="zstd", row_group_size=PREPARED_ROW_GROUP_ROWS)
        with local_path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        with atomic_rename(path, fs=filesystem) as temporary_path:
            filesystem.put(str(local_path), temporary_path)
    result = {"output_url": output_url, "rows": expected_rows, "sha256": digest, "reused": False}
    write_json(source_report_url, result)
    return result


def prepare_rung(rung: str) -> dict[str, Any]:
    """Pack all source rows in one audited expanded teacher rung."""
    expanded_manifest_url = f"{expanded_root(rung)}/manifest.json"
    teacher_audit_url = f"{teacher_root(rung)}/audit.json"
    manifest = read_json(expanded_manifest_url)
    audit_filesystem, audit_path = fsspec.core.url_to_fs(teacher_audit_url)
    with audit_filesystem.open(audit_path, "r") as file:
        audit_text = file.read()
    audit_sha256 = hashlib.sha256(audit_text.encode()).hexdigest()
    audit = json.loads(audit_text)
    base_prepared = read_json(BASE_PREPARED_MANIFEST_URL)
    if not audit["all_sources_passed"] or audit["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("The expanded teacher audit does not pass for this manifest")
    if base_prepared["manifest_sha256"] != manifest["base_manifest_sha256"]:
        raise ValueError("The fixed tokenizer map has a different base manifest")
    raw_to_compact = load_numpy(base_prepared["raw_to_compact_url"])
    metadata = expected_metadata(manifest["sha256"], audit_sha256, base_prepared["raw_to_compact_sha256"])
    sources = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Preparing source %d/%d: %s", index, len(manifest["sources"]), source)
        teacher_url = audit["sources"][source]["output_url"]
        sources[source] = prepare_source(
            rung,
            source,
            result["output_url"],
            teacher_url,
            int(result["counts"][f"train_{rung}"]),
            raw_to_compact,
            metadata,
        )
    row_count = sum(result["rows"] for result in sources.values())
    if row_count != int(manifest["training_targets"][rung]):
        raise ValueError(f"Prepared {row_count} rows; expected {manifest['training_targets'][rung]}")
    return {
        "rung": rung,
        "manifest_url": expanded_manifest_url,
        "manifest_sha256": manifest["sha256"],
        "teacher_audit_url": teacher_audit_url,
        "teacher_audit_sha256": audit_sha256,
        "raw_to_compact_url": base_prepared["raw_to_compact_url"],
        "raw_to_compact_sha256": base_prepared["raw_to_compact_sha256"],
        "actual_compact_vocab_size": base_prepared["actual_compact_vocab_size"],
        "rows": row_count,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=("10m", "30m"), required=True)
    arguments = parser.parse_args()
    report = prepare_rung(arguments.rung)
    report_url = f"{expanded_root(arguments.rung)}/prepared/manifest.json"
    write_json(report_url, report)
    summary = {
        "rung": arguments.rung,
        "rows": report["rows"],
        "sources": len(report["sources"]),
        "report_url": report_url,
    }
    RESULT_FILE.with_name(f"{RESULT_FILE.name}-{arguments.rung}").write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_EXPANDED_PREPARE=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
