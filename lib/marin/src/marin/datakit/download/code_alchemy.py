# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize the finalized Code Alchemy training subsets.

The three source-complete subsets download from the pinned public release.
``code-dev`` and ``code-dialogue`` consume a validated, pre-staged hydration
because the public release replaces its seed source bytes with placeholders.
The one-off source reconstruction is provenance for that raw boundary, not part
of the repeatable registry chain.
"""

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from fray.types import ResourceConfig

import pyarrow as pa
import pyarrow.parquet as pq

from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import DedupMode, normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-alchemy/code-alchemy"
HF_REVISION = "d367da91def5024929d0fa8d46d47d4ef616b467"
HYDRATED_OUTPUT_PATH = "raw/code-alchemy-hydrated-d367da9"
HYDRATED_REGISTRATION_MANIFEST = ".registration.json"
HYDRATION_GIT_COMMIT = "e52fd794d3"
HYDRATION_PROVENANCE_MD5_BASE64 = "0bqbZvfo9gv5OtiaI8cSjg=="
HYDRATION_SUMMARY_MD5_BASE64 = "jsAtjeKbgHAJvt7jXU0BfQ=="

_DIRECT_SUBSETS = ("code-enhance", "code-qa", "code-trace")
_HYDRATED_SUBSETS = ("code-dev", "code-dialogue")
_PARTITIONS_PER_HYDRATED_SUBSET = 256
_EXPECTED_HYDRATION_TASKS = len(_HYDRATED_SUBSETS) * _PARTITIONS_PER_HYDRATED_SUBSET
_EXPECTED_HYDRATED_ROWS_BY_SUBSET = {
    "code-dev": 62_187_373,
    "code-dialogue": 30_908_028,
}
_EXPECTED_HYDRATED_ROWS = sum(_EXPECTED_HYDRATED_ROWS_BY_SUBSET.values())
_PARQUET_METADATA_WORKERS = 32
_NORMALIZE_MAX_WORKERS = 128
_NORMALIZE_HEARTBEAT_TIMEOUT = 30 * 60.0
_HYDRATED_NORMALIZE_RESOURCES = ResourceConfig(cpu=4, ram="64g", disk="16g")
_EXPECTED_PREFIXES = frozenset(f"{prefix:02x}" for prefix in range(_PARTITIONS_PER_HYDRATED_SUBSET))
_HYDRATED_PARQUET_PATH = re.compile(r"subset=(?P<subset>[^/]+)/blob_prefix=(?P<prefix>[^/]+)/[^/]+\.parquet")
_ZERO_HYDRATION_METRICS = (
    "missing_source_rows",
    "null_blob_id_rows",
    "prefix_mismatch_rows",
    "unresolved_placeholder_rows",
    "unresolved_blob_ids",
)


def _load_json_object(
    path: str,
    *,
    label: str,
    expected_md5_base64: str | None = None,
) -> Mapping[str, Any]:
    storage_path = StoragePath(path)
    if not storage_path.exists():
        raise FileNotFoundError(f"Missing Code Alchemy hydration {label}: {path}")

    payload = storage_path.read_bytes()
    if expected_md5_base64 is not None:
        digest = hashlib.md5(payload, usedforsecurity=False).digest()
        actual_md5_base64 = base64.b64encode(digest).decode("ascii")
        if actual_md5_base64 != expected_md5_base64:
            raise ValueError(
                f"Code Alchemy hydration {label} MD5 must be {expected_md5_base64}, found {actual_md5_base64}: {path}"
            )
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"Code Alchemy hydration {label} must contain a JSON object: {path}")
    return value


def _require_int_metric(summary: Mapping[str, Any], key: str, expected: int) -> None:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Code Alchemy hydration metric {key!r} must be an integer, found {value!r}")
    if value != expected:
        raise ValueError(f"Code Alchemy hydration metric {key!r} must be {expected:,}, found {value:,}")


def _validate_registration_manifest(registration: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "source_dataset": HF_DATASET_ID,
        "source_revision": HF_REVISION,
        "hydration_git_commit": HYDRATION_GIT_COMMIT,
        "hydration_provenance_md5_base64": HYDRATION_PROVENANCE_MD5_BASE64,
        "hydrate_summary_md5_base64": HYDRATION_SUMMARY_MD5_BASE64,
        "subsets": {
            subset: {
                "parquet_files": _PARTITIONS_PER_HYDRATED_SUBSET,
                "rows": rows,
            }
            for subset, rows in _EXPECTED_HYDRATED_ROWS_BY_SUBSET.items()
        },
    }
    for key, expected_value in expected.items():
        if registration.get(key) != expected_value:
            raise ValueError(
                f"Code Alchemy hydration registration {key!r} must be {expected_value!r}, "
                f"found {registration.get(key)!r}"
            )


def _read_parquet_metadata(path: str) -> int:
    with StoragePath(path).open("rb") as source:
        parquet = pq.ParquetFile(source)
        schema = parquet.schema_arrow
        missing = {"blob_id", "text_with_placeholders"} - set(schema.names)
        if missing:
            raise ValueError(f"Hydrated Code Alchemy Parquet {path} is missing columns: {sorted(missing)}")
        for name in ("blob_id", "text_with_placeholders"):
            data_type = schema.field(name).type
            if not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
                raise ValueError(f"Hydrated Code Alchemy Parquet {path} column {name!r} has type {data_type}")
        return parquet.metadata.num_rows


def _validate_hydrated_code_alchemy(output_path: str) -> None:
    """Reject incomplete or unresolved copies of the pre-staged hydrated artifact."""
    registration = _load_json_object(
        prefix_join(output_path, HYDRATED_REGISTRATION_MANIFEST),
        label="registration manifest",
    )
    provenance = _load_json_object(
        prefix_join(output_path, ".provenance.json"),
        label="provenance",
        expected_md5_base64=HYDRATION_PROVENANCE_MD5_BASE64,
    )
    summary = _load_json_object(
        prefix_join(output_path, ".metrics/hydrate-summary.json"),
        label="summary",
        expected_md5_base64=HYDRATION_SUMMARY_MD5_BASE64,
    )
    _validate_registration_manifest(registration)

    if provenance.get("completion_status") != "complete":
        raise ValueError(
            "Code Alchemy hydration provenance 'completion_status' must be 'complete', "
            f"found {provenance.get('completion_status')!r}"
        )
    if provenance.get("diagnostic_only") is not False:
        raise ValueError(
            "Code Alchemy hydration provenance 'diagnostic_only' must be false, "
            f"found {provenance.get('diagnostic_only')!r}"
        )
    if provenance.get("subsets") != list(_HYDRATED_SUBSETS):
        raise ValueError(
            f"Code Alchemy hydration provenance must cover {list(_HYDRATED_SUBSETS)!r}, "
            f"found {provenance.get('subsets')!r}"
        )
    if provenance.get("metrics") != summary:
        raise ValueError("Code Alchemy hydration provenance metrics do not match hydrate-summary.json")

    _require_int_metric(summary, "task_count", _EXPECTED_HYDRATION_TASKS)
    _require_int_metric(summary, "input_rows", _EXPECTED_HYDRATED_ROWS)
    _require_int_metric(summary, "output_rows", _EXPECTED_HYDRATED_ROWS)
    for key in _ZERO_HYDRATION_METRICS:
        _require_int_metric(summary, key, 0)
    _require_int_metric(summary, "rows_with_available_source", _EXPECTED_HYDRATED_ROWS)
    _require_int_metric(summary, "rows_with_placeholder", _EXPECTED_HYDRATED_ROWS)

    parquet_glob = prefix_join(output_path, "**/*.parquet")
    partitions: dict[str, list[str]] = {subset: [] for subset in _HYDRATED_SUBSETS}
    parquet_subsets: dict[str, str] = {}
    unexpected: list[str] = []
    parquet_files = [str(path) for path in StoragePath(parquet_glob).glob()]
    root_prefix = f"{output_path.rstrip('/')}/"
    for parquet_file in parquet_files:
        relative_path = parquet_file.removeprefix(root_prefix)
        match = _HYDRATED_PARQUET_PATH.fullmatch(relative_path)
        if not parquet_file.startswith(root_prefix) or match is None or match.group("subset") not in partitions:
            unexpected.append(parquet_file)
            continue
        subset = match.group("subset")
        partitions[subset].append(match.group("prefix"))
        parquet_subsets[parquet_file] = subset

    if len(parquet_files) != _EXPECTED_HYDRATION_TASKS:
        raise FileNotFoundError(
            f"Expected {_EXPECTED_HYDRATION_TASKS} hydrated Code Alchemy Parquet files under "
            f"{output_path}, found {len(parquet_files)}"
        )
    if unexpected:
        raise ValueError(f"Unexpected hydrated Code Alchemy Parquet path: {unexpected[0]}")

    for subset, prefixes in partitions.items():
        prefix_set = set(prefixes)
        if len(prefixes) != _PARTITIONS_PER_HYDRATED_SUBSET or prefix_set != _EXPECTED_PREFIXES:
            missing = sorted(_EXPECTED_PREFIXES - prefix_set)
            duplicates = len(prefixes) - len(prefix_set)
            raise FileNotFoundError(
                f"Expected one Parquet file in each of {_PARTITIONS_PER_HYDRATED_SUBSET} blob-prefix "
                f"partitions for {subset}; found {len(prefixes)} files across {len(prefix_set)} partitions "
                f"(missing={missing[:5]}, duplicate_files={duplicates})"
            )
    row_counts = dict.fromkeys(_HYDRATED_SUBSETS, 0)
    with ThreadPoolExecutor(max_workers=_PARQUET_METADATA_WORKERS) as pool:
        for parquet_file, rows in zip(parquet_files, pool.map(_read_parquet_metadata, parquet_files), strict=True):
            row_counts[parquet_subsets[parquet_file]] += rows
    if row_counts != _EXPECTED_HYDRATED_ROWS_BY_SUBSET:
        raise ValueError(
            f"Hydrated Code Alchemy Parquet row counts are {row_counts}, expected {_EXPECTED_HYDRATED_ROWS_BY_SUBSET}"
        )


def _direct_download_step() -> StepSpec:
    return download_hf_step(
        "raw/code-alchemy-d367da9",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subset}/*.parquet" for subset in _DIRECT_SUBSETS],
    )


def _hydrated_source_step() -> StepSpec:
    return StepSpec(
        name=HYDRATED_OUTPUT_PATH,
        override_output_path=HYDRATED_OUTPUT_PATH,
        fn=_validate_hydrated_code_alchemy,
        hash_attrs={
            "version": "2026-08-25.2",
            "source_dataset": HF_DATASET_ID,
            "source_revision": HF_REVISION,
            "hydration_git_commit": HYDRATION_GIT_COMMIT,
            "registration_manifest": HYDRATED_REGISTRATION_MANIFEST,
            "hydration_provenance_md5_base64": HYDRATION_PROVENANCE_MD5_BASE64,
            "hydrate_summary_md5_base64": HYDRATION_SUMMARY_MD5_BASE64,
            "format": "parquet",
            "subsets": list(_HYDRATED_SUBSETS),
            "partition_key": "blob_prefix",
            "partitions_per_subset": _PARTITIONS_PER_HYDRATED_SUBSET,
            "tasks": _EXPECTED_HYDRATION_TASKS,
            "rows_by_subset": _EXPECTED_HYDRATED_ROWS_BY_SUBSET,
            "rows": _EXPECTED_HYDRATED_ROWS,
            "completion_status": "complete",
        },
    )


def _normalize_subset(source: StepSpec, subset: str, *, hydrated: bool) -> StepSpec:
    return normalize_step(
        name=f"normalized/code-alchemy/{subset}",
        download=source,
        relative_input_path=f"subset={subset}" if hydrated else subset,
        text_field="text_with_placeholders" if hydrated else "text",
        id_field="blob_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.NONE,
        max_workers=_NORMALIZE_MAX_WORKERS,
        heartbeat_timeout=_NORMALIZE_HEARTBEAT_TIMEOUT,
        worker_resources=_HYDRATED_NORMALIZE_RESOURCES if hydrated else None,
    )


def code_alchemy_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return one shared source dependency and normalize terminal per registry subset."""
    direct_download = _direct_download_step()
    hydrated_source = _hydrated_source_step()

    chains: dict[str, tuple[StepSpec, ...]] = {}
    for subset in _DIRECT_SUBSETS:
        chains[f"code-alchemy/{subset}"] = (
            direct_download,
            _normalize_subset(direct_download, subset, hydrated=False),
        )
    for subset in _HYDRATED_SUBSETS:
        chains[f"code-alchemy/{subset}"] = (
            hydrated_source,
            _normalize_subset(hydrated_source, subset, hydrated=True),
        )
    return chains
