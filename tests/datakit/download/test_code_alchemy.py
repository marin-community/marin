# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
from pathlib import Path

from fray.types import ResourceConfig
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marin.datakit.download.code_alchemy as code_alchemy
import marin.datakit.normalize as normalize_module

from marin.datakit.download.code_alchemy import (
    HF_DATASET_ID,
    HF_REVISION,
    HYDRATED_OUTPUT_PATH,
    HYDRATED_REGISTRATION_MANIFEST,
    HYDRATION_GIT_COMMIT,
    HYDRATION_PROVENANCE_MD5_BASE64,
    HYDRATION_SUMMARY_MD5_BASE64,
    _validate_hydrated_code_alchemy,
    code_alchemy_normalize_steps,
)
from marin.datakit.normalize import DedupMode

_EXPECTED_KEYS = {
    "code-alchemy/code-enhance",
    "code-alchemy/code-qa",
    "code-alchemy/code-dev",
    "code-alchemy/code-dialogue",
    "code-alchemy/code-trace",
}
_DIRECT_KEYS = {
    "code-alchemy/code-enhance",
    "code-alchemy/code-qa",
    "code-alchemy/code-trace",
}
_HYDRATED_KEYS = {
    "code-alchemy/code-dev",
    "code-alchemy/code-dialogue",
}


def _valid_summary(**overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "task_count": 512,
        "input_rows": 93_095_401,
        "output_rows": 93_095_401,
        "missing_source_rows": 0,
        "null_blob_id_rows": 0,
        "prefix_mismatch_rows": 0,
        "unresolved_placeholder_rows": 0,
        "unresolved_blob_ids": 0,
        "rows_with_available_source": 93_095_401,
        "rows_with_placeholder": 93_095_401,
    }
    summary.update(overrides)
    return summary


def _valid_registration(**overrides: object) -> dict[str, object]:
    registration: dict[str, object] = {
        "schema_version": 1,
        "source_dataset": HF_DATASET_ID,
        "source_revision": HF_REVISION,
        "hydration_git_commit": HYDRATION_GIT_COMMIT,
        "hydration_provenance_md5_base64": code_alchemy.HYDRATION_PROVENANCE_MD5_BASE64,
        "hydrate_summary_md5_base64": code_alchemy.HYDRATION_SUMMARY_MD5_BASE64,
        "subsets": {
            "code-dev": {"parquet_files": 256, "rows": 62_187_373},
            "code-dialogue": {"parquet_files": 256, "rows": 30_908_028},
        },
    }
    registration.update(overrides)
    return registration


def _write_metadata(
    root: Path,
    summary: dict[str, object] | None = None,
    *,
    monkeypatch: pytest.MonkeyPatch,
    provenance_overrides: dict[str, object] | None = None,
    registration_overrides: dict[str, object] | None = None,
) -> None:
    hydration_summary = summary if summary is not None else _valid_summary()
    provenance: dict[str, object] = {
        "completion_status": "complete",
        "diagnostic_only": False,
        "subsets": ["code-dev", "code-dialogue"],
        "metrics": hydration_summary,
    }
    provenance.update(provenance_overrides or {})
    provenance_payload = json.dumps(provenance).encode()
    summary_payload = json.dumps(hydration_summary).encode()
    provenance_digest = base64.b64encode(hashlib.md5(provenance_payload, usedforsecurity=False).digest()).decode("ascii")
    summary_digest = base64.b64encode(hashlib.md5(summary_payload, usedforsecurity=False).digest()).decode("ascii")
    monkeypatch.setattr(code_alchemy, "HYDRATION_PROVENANCE_MD5_BASE64", provenance_digest)
    monkeypatch.setattr(code_alchemy, "HYDRATION_SUMMARY_MD5_BASE64", summary_digest)

    (root / ".metrics").mkdir(parents=True)
    (root / HYDRATED_REGISTRATION_MANIFEST).write_text(json.dumps(_valid_registration(**(registration_overrides or {}))))
    (root / ".provenance.json").write_bytes(provenance_payload)
    (root / ".metrics" / "hydrate-summary.json").write_bytes(summary_payload)


def _write_partitions(root: Path, *, omit: tuple[str, str] | None = None) -> None:
    for subset in ("code-dev", "code-dialogue"):
        for prefix in range(256):
            blob_prefix = f"{prefix:02x}"
            if omit == (subset, blob_prefix):
                continue
            partition = root / f"subset={subset}" / f"blob_prefix={blob_prefix}"
            partition.mkdir(parents=True)
            (partition / "part-00000.parquet").touch()


def _mock_valid_parquet_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_rows = {
        "code-dev": 62_187_373,
        "code-dialogue": 30_908_028,
    }

    def rows_for_path(path: str) -> int:
        subset = next(subset for subset in expected_rows if f"subset={subset}/" in path)
        prefix = int(path.split("blob_prefix=", 1)[1].split("/", 1)[0], 16)
        quotient, remainder = divmod(expected_rows[subset], 256)
        return quotient + int(prefix < remainder)

    monkeypatch.setattr(code_alchemy, "_read_parquet_metadata", rows_for_path)


def test_factory_has_exact_keys_and_shared_source_dependencies():
    chains = code_alchemy_normalize_steps()

    assert set(chains) == _EXPECTED_KEYS
    assert all(len(chain) == 2 for chain in chains.values())

    direct_sources = [chains[key][0] for key in _DIRECT_KEYS]
    hydrated_sources = [chains[key][0] for key in _HYDRATED_KEYS]
    assert all(source is direct_sources[0] for source in direct_sources)
    assert all(source is hydrated_sources[0] for source in hydrated_sources)
    assert direct_sources[0] is not hydrated_sources[0]
    assert all(chain[-1].deps == [chain[0]] for chain in chains.values())


def test_direct_download_is_pinned_to_only_public_subset_parquet_globs():
    chains = code_alchemy_normalize_steps()
    download = chains["code-alchemy/code-enhance"][0]

    assert download.name == "raw/code-alchemy-d367da9"
    assert download.hash_attrs == {
        "hf_dataset_id": HF_DATASET_ID,
        "revision": HF_REVISION,
        "hf_urls_glob": [
            "code-enhance/*.parquet",
            "code-qa/*.parquet",
            "code-trace/*.parquet",
        ],
        "append_sha_to_path": False,
    }


def test_hydrated_source_uses_durable_override_and_complete_identity():
    chains = code_alchemy_normalize_steps()
    hydrated = chains["code-alchemy/code-dev"][0]

    assert hydrated.name == HYDRATED_OUTPUT_PATH
    assert hydrated.override_output_path == HYDRATED_OUTPUT_PATH
    assert hydrated.hash_attrs == {
        "version": "2026-08-25.2",
        "source_dataset": HF_DATASET_ID,
        "source_revision": HF_REVISION,
        "hydration_git_commit": HYDRATION_GIT_COMMIT,
        "registration_manifest": HYDRATED_REGISTRATION_MANIFEST,
        "hydration_provenance_md5_base64": HYDRATION_PROVENANCE_MD5_BASE64,
        "hydrate_summary_md5_base64": HYDRATION_SUMMARY_MD5_BASE64,
        "format": "parquet",
        "subsets": ["code-dev", "code-dialogue"],
        "partition_key": "blob_prefix",
        "partitions_per_subset": 256,
        "tasks": 512,
        "rows_by_subset": {
            "code-dev": 62_187_373,
            "code-dialogue": 30_908_028,
        },
        "rows": 93_095_401,
        "completion_status": "complete",
    }


def test_normalize_terminals_have_subset_specific_identities():
    chains = code_alchemy_normalize_steps()

    for registry_name, (source, normalized) in chains.items():
        subset = registry_name.removeprefix("code-alchemy/")
        hydrated = registry_name in _HYDRATED_KEYS
        assert normalized.name == f"normalized/code-alchemy/{subset}"
        assert normalized.deps == [source]
        assert normalized.hash_attrs["relative_input_path"] == (f"subset={subset}" if hydrated else subset)
        assert normalized.hash_attrs["text_field"] == ("text_with_placeholders" if hydrated else "text")
        assert normalized.hash_attrs["id_field"] == "blob_id"
        assert normalized.hash_attrs["file_extensions"] == (".parquet",)
        assert normalized.hash_attrs["dedup_mode"] is DedupMode.NONE
        assert "bare" not in normalized.hash_attrs
        assert "drop_fields" not in normalized.hash_attrs


def test_normalize_execution_policy_is_forwarded_without_rekeying(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    def fake_normalize_to_parquet(**kwargs: object) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(normalize_module, "normalize_to_parquet", fake_normalize_to_parquet)
    normalized = code_alchemy_normalize_steps()["code-alchemy/code-dev"][-1]
    assert normalized.fn is not None
    normalized.fn("normalized-output")

    assert calls["max_workers"] == 128
    assert calls["heartbeat_timeout"] == 30 * 60.0
    assert calls["worker_resources"] == ResourceConfig(cpu=4, ram="64g", disk="16g")
    assert "max_workers" not in normalized.hash_attrs
    assert "heartbeat_timeout" not in normalized.hash_attrs
    assert "worker_resources" not in normalized.hash_attrs


def test_hydrated_validator_accepts_complete_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_metadata(tmp_path, monkeypatch=monkeypatch)
    _write_partitions(tmp_path)
    _mock_valid_parquet_rows(monkeypatch)

    _validate_hydrated_code_alchemy(str(tmp_path))


@pytest.mark.parametrize(
    "relative_path",
    [
        HYDRATED_REGISTRATION_MANIFEST,
        ".provenance.json",
        ".metrics/hydrate-summary.json",
    ],
)
def test_hydrated_validator_rejects_missing_metadata(
    tmp_path: Path, relative_path: str, monkeypatch: pytest.MonkeyPatch
):
    _write_metadata(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / relative_path).unlink()

    with pytest.raises(FileNotFoundError):
        _validate_hydrated_code_alchemy(str(tmp_path))


@pytest.mark.parametrize(
    ("override", "metric"),
    [
        ({"task_count": 511}, "task_count"),
        ({"input_rows": 93_095_400}, "input_rows"),
        ({"output_rows": 93_095_400}, "output_rows"),
        ({"missing_source_rows": 1}, "missing_source_rows"),
        ({"null_blob_id_rows": 1}, "null_blob_id_rows"),
        ({"prefix_mismatch_rows": 1}, "prefix_mismatch_rows"),
        ({"unresolved_placeholder_rows": 1}, "unresolved_placeholder_rows"),
        ({"unresolved_blob_ids": 1}, "unresolved_blob_ids"),
    ],
)
def test_hydrated_validator_rejects_incomplete_or_unresolved_metrics(
    tmp_path: Path,
    override: dict[str, object],
    metric: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_metadata(tmp_path, _valid_summary(**override), monkeypatch=monkeypatch)

    with pytest.raises(ValueError, match=metric):
        _validate_hydrated_code_alchemy(str(tmp_path))


@pytest.mark.parametrize("metric", ["rows_with_available_source", "rows_with_placeholder"])
def test_hydrated_validator_rejects_missing_hydration_metrics(
    tmp_path: Path, metric: str, monkeypatch: pytest.MonkeyPatch
):
    summary = _valid_summary()
    summary.pop(metric)
    _write_metadata(tmp_path, summary, monkeypatch=monkeypatch)

    with pytest.raises(ValueError, match=metric):
        _validate_hydrated_code_alchemy(str(tmp_path))


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"completion_status": "partial"}, "completion_status"),
        ({"diagnostic_only": True}, "diagnostic_only"),
        ({"subsets": ["code-dev"]}, "cover"),
        ({"metrics": {}}, "metrics do not match"),
    ],
)
def test_hydrated_validator_rejects_incomplete_provenance(
    tmp_path: Path,
    override: dict[str, object],
    field: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_metadata(tmp_path, monkeypatch=monkeypatch, provenance_overrides=override)

    with pytest.raises(ValueError, match=field):
        _validate_hydrated_code_alchemy(str(tmp_path))


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"source_dataset": "other/dataset"}, "source_dataset"),
        ({"source_revision": "other-revision"}, "source_revision"),
        ({"hydration_git_commit": "other-commit"}, "hydration_git_commit"),
    ],
)
def test_hydrated_validator_binds_registration_identity(
    tmp_path: Path,
    override: dict[str, object],
    field: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_metadata(tmp_path, monkeypatch=monkeypatch, registration_overrides=override)

    with pytest.raises(ValueError, match=field):
        _validate_hydrated_code_alchemy(str(tmp_path))


def test_hydrated_validator_rejects_wrong_parquet_row_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_metadata(tmp_path, monkeypatch=monkeypatch)
    _write_partitions(tmp_path)
    monkeypatch.setattr(code_alchemy, "_read_parquet_metadata", lambda _path: 0)

    with pytest.raises(ValueError, match="row counts"):
        _validate_hydrated_code_alchemy(str(tmp_path))


def test_read_parquet_metadata_validates_schema_and_rows(tmp_path: Path):
    parquet_path = tmp_path / "part.parquet"
    pq.write_table(
        pa.table(
            {
                "blob_id": ["a", "b"],
                "text_with_placeholders": ["first", "second"],
            }
        ),
        parquet_path,
    )

    assert code_alchemy._read_parquet_metadata(str(parquet_path)) == 2


def test_read_parquet_metadata_rejects_missing_columns(tmp_path: Path):
    parquet_path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"blob_id": ["a"]}), parquet_path)

    with pytest.raises(ValueError, match="text_with_placeholders"):
        code_alchemy._read_parquet_metadata(str(parquet_path))


def test_hydrated_validator_rejects_wrong_file_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_metadata(tmp_path, monkeypatch=monkeypatch)
    _write_partitions(tmp_path, omit=("code-dev", "ff"))

    with pytest.raises(FileNotFoundError, match="512.*found 511"):
        _validate_hydrated_code_alchemy(str(tmp_path))


def test_hydrated_validator_rejects_missing_partition_even_with_512_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_metadata(tmp_path, monkeypatch=monkeypatch)
    _write_partitions(tmp_path, omit=("code-dev", "ff"))
    duplicate_partition = tmp_path / "subset=code-dev" / "blob_prefix=fe"
    (duplicate_partition / "part-00001.parquet").touch()

    with pytest.raises(FileNotFoundError, match="code-dev"):
        _validate_hydrated_code_alchemy(str(tmp_path))


def test_hydrated_validator_rejects_nested_partition_substitute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_metadata(tmp_path, monkeypatch=monkeypatch)
    _write_partitions(tmp_path, omit=("code-dev", "ff"))
    nested_partition = tmp_path / "archive" / "subset=code-dev" / "blob_prefix=ff"
    nested_partition.mkdir(parents=True)
    (nested_partition / "part-00000.parquet").touch()

    with pytest.raises(ValueError, match="Unexpected"):
        _validate_hydrated_code_alchemy(str(tmp_path))


def test_load_json_object_rejects_digest_mismatch(tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")

    with pytest.raises(ValueError, match="MD5"):
        code_alchemy._load_json_object(
            str(metadata),
            label="test",
            expected_md5_base64="AAAAAAAAAAAAAAAAAAAAAA==",
        )
