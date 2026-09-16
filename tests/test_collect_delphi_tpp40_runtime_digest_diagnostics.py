# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from typing import cast

import pytest

from experiments.domain_phase_mix.collect_delphi_tpp40_runtime_digest_diagnostics import (
    build_acceptance_reports,
    build_diagnostic_reports,
)
from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import (
    ALGORITHM,
    artifact_contract_sha256,
    digest_payload_sha256,
)


def _report(cache_path: str, *, block_sha256: str = "a" * 64, selected_rows: int = 2) -> dict[str, object]:
    report: dict[str, object] = {
        "status": "complete",
        "algorithm": ALGORITHM,
        "block_rows": 4_096,
        "selected_rows": selected_rows,
        "source_rows": selected_rows,
        "selected_tokens": 3,
        "source_tokens": 3,
        "dtype": "int64",
        "field_names": ["input_ids"],
        "blocks": [
            {
                "output_row_start": 0,
                "output_row_stop": selected_rows,
                "token_count": 3,
                "sha256": block_sha256,
            }
        ],
        "excluded_row_ranges": [],
        "binding": {
            "algorithm": ALGORITHM,
            "cache_path": cache_path,
            "block_rows": 4_096,
            "expected_rows": selected_rows,
            "expected_tokens": 3,
            "ledger_sha256": "ledger",
            "preprocessor_metadata_sha256": "metadata",
            "runtime_object_manifest": {
                "sha256": "objects",
                "objects": 4,
                "bytes": 100,
                "field_names": ["input_ids"],
            },
            "excluded_shards": [],
        },
    }
    report["logical_payload_sha256"] = digest_payload_sha256(report)
    report["artifact_contract_sha256"] = artifact_contract_sha256(report)
    return report


def _manifest() -> dict[str, object]:
    return {
        "algorithm": ALGORITHM,
        "mode": "diagnostic_only",
        "jobs": [
            {
                "component": "component",
                "region_key": "east5",
                "cache_path": "gs://east/cache",
                "output": "gs://east/digest.json",
                "expected_rows": 2,
                "expected_tokens": 3,
                "excluded_shards": [],
            },
            {
                "component": "component",
                "region_key": "europe",
                "cache_path": "gs://europe/cache",
                "output": "gs://europe/digest.json",
                "expected_rows": 2,
                "expected_tokens": 3,
                "excluded_shards": [],
            },
        ],
    }


def _acceptance_manifest() -> dict[str, object]:
    return {**_manifest(), "mode": "acceptance"}


def test_build_diagnostic_reports_preserves_payload_mismatch_as_evidence() -> None:
    artifacts = {
        "gs://east/digest.json": _report("gs://east/cache"),
        "gs://europe/digest.json": _report("gs://europe/cache", block_sha256="b" * 64),
    }

    def read(path: str) -> tuple[dict[str, object], str]:
        report = artifacts[path]
        return report, hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()

    reports = build_diagnostic_reports(_manifest(), artifact_reader=read)

    assert reports["component"]["status"] == "mismatch"
    assert reports["component"]["block_mismatch_count"] == 1


def test_build_diagnostic_reports_rejects_incomparable_artifacts() -> None:
    artifacts = {
        "gs://east/digest.json": _report("gs://east/cache"),
        "gs://europe/digest.json": _report("gs://europe/cache", selected_rows=3),
    }

    def read(path: str) -> tuple[dict[str, object], str]:
        return artifacts[path], "artifact"

    manifest = _manifest()
    cast(list[dict[str, object]], manifest["jobs"])[1]["expected_rows"] = 3
    with pytest.raises(ValueError, match="incomparable"):
        build_diagnostic_reports(manifest, artifact_reader=read)


def test_build_acceptance_reports_requires_equivalent_zero_exclusion_payloads() -> None:
    artifacts = {
        "gs://east/digest.json": _report("gs://east/cache"),
        "gs://europe/digest.json": _report("gs://europe/cache"),
    }

    def read(path: str) -> tuple[dict[str, object], str]:
        report = artifacts[path]
        return report, hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()

    reports = build_acceptance_reports(_acceptance_manifest(), artifact_reader=read)
    assert reports["component"]["equivalent"] is True
    assert reports["component"]["exclusion_gate_passes"] is True

    artifacts["gs://europe/digest.json"] = _report("gs://europe/cache", block_sha256="b" * 64)
    with pytest.raises(ValueError, match="not equivalent"):
        build_acceptance_reports(_acceptance_manifest(), artifact_reader=read)


def test_build_reports_rejects_artifact_count_binding_drift() -> None:
    artifacts = {
        "gs://east/digest.json": _report("gs://east/cache"),
        "gs://europe/digest.json": _report("gs://europe/cache"),
    }
    europe_binding = cast(dict[str, object], artifacts["gs://europe/digest.json"]["binding"])
    europe_binding["expected_rows"] = 3

    def read(path: str) -> tuple[dict[str, object], str]:
        return artifacts[path], "artifact"

    with pytest.raises(ValueError, match="expected_rows differs"):
        build_acceptance_reports(_acceptance_manifest(), artifact_reader=read)


def test_build_acceptance_reports_rejects_manifest_or_artifact_exclusions() -> None:
    artifacts = {
        "gs://east/digest.json": _report("gs://east/cache"),
        "gs://europe/digest.json": _report("gs://europe/cache"),
    }

    def read(path: str) -> tuple[dict[str, object], str]:
        return artifacts[path], "artifact"

    manifest_with_exclusion = _acceptance_manifest()
    cast(list[dict[str, object]], manifest_with_exclusion["jobs"])[0]["excluded_shards"] = ["part-00000"]
    with pytest.raises(ValueError, match="manifest has shard exclusions"):
        build_acceptance_reports(manifest_with_exclusion, artifact_reader=read)

    europe_binding = cast(dict[str, object], artifacts["gs://europe/digest.json"]["binding"])
    europe_binding["excluded_shards"] = [{"name": "part-00000"}]
    with pytest.raises(ValueError, match="artifact has shard exclusions"):
        build_acceptance_reports(_acceptance_manifest(), artifact_reader=read)
