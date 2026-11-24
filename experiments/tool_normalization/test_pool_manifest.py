# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GLM pool manifest client (2026-11-24 serving handoff contract)."""

import json
import time
from pathlib import Path

import pytest

from experiments.tool_normalization.pool_manifest import (
    PoolManifestError,
    check_status,
    load_pool,
    refresh_pool,
    split_slots,
)


def _write_pool(tmp_path: Path, entries: list[dict], status_overrides: dict | None = None,
                credentials: dict[str, str] | None = None) -> str:
    """Materialize a manifest/status/credential set under file:// refs."""
    credentials = credentials or {}
    for i, entry in enumerate(entries):
        (tmp_path / f"key-{i}.jwt").write_text(credentials.get(f"key-{i}", f"jwt-{i}-token"))
        entry = dict(entry)
        entry["api_key_ref"] = str(tmp_path / f"key-{i}.jwt")
        entries[i] = entry
    (tmp_path / "manifest.json").write_text(json.dumps(entries))
    status = {
        "model": "glm-5.3",
        "cluster": "cw-us-east-08a",
        "healthy_entry_count": len(entries),
        "per_engine_concurrency": 24,
        "total_concurrency": 24 * len(entries),
        "validated_at": int(time.time()),
        "proof_ref": "s3://bucket/proof.json",
        "sustained_ref": "s3://bucket/sustained.json",
    }
    status.update(status_overrides or {})
    (tmp_path / "status.json").write_text(json.dumps(status))
    return f"file://{tmp_path}/manifest.json"


def test_split_slots_exact_division() -> None:
    # The full-32 fleet geometry: 768 fleet-wide across 32 pods = 24 each.
    assert [split_slots(768, 32, i) for i in range(32)] == [24] * 32


def test_split_slots_remainder_goes_to_first_pods() -> None:
    assert split_slots(10, 3, 0) == 4
    assert split_slots(10, 3, 1) == 3
    assert split_slots(10, 3, 2) == 3


def test_split_slots_zero_share_pods_pause() -> None:
    # 24 slots across 32 pods: a floor of one per pod would violate the aggregate cap.
    shares = [split_slots(24, 32, i) for i in range(32)]
    assert shares == [1] * 24 + [0] * 8
    assert sum(shares) == 24


def test_split_slots_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        split_slots(24, 0, 0)
    with pytest.raises(ValueError):
        split_slots(24, 4, 4)


def test_check_status_enforces_contract() -> None:
    good = {
        "model": "glm-5.3",
        "healthy_entry_count": 2,
        "per_engine_concurrency": 24,
        "total_concurrency": 48,
        "validated_at": 1,
        "proof_ref": "s3://x",
        "sustained_ref": "s3://y",
    }
    check_status(good, 2, "glm-5.3")
    with pytest.raises(PoolManifestError, match="missing"):
        check_status({k: v for k, v in good.items() if k != "proof_ref"}, 2)
    with pytest.raises(PoolManifestError, match="validated_at"):
        check_status({**good, "validated_at": 0}, 2)
    with pytest.raises(PoolManifestError, match="mismatch"):
        check_status(good, 3)
    with pytest.raises(PoolManifestError, match="inconsistent"):
        check_status({**good, "total_concurrency": 47}, 2)
    with pytest.raises(PoolManifestError, match="model"):
        check_status(good, 2, "other-model")


def test_load_pool_happy_path(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""},
               {"base_url": "http://entry-b/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries, credentials={"key-0": "jwt-a", "key-1": "jwt-b"})
    plan = load_pool(ref, tmp_path / "work", expected_model="glm-5.3", expected_pods=2, pod_index=1)
    assert [e.base_url for e in plan.entries] == ["http://entry-a/v1", "http://entry-b/v1"]
    assert plan.entries[0].api_key_file.read_text() == "jwt-a"
    assert plan.entries[1].api_key_file.read_text() == "jwt-b"
    assert plan.request_slots == plan.total_concurrency // 2
    assert plan.rr_offset == 1
    assert plan.model == "glm-5.3"
    # Credential files must be private.
    assert (plan.entries[0].api_key_file.stat().st_mode & 0o777) == 0o600


def test_load_pool_fails_fast_on_empty_credential(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries, credentials={"key-0": ""})
    with pytest.raises(PoolManifestError, match="EMPTY"):
        load_pool(ref, tmp_path / "work")


def test_load_pool_rejects_non_list_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"entries": []}))
    (tmp_path / "status.json").write_text("{}")
    with pytest.raises(PoolManifestError, match="non-empty entry list"):
        load_pool(f"file://{tmp_path}/manifest.json", tmp_path / "work")


def test_load_pool_rejects_entry_missing_fields(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps([{"base_url": "http://x/v1"}]))
    (tmp_path / "status.json").write_text("{}")
    with pytest.raises(PoolManifestError, match="base_url/api_key_ref"):
        load_pool(f"file://{tmp_path}/manifest.json", tmp_path / "work")


def test_load_pool_rejects_gate_failures(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries, status_overrides={"validated_at": 0})
    with pytest.raises(PoolManifestError, match="gates not passed"):
        load_pool(ref, tmp_path / "work")


def test_refresh_pool_adopts_rolled_generation(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries)
    first = load_pool(ref, tmp_path / "work", expected_pods=2, pod_index=0)
    # The serving side rolls the generation: a second entry joins and the budget doubles.
    _write_pool(tmp_path, [
        {"base_url": "http://entry-a/v1", "api_key_ref": ""},
        {"base_url": "http://entry-b/v1", "api_key_ref": ""},
    ], status_overrides={"validated_at": first.validated_at + 30})
    fresh = refresh_pool(ref, tmp_path / "work", first, expected_pods=2, pod_index=0)
    assert fresh is not None and len(fresh.entries) == 2
    assert fresh.request_slots == first.total_concurrency
    assert fresh.validated_at == first.validated_at + 30


def test_refresh_pool_pauses_on_inconsistent_publication(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries)
    plan = load_pool(ref, tmp_path / "work")
    # Mid-roll publication: the manifest has two entries but the stale status still
    # claims one healthy entry — a mixed generation must never be consumed.
    _write_pool(tmp_path, [
        {"base_url": "http://entry-a/v1", "api_key_ref": ""},
        {"base_url": "http://entry-b/v1", "api_key_ref": ""},
    ])
    stale_status = {
        "model": "glm-5.3",
        "cluster": "cw-us-east-08a",
        "healthy_entry_count": 1,
        "per_engine_concurrency": 24,
        "total_concurrency": 24,
        "validated_at": plan.validated_at,
        "proof_ref": "s3://bucket/proof.json",
        "sustained_ref": "s3://bucket/sustained.json",
    }
    (tmp_path / "status.json").write_text(json.dumps(stale_status))
    assert refresh_pool(ref, tmp_path / "work", plan) is None


def test_refresh_pool_pauses_on_zero_share(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries)
    # 24 budget across 32 pods: pod 24 owns zero slots and pauses.
    assert refresh_pool(ref, tmp_path / "work", None, expected_pods=32, pod_index=24) is None


def test_refresh_pool_pauses_on_transient_fetch_error(tmp_path: Path) -> None:
    entries = [{"base_url": "http://entry-a/v1", "api_key_ref": ""}]
    ref = _write_pool(tmp_path, entries)
    plan = load_pool(ref, tmp_path / "work")
    # Simulate a transient S3 blip: the manifest disappears.
    (tmp_path / "manifest.json").unlink()
    assert refresh_pool(ref, tmp_path / "work", plan) is None
