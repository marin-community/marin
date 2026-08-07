# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from tile_lifetime import ExpertParallelConfig

SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "artifacts" / "gb200_moe_v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoded_npz(directory: Path, filename: str) -> bytes:
    parts = sorted(directory.glob(f"{filename}.b64.part*"))
    assert parts
    encoded = "".join("".join(part.read_text().split()) for part in parts)
    return base64.b64decode(encoded, validate=True)


def _route_content_sha256(payload: bytes, array_order: list[str]) -> str:
    digest = hashlib.sha256()
    with np.load(io.BytesIO(payload), allow_pickle=False) as fixture:
        for name in array_order:
            array = np.ascontiguousarray(fixture[name])
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(array.dtype.str.encode())
            digest.update(b"\0")
            digest.update(struct.pack("<I", array.ndim))
            for dimension in array.shape:
                digest.update(struct.pack("<q", dimension))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _framed_tensor_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    digest = hashlib.sha256()
    encoded_dtype = dtype.encode()
    digest.update(struct.pack("<Q", len(encoded_dtype)))
    digest.update(encoded_dtype)
    digest.update(struct.pack("<Q", len(shape)))
    for dimension in shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(payload)
    return digest.hexdigest()


def test_gb200_snapshot_artifacts_match_manifest_and_cache() -> None:
    manifest = _json(SNAPSHOT_ROOT / "manifest.json")
    for relative_path, identity in manifest["artifacts"].items():
        artifact = SNAPSHOT_ROOT / relative_path
        assert artifact.stat().st_size == identity["bytes"]
        assert _sha256(artifact) == identity["sha256"]

    cache = _json(SNAPSHOT_ROOT / "benchmark_cache.json")
    selected_entries = []
    for entry in cache["entries"]:
        artifact = SNAPSHOT_ROOT / entry["artifact"]
        result = _json(artifact)
        samples = result["timing"][entry["phase"]]["rank_max_samples_ms"]
        assert len(samples) == entry["sample_count"]
        assert _sha256(artifact) == entry["artifact_sha256"]
        if entry["selection_status"] == "selected_confirmation":
            selected_entries.append(entry)
    assert len(selected_entries) >= 2
    assert {entry["candidate"]["fingerprint_sha256"] for entry in selected_entries} == {
        cache["selected_candidate_fingerprint_sha256"]
    }


def test_gb200_route_fixture_and_selected_plan_match_snapshot_contract() -> None:
    identity = _json(SNAPSHOT_ROOT / "fixtures" / "mok-route-fixture-content-identity.json")
    route_fixture = _decoded_npz(
        SNAPSHOT_ROOT / "fixtures", "mok_routes_t2048_e384_k6_seed1234_torch2.10-reserialized.npz"
    )
    assert hashlib.sha256(route_fixture).hexdigest() == identity["npz_container_sha256"]["reserialized"]
    assert (
        _route_content_sha256(route_fixture, identity["content_hash_framing"]["array_order"])
        == identity["canonical_tensor_content_sha256"]
    )

    candidate_space = _json(SNAPSHOT_ROOT / "candidate_space.json")
    selected = candidate_space["selected"]
    config = ExpertParallelConfig(expert_parallel_size=4)
    assert config.exchange_workers == selected["exchange_workers"]
    assert config.gate_up_layout.value == selected["gate_up_layout"]
    assert config.overlap_policy.value == selected["overlap_policy"]
    assert config.materialization_schedule.value == selected["materialization_schedule"]


def test_gb200_replay_preserves_raw_samples_hashes_fixtures_and_clocks() -> None:
    replay_root = SNAPSHOT_ROOT / "raw" / "schema2"
    distributed_runs = sorted(replay_root.glob("deepep-mok-distributed-*.json"))
    assert len(distributed_runs) == 2
    for run_path in distributed_runs:
        result = _json(run_path)
        assert result["source"]["shuttle_revision"] == "3dd61fad063bae54ac5e337d8f1657264011d6ff"
        assert result["source"]["deepep_commit"] == "7febc6e25660af0f54d95dd781ecdcd62265ecca"
        for timing in result["timing"].values():
            assert len(timing["rank_max_samples_ms"]) == 50
            assert [len(samples) for samples in timing["per_rank_samples_ms"]] == [50] * 4
        for record in result["rank_records"]:
            correctness = record["correctness"]
            assert {
                correctness["sequential_output_sha256"],
                correctness["overlap_output_sha256"],
                correctness["repeated_overlap_output_sha256"],
                correctness["coarse_output_sha256"],
            } == {correctness["overlap_output_sha256"]}
            semantic = record["independent_semantic_reference"]
            fixture = semantic["fixture"]
            fixture_payload = _decoded_npz(SNAPSHOT_ROOT / "fixtures" / "schema2", Path(fixture["path"]).name)
            assert hashlib.sha256(fixture_payload).hexdigest() == fixture["sha256"]
            with np.load(io.BytesIO(fixture_payload), allow_pickle=False) as arrays:
                generated = np.ascontiguousarray(arrays["generated_bf16_bits"])
                reference = np.ascontiguousarray(arrays["reference_bf16_bits"])
                assert (
                    _framed_tensor_sha256("torch.bfloat16", generated.shape, generated.tobytes(order="C"))
                    == semantic["generated_output_sha256"]
                )
                assert (
                    _framed_tensor_sha256("torch.bfloat16", reference.shape, reference.tobytes(order="C"))
                    == semantic["reference_output_sha256"]
                )

        telemetry = result["environment"]["gpu_telemetry"]
        snapshots = [telemetry["initial"], telemetry["final"]]
        for phase in telemetry["by_phase"].values():
            snapshots.extend((phase["before"], phase["after"]))
        for snapshot in snapshots:
            assert len(snapshot["gpus"]) == 4
            assert {gpu["clocks.current.sm"] for gpu in snapshot["gpus"]} == {"1950"}
            assert {gpu["clocks.current.memory"] for gpu in snapshot["gpus"]} == {"3996"}
            assert {gpu["power.limit"] for gpu in snapshot["gpus"]} == {"1200.00"}

    mok = _json(replay_root / "mok-forward-sweep-selected.json")
    assert mok["source"]["mok_commit"] == "3e1cf43ab93ad040afed52a45ab03cb490ffe4be"
    assert mok["protocol"] == {"repeats": 50, "selection_metric": "median rank-max ms", "warmups": 100}
    assert len(mok["candidates"]) == 1
    timing = mok["candidates"][0]["timing"]
    assert len(timing["rank_max_samples_ms"]) == 50
    assert [len(samples) for samples in timing["per_rank_samples_ms"]] == [50] * 4
    assert len(set(mok["candidates"][0]["per_rank_output_sha256"])) == 4
