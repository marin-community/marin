# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote GB200 MoE measurements into a content-addressed snapshot."""

import argparse
import base64
import json
import shutil
import statistics
import textwrap
from pathlib import Path
from typing import Any

from benchmark_metadata import canonical_json_sha256, file_sha256

SCHEMA1_DISTRIBUTED_GLOB = "deepep-mok-distributed-*.json"
ROUTE_FIXTURE = "mok_routes_t2048_e384_k6_seed1234_torch2.10-reserialized.npz"
ROUTE_IDENTITY = "mok-route-fixture-content-identity.json"
REPLAY_PROVENANCE = "replay-provenance.txt"
BASE64_CHUNK_CHARACTERS = 384 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--shuttle-tag", required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _copy_artifacts(source: Path, destination: Path, names: list[str]) -> list[Path]:
    copied = []
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_path = source / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination_path = destination / name
        shutil.copyfile(source_path, destination_path)
        copied.append(destination_path)
    return copied


def _copy_npz_as_base64_parts(source: Path, destination: Path, name: str) -> list[Path]:
    source_path = source / name
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    encoded = base64.b64encode(source_path.read_bytes()).decode()
    destination.mkdir(parents=True, exist_ok=True)
    parts = []
    for part_index, start in enumerate(range(0, len(encoded), BASE64_CHUNK_CHARACTERS)):
        part = destination / f"{name}.b64.part{part_index:02d}"
        chunk = encoded[start : start + BASE64_CHUNK_CHARACTERS]
        part.write_text("\n".join(textwrap.wrap(chunk, width=120)) + "\n")
        parts.append(part)
    return parts


def _historical_names(root: Path) -> list[str]:
    names = [path.name for path in sorted(root.glob("*.json"))]
    if not names:
        raise ValueError(f"no historical JSON artifacts found under {root}")
    return names


def _distributed_candidate(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    schedule = result["schedule"]
    filename = path.name
    if "concat" in filename:
        layout = "concatenated_e_2i_k"
    elif "separate" in filename or filename.startswith("deepep-mok-distributed-sms"):
        layout = "separate_e_i_k"
    else:
        layout = schedule.get("gate_up_layout", "separate_e_i_k")
    candidate = {
        "exchange_implementation": "deepep",
        "segmented_contraction_implementation": "standalone_sm100_grouped_gemm",
        "exchange_workers": schedule["deepep_sms"],
        "gate_up_layout": layout,
        "overlap_policy": "shared_with_async_dispatch",
        "materialization_schedule": "tile_flow_boundaries",
    }
    return {**candidate, "fingerprint_sha256": canonical_json_sha256(candidate)}


def _cache_entries(raw_root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(raw_root.glob("deepep-mok-distributed-*.json")):
        result = _read_json(path)
        candidate = _distributed_candidate(path, result)
        phases = (
            ("full_overlap_shared_with_async_dispatch", "shared_with_async_dispatch", "tile_flow_boundaries"),
            ("full_sequential", "sequential", "tile_flow_boundaries"),
            ("full_coarse_materialized_sequential", "sequential", "coarse_activation_boundaries"),
        )
        for phase, overlap, materialization in phases:
            if phase not in result["timing"]:
                continue
            phase_candidate = {
                **candidate,
                "overlap_policy": overlap,
                "materialization_schedule": materialization,
            }
            phase_candidate["fingerprint_sha256"] = canonical_json_sha256(
                {key: value for key, value in phase_candidate.items() if key != "fingerprint_sha256"}
            )
            timing = result["timing"][phase]
            entries.append(
                {
                    "cache_key_sha256": canonical_json_sha256(
                        {
                            "workload": "gb200_moe_t2048_e384_top6_h7168_i3072_bf16",
                            "candidate": phase_candidate,
                            "artifact_sha256": file_sha256(path),
                            "phase": phase,
                        }
                    ),
                    "candidate": phase_candidate,
                    "phase": phase,
                    "artifact": str(path.relative_to(raw_root.parent.parent)),
                    "artifact_sha256": file_sha256(path),
                    "run_schema_version": result["schema_version"],
                    "sample_count": len(timing["rank_max_samples_ms"]),
                    "median_rank_max_ms": timing["median_ms"],
                    "correctness": "passed",
                }
            )
    return entries


def _select_candidate(historical_entries: list[dict[str, Any]]) -> dict[str, Any]:
    competitive = [
        entry
        for entry in historical_entries
        if entry["candidate"]["overlap_policy"] == "shared_with_async_dispatch"
        and entry["candidate"]["materialization_schedule"] == "tile_flow_boundaries"
    ]
    if not competitive:
        raise ValueError("no correct overlapped tile-flow candidates were measured")
    best = min(
        competitive,
        key=lambda entry: (
            entry["median_rank_max_ms"],
            entry["candidate"]["exchange_workers"],
            entry["candidate"]["fingerprint_sha256"],
        ),
    )
    fingerprint = best["candidate"]["fingerprint_sha256"]
    confirmation_medians = [
        entry["median_rank_max_ms"] for entry in competitive if entry["candidate"]["fingerprint_sha256"] == fingerprint
    ]
    return {
        **best["candidate"],
        "historical_confirmation_count": len(confirmation_medians),
        "historical_median_of_medians_ms": statistics.median(confirmation_medians),
    }


def _candidate_space(cache_entries: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any]:
    measured_workers = sorted(
        {
            entry["candidate"]["exchange_workers"]
            for entry in cache_entries
            if entry["candidate"]["overlap_policy"] == "shared_with_async_dispatch"
        }
    )
    return {
        "schema_version": 1,
        "workload": "gb200_moe_t2048_e384_top6_h7168_i3072_bf16",
        "selection_objective": "minimum median rank-maximum end-to-end latency among correct candidates",
        "selection_dataset": "raw/schema1 historical search runs; schema2 is a reproducibility replay only",
        "tie_breaker": "lower exchange-worker count, then lexicographic candidate fingerprint",
        "dimensions": {
            "exchange_implementation": ["deepep", "ragged_all_to_all"],
            "segmented_contraction_implementation": ["standalone_sm100_grouped_gemm", "ragged_dot"],
            "exchange_workers": [12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96],
            "gate_up_layout": ["concatenated_e_2i_k", "interleaved_e_2i_k", "separate_e_i_k"],
            "overlap_policy": ["shared_with_async_dispatch", "sequential"],
            "materialization_schedule": ["tile_flow_boundaries", "coarse_activation_boundaries"],
        },
        "measured": {
            "exchange_workers": measured_workers,
            "distributed_gate_up_layouts": ["concatenated_e_2i_k", "separate_e_i_k"],
            "overlap_policies": ["shared_with_async_dispatch", "sequential"],
            "materialization_schedules": sorted(
                {entry["candidate"]["materialization_schedule"] for entry in cache_entries}
            ),
        },
        "unmeasured": [
            {
                "dimension": "gate_up_layout",
                "value": "interleaved_e_2i_k",
                "reason": "receiver-local validation passed, but no distributed replay was run",
            }
        ],
        "failed_or_pruned": [
            {
                "candidate": "native JAX ragged_all_to_all",
                "reason": "segmentation fault on first execution on the measured JAX 0.11 toolchain",
            },
            {
                "candidate": "XLA ragged_dot",
                "reason": "95.100 ms component result was pruned before distributed composition",
            },
        ],
        "search_protocol": [
            "sweep DeepEP communication workers with the initial separate gate/up layout",
            "confirm the latency turn at 96 workers",
            "compare concatenated and separate gate/up at 56 and 80 workers",
            "repeat the lowest-latency correct candidate",
            "retain sequential and coarse-materialization phases as ablations, not selection candidates",
        ],
        "selected": selected,
    }


def _manifest(
    output_root: Path,
    *,
    shuttle_revision: str,
    shuttle_tag: str,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(output_root)): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(artifact_paths)
    }
    return {
        "schema_version": 1,
        "snapshot": {
            "name": "Shuttle distributed BF16 MoE GB200 proof of life",
            "shuttle_revision": shuttle_revision,
            "shuttle_tag": shuttle_tag,
        },
        "sources": {
            "marin_base_commit": "c26285a61654a9e6a9029cfdb3d018badc35d71c",
            "coda_commit": "8fa88065e541f6a5b52fb400d94d4be02f18c543",
            "coda_quack_commit": "02c7f69881737731173a6a009aeb6f032e449b61",
            "consumer_prologue_quack_commit": "84ef91df9bec87c7e4938517234fafb07ef844dd",
            "consumer_prologue_patch_sha256": "40318b9b390e111c38f4838a50cf8913695c9f94142122b374bf09c220cfd9a1",
            "flash_attention_commit": "3fa810570e17bb4354155bdb71d826eca6079208",
            "mixture_of_kittens_commit": "3e1cf43ab93ad040afed52a45ab03cb490ffe4be",
            "thunderkittens_commit": "1c3920d993404dd49a6d4c7267ea11d583bd5c68",
            "deepep_commit": "7febc6e25660af0f54d95dd781ecdcd62265ecca",
            "cutlass_dsl": "4.6.0",
        },
        "toolchain": {
            "python": "3.12.13",
            "torch": "2.10.0+cu130",
            "torch_cuda": "13.0",
            "cuda_toolkit_package": "13.0.2",
            "cuda_runtime_package": "13.0.96",
            "nvcc": "13.0.88",
            "ptxas": "13.0.88",
            "cccl": "13.0.85",
            "cuda_crt": "13.0.88",
            "nvvm": "13.0.88",
            "nccl": "2.28.9",
            "nvidia_driver": "595.71.05",
            "target": "sm100a",
        },
        "hardware": {
            "world_size": 4,
            "gpu": "NVIDIA GB200",
            "compute_capability": "10.0",
            "placement": "one four-GPU low-priority GB200 tray",
            "replay_reservation": {
                "priority": "batch",
                "cpu_cores": 8,
                "host_memory_gb": 64,
                "local_disk_gb": 100,
            },
            "historical_clock_policy": "cluster default; clocks and power telemetry were not captured",
            "replay_clock_policy": {
                "policy": "cluster default, application clocks deprecated and unpinned",
                "pre_benchmark_idle_sm_clock_mhz": 120,
                "timed_phase_telemetry_sm_clock_mhz": 1950,
                "memory_clock_mhz": 3996,
                "advertised_max_sm_clock_mhz": 2062,
                "power_limit_watts": 1200,
                "sampled_power_draw_watts": [199.83, 757.32],
                "detail": "raw/schema2 replay records contain per-GPU UUID and per-phase telemetry",
            },
        },
        "workload": {
            "dtype": "bfloat16",
            "accumulation": "FP32 GEMM and fixed route-slot FP32 merge",
            "local_tokens": 2048,
            "global_experts": 384,
            "local_experts": 96,
            "top_k": 6,
            "hidden_size": 7168,
            "intermediate_size": 3072,
            "padding_quantum": 256,
            "route_seed_by_rank": "1234 + rank in official MoK input generation",
            "benchmark_seed": 0,
            "warmups": 10,
            "measured_iterations": 50,
            "selection_metric": "median rank-maximum CUDA-event latency",
        },
        "oracle": {
            "implementation": "Mixture-of-Kittens BF16 forward",
            "communication_sms": 20,
            "minibatch_size": 2048,
            "macrobatch_size": 65536,
            "historical_latency_ms": 3.613,
            "historical_tflops": 524.2,
            "replay_warmups": 100,
            "replay_iterations": 50,
            "replay_latency_ms": 3.56169593334198,
            "replay_tflops": 531.7917680184346,
            "raw_replay": "raw/schema2/mok-forward-sweep-selected.json",
        },
        "correctness": {
            "stablehlo_fixture": "../../../tests/fixtures/stablehlo/moe_region_v1_14_1.mlir.bc.b64",
            "route_tensor_content_sha256": "f1b5d8b3a53372eca228261b48b7ad9cfe925f1f8083f9cae07f9a24713f6908",
            "merge_order": "ascending route slot within owner, then ascending owner rank; no atomics",
            "schema1_limit": "bitwise parity was recorded, but output tensors were not hashed",
            "schema2_output_hashes": "stored per rank in replay run records",
        },
        "artifacts": artifacts,
    }


def main() -> None:
    args = _parser().parse_args()
    historical_root = args.historical_root.resolve()
    output_root = args.output_root.resolve()
    schema1_root = output_root / "raw" / "schema1"
    schema2_root = output_root / "raw" / "schema2"
    fixture_root = output_root / "fixtures"
    for generated_root in (schema1_root, schema2_root, fixture_root):
        if generated_root.exists():
            shutil.rmtree(generated_root)

    artifact_paths = _copy_artifacts(historical_root, schema1_root, _historical_names(historical_root))
    artifact_paths.extend(_copy_artifacts(historical_root, fixture_root, [ROUTE_IDENTITY]))
    artifact_paths.extend(_copy_npz_as_base64_parts(historical_root, fixture_root, ROUTE_FIXTURE))
    if args.replay_root is not None:
        replay_root = args.replay_root.resolve()
        replay_names = [path.name for path in sorted(replay_root.glob("*.json"))]
        artifact_paths.extend(_copy_artifacts(replay_root, schema2_root, replay_names))
        artifact_paths.extend(_copy_artifacts(replay_root, schema2_root, [REPLAY_PROVENANCE]))
        semantic_fixture_names = [path.name for path in sorted(replay_root.glob("semantic-fixture-*.npz"))]
        for fixture_name in semantic_fixture_names:
            artifact_paths.extend(_copy_npz_as_base64_parts(replay_root, fixture_root / "schema2", fixture_name))

    historical_entries = _cache_entries(schema1_root)
    replay_entries = _cache_entries(schema2_root)
    selected = _select_candidate(historical_entries)
    selected_fingerprint = selected["fingerprint_sha256"]
    cache_entries = historical_entries + replay_entries
    for entry in cache_entries:
        entry["selection_status"] = (
            "selected_confirmation"
            if entry["candidate"]["fingerprint_sha256"] == selected_fingerprint
            else "measured_ablation"
        )
    candidate_path = output_root / "candidate_space.json"
    cache_path = output_root / "benchmark_cache.json"
    _write_json(candidate_path, _candidate_space(cache_entries, selected))
    _write_json(
        cache_path,
        {
            "schema_version": 1,
            "workload_fingerprint_sha256": canonical_json_sha256(
                {
                    "shape": [4, 2048, 384, 96, 6, 7168, 3072],
                    "dtype": "bfloat16",
                    "route_fixture": "f1b5d8b3a53372eca228261b48b7ad9cfe925f1f8083f9cae07f9a24713f6908",
                    "gpu": "NVIDIA GB200",
                    "driver": "595.71.05",
                    "cuda": "13.0.88",
                }
            ),
            "entries": cache_entries,
            "selected_candidate_fingerprint_sha256": selected_fingerprint,
            "selection_note": (
                "Only measurements listed in entries participate in selection; failed, pruned, and unmeasured "
                "alternatives are recorded in candidate_space.json"
            ),
        },
    )
    artifact_paths.extend((candidate_path, cache_path))
    manifest_path = output_root / "manifest.json"
    _write_json(
        manifest_path,
        _manifest(
            output_root,
            shuttle_revision=args.shuttle_revision,
            shuttle_tag=args.shuttle_tag,
            artifact_paths=artifact_paths,
        ),
    )


if __name__ == "__main__":
    main()
