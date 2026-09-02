# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec>=2025.7.0", "gcsfs>=2025.7.0"]
# ///

"""Compare two region-local logical runtime-cache digest artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fsspec

from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import ALGORITHM, validate_digest_artifact

ACCEPTANCE_MODE = "acceptance"
DIAGNOSTIC_MODE = "diagnostic"
MODES = (ACCEPTANCE_MODE, DIAGNOSTIC_MODE)
MAX_REPORTED_BLOCK_MISMATCHES = 20


def read_digest_artifact(path: str) -> tuple[dict[str, object], str]:
    with fsspec.open(path, "rb") as handle:
        payload = handle.read()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _binding(report: dict[str, object]) -> dict[str, object]:
    binding = report.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Digest artifact lacks a binding object")
    return binding


def _blocks(report: dict[str, object]) -> list[dict[str, object]]:
    blocks = report.get("blocks")
    if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
        raise ValueError("Digest artifact lacks a block list")
    return blocks


def _excluded_shards(binding: dict[str, object]) -> list[dict[str, object]]:
    excluded = binding.get("excluded_shards")
    if not isinstance(excluded, list) or not all(isinstance(item, dict) for item in excluded):
        raise ValueError("Digest binding lacks an excluded_shards list")
    return excluded


def compare_digest_reports(
    east5: dict[str, object],
    europe: dict[str, object],
    *,
    mode: str,
) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"Unknown comparison mode: {mode!r}")
    validate_digest_artifact(east5)
    validate_digest_artifact(europe)
    east5_binding = _binding(east5)
    europe_binding = _binding(europe)

    incomparable_fields = [
        field
        for field in ("algorithm", "block_rows", "selected_rows", "dtype", "field_names")
        if east5.get(field) != europe.get(field)
    ]
    if east5.get("algorithm") == europe.get("algorithm") and east5.get("algorithm") != ALGORITHM:
        incomparable_fields.append("unsupported_algorithm")
    provenance_mismatches = []
    if east5_binding["preprocessor_metadata_sha256"] != europe_binding["preprocessor_metadata_sha256"]:
        provenance_mismatches.append("preprocessor_metadata_sha256")

    east5_excluded = _excluded_shards(east5_binding)
    europe_excluded = _excluded_shards(europe_binding)
    zero_exclusion_payload = (
        not east5_excluded
        and not europe_excluded
        and not east5["excluded_row_ranges"]
        and not europe["excluded_row_ranges"]
        and east5["selected_rows"] == east5["source_rows"]
        and europe["selected_rows"] == europe["source_rows"]
    )
    exclusion_gate_passes = mode == DIAGNOSTIC_MODE or zero_exclusion_payload

    east5_blocks = _blocks(east5)
    europe_blocks = _blocks(europe)
    block_shape_matches = len(east5_blocks) == len(europe_blocks) and all(
        east5_block.get("output_row_start") == europe_block.get("output_row_start")
        and east5_block.get("output_row_stop") == europe_block.get("output_row_stop")
        for east5_block, europe_block in zip(east5_blocks, europe_blocks, strict=False)
    )
    block_mismatches: list[dict[str, object]] = []
    if block_shape_matches:
        for east5_block, europe_block in zip(east5_blocks, europe_blocks, strict=True):
            if east5_block != europe_block:
                block_mismatches.append(
                    {
                        "output_row_start": east5_block["output_row_start"],
                        "output_row_stop": east5_block["output_row_stop"],
                        "east5_token_count": east5_block["token_count"],
                        "europe_token_count": europe_block["token_count"],
                        "east5_sha256": east5_block["sha256"],
                        "europe_sha256": europe_block["sha256"],
                    }
                )

    payload_matches = (
        not incomparable_fields
        and block_shape_matches
        and east5.get("selected_tokens") == europe.get("selected_tokens")
        and not block_mismatches
        and east5.get("logical_payload_sha256") == europe.get("logical_payload_sha256")
    )
    equivalent = payload_matches and not provenance_mismatches and exclusion_gate_passes
    if incomparable_fields or not block_shape_matches:
        status = "incomparable"
    elif equivalent:
        status = "equivalent"
    else:
        status = "mismatch"
    return {
        "status": status,
        "mode": mode,
        "equivalent": equivalent,
        "payload_matches": payload_matches,
        "exclusion_gate_passes": exclusion_gate_passes,
        "incomparable_fields": incomparable_fields,
        "provenance_mismatches": provenance_mismatches,
        "selected_rows": {"east5": east5.get("selected_rows"), "europe": europe.get("selected_rows")},
        "selected_tokens": {"east5": east5.get("selected_tokens"), "europe": europe.get("selected_tokens")},
        "excluded_shards": {"east5": east5_excluded, "europe": europe_excluded},
        "excluded_row_ranges": {
            "east5": east5["excluded_row_ranges"],
            "europe": europe["excluded_row_ranges"],
        },
        "cache_paths": {
            "east5": east5_binding["cache_path"],
            "europe": europe_binding["cache_path"],
        },
        "block_shape_matches": block_shape_matches,
        "block_mismatch_count": len(block_mismatches),
        "first_block_mismatches": block_mismatches[:MAX_REPORTED_BLOCK_MISMATCHES],
        "logical_payload_sha256": {
            "east5": east5.get("logical_payload_sha256"),
            "europe": europe.get("logical_payload_sha256"),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--east5", required=True)
    parser.add_argument("--europe", required=True)
    parser.add_argument("--mode", choices=MODES, default=ACCEPTANCE_MODE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    east5, east5_artifact_sha256 = read_digest_artifact(args.east5)
    europe, europe_artifact_sha256 = read_digest_artifact(args.europe)
    report = compare_digest_reports(east5, europe, mode=args.mode)
    report["artifacts"] = {
        "east5": {"path": args.east5, "sha256": east5_artifact_sha256},
        "europe": {"path": args.europe, "sha256": europe_artifact_sha256},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
