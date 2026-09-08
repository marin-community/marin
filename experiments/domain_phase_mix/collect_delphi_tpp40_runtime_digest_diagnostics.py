# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec>=2025.7.0", "gcsfs>=2025.7.0"]
# ///

"""Collect paired Delphi TPP40 runtime-cache diagnostic digest reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from experiments.domain_phase_mix.compare_delphi_tpp40_runtime_digests import (
    ACCEPTANCE_MODE,
    DIAGNOSTIC_MODE,
    compare_digest_reports,
    read_digest_artifact,
)
from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import ALGORITHM

ArtifactReader = Callable[[str], tuple[dict[str, object], str]]


MANIFEST_COMPARISON_MODES = {
    "acceptance": ACCEPTANCE_MODE,
    "diagnostic_only": DIAGNOSTIC_MODE,
}


def _comparison_mode(manifest: dict[str, object]) -> str:
    if manifest.get("algorithm") != ALGORITHM or manifest.get("mode") not in MANIFEST_COMPARISON_MODES:
        raise ValueError("Runtime-digest manifest has the wrong algorithm or mode")
    return MANIFEST_COMPARISON_MODES[str(manifest["mode"])]


def _jobs_by_component(manifest: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    _comparison_mode(manifest)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not all(isinstance(job, dict) for job in jobs):
        raise ValueError("Runtime-digest manifest lacks a job list")

    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for job in jobs:
        component = job.get("component")
        region_key = job.get("region_key")
        output = job.get("output")
        if not isinstance(component, str) or region_key not in {"east5", "europe"} or not isinstance(output, str):
            raise ValueError("Diagnostic manifest has a malformed job")
        regional_jobs = grouped.setdefault(component, {})
        if region_key in regional_jobs:
            raise ValueError(f"Diagnostic manifest repeats {component!r}/{region_key}")
        regional_jobs[region_key] = job
    for component, regional_jobs in grouped.items():
        if set(regional_jobs) != {"east5", "europe"}:
            raise ValueError(f"Diagnostic manifest does not pair both regions for {component!r}")
    return grouped


def build_reports(
    manifest: dict[str, object],
    *,
    artifact_reader: ArtifactReader = read_digest_artifact,
) -> dict[str, dict[str, object]]:
    mode = _comparison_mode(manifest)
    reports: dict[str, dict[str, object]] = {}
    for component, regional_jobs in sorted(_jobs_by_component(manifest).items()):
        regional_artifacts: dict[str, dict[str, object]] = {}
        regional_digests: dict[str, dict[str, object]] = {}
        for region_key in ("east5", "europe"):
            job = regional_jobs[region_key]
            output = job["output"]
            assert isinstance(output, str)
            digest, artifact_sha256 = artifact_reader(output)
            binding = digest.get("binding")
            if not isinstance(binding, dict) or binding.get("cache_path") != job.get("cache_path"):
                raise ValueError(f"Diagnostic artifact cache path differs for {component!r}/{region_key}")
            for field in ("expected_rows", "expected_tokens"):
                expected_value = job.get(field)
                if not isinstance(expected_value, int) or binding.get(field) != expected_value:
                    raise ValueError(f"Diagnostic artifact {field} differs for {component!r}/{region_key}")
            if mode == ACCEPTANCE_MODE:
                manifest_exclusions = job.get("excluded_shards")
                if not isinstance(manifest_exclusions, (list, tuple)) or manifest_exclusions:
                    raise ValueError(f"Acceptance manifest has shard exclusions for {component!r}/{region_key}")
                if binding.get("excluded_shards") != []:
                    raise ValueError(f"Acceptance artifact has shard exclusions for {component!r}/{region_key}")
            regional_digests[region_key] = digest
            regional_artifacts[region_key] = {"path": output, "sha256": artifact_sha256}

        comparison = compare_digest_reports(
            regional_digests["east5"],
            regional_digests["europe"],
            mode=mode,
        )
        if comparison["status"] == "incomparable":
            raise ValueError(f"Runtime-digest artifacts are incomparable for {component!r}: {comparison}")
        if mode == ACCEPTANCE_MODE and comparison["equivalent"] is not True:
            raise ValueError(f"Acceptance artifacts are not equivalent for {component!r}: {comparison}")
        comparison["artifacts"] = regional_artifacts
        reports[component] = comparison
    return reports


def build_diagnostic_reports(
    manifest: dict[str, object],
    *,
    artifact_reader: ArtifactReader = read_digest_artifact,
) -> dict[str, dict[str, object]]:
    if manifest.get("mode") != "diagnostic_only":
        raise ValueError("Expected a diagnostic-only manifest")
    return build_reports(manifest, artifact_reader=artifact_reader)


def build_acceptance_reports(
    manifest: dict[str, object],
    *,
    artifact_reader: ArtifactReader = read_digest_artifact,
) -> dict[str, dict[str, object]]:
    if manifest.get("mode") != "acceptance":
        raise ValueError("Expected an acceptance manifest")
    return build_reports(manifest, artifact_reader=artifact_reader)


def write_reports(
    reports: dict[str, dict[str, object]],
    output_dir: Path,
    *,
    mode: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for component, report in sorted(reports.items()):
        filename = component.replace("/", "_") + ".json"
        (output_dir / filename).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        summary_rows.append(
            {
                "component": component,
                "status": report["status"],
                "payload_matches": report["payload_matches"],
                "block_mismatch_count": report["block_mismatch_count"],
                "selected_rows": report["selected_rows"],
                "selected_tokens": report["selected_tokens"],
            }
        )
    summary: dict[str, object] = {
        "algorithm": ALGORITHM,
        "mode": mode,
        "components": summary_rows,
    }
    (output_dir / "index.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_diagnostic_reports(reports: dict[str, dict[str, object]], output_dir: Path) -> dict[str, object]:
    return write_reports(reports, output_dir, mode=DIAGNOSTIC_MODE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Diagnostic manifest is not a JSON object")
    mode = _comparison_mode(manifest)
    reports = build_reports(manifest)
    summary = write_reports(reports, args.output_dir, mode=mode)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
